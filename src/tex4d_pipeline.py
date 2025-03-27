import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL
from PIL import Image
from einops import rearrange
import numpy as np
import torch
from torch import functional as F
import torchvision.transforms as transforms
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode
from diffusers import (
    StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler,
    AutoencoderKL, UNet2DConditionModel
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import (
    BaseOutput, numpy_to_pil, pt_to_pil, is_accelerate_available, is_accelerate_version, logging, replace_example_docstring
)
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from controlnet.multicontrolnet import MultiControlNetModel
from controlnet.controlnet import ControlNetModel
from model.ctrl_adapter import ControlNetAdapter
from model.ctrl_router import ControlNetRouter
from model.ctrl_helper import ControlNetHelper

from i2vgen_xl.pipelines.i2vgen_xl_controlnet_adapter_pipeline import I2VGenXLControlNetAdapterPipeline
from i2vgen_xl.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from .renderer.project import UVProjection as UVP
from .syncmvd.prompt import *
from .syncmvd.step import step_tex
from .utils import *

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")

# Background colors
color_constants = {"black": [-1, -1, -1], "white": [1, 1, 1], "maroon": [0, -1, -1],
			"red": [1, -1, -1], "olive": [0, 0, -1], "yellow": [1, 1, -1],
			"green": [-1, 0, -1], "lime": [-1 ,1, -1], "teal": [-1, 0, 0],
			"aqua": [-1, 1, 1], "navy": [-1, -1, 0], "blue": [-1, -1, 1],
			"purple": [0, -1 , 0], "fuchsia": [1, -1, 1]}
color_names = list(color_constants.keys())


# Used to generate depth or normal conditioning images
@torch.no_grad()
def get_conditioning_images(uvp, output_size, render_size=512, blur_filter=5, cond_type="normal"):
	verts, normals, depths, cos_maps, texels, fragments = uvp.render_geometry(image_size=render_size)
	conditional_images_list = []
	masks_list = []
	for normals, depths in zip(normals, depths):
		masks = normals[...,3][:,None,...]
		masks = Resize((output_size//8,)*2, antialias=True)(masks)
		normals_transforms = Compose([
			Resize((output_size,)*2, interpolation=InterpolationMode.BILINEAR, antialias=True),
			GaussianBlur(blur_filter, blur_filter//3+1)]
		)
		if cond_type == "normal":
			view_normals = uvp.decode_view_normal(normals).permute(0,3,1,2) *2 - 1
			conditional_images = normals_transforms(view_normals)
		# Some problem here, depth controlnet don't work when depth is normalized
		# But it do generate using the unnormalized form as below
		elif cond_type == "depth":
			view_depths = uvp.decode_normalized_depth(depths).permute(0,3,1,2)
			conditional_images = normals_transforms(view_depths)
		conditional_images_list.append(conditional_images)
		masks_list.append(masks)
	return conditional_images_list, masks_list


# Revert time 0 background to time t to composite with time t foreground
@torch.no_grad()
def composite_rendered_view(scheduler, backgrounds, foregrounds, masks, t):
	composited_images = []
	for i, (background, foreground, mask) in enumerate(zip(backgrounds, foregrounds, masks)):
		if t > 0:
			alphas_cumprod = scheduler.alphas_cumprod[t]
			noise = torch.normal(0, 1, background.shape, device=background.device)
			background = (1-alphas_cumprod) * noise + alphas_cumprod * background
		composited = foreground * mask + background * (1-mask)
		composited_images.append(composited)
	composited_tensor = torch.stack(composited_images)
	return composited_tensor


# Split into micro-batches to use less memory in each unet prediction
# But need more investigation on reducing memory usage
# Assume it has no possitive effect and use a large "max_batch_size" to skip splitting
def split_groups(attention_mask, max_batch_size, ref_view=[]):
	group_sets = []
	group = set()
	ref_group = set()
	idx = 0
	while idx < len(attention_mask):
		new_group = group | set([idx])
		new_ref_group = (ref_group | set(attention_mask[idx] + ref_view)) - new_group
		if len(new_group) + len(new_ref_group) <= max_batch_size:
			group = new_group
			ref_group = new_ref_group
			idx += 1
		else:
			assert len(group) != 0, "Cannot fit into a group"
			group_sets.append((group, ref_group))
			group = set()
			ref_group = set()
	if len(group)>0:
		group_sets.append((group, ref_group))

	group_metas = []
	for group, ref_group in group_sets:
		in_mask = sorted(list(group | ref_group))
		out_mask = []
		group_attention_masks = []
		for idx in in_mask:
			if idx in group:
				out_mask.append(in_mask.index(idx))
			group_attention_masks.append([in_mask.index(idxx) for idxx in attention_mask[idx] if idxx in in_mask])
		ref_attention_mask = [in_mask.index(idx) for idx in ref_view]
		group_metas.append([in_mask, out_mask, group_attention_masks, ref_attention_mask])

	return group_metas

'''

	MultiView-Diffusion Stable-Diffusion Pipeline
	Modified from a Diffusers StableDiffusionControlNetPipeline
	Just mimic the pipeline structure but did not follow any API convention

'''
def tensor2vid(video: torch.Tensor, processor: "VaeImageProcessor", output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs

class I2VGenXLPipelineOutput(BaseOutput):
    r"""
     Output class for image-to-video pipeline.

     Args:
         frames (`torch.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
             List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing denoised
     PIL image sequences of length `num_frames.` It can also be a NumPy array or Torch tensor of shape
    `(batch_size, num_frames, channels, height, width)`
    """

    frames: Union[torch.Tensor, np.ndarray, List[List[PIL.Image.Image]]]
    down_block_weights: List ### output controlnet weights if using multi-condition control
    mid_block_weights: List ### output controlnet weights if using multi-condition control

class Tex4DPipeline(I2VGenXLControlNetAdapterPipeline):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            image_encoder: CLIPVisionModelWithProjection,
            feature_extractor: CLIPImageProcessor,
            unet: I2VGenXLUNet,
            scheduler: DDIMScheduler,

            ### controlnet, adapter, helper, and router are newly added
            controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
            adapter: ControlNetAdapter,
            helper: ControlNetHelper,
            router: ControlNetRouter = None,

    ):
        # Here we use the same structure as the original pipeline
        super().__init__(vae, text_encoder, tokenizer, image_encoder, feature_extractor, unet, scheduler, controlnet, adapter, helper, router)

        # self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
        # self.enable_model_cpu_offload()
        # self.model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
        # self.enable_model_cpu_offload()
        # we define something needed by SyncMVD(rendering!) Under

    def initialize_pipeline(
            self,
            mesh_path=None,
            mesh_transform=None,
            mesh_autouv=None,
            camera_azims=None,
            camera_centers=None,
            top_cameras=False,
            down_cameras=False,
            ref_views=[],
            latent_size=None,
            render_rgb_size=None,
            texture_size=None,
            texture_rgb_size=None,

            max_batch_size=24,
            logging_config=None,
    ):
        # Make output dir
        output_dir = logging_config["output_dir"]

        self.result_dir = f"{output_dir}/results"
        self.intermediate_dir = f"{output_dir}/intermediate"
        self.first_key_frame = f"{output_dir}/first_key_frame"
        self.conditional_image = f"{output_dir}/conditional_image"

        dirs = [output_dir, self.result_dir, self.intermediate_dir, self.first_key_frame, self.conditional_image]
        for dir_ in dirs:
            if not os.path.isdir(dir_):
                os.mkdir(dir_)

        # Define the cameras for rendering
        self.camera_poses = []
        self.attention_mask = []
        self.centers = camera_centers

        cam_count = len(camera_azims)
        front_view_diff = 360
        back_view_diff = 360
        front_view_idx = 0
        back_view_idx = 0
        for i, azim in enumerate(camera_azims):
            if azim < 0:
                azim += 360
            self.camera_poses.append((0, azim))
            self.attention_mask.append([(cam_count + i - 1) % cam_count, i, (i + 1) % cam_count])
            if abs(azim) < front_view_diff:
                front_view_idx = i
                front_view_diff = abs(azim)
            if abs(azim - 180) < back_view_diff:
                back_view_idx = i
                back_view_diff = abs(azim - 180)

        # Add two additional cameras for painting the top surfaces
        if top_cameras:
            self.camera_poses.append((30, 0))
            self.camera_poses.append((30, 180))

            self.attention_mask.append([front_view_idx, cam_count])
            self.attention_mask.append([back_view_idx, cam_count + 1])

        # Add two additional cameras for painting the top surfaces
        if down_cameras:
            self.camera_poses.append((210, 0))
            self.camera_poses.append((210, 180))

            self.attention_mask.append([front_view_idx, cam_count + 2])
            self.attention_mask.append([back_view_idx, cam_count + 3])

        # Reference view for attention (all views attend the the views in this list)
        # A forward view will be used if not specified
        if len(ref_views) == 0:
            ref_views = [front_view_idx]

        # Calculate in-group attention mask
        # It is useless now!
        self.group_metas = split_groups(self.attention_mask, max_batch_size, ref_views)

        # Set up pytorch3D for projection between screen space and UV space
        # uvp is for latent and uvp_rgb for rgb color
        self.uvp = UVP(texture_size=texture_size, render_size=latent_size, sampling_mode="nearest", channels=4,
                       device=self._execution_device)

        if mesh_path.lower().endswith(".obj"):
            self.uvp.load_mesh([mesh_path], scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        elif mesh_path.lower().endswith(".glb"):
            self.uvp.load_glb_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)

        elif os.path.isdir(mesh_path):
            mesh_list = []
            for file in os.listdir(mesh_path):
                if file.endswith(".obj"):  # 确保是.obj文件
                    mesh_list.append(os.path.join(mesh_path, file))
            mesh_list.sort(key=lambda x: int(
                os.path.splitext(os.path.basename(x))[0][4:]))  # assume that the number comes after text
            self.uvp.load_mesh(mesh_list, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
        else:
            assert False, "The mesh file format is not supported. Use .obj or .glb."
        self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0)

        self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest",
                           channels=3, device=self._execution_device)

        self.uvp_rgb.mesh = [mesh.clone() for mesh in self.uvp.mesh]
        self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0)
        _, _, _, cos_maps, _, _ = self.uvp_rgb.render_geometry()
        self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

        # Save some VRAM
        del _, cos_maps
        self.uvp.to("cpu")
        self.uvp_rgb.to("cpu")

        # check
        color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1, 3, 1, 1).to(
        dtype=self.text_encoder.dtype, device=self._execution_device)
        color_images = torch.ones(
            (1, 1, latent_size * 8, latent_size * 8),
            device=self._execution_device,
            dtype=self.text_encoder.dtype
        ) * color_images
        color_images = ((0.5 * color_images) + 0.5)
        color_latents = encode_latents(self.vae, color_images)

        self.color_latents = {color[0]: color[1] for color in
                              zip(color_names, [latent for latent in color_latents])}
        self.vae = self.vae.to("cpu")
        torch.cuda.empty_cache()
        print("Done Initialization")

    '''
        Modified from a StableDiffusion ControlNet pipeline
        Multi ControlNet not supported yet
    '''

    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            height: Optional[int] = 704,
            width: Optional[int] = 1280,
            target_fps: Optional[int] = 16,
            num_frames: int = 16,
            num_inference_steps: int = 50,
            guidance_scale: float = 9.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: float = 0.0,
            num_videos_per_prompt: Optional[int] = 1,
            decode_chunk_size: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = 1,

            ### the following arguments are newly added
            control_images: List[PIL.Image.Image] = None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            num_images_per_prompt: Optional[int] = 1,
            guess_mode: bool = False,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
            negative_target_size: Optional[Tuple[int, int]] = None,
            sparse_frames=None,
            skip_conv_in=False,
            skip_time_emb=False,
            fixed_controlnet_timestep=-1,
            use_size_512=True,
            adapter_locations=None,
            inference_expert_masks=None,
            fixed_weights=None,

            ### the following arguments are for Tex4D
            mesh_path: str = None,
            mesh_transform: dict = None,
            mesh_autouv=False,
            camera_azims=None,
            camera_centers=None,
            top_cameras=True,
            texture_size=1536,
            render_rgb_size=1024,
            texture_rgb_size=1024,
            multiview_diffusion_end=0.8,
            exp_start=0.0,
            exp_end=6.0,
            ref_attention_end=0.2,
            logging_config=None,
            cond_type="depth",
            max_batch_size=6,
            progress_callback=None,

    ):
        r"""
        The call function to the pipeline for image-to-video generation with [`I2VGenXLPipeline`].

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            target_fps (`int`, *optional*):
                Frames per second. The rate at which the generated images shall be exported to a video after generation. This is also used as a "micro-condition" while generation.
            num_frames (`int`, *optional*):
                The number of video frames to generate.
            num_inference_steps (`int`, *optional*):
                The number of denoising steps.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            eta (`float`, *optional*):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            num_videos_per_prompt (`int`, *optional*):
                The number of images to generate per prompt.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`pipelines.i2vgen_xl.pipeline_i2vgen_xl.I2VGenXLPipelineOutput`] is
                returned, otherwise a `tuple` is returned where the first element is a list with the generated frames.
        """
        self.initialize_pipeline(
            mesh_path=mesh_path,
            mesh_transform=mesh_transform,
            mesh_autouv=mesh_autouv,
            camera_azims=camera_azims,
            camera_centers=camera_centers,
            top_cameras=top_cameras,
            down_cameras=True,
            ref_views=[],
            latent_size=height // 8,
            render_rgb_size=render_rgb_size,
            texture_size=texture_size,
            texture_rgb_size=texture_rgb_size,

            max_batch_size=max_batch_size,

            logging_config=logging_config
        )
        ### newly added
        if adapter_locations is None:
            adapter_locations = ['A', 'B', 'C', 'D', 'M']

        video_length = num_frames

        if video_length > 1 and num_images_per_prompt > 1:
            print(f"Warning - setting num_images_per_prompt = 1 because video_length = {video_length}")
            num_images_per_prompt = 1

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]
        ###

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, image, height, width, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = guidance_scale

        ### newly added
        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = False
        ###

        # 3.1 Encode input text prompt

        # TODO: I comment the direction prompt for now
        # prompt, negative_prompt = prepare_directional_prompt(prompt, negative_prompt)

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        # if self.do_classifier_free_guidance:
        #    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        ### newly added
        (
            controlnet_prompt_embeds,
            controlnet_negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds
        ) = self.helper.encode_controlnet_prompt(
            prompt,
            device,
            1,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            lora_scale=None,
            clip_skip=None,
        )

        # 3.1.5 Prepare image
        if isinstance(controlnet, ControlNetModel):

            assert len(control_images) == video_length * batch_size

            images = self.helper.prepare_images(
                images=control_images,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )

            height, width = images.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):

            images = []

            for image_ in control_images:
                image_ = self.helper.prepare_images(
                    images=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            # image = images
            height, width = images[0].shape[-2:]

        else:
            assert False
        ###

        # 3.2 Encode image prompt
        # 3.2.1 Image encodings.
        # https://github.com/ali-vilab/i2vgen-xl/blob/2539c9262ff8a2a22fa9daecbfd13f0a2dbc32d0/tools/inferences/inference_i2vgen_entrance.py#L114
        cropped_image = _center_crop_wide(image, (width, width))
        cropped_image = _resize_bilinear(
            cropped_image, (self.feature_extractor.crop_size["width"], self.feature_extractor.crop_size["height"])
        )
        image_embeddings = self._encode_image(cropped_image, device, num_videos_per_prompt)
        self.image_encoder.to('cpu')
        # 3.2.2 Image latents.
        resized_image = _center_crop_wide(image, (width, height))
        image = self.image_processor.preprocess(resized_image).to(device=device, dtype=image_embeddings.dtype)
        self.vae = self.vae.to("cuda")
        image_latents = self.prepare_image_latents(
            image,
            device=device,
            num_frames=num_frames,
            num_videos_per_prompt=num_videos_per_prompt,
        )
        self.vae = self.vae.to("cpu")
        torch.cuda.empty_cache()
        # 3.3 Prepare additional conditions for the UNet.
        if self.do_classifier_free_guidance:
            fps_tensor = torch.tensor([target_fps, target_fps]).to(device)
        else:
            fps_tensor = torch.tensor([target_fps]).to(device)
        fps_tensor = fps_tensor.repeat(batch_size * num_videos_per_prompt, 1).ravel()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        timesteps = timesteps.to(dtype=torch.bfloat16)
        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ### newly added
        # 6.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 6.2 Prepare added time ids & embeddings
        if isinstance(images, list):
            original_size = images[0].shape[-2:]
        else:
            original_size = images.shape[-2:]
        target_size = (height, width)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.helper._get_add_time_ids(original_size, crops_coords_top_left, target_size,
                                                     dtype=prompt_embeds.dtype)

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.helper._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
            controlnet_prompt_embeds = torch.cat([controlnet_negative_prompt_embeds, controlnet_prompt_embeds])

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
        controlnet_prompt_embeds = controlnet_prompt_embeds.to(device)

        if isinstance(images, list):
            images = [rearrange(img, "b f c h w -> (b f) c h w") for img in images]
        else:
            images = rearrange(images, "b f c h w -> (b f) c h w")

        if video_length > 1:
            # use repeat_interleave as we need to match the rearrangement above.
            controlnet_prompt_embeds = controlnet_prompt_embeds.repeat_interleave(video_length, dim=0)

        output_down_block_weights = []
        output_mid_block_weights = []
        ###

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                ### newly added
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                # controlnet(s) inference
                if guess_mode and self.do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_added_cond_kwargs = added_cond_kwargs

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    # this will be non interlaced when arranged!
                control_model_input = rearrange(control_model_input, "b c f h w -> (b f) c h w")
                # if we chunked this by 2 - the top 8 frames will be positive for cfg
                # the bottom half will be negative for cfg...

                if video_length > 1:
                    controlnet_added_cond_kwargs = {
                        "text_embeds": controlnet_added_cond_kwargs['text_embeds'].repeat_interleave(video_length,
                                                                                                     dim=0),
                        "time_ids": controlnet_added_cond_kwargs['time_ids'].repeat_interleave(video_length, dim=0)
                    }

                _, _, control_model_input_h, control_model_input_w = control_model_input.shape
                if (control_model_input_h, control_model_input_w) != (64, 64) and use_size_512:
                    reshaped_control_model_input = F.adaptive_avg_pool2d(control_model_input, (64, 64))
                    reshaped_images = F.adaptive_avg_pool2d(images, (512, 512))
                else:
                    reshaped_control_model_input = control_model_input
                    reshaped_images = images

                # todo - check if video_length > 1 this needs to produce num_frames * batch_size samples...
                with torch.no_grad():

                    if fixed_controlnet_timestep >= 0:
                        controlnet_timesteps = (torch.zeros_like(t) + fixed_controlnet_timestep).long().to(t.device)
                    else:
                        controlnet_timesteps = t
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        reshaped_control_model_input.to(dtype=torch.bfloat16),  # 强制转换为 float16
                        controlnet_timesteps.to(dtype=torch.bfloat16),  # 确保时间步也是 float16
                        encoder_hidden_states=controlnet_prompt_embeds.to(dtype=torch.bfloat16),
                        controlnet_cond=reshaped_images.to(dtype=torch.bfloat16),
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        added_cond_kwargs=controlnet_added_cond_kwargs,
                        return_dict=False,
                        skip_conv_in=skip_conv_in,
                        skip_time_emb=skip_time_emb,
                    )

                # this part is for MoE router
                if self.router is not None:
                    with torch.no_grad():
                        if self.router.router_type == 'timestep_weights':
                            input_t = t.unsqueeze(0) if t.dim() == 0 else t
                            down_block_weights, mid_block_weights = self.router(input_t.to(self.router.dtype),
                                                                                sparse_mask=inference_expert_masks)
                        elif self.router.router_type == 'simple_weights' or self.router.router_type == 'equal_weights':
                            down_block_weights, mid_block_weights = self.router(sparse_mask=inference_expert_masks)

                        elif self.router.router_type == 'embedding_weights':
                            down_block_weights, mid_block_weights = self.router(
                                router_input=image_embeddings[-1].unsqueeze(0).to(self.router.dtype),
                                sparse_mask=inference_expert_masks)
                        elif self.router.router_type == 'timestep_embedding_weights':
                            input_t = t.unsqueeze(0) if t.dim() == 0 else t
                            down_block_weights, mid_block_weights = self.router(
                                router_input=[input_t.to(self.router.dtype),
                                              image_embeddings[-1].unsqueeze(0).to(self.router.dtype)],
                                sparse_mask=inference_expert_masks)

                    output_down_block_weights.append(down_block_weights.cpu().numpy().tolist())
                    if mid_block_weights is not None:
                        output_mid_block_weights.append(mid_block_weights.cpu().numpy().tolist())
                    else:
                        output_mid_block_weights.append(None)

                    num_routers = self.router.num_routers
                    num_experts = self.router.num_experts

                    # merge the controlnets' features according to router weights
                    if mid_block_weights is not None:
                        mid_block_res_sample_merged = 0
                        idx_e = 0
                        for e in range(num_experts):
                            if inference_expert_masks[e] == True:
                                mid_block_res_sample_merged = mid_block_res_sample_merged + \
                                                              mid_block_res_sample[idx_e] * \
                                                              mid_block_weights.repeat_interleave(num_frames, dim=0)[e]
                                idx_e += 1
                    else:
                        mid_block_res_sample_merged = None

                    down_block_res_samples_merged = [0 for k in range(num_routers)]
                    for k in range(num_routers):
                        idx_e = 0
                        for e in range(num_experts):
                            if inference_expert_masks[e] == True:
                                down_block_res_samples_merged[k] = down_block_res_samples_merged[k] + \
                                                                   down_block_res_samples[idx_e][k] * \
                                                                   down_block_weights[k].repeat_interleave(num_frames,
                                                                                                           dim=0)[e]
                                idx_e += 1

                    down_block_res_samples = down_block_res_samples_merged
                    mid_block_res_sample = mid_block_res_sample_merged

                # this part is for sparse control
                # we only give the sparse key frames to our adapter below
                if sparse_frames is not None:
                    sparse_frames = [int(sparse_frames[k]) for k in range(len(sparse_frames))]
                    # print("sparse_frames", sparse_frames)
                    if self.do_classifier_free_guidance:
                        double_sparse_frames = sparse_frames + [(sparse_frames[k] + num_frames) for k in
                                                                range(len(sparse_frames))]
                    down_block_res_samples = [down_block_res_samples[i][double_sparse_frames, :] for i in
                                              range(len(down_block_res_samples))]
                    mid_block_res_sample = mid_block_res_sample[double_sparse_frames, :]

                with torch.no_grad():
                    mid_block_res_sample = mid_block_res_sample.to(
                        self.adapter.dtype) if 'M' in adapter_locations else None

                    adapter_input_num_frames = len(sparse_frames) if sparse_frames is not None else num_frames

                    # get the mid block and output block features output from adapters
                    adapted_down_block_res_samples, adapted_mid_block_res_sample = self.adapter(
                        down_block_res_samples=[down_block.to(self.adapter.dtype) for down_block in
                                                down_block_res_samples],
                        mid_block_res_sample=mid_block_res_sample,
                        sparsity_masking=sparse_frames,
                        num_frames=adapter_input_num_frames,
                        timestep=t,
                        encoder_hidden_states=image_embeddings[-1].unsqueeze(0)
                    )

                    # transform from sparse frame to dense frame, since I2VGen-XL UNet needs dense frames as input
                    if sparse_frames is not None:
                        if self.do_classifier_free_guidance:
                            full_n_sample_frames = num_frames * 2
                            full_sparsity_masking = double_sparse_frames
                        else:
                            full_n_sample_frames = num_frames
                            full_sparsity_masking = sparse_frames

                        full_adapted_down_block_res_samples = []
                        for k in range(len(adapted_down_block_res_samples)):
                            _, c, h, w = adapted_down_block_res_samples[k].shape
                            full_adapted_down_block_res_samples.append(
                                torch.zeros((full_n_sample_frames, c, h, w)).to(device))
                            for j, pos in enumerate(full_sparsity_masking):
                                full_adapted_down_block_res_samples[k][pos] = adapted_down_block_res_samples[k][j]
                        if adapted_mid_block_res_sample is not None:
                            _, c, h, w = adapted_mid_block_res_sample.shape
                            full_adapted_mid_block_res_sample = torch.zeros((full_n_sample_frames, c, h, w)).to(device)
                            for j, pos in enumerate(full_sparsity_masking):
                                full_adapted_mid_block_res_sample[pos] = adapted_mid_block_res_sample[j]
                        else:
                            full_adapted_mid_block_res_sample = None

                    else:
                        full_adapted_mid_block_res_sample = adapted_mid_block_res_sample
                        full_adapted_down_block_res_samples = adapted_down_block_res_samples

                    if full_adapted_mid_block_res_sample is not None:
                        full_adapted_mid_block_res_sample = rearrange(full_adapted_mid_block_res_sample,
                                                                      "(bs nf) c h w -> bs c nf h w", bs=2)
                    full_adapted_down_block_res_samples = [rearrange(down_block, "(bs nf) c h w -> bs c nf h w", bs=2)
                                                           for down_block in full_adapted_down_block_res_samples]
                    if cond_scale == 0:
                        full_adapted_down_block_res_samples = None
                ###

                # predict the noise residual
                self.unet.to("cuda")
                torch.cuda.empty_cache()
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    fps=fps_tensor,
                    image_latents=image_latents,
                    image_embeddings=image_embeddings,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=full_adapted_down_block_res_samples,  ### newly added
                    mid_block_additional_residual=full_adapted_mid_block_res_sample,  ### newly added
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # reshape latents
                batch_size, channel, frames, width, height = latents.shape
                latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)
                noise_pred = noise_pred.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channel, width, height)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # reshape latents back
                latents = latents[None, :].reshape(batch_size, frames, channel, width, height).permute(0, 2, 1, 3, 4)
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # if output_type == "latent":
        #    return I2VGenXLPipelineOutput(frames=latents)

        # video = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)

        # # Convert to tensor
        # if output_type == "tensor":
        #     video = torch.from_numpy(video)

        # if not return_dict:
        #     return video

        # return I2VGenXLPipelineOutput(videos=video)

        video_tensor = self.decode_latents(latents, decode_chunk_size=decode_chunk_size)
        video = tensor2vid(video_tensor, self.image_processor, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        ### additionally return the mid/output block router weights
        # return I2VGenXLPipelineOutput(frames=video, down_block_weights=None, mid_block_weights=None)
        return I2VGenXLPipelineOutput(frames=video, down_block_weights=output_down_block_weights,
                                      mid_block_weights=output_mid_block_weights)





def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        image = image_pil

    return image


def _resize_bilinear(
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        image = [u.resize(resolution, PIL.Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, PIL.Image.BILINEAR)
    return image


def _center_crop_wide(
    image: Union[torch.Tensor, List[torch.Tensor], PIL.Image.Image, List[PIL.Image.Image]], resolution: Tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        scale = min(image[0].size[0] / resolution[0], image[0].size[1] / resolution[1])
        image = [u.resize((round(u.width // scale), round(u.height // scale)), resample=PIL.Image.BOX) for u in image]

        # center crop
        x1 = (image[0].width - resolution[0]) // 2
        y1 = (image[0].height - resolution[1]) // 2
        image = [u.crop((x1, y1, x1 + resolution[0], y1 + resolution[1])) for u in image]
        return image
    else:
        scale = min(image.size[0] / resolution[0], image.size[1] / resolution[1])
        image = image.resize((round(image.width // scale), round(image.height // scale)), resample=PIL.Image.BOX)
        x1 = (image.width - resolution[0]) // 2
        y1 = (image.height - resolution[1]) // 2
        image = image.crop((x1, y1, x1 + resolution[0], y1 + resolution[1]))
        return image
