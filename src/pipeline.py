import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from IPython.display import display
import numpy as np
import math
import random
import torch
from torch import functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, GaussianBlur, InterpolationMode

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
	BaseOutput,
	numpy_to_pil,
	pt_to_pil,
	# make_image_grid,
	is_accelerate_available,
	is_accelerate_version,
	logging,
	replace_example_docstring
	)
from diffusers.utils.torch_utils import (randn_tensor, is_compiled_module)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.models.attention_processor import Attention, AttentionProcessor

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from .renderer.project import UVProjection as UVP


from .syncmvd.attention import SamplewiseAttnProcessor2_0, replace_attention_processors
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

class StableSyncMVDPipeline(StableDiffusionControlNetPipeline):
	def __init__(
		self, 
		vae: AutoencoderKL,
		text_encoder: CLIPTextModel,
		tokenizer: CLIPTokenizer,
		unet: UNet2DConditionModel,
		controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel]],
		scheduler: KarrasDiffusionSchedulers,
		safety_checker: StableDiffusionSafetyChecker,
		feature_extractor: CLIPImageProcessor,
		image_encoder=None,
		requires_safety_checker: bool = False,

	):
		super().__init__(
			vae, text_encoder, tokenizer, unet, 
			controlnet, scheduler, safety_checker, 
			feature_extractor, image_encoder, requires_safety_checker
		)

		self.scheduler = DDPMScheduler.from_config(self.scheduler.config)
		self.model_cpu_offload_seq = "vae->text_encoder->unet->vae"
		self.enable_model_cpu_offload()
		self.enable_vae_slicing()
		self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

	
	def initialize_pipeline(
			self,
			mesh_path=None,
			mesh_transform=None,
			mesh_autouv=None,
			camera_azims=None,
			camera_centers=None,
			top_cameras=True,
			down_cameras=True,
			ref_views=[],
			latent_size=None,
			render_rgb_size=None,
			texture_size=None,
			texture_rgb_size=None,

			max_batch_size=24,
			logging_config=None,
   
			view_matrices=None,
		):
		# Make output dir
		output_dir = logging_config["output_dir"]

		self.result_dir = f"{output_dir}/results"
		self.intermediate_dir = f"{output_dir}/intermediate"

		dirs = [output_dir, self.result_dir, self.intermediate_dir]
		for dir_ in dirs:
			if not os.path.isdir(dir_):
				os.mkdir(dir_)


		# Define the cameras for rendering
		self.camera_poses = []
		self.attention_mask = []
		self.centers = camera_centers

		front_view_diff = 360
		back_view_diff = 360
		front_view_idx = 0
		back_view_idx = 0
		if view_matrices is None:
			cam_count = len(camera_azims)
			for i, azim in enumerate(camera_azims):
				if azim < 0:
					azim += 360
				self.camera_poses.append((0, azim))
				self.attention_mask.append([(cam_count+i-1)%cam_count, i, (i+1)%cam_count])
				if abs(azim) < front_view_diff:
					front_view_idx = i
					front_view_diff = abs(azim)
				if abs(azim - 180) < back_view_diff:
					back_view_idx = i
					back_view_diff = abs(azim - 180)

			# Additional cameras for top or down views
			if top_cameras:
				self.camera_poses.append((30, 0))
				self.camera_poses.append((30, 180))
				self.attention_mask.append([front_view_idx, cam_count])
				self.attention_mask.append([back_view_idx, cam_count+1])
			if down_cameras:
				self.camera_poses.append((210, 0))
				self.camera_poses.append((210, 180))
				self.attention_mask.append([front_view_idx, cam_count+2])
				self.attention_mask.append([back_view_idx, cam_count+3])
		else: # if view matrices are provided
			cam_count = len(view_matrices)

			# Process each view matrix: extract pose and update front/back indices
			for i, view_matrix in enumerate(view_matrices):
				tilt, azim = rotation_matrix_to_tilt_azim(view_matrix[:3, :3])
				# Normalize azimuth to [0, 360)
				if azim < 0:
					azim += 360
				self.camera_poses.append((tilt, azim))
				
				# Build the attention mask for the basic views as in the original code.
				# Using modulo arithmetic to create a circular neighborhood.
				self.attention_mask.append([(cam_count+i-1) % cam_count, i, (i+1) % cam_count])
				
				# Determine which view is closest to 0° (front) and which is closest to 180° (back)
				if abs(azim) < front_view_diff:
					front_view_diff = abs(azim)
					front_view_idx = i
				if abs(azim - 180) < back_view_diff:
					back_view_diff = abs(azim - 180)
					back_view_idx = i
		print(f"Camera Poses: {self.camera_poses}")

		# Reference view for attention (if none provided, default to front view)
		if len(ref_views) == 0:
			ref_views = [front_view_idx]

		# Calculate in-group attention mask
		self.group_metas = split_groups(self.attention_mask, max_batch_size, ref_views)


		# Set up pytorch3D for projection between screen space and UV space
		# uvp is for latent and uvp_rgb for rgb color
		self.uvp = UVP(texture_size=texture_size, render_size=latent_size, sampling_mode="nearest", channels=4, device=self._execution_device)

		if mesh_path.lower().endswith(".obj"):
			self.uvp.load_mesh([mesh_path], scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
		elif mesh_path.lower().endswith(".glb"):
			self.uvp.load_glb_mesh(mesh_path, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)

		elif os.path.isdir(mesh_path):
			mesh_list = []
			for file in os.listdir(mesh_path):
				mesh_list.append(os.path.join(mesh_path, file))
			self.uvp.load_mesh(mesh_list, scale_factor=mesh_transform["scale"] or 1, autouv=mesh_autouv)
		else:
			assert False, "The mesh file format is not supported. Use .obj or .glb."
		
		if view_matrices is None: 
			self.uvp.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0)
		else:
			self.uvp.set_cameras_and_render_settings_matrices(view_matrices)


		self.uvp_rgb = UVP(texture_size=texture_rgb_size, render_size=render_rgb_size, sampling_mode="nearest", channels=3, device=self._execution_device)

		self.uvp_rgb.mesh = [mesh.clone() for mesh in self.uvp.mesh]

		if view_matrices is None:
			self.uvp_rgb.set_cameras_and_render_settings(self.camera_poses, centers=camera_centers, camera_distance=4.0)
		else:
			self.uvp_rgb.set_cameras_and_render_settings_matrices(view_matrices)

		_, _, _, cos_maps, _, _= self.uvp_rgb.render_geometry()
		self.uvp_rgb.calculate_cos_angle_weights(cos_maps, fill=False)

		# Save some VRAM
		del _, cos_maps
		self.uvp.to("cpu")
		self.uvp_rgb.to("cpu")


		color_images = torch.FloatTensor([color_constants[name] for name in color_names]).reshape(-1,3,1,1).to(dtype=self.text_encoder.dtype, device=self._execution_device)
		color_images = torch.ones(
			(1,1,latent_size*8, latent_size*8), 
			device=self._execution_device, 
			dtype=self.text_encoder.dtype
		) * color_images
		color_images = ((0.5*color_images)+0.5)
		color_latents = encode_latents(self.vae, color_images)

		self.color_latents = {color[0]:color[1] for color in zip(color_names, [latent for latent in color_latents])}
		self.vae = self.vae.to("cpu")

		print("Done Initialization")




	'''
		Modified from a StableDiffusion ControlNet pipeline
		Multi ControlNet not supported yet
	'''
	@torch.no_grad()
	def __call__(
		self,
		prompt: str = None,
		height: Optional[int] = None,
		width: Optional[int] = None,
		num_inference_steps: int = 50,
		guidance_scale: float = 7.5,
		negative_prompt: str = None,
		
		num_images_per_prompt: Optional[int] = 1,
		eta: float = 0.0,
		generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
		return_dict: bool = False,
		callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
		callback_steps: int = 1,
		max_batch_size=6,
		
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		controlnet_guess_mode: bool = False,
		controlnet_conditioning_scale: Union[float, List[float]] = 0.7,
		controlnet_conditioning_end_scale: Union[float, List[float]] = 0.9,
		control_guidance_start: Union[float, List[float]] = 0.0,
		control_guidance_end: Union[float, List[float]] = 0.99,
		guidance_rescale: float = 0.0,

		mesh_path: str = None,
		mesh_transform: dict = None,
		mesh_autouv = False,
		camera_azims=None,
		camera_centers=None,
		top_cameras=True,
		texture_size = 1536,
		render_rgb_size=1024,
		texture_rgb_size = 1024,
		multiview_diffusion_end=0.8,
		exp_start=0.0,
		exp_end=6.0,
		shuffle_background_change=0.4,
		shuffle_background_end=0.99, #0.4

		use_directional_prompt=True,
		
		ref_attention_end=0.2,

		logging_config=None,
		cond_type="depth",
  
		progress_callback=None,
		view_matrices=None,
	):
		

		# Setup pipeline settings
		self.initialize_pipeline(
				mesh_path=mesh_path,
				mesh_transform=mesh_transform,
				mesh_autouv=mesh_autouv,
				camera_azims=camera_azims,
				camera_centers=camera_centers,
				top_cameras=top_cameras,
				down_cameras=True,
				ref_views=[],
				latent_size=height//8,
				render_rgb_size=render_rgb_size,
				texture_size=texture_size,
				texture_rgb_size=texture_rgb_size,

				max_batch_size=max_batch_size,

				logging_config=logging_config, 

				view_matrices=view_matrices,
			)


		
		num_timesteps = self.scheduler.config.num_train_timesteps
		initial_controlnet_conditioning_scale = controlnet_conditioning_scale
		log_interval = logging_config.get("log_interval", 10)
		log_interval = 2
		view_fast_preview = logging_config.get("view_fast_preview", True)
		tex_fast_preview = logging_config.get("tex_fast_preview", True)

		controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

		# align format for control guidance
		if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
			control_guidance_start = len(control_guidance_end) * [control_guidance_start]
		elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
			control_guidance_end = len(control_guidance_start) * [control_guidance_end]
		elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
			# mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
			mult = 1
			control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
				control_guidance_end
			]


		# 0. Default height and width to unet
		height = height or self.unet.config.sample_size * self.vae_scale_factor
		width = width or self.unet.config.sample_size * self.vae_scale_factor


		# 1. Check inputs. Raise error if not correct
		# self.check_inputs(
		# 	prompt,
		# 	torch.zeros((1,3,height,width), device=self._execution_device),
		# 	callback_steps,
		# 	negative_prompt,
		# 	None,
		# 	None,
		# 	controlnet_conditioning_scale,
		# 	control_guidance_start,
		# 	control_guidance_end,
		# )


		# 2. Define call parameters
		if prompt is not None and isinstance(prompt, list):
			assert len(prompt) == 1 and len(negative_prompt) == 1, "Only implemented for 1 (negative) prompt"  
		assert num_images_per_prompt == 1, "Only implemented for 1 image per-prompt"
		batch_size = len(self.uvp.cameras)


		device = self._execution_device
		# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
		# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
		# corresponds to doing no classifier free guidance.
		do_classifier_free_guidance = guidance_scale > 1.0

		# if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
		# 	controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

		global_pool_conditions = (
			controlnet.config.global_pool_conditions
			if isinstance(controlnet, ControlNetModel)
			else controlnet.nets[0].config.global_pool_conditions
		)
		guess_mode = controlnet_guess_mode or global_pool_conditions


		# 3. Encode input prompt
		prompt, negative_prompt = prepare_directional_prompt(prompt, negative_prompt)

		text_encoder_lora_scale = (
			cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
		)
		prompt_embeds = self._encode_prompt(
			prompt,
			device,
			num_images_per_prompt,
			do_classifier_free_guidance,
			negative_prompt,
			prompt_embeds=None,
			negative_prompt_embeds=None,
			lora_scale=text_encoder_lora_scale,
		)

		negative_prompt_embeds, prompt_embeds = torch.chunk(prompt_embeds, 2)
		prompt_embed_dict = dict(zip(direction_names, [emb for emb in prompt_embeds]))
		negative_prompt_embed_dict = dict(zip(direction_names, [emb for emb in negative_prompt_embeds]))

		# (4. Prepare image) This pipeline use internal conditional images from Pytorch3D
		self.uvp.to(self._execution_device)
		conditioning_images, masks = get_conditioning_images(self.uvp, height, cond_type=cond_type)
		conditioning_images = [i.type(prompt_embeds.dtype)for i in conditioning_images]
		cond = [(conditioning_images/2+0.5).permute(0,2,3,1).cpu().numpy() for conditioning_images in conditioning_images]
		cond = np.concatenate([img for img in cond], axis=1)
		cond = np.concatenate(cond, axis=1)
		numpy_to_pil(cond)[0].save(f"{self.intermediate_dir}/cond.jpg")

		# 5. Prepare timesteps
		self.scheduler.set_timesteps(num_inference_steps, device=device)
		timesteps = self.scheduler.timesteps
		batch_size = len(conditioning_images) * len(self.camera_poses)
		# 6. Prepare latent variables
		num_channels_latents = self.unet.config.in_channels
		latents = self.prepare_latents(
			batch_size,
			num_channels_latents,
			height,
			width,
			prompt_embeds.dtype,
			device,
			generator,
			None,
		)

		latent_tex = self.uvp.set_noise_texture()
		noise_views = self.uvp.render_textured_views()
		foregrounds = [view[:-1] for view in noise_views]
		masks = [view[-1:] for view in noise_views]
		composited_tensor = composite_rendered_view(self.scheduler, latents, foregrounds, masks, timesteps[0]+1)
		latents = composited_tensor.type(latents.dtype)
		self.uvp.to("cpu")


		# 7. Prepare extra step kwargs.
		extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

		# 7.1 Create tensor stating which controlnets to keep
		controlnet_keep = []

		for i in range(len(timesteps)):
			keeps = [
				1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
				for s, e in zip(control_guidance_start, control_guidance_end)
			]
			controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

		# 8. Denoising loop
		num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
		intermediate_results = []
		self.key_frames_num = len(self.uvp.mesh)
		background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses)*self.key_frames_num)]
		dbres_sizes_list = []
		mbres_size_list = []

		# 9. Tex4D: Reference UV Texture
		# The reference map is regenerated at every step by sequentially combining the texture of every key frame
		# Therefore, there is no need to define it and store it outside the diffusion step. (Nach meiner Meinung)
		# reference_uv = torch.zeros_like(latent_tex)


		with self.progress_bar(total=num_inference_steps) as progress_bar:
			for i, t in enumerate(timesteps):

				if progress_callback:
					progress_percentage = (progress_bar.n / progress_bar.total)
					progress_callback(progress_percentage) 
	 
				# mix prompt embeds  according to azim angle
				positive_prompt_embeds = [azim_prompt(prompt_embed_dict, pose) for pose in self.camera_poses]
				positive_prompt_embeds = torch.stack(positive_prompt_embeds, axis=0)

				negative_prompt_embeds = [azim_neg_prompt(negative_prompt_embed_dict, pose) for pose in self.camera_poses]
				negative_prompt_embeds = torch.stack(negative_prompt_embeds, axis=0)


				# expand the latents if we are doing classifier free guidance
				latent_model_input = self.scheduler.scale_model_input(latents, t)

				'''
					Use groups to manage prompt and results
					Make sure negative and positive prompt does not perform attention together
				'''
				prompt_embeds_groups = {"positive": positive_prompt_embeds}
				result_groups = {}
				if do_classifier_free_guidance:
					prompt_embeds_groups["negative"] = negative_prompt_embeds

				for prompt_tag, prompt_embeds in prompt_embeds_groups.items():
					if prompt_tag == "positive" or not guess_mode:
						# controlnet(s) inference
						control_model_input = latent_model_input
						controlnet_prompt_embeds = prompt_embeds


						if isinstance(controlnet_keep[i], list):
							cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
						else:
							controlnet_cond_scale = controlnet_conditioning_scale
							if isinstance(controlnet_cond_scale, list):
								controlnet_cond_scale = controlnet_cond_scale[0]
							cond_scale = controlnet_cond_scale * controlnet_keep[i]

						# Split into micro-batches according to group meta info
						# Ignore this feature for now
						down_block_res_samples_list = []
						mid_block_res_sample_list = []

						control_model_input = control_model_input.reshape(self.key_frames_num,-1,4,control_model_input.shape[2],
																		  control_model_input.shape[3])
						control_model_input = torch.unbind(control_model_input, dim=0)
						model_input_batches = []
						prompt_embeds_batches = []
						conditioning_images_batches = []

						for control_model_input_tmp, conditioning_images_tmp in zip(control_model_input, conditioning_images):
							model_input_batches.extend([torch.index_select(control_model_input_tmp, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas])
							prompt_embeds_batches.extend([torch.index_select(controlnet_prompt_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas])
							conditioning_images_batches.extend([torch.index_select(conditioning_images_tmp, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas])

						for model_input_batch ,prompt_embeds_batch, conditioning_images_batch \
							in zip (model_input_batches, prompt_embeds_batches, conditioning_images_batches):
							down_block_res_samples, mid_block_res_sample = self.controlnet(
								model_input_batch,
								t,
								encoder_hidden_states=prompt_embeds_batch,
								controlnet_cond=conditioning_images_batch,
								conditioning_scale=cond_scale,
								guess_mode=guess_mode,
								return_dict=False,
							)
							down_block_res_samples_list.append(down_block_res_samples)
							mid_block_res_sample_list.append(mid_block_res_sample)

						''' For the ith element of down_block_res_samples, concat the ith element of all mini-batch result '''
						model_input_batches = prompt_embeds_batches = conditioning_images_batches = None

						if guess_mode:
							for dbres in down_block_res_samples_list:
								dbres_sizes = []
								for res in dbres:
									dbres_sizes.append(res.shape)
								dbres_sizes_list.append(dbres_sizes)

							for mbres in mid_block_res_sample_list:
								mbres_size_list.append(mbres.shape)

					else:
						# Infered ControlNet only for the conditional batch.
						# To apply the output of ControlNet to both the unconditional and conditional batches,
						# add 0 to the unconditional batch to keep it unchanged.
						# We copy the tensor shapes from a conditional batch
						down_block_res_samples_list = []
						mid_block_res_sample_list = []
						for dbres_sizes in dbres_sizes_list:
							down_block_res_samples_list.append([torch.zeros(shape, device=self._execution_device, dtype=latents.dtype) for shape in dbres_sizes])
						for mbres in mbres_size_list:
							mid_block_res_sample_list.append(torch.zeros(mbres, device=self._execution_device, dtype=latents.dtype))
						dbres_sizes_list = []
						mbres_size_list = []


					'''
					
						predict the noise residual, split into mini-batches
						Downblock res samples has n samples, we split each sample into m batches
						and re group them into m lists of n mini batch samples.
					
					'''
					noise_pred_list = []
					model_input_batches = []
					prompt_embeds_batches = []
					latent_model_input_tmp = latent_model_input.reshape(self.key_frames_num, -1, 4, latent_model_input.shape[2],
																	  latent_model_input.shape[3])
					latent_model_input_tmp = torch.unbind(latent_model_input_tmp, dim=0)
					for input_tmp in latent_model_input_tmp:
						model_input_batches.extend([torch.index_select(input_tmp, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas])
						prompt_embeds_batches.extend([torch.index_select(prompt_embeds, dim=0, index=torch.tensor(meta[0], device=self._execution_device)) for meta in self.group_metas])

					group_metas_batches = self.group_metas * self.key_frames_num
					for model_input_batch, prompt_embeds_batch, down_block_res_samples_batch, mid_block_res_sample_batch, meta \
						in zip(model_input_batches, prompt_embeds_batches, down_block_res_samples_list, mid_block_res_sample_list, group_metas_batches):
						if t > num_timesteps * (1- ref_attention_end):
							replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=1)
						else:
							replace_attention_processors(self.unet, SamplewiseAttnProcessor2_0, attention_mask=meta[2], ref_attention_mask=meta[3], ref_weight=0)

						noise_pred = self.unet(
							model_input_batch,
							t,
							encoder_hidden_states=prompt_embeds_batch,
							cross_attention_kwargs=cross_attention_kwargs,
							down_block_additional_residuals=down_block_res_samples_batch,
							mid_block_additional_residual=mid_block_res_sample_batch,
							return_dict=False,
						)[0]
						noise_pred_list.extend(noise_pred)

					# for noise_pred in noise_pred_list:
					# 	noise_pred_list.extend([torch.index_select(noise_pred, dim=0, index=torch.tensor(meta[1], device=self._execution_device)) for noise_pred, meta in zip(noise_pred_list, self.group_metas)])
					noise_pred = torch.stack(noise_pred_list, dim=0)
					down_block_res_samples_list = None
					mid_block_res_sample_list = None
					# noise_pred_list = None
					model_input_batches = prompt_embeds_batches = down_block_res_samples_batches = mid_block_res_sample_batches = None

					result_groups[prompt_tag] = noise_pred

				positive_noise_pred = result_groups["positive"]

				# perform guidance
				if do_classifier_free_guidance:
					noise_pred = result_groups["negative"] + guidance_scale * (positive_noise_pred - result_groups["negative"])


				if do_classifier_free_guidance and guidance_rescale > 0.0:
					# Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
					noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

				self.uvp.to(self._execution_device)
				# compute the previous noisy sample x_t -> x_t-1
				# Multi-View step or individual step
				current_exp = ((exp_end-exp_start) * i / num_inference_steps) + exp_start
				if t > (1-multiview_diffusion_end)*num_timesteps:
					step_results = step_tex(
						scheduler=self.scheduler, 
						uvp=self.uvp, 
						model_output=noise_pred, 
						timestep=t, 
						sample=latents, 
						texture=latent_tex,
						# reference_uv=reference_uv,
						return_dict=True, 
						main_views=[], 
						exp= current_exp,
						**extra_step_kwargs
					)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = step_results["prev_tex"]
					reference_uv = step_results["reference_uv"]

					# Composit latent foreground with random color background
					background_latents = [self.color_latents[color] for color in background_colors]
					composited_tensor = composite_rendered_view(self.scheduler, background_latents, latents, masks, t)
					latents = composited_tensor.type(latents.dtype)

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))
				else:
					# TODO: The last 200 steps directly refine the texture????
					# No, it seems that in the last 200 steps thetexture is not refined at all???
					step_results = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=True)

					pred_original_sample = step_results["pred_original_sample"]
					latents = step_results["prev_sample"]
					latent_tex = None

					intermediate_results.append((latents.to("cpu"), pred_original_sample.to("cpu")))

				del noise_pred, result_groups
					


				# Update pipeline settings after one step:
				# 1. Annealing ControlNet scale
				if (1-t/num_timesteps) < control_guidance_start[0]:
					controlnet_conditioning_scale = initial_controlnet_conditioning_scale
				elif (1-t/num_timesteps) > control_guidance_end[0]:
					controlnet_conditioning_scale = controlnet_conditioning_end_scale
				else:
					alpha = ((1-t/num_timesteps) - control_guidance_start[0]) / (control_guidance_end[0] - control_guidance_start[0])
					controlnet_conditioning_scale = alpha * initial_controlnet_conditioning_scale + (1-alpha) * controlnet_conditioning_end_scale

				# 2. Shuffle background colors; only black and white used after certain timestep
				if (1-t/num_timesteps) < shuffle_background_change:
					background_colors = [random.choice(list(color_constants.keys())) for i in range(len(self.camera_poses)*self.key_frames_num)]
				elif (1-t/num_timesteps) < shuffle_background_end:
					background_colors = [random.choice(["black","white"]) for i in range(len(self.camera_poses)*self.key_frames_num)]
				else:
					background_colors = background_colors



				# Logging at "log_interval" intervals and last step
				# Choose to uses color approximation or vae decoding
				if i % log_interval == log_interval-1 or t == 1:
					if view_fast_preview:
						decoded_results = []
						for latent_images in intermediate_results[-1]:
							images = latent_preview(latent_images.to(self._execution_device))
							images = np.concatenate([img for img in images], axis=1)
							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")
					else:
						decoded_results = []
						for latent_images in intermediate_results[-1]:
							images = decode_latents(self.vae, latent_images.to(self._execution_device))

							images = np.concatenate([img for img in images], axis=1)

							decoded_results.append(images)
						result_image = np.concatenate(decoded_results, axis=0)
						numpy_to_pil(result_image)[0].save(f"{self.intermediate_dir}/step_{i:02d}.jpg")

					if not t < (1-multiview_diffusion_end)*num_timesteps:
						if tex_fast_preview:
							for index, tex in enumerate(latent_tex):
								texture_color = latent_preview(tex[None, ...])
								numpy_to_pil(texture_color)[0].save(f"{self.intermediate_dir}/texture_{i:02d}_key_frames_{index:02d}.jpg")
							texture_color = latent_preview(reference_uv[None, ...])
							numpy_to_pil(texture_color)[0].save(f"{self.intermediate_dir}/reference_texture_{i:02d}.jpg")
						else:
							self.uvp_rgb.to(self._execution_device)
							result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, pred_original_sample)
							numpy_to_pil(result_tex_rgb_output)[0].save(f"{self.intermediate_dir}/texture_{i:02d}.png")
							self.uvp_rgb.to("cpu")

				self.uvp.to("cpu")

				# call the callback, if provided
				if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
					progress_bar.update()
					if callback is not None and i % callback_steps == 0:
						callback(i, t, latents)

				# Signal the program to skip or end
				import select
				import sys
				if select.select([sys.stdin],[],[],0)[0]:
					userInput = sys.stdin.readline().strip()
					if userInput == "skip":
						return None
					elif userInput == "end":
						exit(0)
		
		self.uvp.to(self._execution_device)
		self.uvp_rgb.to(self._execution_device)
		result_tex_rgb, result_tex_rgb_output = get_rgb_texture(self.vae, self.uvp_rgb, latents)
		for i, result_tex_rgb_tmp in enumerate(result_tex_rgb):
			self.uvp.save_mesh(f"{self.result_dir}/textured_{i:02d}.obj", result_tex_rgb_tmp,i)

		# self.uvp_rgb.set_texture_map(result_tex_rgb)
		# textured_views = self.uvp_rgb.render_textured_views()
		# textured_views_rgb = torch.cat(textured_views, axis=-1)[:-1,...]
		# textured_views_rgb = textured_views_rgb.permute(1,2,0).cpu().numpy()[None,...]
		# v = numpy_to_pil(textured_views_rgb)[0]
		# v.save(f"{self.result_dir}/textured_views_rgb_{i:02d}.jpg")
		# display(v)

		# Offload last model to CPU
		if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
			self.final_offload_hook.offload()

		self.uvp.to("cpu")
		self.uvp_rgb.to("cpu")

		return result_tex_rgb   #, textured_views, v