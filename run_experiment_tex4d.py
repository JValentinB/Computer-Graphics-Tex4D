import os
from os.path import join, isdir, abspath, dirname, basename, splitext
from IPython.display import display
from datetime import datetime
import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers import DDPMScheduler, DDIMScheduler, UniPCMultistepScheduler
# from src.pipeline import StableSyncMVDPipeline
from src.tex4d_pipeline import Tex4DPipeline
from i2vgen_xl.pipelines.i2vgen_xl_controlnet_adapter_pipeline import I2VGenXLControlNetAdapterPipeline
from i2vgen_xl.models.unets.unet_i2vgen_xl import I2VGenXLUNet
from src.configs import *
from shutil import copy

import time
import json
import argparse
from PIL import Image

import torch
import torchvision.transforms as transforms
from controlnet.controlnet import ControlNetModel
from model.ctrl_adapter import ControlNetAdapter
from model.ctrl_helper import ControlNetHelper

from utils.utils import center_crop_and_resize, bool_flag, save_as_gif, save_concatenated_gif



opt = parse_config()
# print(opt)

if opt.mesh_config_relative:
	mesh_path = join(dirname(opt.config), opt.mesh)
else:
	mesh_path = abspath(opt.mesh)

if opt.output:
	output_root = abspath(opt.output)
else:
	output_root = join(dirname(opt.config), 'result')

output_name_components = []
if opt.prefix and opt.prefix != "":
	output_name_components.append(opt.prefix)
if opt.use_mesh_name:
	mesh_name = splitext(basename(mesh_path))[0].replace(" ", "_")
	output_name_components.append(mesh_name)

if opt.timeformat and opt.timeformat != "":
	output_name_components.append(datetime.now().strftime(opt.timeformat))
output_name = "_".join(output_name_components)
output_dir = join(output_root, output_name)

if not isdir(output_dir):
	os.mkdir(output_dir)
else:
	print(f"Results exist in the output directory, use time string to avoid name collision.")
	exit(0)

print(f"Saving to {output_dir}")

copy(opt.config, join(output_dir, "config.yaml"))

logging_config = {
	"output_dir":output_dir, 
	# "output_name":None, 
	# "intermediate":False, 
	"log_interval":opt.log_interval,
	"view_fast_preview": opt.view_fast_preview,
	"tex_fast_preview": opt.tex_fast_preview,
	}


data_type = torch.bfloat16

helper = ControlNetHelper(use_size_512= True)
adapter = ControlNetAdapter.from_pretrained(
            "hanlincs/Ctrl-Adapter",
            subfolder=opt.huggingface_checkpoint_folder,
            low_cpu_mem_usage=False,
            device_map=None
            )

adapter = adapter.to(data_type)

pipe_line_args = {
	"torch_dtype": data_type,
	"use_safetensors": True,
	'helper': helper,
	'adapter': adapter
}

pipe_line_args['controlnet'] = {}
model_path = "lllyasviel/control_v11f1p_sd15_depth"
pipe_line_args['controlnet'] = ControlNetModel.from_pretrained(model_path, torch_dtype=data_type,
																			 use_safetensors=True)
pretrained_model_name_or_path = "ali-vilab/i2vgen-xl"
pipe = I2VGenXLControlNetAdapterPipeline.from_pretrained(pretrained_model_name_or_path, **pipe_line_args).to("cuda", dtype=data_type)
pipe.unet = I2VGenXLUNet.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to("cuda", dtype=data_type)
components = pipe.components  # Get the components dictionary

# define the pipeline
tex4d = Tex4DPipeline(**components)
kwargs = {
	'controlnet_conditioning_scale': opt.conditioning_scale,
	'control_guidance_start': opt.control_guidance_start,
	'control_guidance_end': opt.control_guidance_end,
	'use_size_512': opt.use_size_512,
}

base_path = "assets/evaluation/frames/depth"
all_batch_conditioning_images_pil = []
# Traverse through all directories in the base path
for folder_name in sorted(os.listdir(base_path)):
	condition_images_path = os.path.join(base_path, folder_name)

	if os.path.isdir(condition_images_path):
		condition_frames = sorted(os.listdir(condition_images_path))
		conditioning_images_pil = [Image.open(os.path.join(condition_images_path, frame)) for frame in condition_frames]
		conditioning_images_pil = [center_crop_and_resize(img, output_size=(opt.width, opt.height)) for img in
								   conditioning_images_pil]
		conditioning_images_pil = [img.resize((512, 512)) for img in conditioning_images_pil]
		all_batch_conditioning_images_pil.append(conditioning_images_pil)

base_path = "assets/evaluation/frames/raw_input"
first_frame = []
for folder_name in sorted(os.listdir(base_path)):
	condition_images_path = os.path.join(base_path, folder_name,"first_key_frame.jpg")
	conditioning_images_pil = Image.open(condition_images_path)
	conditioning_images_pil = center_crop_and_resize(conditioning_images_pil, output_size=(opt.width, opt.height))
	conditioning_images_pil = conditioning_images_pil.resize((512, 512))
	first_frame.append(conditioning_images_pil)


i2vgenxl_outputs = tex4d(
                    prompt=opt.prompt,
                    negative_prompt="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
                    height = opt.height,
                    width = opt.width,
                    image= first_frame[0],
                    control_images = all_batch_conditioning_images_pil[0],
                    num_inference_steps=opt.num_inference_steps,
                    guidance_scale=9.0,
                    generator=torch.manual_seed(opt.seed),
                    target_fps = 16,
                    num_frames = opt.n_sample_frames,
                    output_type="pil",
					mesh_path=mesh_path,
					mesh_autouv=not opt.keep_mesh_uv,
					mesh_transform={"scale":opt.mesh_scale},
					camera_azims=opt.camera_azims,
					top_cameras=not opt.top_cameras,
					texture_size=opt.latent_tex_size,
					render_rgb_size=opt.rgb_view_size,
					texture_rgb_size=opt.rgb_tex_size,
					multiview_diffusion_end=opt.mvd_end,
					max_batch_size=48,
					exp_start=opt.mvd_exp_start,
					exp_end=opt.mvd_exp_end,
					logging_config=logging_config,
					ref_attention_end=opt.ref_attention_end,
					cond_type=opt.cond_type,
                    **kwargs
                )

