from src.pipeline import StableSyncMVDPipeline
import torch 
import os
import base64
import traceback

from PIL import Image
from io import BytesIO
from src.configs import *

def process_mesh_with_preloaded_models(
        tex4d_pipeline: StableSyncMVDPipeline, 
        mesh_path, 
        output_path: str, 
        prompt: str = "", 
        inference_steps: int = 10,
        latent_tex_size: int = 512,
        rgb_tex_size: int = 1024,
        progress_callback=None,
        view_matrices=None
    ):
    try:

        # Default options
        opt = {
            'seed': 1,
            'negative_prompt': 'oversmoothed, blurry, depth of field, out of focus, low quality, bloom, glowing effect.',
            'guidance_scale': 15.5,
            'guess_mode': False,
            'conditioning_scale': 0.7,                                                                                                                          
            'conditioning_scale_end': 0.9,
            'control_guidance_start': 0.0,
            'control_guidance_end': 0.99,                                                                                                       
            'guidance_rescale': 0.0,
            'latent_view_size': 96,
            'latent_tex_size': latent_tex_size,
            'rgb_view_size': 1536,
            'rgb_tex_size': rgb_tex_size,
            'camera_azims': [-180, -120, -60, 0, 60, 120],
            'no_top_cameras': False,
            'mvd_end': 0.8,
            'mvd_exp_start': 0.0,
            'mvd_exp_end': 6.0,
            'ref_attention_end': 0.2,
            'shuffle_bg_change': 0.4,                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            'shuffle_bg_end': 0.8,
            'mesh_scale': 1.0,
            'keep_mesh_uv': False,
        }
        #opt = parse_config()
        logging_config = {
	    "output_dir": os.path.dirname(output_path), 
	    # "output_name":None, 
	    "intermediate": False, 
	    "log_interval": 10,
	    "view_fast_preview": True,
	    "tex_fast_preview": True,
	    }

        # Process the mesh with the user-provided prompt
        print(f"Running SyncMVD with mesh: {mesh_path} and prompt: {prompt}")
        result_tex_rgb = tex4d_pipeline(  #result_tex_rgb, textured_views, v = syncmvd(
            prompt=prompt,  # Use the prompt provided by the client                                                 
            height=opt['latent_view_size'] * 8,
            width=opt['latent_view_size'] * 8,
            num_inference_steps=inference_steps,
            guidance_scale=opt['guidance_scale'],
            negative_prompt=opt['negative_prompt'],
            
            generator=torch.manual_seed(opt['seed']),
            max_batch_size=48,
            controlnet_guess_mode=opt['guess_mode'],
            controlnet_conditioning_scale=opt['conditioning_scale'],
            controlnet_conditioning_end_scale=opt['conditioning_scale_end'],
            control_guidance_start=opt['control_guidance_start'],
            control_guidance_end=opt['control_guidance_end'],
            guidance_rescale=opt['guidance_rescale'],
            use_directional_prompt=True,
            
            mesh_path=mesh_path,
            mesh_transform={"scale": opt['mesh_scale']},
            mesh_autouv=False,
            
            camera_azims=opt['camera_azims'],
            top_cameras= not opt['no_top_cameras'],
            texture_size=opt['latent_tex_size'],
            render_rgb_size=opt['rgb_view_size'],
            texture_rgb_size=opt['rgb_tex_size'],
            multiview_diffusion_end=opt['mvd_end'],
            exp_start=opt['mvd_exp_start'],
            exp_end=opt['mvd_exp_end'],
            ref_attention_end=opt['ref_attention_end'],
            shuffle_background_change=opt['shuffle_bg_change'],
            shuffle_background_end=opt['shuffle_bg_end'],
            logging_config=logging_config,
            cond_type='depth',  
            progress_callback=progress_callback,
            view_matrices=view_matrices,
        )
        print("Finished SyncMVD.")

        #return handle_single_mesh(output_path)


    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())

def handle_single_mesh(output_path):
        # open the textured mesh from output_path/results/textured.obj
        textured_mesh_path = os.path.join(os.path.dirname(output_path), 'results', 'textured.obj')
        with open(textured_mesh_path, 'r') as f:
            textured_mesh = f.read()
        

        #for i, texture in enumerate(result_tex_rgb):
        # Save the result
        if result_tex_rgb.dim() == 3 and result_tex_rgb.size(0) == 3:
            # Convert tensor from [C, H, W] to [H, W, C] for PIL compatibility
            result_tex_rgb = result_tex_rgb.permute(1, 2, 0)
            # Denormalize if necessary, assuming the values are in the range [0, 1]
            img_array = (result_tex_rgb * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            # Create PIL image and convert to base64
            pil_image = Image.fromarray(img_array)
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            texture_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return {
                'texture': texture_base64, 
                'mesh': textured_mesh
            }
        else:
            raise ValueError("Tensor does not have the correct shape or number of channels.")
