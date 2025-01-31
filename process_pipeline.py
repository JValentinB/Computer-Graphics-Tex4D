from src.pipeline import StableSyncMVDPipeline
import torch 
import os
import base64
import traceback

from PIL import Image
from io import BytesIO

def process_mesh_with_preloaded_models(pipe, mesh_path, output_path, prompt, inference_steps=10, progress_callback=None):
    try:
        # Initialize StableSyncMVDPipeline with preloaded components
        syncmvd = StableSyncMVDPipeline(**pipe.components)

        # Example options (adjust based on your configuration)
        opt = {
            'latent_view_size': 64,
            'guidance_scale': 7.5,
            'negative_prompt': 'distorted, disfigured, disconnected limbs, ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, ugly faces, incomplete arms',
            'seed': 42,
            'guess_mode': False,
            'conditioning_scale': 1.0,
            'conditioning_scale_end': 1.0,
            'control_guidance_start': 0.0,
            'control_guidance_end': 1.0,
            'guidance_rescale': 0.0,
            'mesh_scale': 1.0,
            'camera_azims': [0, 90, 180, 270],
            'latent_tex_size': 128,
            'rgb_view_size': 256,
            'rgb_tex_size': 512,
            'mvd_end': 0.8,
            'mvd_exp_start': 0.2,
            'mvd_exp_end': 1.0,
            'ref_attention_end': 0.9,
            'shuffle_bg_change': False,
            'shuffle_bg_end': 1.0,
        }

        # Process the mesh with the user-provided prompt
        print(f"Running SyncMVD with mesh: {mesh_path} and prompt: {prompt}")
        result_tex_rgb = syncmvd(  #result_tex_rgb, textured_views, v = syncmvd(
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
            top_cameras=True,
            texture_size=opt['latent_tex_size'],
            render_rgb_size=opt['rgb_view_size'],
            texture_rgb_size=opt['rgb_tex_size'],
            multiview_diffusion_end=opt['mvd_end'],
            exp_start=opt['mvd_exp_start'],
            exp_end=opt['mvd_exp_end'],
            ref_attention_end=opt['ref_attention_end'],
            shuffle_background_change=opt['shuffle_bg_change'],
            shuffle_background_end=opt['shuffle_bg_end'],
            logging_config={"output_dir": os.path.dirname(output_path)},
            cond_type='depth',  # Adjust based on your logic
            
            progress_callback=progress_callback,
        )
        print("Finished SyncMVD.")
        
        # open the textured mesh from output_path/results/textured.obj
        textured_mesh_path = os.path.join(os.path.dirname(output_path), 'results', 'textured.obj')
        with open(textured_mesh_path, 'r') as f:
            textured_mesh = f.read()
        
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

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
