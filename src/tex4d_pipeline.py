import torch
from torch import nn
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
from .renderer.project import UVProjection as UVP

class StableSyncTex4DPipeline(StableDiffusionPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        controlnet,
        scheduler,
        feature_extractor,
        safety_checker=None,
        requires_safety_checker=False,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)

        self.controlnet = controlnet
        self.feature_extractor = feature_extractor
        self.safety_checker = safety_checker
        self.requires_safety_checker = requires_safety_checker

        # Default configurations
        self.uv_projection = None  # UV Projection system
        self.ref_uv_map = None     # Reference UV map for temporal blending

    def initialize_pipeline(
        self,
        mesh_path,
        keyframes,
        texture_size,
        render_size,
        camera_poses,
        max_batch_size=24,
    ):
        """Initialize the pipeline by setting up cameras, UV projections, and other configurations."""

        self.keyframes = keyframes

        # Initialize UV projection system for latent and RGB spaces
        self.uv_projection = UVP(texture_size=texture_size, render_size=render_size, channels=4, device=self.device)
        self.uv_projection.load_mesh(mesh_path)
        self.uv_projection.set_cameras_and_render_settings(camera_poses)

        # Prepare a reference UV map for temporal blending
        self.ref_uv_map = torch.zeros_like(self.uv_projection.get_uv_map(), device=self.device)

        # Additional configurations
        self.camera_poses = camera_poses
        self.max_batch_size = max_batch_size

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        num_inference_steps=50,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.7,
        keyframe_interpolation=True,
        output_dir="outputs",
    ):
        """Main function to generate 4D textures for an animated sequence."""

        # 1. Encode the input prompt
        prompt_embeds = self._encode_prompt(prompt)
        negative_prompt_embeds = self._encode_prompt(negative_prompt) if negative_prompt else None

        # 2. Prepare keyframe UV maps
        keyframe_uv_maps = self._prepare_keyframe_uv_maps()

        # 3. Initialize latent variables
        latents = self._initialize_latents(keyframe_uv_maps)

        # 4. Denoising loop with temporal blending
        for step, t in enumerate(self.scheduler.timesteps):
            latents = self._denoising_step(latents, prompt_embeds, negative_prompt_embeds, t, guidance_scale)

            # Temporal blending with the reference UV map
            latents = self._temporal_blending(latents)

        # 5. Post-process latents into textures
        textures = self._generate_final_textures(latents, keyframe_uv_maps)

        # 6. Interpolate non-keyframe textures (optional)
        if keyframe_interpolation:
            textures = self._interpolate_textures(textures)

        # 7. Save textures to output directory
        self._save_textures(textures, output_dir)

        return textures

    def _prepare_keyframe_uv_maps(self):
        """Generate UV maps for the selected keyframes."""
        keyframe_uv_maps = []
        for keyframe in self.keyframes:
            uv_map = self.uv_projection.render_uv_map(keyframe)
            keyframe_uv_maps.append(uv_map)
        return keyframe_uv_maps

    def _initialize_latents(self, keyframe_uv_maps):
        """Initialize latent variables for diffusion based on keyframe UV maps."""
        latents = [torch.randn_like(uv_map) for uv_map in keyframe_uv_maps]
        return latents

    def _denoising_step(self, latents, prompt_embeds, negative_prompt_embeds, timestep, guidance_scale):
        """Perform a single denoising step for the diffusion process."""
        noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs={"guidance_scale": guidance_scale},
        )[0]

        if negative_prompt_embeds is not None:
            noise_pred_neg = self.unet(
                latents,
                timestep,
                encoder_hidden_states=negative_prompt_embeds,
            )[0]
            noise_pred = noise_pred + guidance_scale * (noise_pred - noise_pred_neg)

        latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
        return latents

    def _temporal_blending(self, latents):
        """Blend latents with the reference UV map for temporal consistency."""
        alpha = 0.2  # Blending weight
        blended_latents = alpha * latents + (1 - alpha) * self.ref_uv_map
        self.ref_uv_map = blended_latents  # Update reference UV map
        return blended_latents

    def _generate_final_textures(self, latents, keyframe_uv_maps):
        """Decode latents into textures using the VAE decoder."""
        textures = [self.vae.decode(latent) for latent in latents]
        return textures

    def _interpolate_textures(self, textures):
        """Interpolate textures for non-keyframe frames."""
        interpolated_textures = []
        for i in range(len(textures) - 1):
            interpolated_textures.append(textures[i])
            # Linear interpolation between keyframe textures
            interp = (textures[i] + textures[i + 1]) / 2
            interpolated_textures.append(interp)
        interpolated_textures.append(textures[-1])
        return interpolated_textures

    def _save_textures(self, textures, output_dir):
        """Save generated textures to the output directory."""
        for i, texture in enumerate(textures):
            texture.save(f"{output_dir}/texture_frame_{i}.png")
