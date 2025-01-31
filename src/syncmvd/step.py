import torch
from diffusers.utils import randn_tensor
from diffusers.utils.torch_utils import randn_tensor
from src.renderer.voronoi import voronoi_solve
'''

	Customized Step Function
	step on texture
	texture

'''
@torch.no_grad()
def step_tex(
		scheduler,
		uvp,
		model_output: torch.FloatTensor,
		timestep: int,
		sample: torch.FloatTensor,
		texture: None,
		# reference_uv: torch.FloatTensor,
		generator=None,
		return_dict: bool = True,
		guidance_scale = 1,
		main_views = [],
		hires_original_views = True,
		exp=None,
		blending_weight = 0.2,
		cos_weighted=True
):
	t = timestep

	prev_t = scheduler.previous_timestep(t)

	if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
		model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
	else:
		predicted_variance = None

	# 1. compute alphas, betas
	alpha_prod_t = scheduler.alphas_cumprod[t]
	alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
	beta_prod_t = 1 - alpha_prod_t
	beta_prod_t_prev = 1 - alpha_prod_t_prev
	current_alpha_t = alpha_prod_t / alpha_prod_t_prev
	current_beta_t = 1 - current_alpha_t

	# 2. compute predicted original sample from predicted noise also called
	# "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
	if scheduler.config.prediction_type == "epsilon":
		pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
	elif scheduler.config.prediction_type == "sample":
		pred_original_sample = model_output
	elif scheduler.config.prediction_type == "v_prediction":
		pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
	else:
		raise ValueError(
			f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
			" `v_prediction`  for the DDPMScheduler."
		)

	# 3. Clip or threshold "predicted x_0"
	if scheduler.config.thresholding:
		pred_original_sample = scheduler._threshold_sample(pred_original_sample)
	elif scheduler.config.clip_sample:
		pred_original_sample = pred_original_sample.clamp(
			-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
		)

	# 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
	pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
	current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

	'''
		Add multidiffusion here
	'''

	if texture is None:
		sample_views = [view for view in sample]
		sample_views, texture, _, _= uvp.bake_texture(views=sample_views, main_views=main_views, exp=exp)
		sample_views = torch.stack(sample_views, axis=0)[:,:-1,...]

	#  TODO: According to the algorithm in Tex4D, the introduction of the reference module is after the defused step, so right here.
	#  TODO: The key step, Expansion happen before the voronoi-based filling! Therefore it needs to be implemented
	#  TODO: in bake_texture funtion. After that, reference UV map might be processed by voronoi-based filling and used to combine
	#  TODO: with the key frame based texture.
	# For example, from 12 views we can bake a texture, we bake 2 texture and combine them sequentially,
	# and then got 1 texture, then we do voronoi-based filling and get a full-filled texture of 512 * 512 as
	# UV reference map. After diffused, we combined it with the diffused view-based texture.

	original_views = [view for view in pred_original_sample]
	original_views, original_tex, visibility_weights, reference_mask_update_list, reference_mask = uvp.bake_texture(views=original_views, main_views=main_views, exp=exp)
	uvp.set_texture_map(original_tex)	
	# original_views = uvp.render_textured_views()
	# original_views = torch.stack(original_views, axis=0)[:,:-1,...]


	# 5. Compute predicted previous sample µ_t
	# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
	# pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
	original_tex = torch.stack(original_tex, dim = 0)
	if isinstance(texture, (tuple, list)):
		# 如果是 tuple 或 list，使用 torch.stack 合并为一个 tensor
		texture = torch.stack(texture, dim=0)

	# 5.1 Tex4D previous texture calculation
	prev_tex = pred_original_sample_coeff * original_tex + current_sample_coeff * texture

	# reference_uv = torch.zeros_like(prev_tex)
	# The reference map is regenerated at every step by sequentially combining the texture of every key frame

	#prev_tex , reference_uv = previous_texture_tex4d(texture, original_tex, reference_uv, alpha_prod_t, beta_prod_t, alpha_prod_t_prev, beta_prod_t_prev)

	# 6. Add noise
	variance = 0

	if predicted_variance is not None:
		variance_views = [view for view in predicted_variance]
		variance_views, variance_tex, visibility_weights, _ = uvp.bake_texture(views=variance_views, main_views=main_views, cos_weighted=cos_weighted, exp=exp)
		variance_views = torch.stack(variance_views, axis=0)[:,:-1,...]
	else:
		variance_tex = None

	if t > 0:
		device = texture.device
		variance_noise = randn_tensor(
			texture.shape, generator=generator, device=device, dtype=texture.dtype
		)
		if scheduler.variance_type == "fixed_small_log":
			variance = scheduler._get_variance(t, predicted_variance=variance_tex) * variance_noise
		elif scheduler.variance_type == "learned_range":
			variance = scheduler._get_variance(t, predicted_variance=variance_tex)
			variance = torch.exp(0.5 * variance) * variance_noise
		else:
			variance = (scheduler._get_variance(t, predicted_variance=variance_tex) ** 0.5) * variance_noise

	reference_uv = torch.zeros_like(prev_tex[0])

	prev_tex = prev_tex + variance
	prev_tex = list(torch.unbind(prev_tex, dim=0))

	for i in range(len(prev_tex)):
		tex = torch.masked_select(prev_tex[i], reference_mask_update_list[i].unsqueeze(0).expand(4, -1, -1))
		reference_uv.masked_scatter_(reference_mask_update_list[i].unsqueeze(0).expand(4, -1, -1), tex)
	reference_uv = voronoi_solve(reference_uv.permute(1, 2, 0), reference_mask).permute(2, 0, 1)


	# for i in range(len(prev_tex)):
	# 	reference_uv.masked_scatter_(reference_mask_update_list[i], prev_tex[i])


	for i in range(len(prev_tex)):
		mask = (visibility_weights[i] > 0)
		prev_tex[i] = ((1 - blending_weight) * prev_tex[i] + blending_weight * reference_uv) * mask + reference_uv * ~mask

	uvp.set_texture_map(prev_tex)
	prev_views = uvp.render_textured_views()
	pred_prev_sample = torch.clone(sample)
	for i, view in enumerate(prev_views):
		pred_prev_sample[i] = view[:-1]
	masks = [view[-1:] for view in prev_views]

	return {"prev_sample": pred_prev_sample, "pred_original_sample":pred_original_sample, "prev_tex": prev_tex ,"reference_uv": reference_uv}

	if not return_dict:
		return pred_prev_sample, pred_original_sample, prev_tex, reference_uv
	pass

@torch.no_grad()
def previous_texture_tex4d(current_tex, original_tex, reference_uv, alpha_t, beta_t, alpha_t_prev, beta_t_prev, blending_weight = 0.2):
	# Eq. (6) in Tex4D
	factor = (alpha_t / beta_t) ** (0.5) * (alpha_t **(0.5) * current_tex - original_tex) + beta_t ** (0.5) * current_tex
	previous_tex = alpha_t_prev ** (0.5) * original_tex + beta_t_prev ** (0.5) * factor

	# Update reference uv and mask
	reference_mask = (reference_uv == 0).all(dim=1, keepdim=True).float()
	reference_uv =  reference_uv * (1 - reference_mask) + original_tex * reference_mask

	# Eq. (8) in Tex4D: Reference UV blending
	new_tex = (1- blending_weight) * previous_tex + blending_weight * reference_uv
	previous_tex = new_tex * reference_mask + reference_uv * (1- reference_mask)

	return previous_tex, reference_uv