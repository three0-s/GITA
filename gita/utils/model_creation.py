# Motivated from https://github.com/openai/glide-text2im/blob/main/glide_text2im/model_creation.py
from gita.model.img2img import GITA, SuperResGITA
from gita.model.unet import SuperResUNetModel
from .gaussian_diffusion import get_named_beta_schedule
from .respace import SpacedDiffusion, space_timesteps
from . import gaussian_diffusion as gd


def model_and_diffusion_defaults():
    return dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="",
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        super_res=False,
    )

def model_and_diffusion_defaults_upsampler():
    result = model_and_diffusion_defaults()
    result.update(
        dict(
            image_size=256,
            num_res_blocks=2,
            noise_schedule="linear",
            super_res=True,
        )
    )
    return result

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    img_encoder,
    encoding_dim,
    super_res,
    **kwargs,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        assert 2 ** (len(channel_mult) + 2) == image_size

    attention_ds = []
    if attention_resolutions != "":
        for res in attention_resolutions.split(","):
            attention_ds.append(image_size // int(res))
            
    # Only supports image to image translation and Upsampling
    model_cls = GITA if not super_res else  SuperResGITA
    return model_cls(
        img_encoder=img_encoder,
        encoding_dim=encoding_dim,  
        in_channels=3,
        model_channels=num_channels,
        out_channels=6,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
    )

# def create_gaussian_diffusion(
#     steps,
#     noise_schedule,
#     timestep_respacing,
# ):
#     betas = get_named_beta_schedule(noise_schedule, steps)
#     if not timestep_respacing:
#         timestep_respacing = [steps]
#     return SpacedDiffusion(
#         use_timesteps=space_timesteps(steps, timestep_respacing),
#         betas=betas,
#     )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    img_encoder,
    aug_level,
    encoding_dim,
    super_res,
    **kwargs,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        img_encoder=img_encoder,
        encoding_dim=encoding_dim,
        super_res=super_res,
        **kwargs,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion