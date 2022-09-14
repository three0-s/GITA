from gita.utils.model_creation import create_model, model_and_diffusion_defaults, model_and_diffusion_defaults_upsampler
import clip
import torch
import numpy as np

clip_model, preprocess = clip.load('ViT-B/16')
img_encoder = clip_model.visual

condi = torch.zeros(1, 3, 224, 224)
low_res = torch.zeros(1, 3, 64, 64)
timestep=torch.tensor([100])
dummy = torch.zeros(1, 3, 256, 256)
model_kwargs = model_and_diffusion_defaults_upsampler()
model_kwargs.update(dict(num_channels=96))
print(img_encoder.output_dim)
model = create_model(img_encoder=img_encoder, encoding_dim=img_encoder.output_dim, aug_level=0.1, **model_kwargs)
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters() if p.requires_grad ]):,}")
out = model(dummy, timestep, condi, low_res=low_res)
print(out.shape)
print(type(model))