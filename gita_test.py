from gita.utils.model_creation import create_model, model_and_diffusion_defaults
import clip
import torch
import numpy as np

clip_model, preprocess = clip.load('ViT-L/14')
img_encoder = clip_model.visual

condi = torch.zeros(1, 3, 224, 224)
timestep=torch.tensor([100])
dummy = torch.zeros(1, 3, 64, 64)
model_kwargs = model_and_diffusion_defaults()
model_kwargs.update(dict(num_head_channels=64, num_res_blocks=3))
print(img_encoder.output_dim)
model = create_model(img_encoder=img_encoder, encoding_dim=img_encoder.output_dim, aug_level=0.1, **model_kwargs)
print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters() ]):,}")
out = model(dummy, timestep, condi)
print(out.shape)