from utils.model_creation import create_model, model_and_diffusion_defaults
import clip
import torch

clip_model, preprocess = clip.load('ViT-B/32')
img_encoder = clip_model.visual

condi = torch.zeros(1, 3, 224, 224)
timestep=torch.tensor([100])
dummy = torch.zeros(1, 3, 64, 64)
model_kwargs = model_and_diffusion_defaults()
model = create_model(img_encoder=img_encoder, aug_level=0.1, **model_kwargs)

out = model(dummy, timestep, condi)