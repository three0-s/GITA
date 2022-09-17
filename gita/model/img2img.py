import torch as th
import torch.nn as nn

from gita.utils.nn import timestep_embedding
from .unet import UNetModel
import torch.nn.functional as F

# @author: ga06033@yonsei.ac.kr (Yewon Lim)
# Genral Image_to_Image Translation Architecture (GITA)
class GITA(UNetModel):
    def __init__(self, img_encoder, encoding_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_encoder = img_encoder
        if img_encoder != None:
            for param in self.img_encoder.parameters():
                param.requires_grad = False
            self.img_encoder.to(self.device)
            self.encoding_dim = encoding_dim
            self.embed_linear_transform = nn.Linear(self.encoding_dim, self.model_channels*4, device=self.device, dtype=self.dtype)
            self.device = list(img_encoder.modules())[1].weight.data.device
            self.dtype = list(img_encoder.modules())[1].weight.data.dtype
        else:
            self.device=th.device('cpu')
            self.dtype=th.float32
        
        # self.cache = None  # We need to cache encoding (or embedding) of condition image to reduce FLOPS.
        self.to(self.device)
        
    # # Motivated by GLIDE (https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/text2im_model.py#L120)
    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        assert next(self.parameters()).device == device
        self.device = device


    def forward(self, x, timesteps, condi_img=None, **kwargs):
        embedding = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if condi_img is not None:
            # # embedding augmentation (Gaussian noise) 
            # condi_img += th.randn_like(condi_img, dtype=self.dtype, device=self.device) * self.aug_level
            #   ==> moved to gita/model/img2img.py
            img_embedding = self.img_encoder(condi_img)
                
            img_embedding = self.embed_linear_transform(img_embedding)
            embedding += img_embedding

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, embedding)
            hs.append(h)
        h = self.middle_block(h, embedding)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, embedding)
        h = h.type(x.dtype)
        h = self.out(h)
        return h

class SuperResGITA(GITA):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, condi_img=None, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, condi_img=None, **kwargs)