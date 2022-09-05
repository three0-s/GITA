import torch as th
import torch.nn as nn

from GITA.utils.nn import timestep_embedding
from .unet import UNetModel

# @author: ga06033@yonsei.ac.kr (Yewon Lim)
# Genral Image_to_Image Translation Architecture (GITA)
class GITA(UNetModel):
    def __init__(self, img_encoder, aug_level, encoding_dim=512, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_encoder = img_encoder
        for param in self.img_encoder.parameters():
            param.requires_grad = False

        self.encoding_dim = encoding_dim
        self.device = list(img_encoder.modules())[1].weight.data.device
        self.dtype = list(img_encoder.modules())[1].weight.data.dtype
        self.aug_level = aug_level 
        self.embed_linear_transform = nn.Linear(self.encoding_dim, self.model_channels*4, device=self.device, dtype=self.dtype)
        # self.cache = None  # We need to cache encoding (or embedding) of condition image to reduce FLOPS.
        self.to(self.device)
        
    # # Motivated by GLIDE (https://github.com/openai/glide-text2im/blob/69b530740eb6cef69442d6180579ef5ba9ef063e/glide_text2im/text2im_model.py#L120)
    # def del_cache(self):
    #     self.cache = None

    def forward(self, x, timesteps, condi_img=None):
        # if self.cache is not None:
        #     img_embedding = self.cache
        # else:
        #     img_embedding = self.embed_linear_transform(self.img_encoder(x))
        #     self.cache = img_embedding
       
        embedding = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if condi_img is not None:
            # embedding augmentation (Gaussian noise)
            condi_img += th.randn_like(condi_img, dtype=self.dtype, device=self.device) * self.aug_level
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