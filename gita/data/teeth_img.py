import os
from clip.clip import BICUBIC
from torch.utils.data import Dataset
import torch
from PIL import Image
import glob
import os
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, RandomCrop, RandomRotation, ColorJitter


class GaussianNoise(torch.nn.Module):
    def __init__(self, noise_level=0.1):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, x):
        out = x + (torch.randn_like(x).to(x) * self.noise_level)
        return out

class PairedTeethImageData(Dataset):
    def __init__(self, img_dir, istrain=True, condi_aug_level=0.07, img_size=64, condi_size=224, transform=None, device=None, meta={}):
        super().__init__()
        self.device = device if device != None else "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transform
        self.img_meta = meta
        self.img_dir = img_dir
        self.img_list = glob.glob(os.path.join(img_dir, '*.png'))
        self.img_size = img_size
        self.condi_size = condi_size
        self.istrain = istrain
        self.condi_aug_level = condi_aug_level

        self.img_resizer = Compose([Resize((int(self.img_size*1.1),int(self.img_size*1.1)), interpolation=BICUBIC), 
                                    # RandomRotation(25),
                                    RandomCrop(self.img_size),
                                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        self.condi_resizer = Compose([Resize((int(self.condi_size*1.1),int(self.condi_size*1.1)), interpolation=BICUBIC), 
                                    ColorJitter(brightness=(0, 0.5), contrast=(0, 0.5)),
                                    RandomRotation(20),
                                    RandomCrop(self.condi_size),
                                    GaussianNoise(self.condi_aug_level),
                                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))]) if self.istrain else \
                             Compose([Resize((int(self.condi_size),int(self.condi_size)),interpolation=BICUBIC), 
                                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_id = img_path.split('/')[-1].split('.')[0]
        img = ToTensor()(Image.open(img_path))
        condi, input_img = torch.chunk(img, 2, dim=-1)
        input_img = self.img_resizer(input_img)
        condi = self.condi_resizer(condi)
        

        # sample = {'input_image':input_img, 'img_id':img_id, 
        #           'cond_image':condi, **self.img_meta}
        return (input_img.to(self.device), {'condi_img':condi.to(self.device), 'id':img_id})