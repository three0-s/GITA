import os
from clip.clip import BICUBIC
from torch.utils.data import Dataset
import torch
from torchdata.datapipes.iter import FSSpecFileLister
from PIL import Image
import glob
import os
from torchvision.transforms import ToTensor, Resize, Compose, Normalize, CenterCrop


class TeethImageData_GCS(Dataset):
    def __init__(self, input_img_dir, cond_img_dir, transform=None, device=None, meta={}):
        self.device = device if device != None else "cuda" if torch.cuda.is_available() else "cpu"
        #img_dir: gcs bucket path

        self.input_datapipe = FSSpecFileLister(root=input_img_dir, masks=['*.jpg'])
        self.input_file_dp = sorted(list(self.input_datapipe.open_files_by_fsspec(mode='rb')))
        self.cond_datapipe = FSSpecFileLister(root=cond_img_dir, masks=['*.jpg'])
        self.cond_file_dp = sorted(list(self.cond_datapipe.open_files_by_fsspec(mode='rb')))

        self.transform = transform
        self.img_meta = meta

    def __len__(self):
        return len(self.input_file_dp)
    
    def __getitem__(self, index):
        input_img_path = self.input_file_dp[index][1]
        input_fname = self.input_file_dp[index][0].split('/')[-1]
        input_img = Image.open(input_img_path)
        if self.transform:
            input_img = self.transform(input_img)
        input_img = input_img.to(self.device)

        cond_img_path = self.cond_file_dp[index][1]
        cond_fname = self.cond_file_dp[index][0].split('/')[-1]
        cond_img = Image.open(cond_img_path)
        if self.transform:
            cond_img = self.transform(cond_img)
        cond_img = cond_img.to(self.device)

        sample = {'input_image':input_img, 'input_fname':input_fname, 
                  'cond_image':cond_img, 'cond_fname':cond_fname ,**self.img_meta}
        return sample

class PairedTeethImageData(TeethImageData_GCS):
    def __init__(self, img_dir, img_size=64, condi_size=224, transform=None, device=None, meta={}):
        super().__init__('', '', transform, device, meta)
        self.img_dir = img_dir
        self.img_list = glob.glob(os.path.join(img_dir, '*.png'))
        self.img_size = img_size
        self.condi_size = condi_size

        self.img_resizer = Compose([Resize(self.img_size, interpolation=BICUBIC), CenterCrop(self.img_size),
                                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

        self.condi_resizer = Compose([Resize(self.condi_size, interpolation=BICUBIC), CenterCrop(self.condi_size),
                                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_id = img_path.split('/')[-1].split('.')[0]
        img = ToTensor()(Image.open(img_path))
        input_img, condi = torch.chunk(img, 2, dim=-1)
        input_img = self.img_resizer(input_img)
        condi = self.condi_resizer(condi)
        

        # sample = {'input_image':input_img, 'img_id':img_id, 
        #           'cond_image':condi, **self.img_meta}
        return (input_img.to(self.device), {'condi_img':condi.to(self.device)})