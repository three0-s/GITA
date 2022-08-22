import torch
import clip
from PIL import Image
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl

from glob import glob
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import torchvision
import shutil
from torchdata.datapipes.iter import FSSpecFileLister
from PIL import Image
import torch_xla.debug.metrics as met

front_bucket = "gs://img2img/front"
left_bucket = "gs://img2img/left"

class TeethImageData(Dataset):
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

SERIAL_EXEC = xmp.MpSerialExecutor()
os.environ['XLA_IR_DEBUG'] = '1'

FLAGS = {}
FLAGS['num_cores'] = 8
FLAGS['batch_size'] = 4
model, preprocess = clip.load("ViT-B/32")
WRAPPED_MODEL = xmp.MpModelWrapper(model)

def run_clip(index, flags):
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    dataset = SERIAL_EXEC.run(lambda : TeethImageData(front_bucket, left_bucket, transform=preprocess))
    loader = DataLoader(dataset, batch_size=flags['batch_size'])
    para_loader = pl.ParallelLoader(loader, [device])

    for i, data in tqdm(enumerate(para_loader.per_device_loader(device))):
        input_image = data['input_image']
        cond_image = data['cond_image']
        with torch.no_grad():
            input_image_features = model.encode_image(input_image)
            cond_image_features = model.encode_image(cond_image)

        xm.master_print("cos similiraty:", torch.nn.CosineSimilarity()(input_image_features, cond_image_features))
        xm.master_print('\n', data['input_fname'], '\t||\t', data['cond_fname'], '\n')
        if i > 10:
            break
xmp.spawn(run_clip, args=(FLAGS,), nprocs=FLAGS['num_cores'],
          start_method='fork')