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
from gita.data.teeth_img import PairedTeethImageData

from PIL import Image
import torch_xla.debug.metrics as met


img_dir = '/home/yewon/GITA/dataset/train'

SERIAL_EXEC = xmp.MpSerialExecutor()
os.environ['XLA_IR_DEBUG'] = '1'

FLAGS = {}
FLAGS['num_cores'] = 8
FLAGS['batch_size'] = 2
FLAGS['model_name'] = 'ViT-L/14'
model, preprocess = clip.load(FLAGS['model_name'])
WRAPPED_MODEL = xmp.MpModelWrapper(model)

def run_clip(index, flags):
    device = xm.xla_device()
    model = WRAPPED_MODEL.to(device)
    # print(f"model device: {model.device}")
    dataset = SERIAL_EXEC.run(lambda : PairedTeethImageData(img_dir))
    loader = DataLoader(dataset, batch_size=flags['batch_size'])
    para_loader = pl.ParallelLoader(loader, [device])

    for i, (input_image, condi) in tqdm(enumerate(para_loader.per_device_loader(device))):
        
        cond_image = condi['condi_img']
        with torch.no_grad():
            # input_image_features = model.encode_image(input_image)
            cond_image_features = model.encode_image(cond_image)

        # xm.master_print("cos similiraty:", torch.nn.CosineSimilarity()(input_image_features, cond_image_features))
        # xm.master_print('\n', data['input_fname'], '\t||\t', data['cond_fname'], '\n')
        if i > 10:
            break
xmp.spawn(run_clip, args=(FLAGS,), nprocs=FLAGS['num_cores'],
          start_method='fork')