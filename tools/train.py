from utils.trainer import TrainLoop
from utils.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
import clip
import torch
import argparse
from utils import logger
from utils.scripts_util import args_to_dict, add_dict_to_argparser
from torch.utils.data import DataLoader

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=1e-4,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        clip_model_name='ViT-L/14',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



def main():
    args = create_argparser().parse_args()
    logger.configure()
    logger.log('='*8+' Creating Clip Encoder... '+'='*8)
    clip_model, preprocess = clip.load(args.clip_model_name)
    img_encoder = clip_model.visual
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    FLAGS = {}
    FLAGS['num_cores'] = 8
    FLAGS['batch_size'] = 2
    FLAGS['model_name'] = 'ViT-L/14'

def train(index, flags, **kwargs):
    pass