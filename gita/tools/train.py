from gita.utils.trainer import TrainLoop
from gita.utils.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
import clip

import argparse
from gita.utils import logger
from gita.utils.scripts_util import args_to_dict, add_dict_to_argparser
from torch.utils.data import DataLoader
from gita.data.teeth_img import PairedTeethImageData

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
        batch_size=2,
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

def build_dataset(dataloader):
    yield from dataloader

def main():
    args = create_argparser().parse_args()
    
    logger.configure()
    logger.log('='*8+' Creating Clip Encoder... '+'='*8)
    clip_model, preprocess = clip.load(args.clip_model_name)
    img_encoder = clip_model.visual
    logger.log('='*8+' Completed ! '+'='*8)
    model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())
    model_kwargs.update(img_encoder=img_encoder, encoding_dim=img_encoder.output_dim, aug_level=0.07)
    logger.log('='*8+' Creating diffusion model... '+'='*8)
    model, diffusion = create_model_and_diffusion(
        **model_kwargs)
    logger.log('='*8+' Completed ! '+'='*8)
    model_kwargs.update(num_cores=8, model=model, diffusion=diffusion, preprocess=preprocess)
    xmp.spawn(train, args=(model_kwargs,), nprocs=model_kwargs['num_cores'])    

def train(index, flags, **kwargs):
    device = xm.xla_device()
    SERIAL_EXEC = xmp.MpSerialExecutor()
    dataset = SERIAL_EXEC.run(lambda: PairedTeethImageData(flags['data_dir'], flags['preprocess']))
    loader = DataLoader(dataset, batch_size=flags['batch_size'])
    para_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    data=build_dataset(para_loader)
    trainer = TrainLoop(**flags, data=data)
    trainer.run_loop()