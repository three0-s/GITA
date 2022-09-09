import sys
sys.path.append('/home/yewon/GITA')

from gita.utils.trainer import TrainLoop
from gita.utils.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
import clip

import torch
from gita.utils.tables import print_table
from gita.utils import logger
from gita.utils.scripts_util import args_to_dict, add_dict_to_argparser
from torch.utils.data import DataLoader
from gita.data.teeth_img import PairedTeethImageData

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp



def create_argparser():
    defaults = dict(
        data_dir="/home/yewon/GITA/dataset/train",
        schedule_sampler=None,
        lr=1e-4,
        weight_decay=1e-4,
        lr_anneal_steps=0,
        batch_size=40,
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
    # parser = argparse.ArgumentParser()
    # add_dict_to_argparser(parser, defaults)
    return defaults



def main():
    args = create_argparser()#.parse_args()
    args.update(num_channels=128, 
                clip_model_name='ViT-B/16',)
    logger.configure()
    
    logger.log('='*8+' Creating Clip Encoder... '.center(34)+'='*8)
    clip_model, preprocess = clip.load(args['clip_model_name'])
    img_encoder = clip_model.visual
    logger.log('='*8+' Completed ! '.center(34)+'='*8)
    # model_kwargs = args_to_dict(args, model_and_diffusion_defaults().keys())

    args.update(img_encoder=img_encoder, 
                encoding_dim=img_encoder.output_dim, 
                aug_level=0.07,
                seed=928,
                # resume_checkpoint='/home/yewon/gita-log/gita-2022-09-09-05-27-12-593841/model000000.pt',
                )

    logger.log('='*8+' Creating diffusion model... '.center(34)+'='*8)
    model, diffusion = create_model_and_diffusion(
        **args)
    logger.log('='*8+' Completed ! '.center(34)+'='*8)
    args.update(num_cores=8, diffusion=diffusion)

    logger.log('='*8+' INPUT PARAMETERS '.center(34)+'='*8)
    row_names=[]
    rows=[]
    for key, values in args.items():
        if key in ['model', 'img_encoder']:
            continue
        row_names.append(key)
        rows.append(values)
    logger.log(print_table(row_names, rows))
    logger.log('='*8+' GITA Training started... '.center(34)+'='*8)
    xmp.spawn(train, args=(args, model), nprocs=args['num_cores'], start_method='fork')    

def train(index, flags, model, **kwargs):
    device = xm.xla_device()
    torch.manual_seed(flags['seed'])
    dataset = PairedTeethImageData(flags['data_dir'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal(),
                    shuffle=True)
    loader = DataLoader(dataset, batch_size=flags['batch_size'], 
                        sampler=train_sampler, num_workers=flags['num_cores'],
                        drop_last=True)
    if xm.is_master_ordinal():
        logger.log('='*8+' Creating Trainer... '.center(34)+'='*8)
    
    model.to(device)
    
    trainer = TrainLoop(**flags, model=model, data=loader, device=device, index=index)
    if xm.is_master_ordinal():
        logger.log('='*8+' Created Trainer ! '.center(34)+'='*8)
    trainer.run_loop()
    xm.mesh_reduce()

if __name__=='__main__':
    main()