from PIL import Image
import torch 
from gita.utils.model_creation import create_model_and_diffusion, model_and_diffusion_defaults
import clip
from torch.utils.data import DataLoader
from gita.data.teeth_img import PairedTeethImageData
from gita.utils import logger

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import os
from torchvision.transforms import Resize
from tqdm.auto import tqdm

def create_argparser():
    defaults = dict(
        data_dir="/home/yewon/GITA/dataset/val",
        out_dir='',
        schedule_sampler="uniform_sampler",
        lr=4e-5,
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

def inverse_normalize(img, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    """
    :param img: numpy array. shape (C, H, W). [-1~1]
    :return: numpy array. shape (height, width, channel). [0~255]
    """
    for i in range(3):
        img[:, i,:, :] = ((img[:, i,:,:]) * std[i]) + mean[i]

    return img*255

# dynamic thr from imagen (Photorealistic Text-to-Image Diffusion Models
#                           with Deep Language Understanding)
# codes are motivated from https://github.com/lucidrains/imagen-pytorch/blob/e10e92e9198982790bc45fc151e880ae3c47ece1/imagen_pytorch/imagen_pytorch.py#L128
def dynamic_thr(x:torch.Tensor, p=0.95):
    def right_pad_dims_to(x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))
    s = torch.quantile(x.reshape(x.shape[0], -1).abs(), p, dim=-1)
    s.clamp_(min=1.0)
    s = right_pad_dims_to(x, s)
    return x.clamp(-s, s) / s

def save_images(batch: torch.Tensor, fnames, dir):
    """ Save a batch of images. """
    scaled = inverse_normalize(batch).round().clamp(0,255)
    # scaled = xm._maybe_convert_to_cpu(scaled)
    # (B, C, H, W) == > (H, B, W, C) == > (H, B x W, C)
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    images = torch.chunk(reshaped, batch.shape[0], dim=1)
    for i, image in tqdm(enumerate(images)):
        Image.fromarray(image.cpu().to(torch.uint8).numpy()).save(os.path.join(dir, fnames[i]+'.png'))
        xm.mark_step()

def main():
    args = create_argparser()#.parse_args()
    args.update(num_channels=128, 
                clip_model_name='ViT-B/16',
                out_dir='/home/yewon/gita-log/dynamic_thr')

    logger.configure(dir=args['out_dir'])
    
    logger.log('='*8+' Creating Clip Encoder... '.center(34)+'='*8)
    clip_model, preprocess = clip.load(args['clip_model_name'])
    img_encoder = clip_model.visual
    logger.log('='*8+' Completed ! '.center(34)+'='*8)

    args.update(img_encoder=img_encoder, 
                encoding_dim=img_encoder.output_dim, 
                seed=928,
                aug_level=0.07,
                save_interval=2000,
                timestep_respacing='150',
                guidance_scale = 10.0,
                resume_checkpoint='/home/yewon/gita-log/checkpoint/model005000.pt',
                )

    logger.log('='*8+' Creating diffusion model... '.center(34)+'='*8)
    model, diffusion = create_model_and_diffusion(**args)
    logger.log('='*8+' Completed ! '.center(34)+'='*8)


    model.eval()
    logger.log(f"loading model from {args['resume_checkpoint']}...")
    model.load_state_dict(torch.load(args['resume_checkpoint'], map_location='cpu'))
    logger.log('loaded !')

    args.update(num_cores=8)

    logger.log('='*8+' GITA Sampling started... '.center(34)+'='*8)
    logger.log('='*8+f"Saving results at {args['out_dir']}".center(34)+'='*8)
    os.makedirs(os.path.join(args['out_dir'], 'gt'), exist_ok=True)
    os.makedirs(os.path.join(args['out_dir'], 'pred'), exist_ok=True)
    # os.makedirs(os.path.join(args['out_dir'], 'compare'), exist_ok=True)
    # test(0, args, model, diffusion)
    xmp.spawn(test, args=(args, model, diffusion), nprocs=args['num_cores'], start_method='fork')   

def test(index, flags, model, diffusion):
    device = xm.xla_device()
    torch.manual_seed(flags['seed'])

    model.to(device)
    
    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + flags['guidance_scale'] * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    dataset = PairedTeethImageData(flags['data_dir'], istrain=False, condi_aug_level=flags['aug_level'])

    test_sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=xm.get_ordinal(),
                    shuffle=True)
    loader = DataLoader(dataset, batch_size=flags['batch_size'], 
                        sampler=test_sampler, num_workers=flags['num_cores'], drop_last=False)

    test_loader = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    resizer = Resize(flags['image_size'])

    for i, (batch, cond) in enumerate(test_loader):
        if xm.is_master_ordinal():
            logger.log(f"{i+1}th batch...")
        # xm.master_print(f"batch: {batch.shape} cond: {cond['condi_img'].shape}")
        cond['condi_img'] = torch.concat([cond['condi_img'], torch.zeros_like(cond['condi_img']).to(cond['condi_img'])], dim=0).to(device)
        samples = diffusion.p_sample_loop(
                    model_fn,
                    (batch.shape[0]*2, 3, flags["image_size"], flags["image_size"]),
                    device=device,
                    clip_denoised=True,
                    denoised_fn=dynamic_thr,
                    progress=True,
                    model_kwargs=cond,
                    cond_fn=None,)[:batch.shape[0]]
        gt = batch
        condi = resizer(cond['condi_img'])
        fnames = cond['id']
        logger.log(f"Saving {i+1}th images...")
        save_images(gt, fnames, os.path.join(flags['out_dir'], 'gt'))
        save_images(samples, fnames, os.path.join(flags['out_dir'], 'pred'))
        # save_images(torch.concat([condi[:flags['batch_size'], ...], samples, gt], axis=-1).to(gt), fnames, os.path.join(flags['out_dir'], 'compare'))
        # xm.rendezvous('init')
               

if __name__ == '__main__':
    main()


