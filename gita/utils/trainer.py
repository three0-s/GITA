import copy
import functools
import os

import torch_xla.distributed.parallel_loader as pl

import blobfile as bf
import torch as th
import torch_xla.core.xla_model as xm

from copy import deepcopy
from torch.optim import AdamW

import collections
from . import logger
from .resample import LossAwareSampler, UniformSampler, LossSecondMomentResampler


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def build_dataset(dataloader):
    while True:
        yield from dataloader


def starts_screen(index, width=35):
    return '\n'.join(['='*width, ' '*width, f'Epoch {index} starts !'.center(width), ' '*width, '='*width]) + '\n'


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        device,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        index=0,
        p_uncond=0.2,
        **kwargs,
    ):
        self.model = model
        self.index=index
        self.diffusion = diffusion
        self.device = device
        self.data = data
        self.val_data = val_data
        self.batch_size = batch_size
        self.p_uncond=p_uncond
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr*8
        # self.ema_rate = (
        #     [ema_rate]
        #     if isinstance(ema_rate, float)
        #     else [float(x) for x in ema_rate.split(",")]
        # )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint

        if schedule_sampler == "loss_aware_sampler":
            self.schedule_sampler = LossSecondMomentResampler(diffusion)
        elif schedule_sampler == "uniform_sampler":
            self.schedule_sampler = UniformSampler(diffusion)
        else:
            raise NotImplementedError("Shedule sampler should be either 'loss_aware_sampler' or 'uniform_sampler'")
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * 8

        self._load_and_sync_parameters()

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, 
        ) 
        # self._load_optimizer_state()
        
        # if self.resume_step:
        #     self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            # self.ema_params = [
            #     self._load_ema_parameters(rate) for rate in self.ema_rate
            # ]
        # else:
            # self.ema_params = [
            #     copy.deepcopy(list(self.model.parameters()))
            #     for _ in range(len(self.ema_rate))
            # ]
        self.model.to(device)


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if xm.is_master_ordinal():
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(th.load(resume_checkpoint, map_location='cpu'))

        self.model.to(self.device)
        return


    def _state_dict_to_master_params(self, state_dict):
        master_params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return master_params


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            # xm.rendezvous('loading optimizer')
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(th.load(opt_checkpoint, map_location='cpu'))
            
            logger.log("Optimizer loaded !")
        return

    def run_loop(self):
        self.model.to(self.device)
        if xm.is_master_ordinal():
            with open(os.path.join(logger.Logger.CURRENT.dir, 'val_loss.csv'), 'a') as f:
                f.write('Val Loss')
                f.write(',')
                f.write('Timesteps')
                f.write('\n')

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps 
        ):  
            para_loader = pl.ParallelLoader(self.data, [self.device]).per_device_loader(self.device)
            val_loader = pl.ParallelLoader(self.val_data, [self.device]).per_device_loader(self.device)
            
            losses = []
            if xm.is_master_ordinal():
                logger.log(f'Validation starts...')
            for val_batch, cond in val_loader:
                losses.append(self._forward_val(val_batch, cond))
            # losses = th.concat(losses, dim=0)
            losses = th.tensor(losses)
            loss = losses.mean().cpu().item()
            with open(os.path.join(logger.Logger.CURRENT.dir, 'val_loss.csv'), 'a') as f:
                f.write(str(loss))
                f.write(',')
                f.write(str(self.step))
                f.write('\n')

            logger.log(f'val loss: {loss}')
            
            for batch, cond in para_loader:
                if xm.is_master_ordinal():
                    logger.log(f'step: {self.step + self.resume_step} starting...')
                self.run_step(batch, cond)
                if self.step % self.log_interval == 0:
                    logger.dumpkvs()

                if self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return
                self.step += 1
            
            
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0 and xm.is_master_ordinal():
            logger.log(f'Saving {self.step + self.resume_step} step checkpoint...')
            self.save()
        logger.log('Breaking loop...')


    def run_step(self, batch, cond):
        # classifier-free guidance training
        cond_ = cond
        if th.rand(1) <= self.p_uncond:
            if xm.is_master_ordinal():
                logger.log(f'Randomly masking the condition image for unconditional image generation...')
            cond_['condi_img'] = th.zeros_like(cond['condi_img']).to(cond['condi_img'])

        self.forward_backward(batch, cond_)
        took_step = self.optimize(self.opt)

        # not using ema 
        # if took_step:
        #     self._update_ema()
        self._anneal_lr()
        self.log_step()


    def optimize(self, opt:th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.log(f"grad_norm: {grad_norm.item()}, param_norm: {param_norm.item()}")
        logger.logkv_mean("grad_norm", grad_norm.item())
        logger.logkv_mean("param_norm", param_norm.item())
        xm.optimizer_step(opt, barrier=True)
        return True


    def _compute_norms(self, grad_scale=1.0):
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.model.parameters():
            with th.no_grad():
                try:
                    param_norm += th.norm(p, p=2) ** 2
                except:
                    logger.log('Error caused while calculating param norm..')
                if p.grad is not None:
                    grad_norm += th.norm(p.grad, p=2) ** 2
        return th.sqrt(grad_norm) / grad_scale, th.sqrt(param_norm)   

    def _zero_grad(self):
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def forward_backward(self, batch, cond):
        self._zero_grad()
        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        s, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                batch,
                t,
                s=s,
                model_kwargs=cond,
            )
        # logger.log("batch loss caclulating...")
        losses = compute_losses()
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        loss = (losses["loss"] * weights).mean()
        # logger.log("batch loss caclulated...")
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        logger.log(f'loss: {loss}')
        # logger.log("batch loss backward...")
        self.opt.zero_grad()
        loss.backward(loss)
        # logger.log("batch loss backward completed !")
        '''
        for i in range(0, batch.shape[0], self.microbatch):
            logger.log("batch entering...")
            logger.log(f'model device: {self.model.device}, saved device: {self.device}')
            micro = batch[i : i + self.microbatch].to(self.device)
            micro_cond = {
                k: v[i : i + self.microbatch].to(self.device)
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            logger.log("batch loss caclulating...")
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            logger.log("batch loss caclulated ")
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            logger.log("batch loss backward...")
            loss.backward(loss)
            logger.log("batch loss backward completed !")
        '''
    

    def _forward_val(self, batch, cond):
        with th.no_grad():
            self._zero_grad()
            t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
            s, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
            compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    batch,
                    t,
                    s=s,
                    model_kwargs=cond,
                )
            
            # logger.log("batch loss caclulating...")
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()
            # logger.log("batch loss caclulated...")
            # logger.log("batch loss backward...")
            self.opt.zero_grad()
            return loss
    # def _update_ema(self):
    #     for rate, params in zip(self.ema_rate, self.ema_params):
    #         update_ema(params, self.model.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)


    def params_to_state_dict(self, params):
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = params[i]

        return state_dict

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.params_to_state_dict(params) if type(params) != collections.OrderedDict else params
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            xm.save(state_dict, os.path.join(get_blob_logdir(), filename), global_master=True)
        
        save_checkpoint(0, self.model.state_dict())
        xm.save(self.opt.state_dict(), os.path.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"), global_master=True)
        if xm.is_master_ordinal():
            logger.log(f"saved !")

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


# def find_ema_checkpoint(main_checkpoint, step, rate):
#     if main_checkpoint is None:
#         return None
#     filename = f"ema_{rate}_{(step):06d}.pt"
#     path = bf.join(bf.dirname(main_checkpoint), filename)
#     if bf.exists(path):
#         return path
#     return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

