import copy
import functools
import os

import blobfile as bf
import torch as th
import numpy as np

import torch_xla.distributed.xla_multiprocessing as xmp

from torch.optim import AdamW

from . import logger
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
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
        **kwargs,
    ):
        self.model = model
        self.index=index
        self.diffusion = diffusion
        self.device = device
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr*xmp._get_world_size()
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * 8
        self._load_and_sync_parameters()

        self.opt = AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(list(self.model.parameters()))
                for _ in range(len(self.ema_rate))
            ]
        self.use_ddp = False
        self.model.to(device)


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(th.load(resume_checkpoint, map_location=self.device))

        self.model.to(self.device)
        return


    def _state_dict_to_master_params(self, state_dict):
        master_params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return master_params


    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)

        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location=self.device)
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        # self.opt.to('cpu')
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=self.device)
            self.opt.load_state_dict(state_dict)
        self.opt.to(self.device)

    def run_loop(self):
        self.model.to(self.device)
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            logger.log(f'step: {self.step} starting...')
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                if self.index==0:
                    self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            if self.index==0:
                logger.log(f'Saving {self.step}step checkpoint...')
                self.save()
        logger.log('Breaking loop...')


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()


    def optimize(self, opt:th.optim.Optimizer):
        grad_norm, param_norm = self._compute_norms()
        logger.log(f"grad_norm: {grad_norm}, param_norm: {param_norm}")
        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)
        opt.step()
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
        # logger.log("batch entering...")
        # logger.log(f'model device: {self.model.device}, saved device: {self.device}')
        t, weights = self.schedule_sampler.sample(batch.shape[0], self.device)
        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                batch,
                t,
                model_kwargs=cond,
            )
        # logger.log("batch loss caclulating...")
        losses = compute_losses()
        loss = (losses["loss"] * weights).mean()
        # logger.log("batch loss caclulated...")
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        logger.log(f'loss: {loss}')
        # logger.log("batch loss backward...")
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
    
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

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

    def save(self):
        def save_checkpoint(rate, state_dict):
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.model.state_dict())
        for rate, params in zip(self.ema_rate, self.ema_params):
            logger.log(f"saving ema checkpoint ...")
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)


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


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

