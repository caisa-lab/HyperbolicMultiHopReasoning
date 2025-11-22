from torch import optim
import os
import sys
import math
from typing import Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler
from transformers import Adafactor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from geoopt.optim import RiemannianAdam

from src.models import SoftPromptModel
from src.config import BaseTrainingConfig


# ---------- DDP helpers ----------

def unwrap_model(m: nn.Module) -> nn.Module:
    """Return the underlying model if wrapped in DDP/DataParallel, else m."""
    return m.module if isinstance(m, (DDP, DataParallel)) else m

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_main_process() -> bool:
    """True on the single process case or rank 0 in distributed."""
    if not dist.is_available():
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def device_index_from_str(device_str: str):
    """
    Convert 'cuda:0' -> 0, 'cuda' -> current_device, 'cpu' -> None.
    Useful for DDP(device_ids=[idx]) single-process-per-GPU setups.
    """
    if device_str.startswith("cuda"):
        dev = torch.device(device_str)
        if dev.index is not None:
            return dev.index
        return torch.cuda.current_device()
    return None


# ---------- Scheduler ----------

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0,
                 gamma=1.0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected positive integer T_up, but got {T_up}")
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [
                (self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.eta_max - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi * (self.T_cur - self.T_up)
                        / (self.T_i - self.T_up)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1),
                            self.T_mult,
                        )
                    )
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult**n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


# ---------- Optimizer / dirs ----------

def get_optimizer(parameters, trainer_config: BaseTrainingConfig):
    lr = trainer_config.learning_rate
    print(
        f"Training with optimizer {trainer_config.optimizer} "
        f"and Learning Rate {trainer_config.learning_rate}"
    )
    if trainer_config.optimizer == "Adam":
        optimizer = optim.Adam(parameters, lr=lr)
    elif trainer_config.optimizer == "AdamW":
        optimizer = optim.AdamW(
            parameters, lr=lr, weight_decay=trainer_config.optimizer_param
        )
    elif trainer_config.optimizer == "AdaFactor":
        optimizer = Adafactor(
            parameters,
            lr=lr,
            weight_decay=trainer_config.optimizer_param,
            relative_step=False,
            scale_parameter=False,
        )
    elif trainer_config.optimizer == "Hyperbolic":
        optimizer = RiemannianAdam(
            parameters, lr=lr, weight_decay=trainer_config.optimizer_param
        )
    else:
        raise ValueError(f"Unsupported optimizer: {trainer_config.optimizer}")
    return optimizer


def setup_directories(trainer_config: BaseTrainingConfig, t5_config):
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    optimizer = trainer_config.optimizer
    final_string = f"{current_time}_{optimizer}_{trainer_config.additional_log_info}"
    log_dir = os.path.join(trainer_config.log_dir, final_string)
    model_dir = os.path.join(trainer_config.model_save_path, final_string)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    return log_dir, model_dir


# ---------- Checkpoint helpers (DDP-safe) ----------

def save_checkpoint(path, model, optimizer=None, epoch=None, extra=None):
    """
    DDP-safe save helper.

    Always unwraps the model so checkpoints never contain a 'module.' prefix.
    """
    model_to_save = unwrap_model(model)
    ckpt = {"model_state_dict": model_to_save.state_dict()}
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        ckpt["epoch"] = epoch
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)
    if is_main_process():
        print(f"Saved checkpoint to {path}")


def load_model_checkpoint(model: nn.Module,
                          checkpoint_path: str,
                          device: Union[str, None] = None,
                          strict: bool = False):
    """
    Load a checkpoint into `model` (possibly DDP-wrapped) in a DDP-agnostic way.

    - Works whether `model` is plain, DDP, or DataParallel.
    - Works whether the saved state_dict has 'module.' prefixes or not.
    - Returns (model, checkpoint_dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 1) pull out the state dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # backward compatibility: checkpoint is a pure state_dict
        state_dict = checkpoint

    # 2) handle optional 'module.' prefix automatically
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    target = unwrap_model(model)
    missing, unexpected = target.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"{missing = }")
        print(f"{unexpected = }")
    print(f"Loaded checkpoint from {checkpoint_path}")

    return model, checkpoint


def load_optimizer_and_start_epoch(optimizer: optim.Optimizer,
                                   checkpoint_path: str,
                                   device: Union[str, None] = None):
    """
    Load optimizer state and epoch from a unified checkpoint.

    Expects the checkpoint to have:
      - 'optimizer_state_dict'
      - 'epoch' (optional, default 0)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0)
        print(f"Loaded optimizer state from {checkpoint_path}, start_epoch={start_epoch}")
        return optimizer, start_epoch
    else:
        print(f"No optimizer_state_dict found in {checkpoint_path}, starting from epoch 0")
        return optimizer, 0


# ---------- Soft prompt & additional layer ----------

def load_soft_prompt_and_additional_layer(model: SoftPromptModel,
                                          soft_prompt_path: str,
                                          device: Union[str, None] = None):
    """
    Load soft prompt and additional layer into a SoftPromptModel, DDP-agnostic.

    Assumes checkpoint has:
      - 'soft_prompt_state_dict' (your stored tensor/params)
      - 'additional_linear_layer' (state_dict of additional layer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(soft_prompt_path, map_location=device)
    soft_prompt = checkpoint["soft_prompt_state_dict"]
    additional_layer = checkpoint["additional_linear_layer"]

    net = unwrap_model(model)
    net.soft_prompt = soft_prompt
    net.knit5.additional_layer.load_state_dict(additional_layer)

    model.to(device)
    print(f"Loading Soft Prompt and Additional Layer from {soft_prompt_path}")


# ---------- Logging ----------

def log_tensorboard(writer: SummaryWriter,
                    value: float,
                    idx,
                    phase: str,
                    eval_metric: str = "loss"):
    if eval_metric not in ["loss", "em", "f1"]:
        raise ValueError(
            f"Unsupported eval_metric: {eval_metric}. Supported are [loss, em, f1]"
        )
    writer.add_scalar(f"{phase}/{eval_metric}", value, idx)


# ---------- Hyperbolic helpers (unchanged) ----------

def geodesic_distance(u, v, c=1.0):
    norm_u = torch.norm(u, p=2, dim=-1)
    norm_v = torch.norm(v, p=2, dim=-1)
    numerator = torch.norm(u - v, p=2, dim=-1) ** 2
    denominator = (1 - c * norm_u**2) * (1 - c * norm_v**2)
    distance = torch.acosh(1 + 2 * c * numerator / denominator)
    return distance


def geodesic_regularization(soft_prompt_input, min_distance=1.0, c=1.0):
    loss = 0.0
    num_components = soft_prompt_input.size(1)
    for i in range(num_components):
        for j in range(i + 1, num_components):
            distance = geodesic_distance(
                soft_prompt_input[:, i], soft_prompt_input[:, j], c
            )
            loss += torch.clamp(min_distance - distance, min=0).mean()
    return loss
