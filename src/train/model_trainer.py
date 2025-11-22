import os
import sys
from datetime import datetime
from typing import Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer

from src.eval import exact_match_score, f1_score
from src.config import Config
from src.utils.trainer_utils import * 
from src.models import T5ModelWithAdditionalLayer

"""Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""
class ModelTrainer:
    def __init__(self,
                 model: Union[nn.Module, T5ModelWithAdditionalLayer, DDP],
                 tokenizer: AutoTokenizer,
                 list_train_dataloader: list,
                 val_dataloader: DataLoader,
                 config: Config,
                 device: str = "cpu",
                 checkpoint_path: str = None,
                 tboard_checkpoint_path: str = None,
                 validation_step: int = 1,
                 method: str = "single_hop_training",
                 load_optimizer: bool = True,
                 gpu_parallelization: bool = False,
                 rank: int = 0):
        self.device = device
        self.gpu_parallelization = gpu_parallelization
        self.rank = rank

        # Model + DDP wrapping
        self.model = model.to(device)
        if gpu_parallelization:
            idx = device_index_from_str(device)
            device_ids = [idx] if idx is not None else None
            self.model = DDP(self.model, device_ids=device_ids, find_unused_parameters=False)
        self.net = unwrap_model(self.model)

        # Tokenizer / env
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = config.t5_model.tokenizer_max_length
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Training data: either [train] or [single_hop, c4]
        if len(list_train_dataloader) == 1:
            self.train_dataloader = list_train_dataloader[0]
            self.single_hop_train_dataloader = None
            self.c4_train_dataloader = None
        elif len(list_train_dataloader) > 1:
            self.single_hop_train_dataloader = list_train_dataloader[0]
            self.c4_train_dataloader = list_train_dataloader[1]
            self.train_dataloader = None
        else:
            raise ValueError("Length of List for Train Dataloaders must be >= 1")

        self.val_dataloader = val_dataloader
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.tboard_checkpoint_path = tboard_checkpoint_path
        self.validation_step = validation_step
        self.load_optimizer = load_optimizer

        self.supported_methods = ["single_hop_training", "one_hop_wiki_training"]
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        self.method = method

        if self.method == "single_hop_training":
            self.training_config = config.single_hop_training
        elif self.method == "one_hop_wiki_training":
            self.training_config = config.one_hop_wiki_training

        # Optimizer built on the underlying model
        self.optimizer = get_optimizer(self.net.parameters(), self.training_config)

        # Training state
        self.start_epoch = 0
        self.patience = self.training_config.epochs
        self.best_loss = float("inf")
        self.best_em = 0.0
        self.early_stop_counter = 0
        self.best_model_path = None

        # Scheduler (optional)
        from transformers import get_linear_schedule_with_warmup

        if self.single_hop_train_dataloader is not None and self.c4_train_dataloader is not None:
            # Two loaders, 50:50 mixture
            num_steps_per_epoch = 2 * min(len(self.single_hop_train_dataloader), len(self.c4_train_dataloader))
        else:
            num_steps_per_epoch = len(self.train_dataloader)
        num_steps = num_steps_per_epoch * self.training_config.epochs

        if self.training_config.scheduler is not None:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_training_steps=num_steps,
                num_warmup_steps=int(0.1 * num_steps),
            )
        else:
            self.scheduler = None
        if is_main_process():
            print(f"Using Scheduler: {self.scheduler}")

        # Directories / TB writer
        self.log_dir, self.model_dir = setup_directories(self.training_config, self.config.t5_model)
        if self.tboard_checkpoint_path is not None:
            self.log_dir = tboard_checkpoint_path
            if is_main_process():
                print(f"Continue writing to {tboard_checkpoint_path}")
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Optional checkpoint loading
        if self.checkpoint_path is not None:
            self.model = load_model_checkpoint(self.model, checkpoint_path, device=self.device)
            self.net = unwrap_model(self.model)
            if self.load_optimizer:
                self.optimizer, self.start_epoch = load_optimizer_and_start_epoch(self.optimizer, checkpoint_path)

    # ----- Public API -----

    def train(self, epochs: int):
        """Train the model. If both single-hop and C4 loaders are present, use the 50:50 mixture."""
        # Gradient checkpointing on the underlying model
        if hasattr(self.net, "gradient_checkpointing_enable"):
            self.net.gradient_checkpointing_enable()

        if self.start_epoch != 0 and is_main_process():
            print(f"Starting training from epoch {self.start_epoch}")

        if self.single_hop_train_dataloader is not None and self.c4_train_dataloader is not None:
            if is_main_process():
                print("Training with C4...")
            self._train_with_c4(epochs)
        else:
            if is_main_process():
                print("Training without C4...")
            self._train_without_c4(epochs)

    # ----- Internal: training loops -----

    def _train_with_c4(self, epochs: int):
        min_len = min(len(self.single_hop_train_dataloader), len(self.c4_train_dataloader))
        num_iterations = 2 * min_len

        for epoch in range(self.start_epoch, epochs):
            self.model.train()

            # DistributedSampler epoch bump (if present)
            if hasattr(self.single_hop_train_dataloader, "sampler") and hasattr(
                self.single_hop_train_dataloader.sampler, "set_epoch"
            ):
                self.single_hop_train_dataloader.sampler.set_epoch(epoch)
            if hasattr(self.c4_train_dataloader, "sampler") and hasattr(
                self.c4_train_dataloader.sampler, "set_epoch"
            ):
                self.c4_train_dataloader.sampler.set_epoch(epoch)

            total_c4_loss = 0.0
            total_single_hop_loss = 0.0

            single_hop_iter = iter(self.single_hop_train_dataloader)
            c4_iter = iter(self.c4_train_dataloader)

            progress_bar = tqdm(
                range(num_iterations),
                leave=True,
                desc=f"Epoch {epoch} - Training - {self.method}",
                file=sys.stdout,
                dynamic_ncols=True,
            )

            for batch_idx in progress_bar:
                # 50:50 mixture of datasets
                if batch_idx % 2 == 0:
                    try:
                        batch = next(single_hop_iter)
                    except StopIteration:
                        single_hop_iter = iter(self.single_hop_train_dataloader)
                        batch = next(single_hop_iter)
                else:
                    try:
                        batch = next(c4_iter)
                    except StopIteration:
                        c4_iter = iter(self.c4_train_dataloader)
                        batch = next(c4_iter)

                input_str, label = batch

                tokenized_inputs = self.tokenizer(
                    input_str,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.config.t5_model.tokenizer_max_length,
                )
                tokenized_labels = self.tokenizer(
                    label,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.config.t5_model.tokenizer_max_length,
                )
                tokenized_labels["input_ids"][tokenized_labels["input_ids"] == self.tokenizer.pad_token_id] = -100

                input_ids = tokenized_inputs["input_ids"].to(self.device)
                attention_mask = tokenized_inputs["attention_mask"].to(self.device)
                labels = tokenized_labels["input_ids"].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    use_cache=False,
                )
                loss = outputs.loss

                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                global_step = epoch * num_iterations + batch_idx

                # Distributed logging: reduce loss across ranks, log only on rank 0
                if self.gpu_parallelization and is_dist_avail_and_initialized():
                    loss_tensor = torch.tensor([loss.item()], device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    global_loss = loss_tensor.item() / dist.get_world_size()

                    if is_main_process():
                        if batch_idx % 2 == 0:
                            total_single_hop_loss += global_loss
                            log_tensorboard(self.writer, global_loss, global_step, "Training/SingleHop", eval_metric="loss")
                        else:
                            total_c4_loss += global_loss
                            log_tensorboard(self.writer, global_loss, global_step, "Training/C4", eval_metric="loss")
                else:
                    if batch_idx % 2 == 0:
                        total_single_hop_loss += loss.item()
                        log_tensorboard(self.writer, loss.item(), global_step, "Training/SingleHop", eval_metric="loss")
                    else:
                        total_c4_loss += loss.item()
                        log_tensorboard(self.writer, loss.item(), global_step, "Training/C4", eval_metric="loss")

                progress_bar.set_description(
                    f"Epoch {epoch} - Training - {self.method} - Loss: {loss.item():.4f}"
                )

                # VRAM + curvature logging (rank-0 only)
                if is_main_process() and torch.cuda.is_available() and self.device.startswith("cuda"):
                    vram_allocated = torch.cuda.memory_allocated(torch.device(self.device)) / (1024 ** 2)
                    vram_reserved = torch.cuda.memory_reserved(torch.device(self.device)) / (1024 ** 2)
                    self.writer.add_scalar("VRAM/Training/Allocated", vram_allocated, global_step)
                    self.writer.add_scalar("VRAM/Training/Reserved", vram_reserved, global_step)

                    # Curvature from underlying model
                    hl = getattr(self.net, "additional_layer", None)
                    curvature = 0.0
                    if hl is not None and hasattr(hl, "manifold") and hasattr(hl.manifold, "c"):
                        curvature = hl.manifold.c.item()
                    self.writer.add_scalar("Training/Curvature", curvature, global_step)

            avg_single_hop_loss = total_single_hop_loss / max(1, min_len)
            avg_c4_loss = total_c4_loss / max(1, min_len)
            if is_main_process():
                print(
                    f"Epoch {epoch} - Training - AVGLoss for SingleHop: {avg_single_hop_loss:.4f} "
                    f"| AVGLoss for C4: {avg_c4_loss:.4f}"
                )

            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate(epoch=epoch):
                    break  # Early stopping

    def _train_without_c4(self, epochs: int):
        for epoch in range(self.start_epoch, epochs):
            self.model.train()
            progress_bar = tqdm(
                self.train_dataloader,
                leave=True,
                desc=f"Epoch {epoch} - Training - {self.method}",
                file=sys.stdout,
                dynamic_ncols=True,
            )
            total_loss = 0.0

            # DistributedSampler epoch bump (if present)
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()

                input_str, label = batch

                tokenized_inputs = self.tokenizer(
                    input_str,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.config.t5_model.tokenizer_max_length,
                ).to(self.device)
                tokenized_labels = self.tokenizer(
                    label,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.config.t5_model.tokenizer_max_length,
                ).to(self.device)

                input_ids = tokenized_inputs["input_ids"]
                attention_mask = tokenized_inputs["attention_mask"]
                labels_ids = tokenized_labels["input_ids"]
                labels_ids[labels_ids == self.tokenizer.pad_token_id] = -100

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_ids)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()
                global_step = epoch * len(self.train_dataloader) + batch_idx
                progress_bar.set_description(
                    f"Epoch {epoch} - Training - {self.method} - Loss: {loss.item():.4f}"
                )
                log_tensorboard(self.writer, loss.item(), global_step, "Training", eval_metric="loss")

                if is_main_process() and torch.cuda.is_available() and self.device.startswith("cuda"):
                    vram_allocated = torch.cuda.memory_allocated(torch.device(self.device)) / (1024 ** 2)
                    vram_reserved = torch.cuda.memory_reserved(torch.device(self.device)) / (1024 ** 2)
                    self.writer.add_scalar("VRAM/Training/Allocated", vram_allocated, global_step)
                    self.writer.add_scalar("VRAM/Training/Reserved", vram_reserved, global_step)

            avg_loss = total_loss / max(1, len(self.train_dataloader))
            if is_main_process():
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate(epoch=epoch):
                    break  # Early stopping

    # ----- Evaluation -----

    def evaluate(self, epoch: int) -> bool:
        self.model.eval()

        total_loss = 0.0
        total_em = 0.0
        total_f1 = 0.0
        total_count = 0

        progress_bar = tqdm(
            self.val_dataloader,
            leave=True,
            desc=f"Epoch {epoch} - Validation - {self.method}",
            file=sys.stdout,
            dynamic_ncols=True,
        )
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_str, label = batch
                inputs = self.tokenizer(
                    input_str,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.config.t5_model.tokenizer_max_length,
                ).to(self.device)
                labels_ids = self.tokenizer(
                    label,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.config.t5_model.tokenizer_max_length,
                )["input_ids"].to(self.device)

                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels_ids,
                )
                loss = outputs.loss
                batch_size = len(input_str)

                total_loss += loss.item() * batch_size

                predictions = torch.argmax(outputs.logits, dim=-1)
                decoded_predictions = [
                    self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions
                ]
                batch_f1 = sum(
                    f1_score(pred, truth)[0] for pred, truth in zip(decoded_predictions, label)
                )
                batch_em = sum(
                    1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, label)
                )

                total_em += batch_em
                total_f1 += batch_f1
                total_count += batch_size

                progress_bar.set_description(
                    f"Epoch {epoch} - Validation - {self.method} - Loss: {loss.item():.4f}"
                )
                if batch_idx <= 5 and is_main_process():
                    self.writer.add_text(
                        f"Validation/Prediction_vs_Label_{epoch}",
                        f"Prediction: {decoded_predictions[0]}\nLabel: {label[0]}",
                        epoch,
                    )

        # Distributed reduction
        if self.gpu_parallelization and is_dist_avail_and_initialized():
            loss_tensor = torch.tensor([total_loss], device=self.device)
            em_tensor = torch.tensor([total_em], device=self.device)
            f1_tensor = torch.tensor([total_f1], device=self.device)
            count_tensor = torch.tensor([total_count], device=self.device)

            for t in (loss_tensor, em_tensor, f1_tensor, count_tensor):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

            global_loss = loss_tensor.item()
            global_em = em_tensor.item()
            global_f1 = f1_tensor.item()
            global_count = max(1.0, count_tensor.item())
        else:
            global_loss = total_loss
            global_em = total_em
            global_f1 = total_f1
            global_count = max(1.0, total_count)

        avg_loss = global_loss / global_count
        avg_em_perc = global_em / global_count
        avg_f1_perc = global_f1 / global_count

        stopped = False

        if is_main_process():
            log_tensorboard(self.writer, avg_loss, epoch, "Validation", eval_metric="loss")
            log_tensorboard(self.writer, avg_em_perc, epoch, "Validation", eval_metric="em")
            log_tensorboard(self.writer, avg_f1_perc, epoch, "Validation", eval_metric="f1")
            print(
                f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | "
                f"AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}"
            )

            model_path = f"{self.model_dir}/knit5.pth"

            # Save best model by EM
            if avg_em_perc > self.best_em:
                if self.best_model_path and os.path.exists(self.best_model_path):
                    try:
                        os.remove(self.best_model_path)
                        print(f"Removed previous best model at {self.best_model_path}")
                    except OSError:
                        pass

                self.best_em = avg_em_perc
                self.early_stop_counter = 0
                self.best_model_path = model_path
                save_checkpoint(self.best_model_path, self.model, self.optimizer, epoch)
            else:
                self.early_stop_counter += 1
                print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
                if self.early_stop_counter >= self.patience:
                    print("Early stopping triggered. Stopping training")
                    stopped = True

        # Ensure all ranks see the same stop decision
        if self.gpu_parallelization and is_dist_avail_and_initialized():
            stop_tensor = torch.tensor([1 if stopped else 0], device=self.device)
            dist.broadcast(stop_tensor, src=0)
            stopped = bool(stop_tensor.item())

        return stopped