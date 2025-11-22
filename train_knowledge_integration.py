import argparse
from math import log, exp
from typing import Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from src.utils.util import load_c4_dataset, set_seed
from src.datasets.dataloader import get_knowledge_integration_dataset
from src.datasets import C4Dataset
from src.knowledge_graph import (
    create_knowledge_graph_metaqa,
    create_knowledge_graph_mlpq,
    create_knowledge_graph_pql,
)
from src.config import Config
from src.train import ModelTrainer
from src.models import T5ModelWithAdditionalLayer

config = Config()


def _knowledge_integration_with_c4(
    dataset: str,
    rank: int,
    world_size: int,
    learning_rate: float = 0.001,
    epochs: int = 50,
    checkpoint_save_path: Union[str, None] = None,
    tboard_logs_save_path: Union[str, None] = None,
    batch_size: int = 64,
    additional_layer: str = "identity",
    curvature: float = 1.0,
):
    # ----- DDP / device -----
    gpu_parallel = config.single_hop_training.gpu_parallelization
    if torch.cuda.is_available():
        if gpu_parallel:
            assert 0 <= rank < torch.cuda.device_count(), (
                f"Rank {rank} out of range for {torch.cuda.device_count()} GPUs"
            )
            device = f"cuda:{rank}"
        else:
            device = "cuda"
    else:
        device = "cpu"

    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    # ----- Data -----
    ki_train = get_knowledge_integration_dataset(dataset)

    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.model_name)

    # Adjust curvature for hyperbolic additional layer
    if additional_layer == "hyperbolic":
        config.single_hop_training.curvature = log(exp(curvature) - 1.0)
        print(f"Setting Curvature (internal) to {config.single_hop_training.curvature}")

    print("Loading T5 model with additional layer...")
    model = T5ModelWithAdditionalLayer(
        model_name=config.t5_model.model_name,
        layer_type=additional_layer,
        curvature=config.single_hop_training.curvature,
        checkpoint_hyperbolic_knit5=config.single_hop_training.model_checkpoint_path,
        with_model_state_dict=True
    )
    model.config.dropout_rate = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1
    model.config.classifier_dropout = 0.1

    # C4 data
    base_path = "c4/en/c4-train.{:05d}-of-01024.json"
    c4_dataset = load_c4_dataset(base_path, number_of_files=3)
    c4_dataset = c4_dataset[: len(ki_train)]

    objective = "prefix_language_modeling"
    c4_train = C4Dataset(c4_dataset, tokenizer=tokenizer, objective=objective)

    # ----- Config overrides -----
    config.single_hop_training.learning_rate = learning_rate
    print(f"Setting Learning Rate to {learning_rate}")
    config.single_hop_training.epochs = epochs
    print(f"Setting Number of Epochs to {epochs}")
    if checkpoint_save_path is not None:
        config.single_hop_training.model_save_path = checkpoint_save_path
        print(f"Setting Checkpoint Save Path to {checkpoint_save_path}")
    if tboard_logs_save_path is not None:
        config.single_hop_training.log_dir = tboard_logs_save_path
        print(f"Setting tboard_logs_save_path to {tboard_logs_save_path}")
    config.t5_model.batch_size = batch_size
    print(f"Setting Batch Size to {batch_size}")

    # ----- DataLoaders (+DistributedSampler if DDP) -----
    if gpu_parallel:
        ki_sampler = DistributedSampler(
            ki_train, shuffle=True, num_replicas=world_size, rank=rank
        )
        c4_sampler = DistributedSampler(
            c4_train, shuffle=True, num_replicas=world_size, rank=rank
        )
        ki_dev_sampler = DistributedSampler(
            ki_train, shuffle=False, num_replicas=world_size, rank=rank
        )

        per_gpu_batch = max(1, config.t5_model.batch_size // world_size)
        single_hop_dataloader_train = DataLoader(
            ki_train, sampler=ki_sampler, batch_size=per_gpu_batch
        )
        c4_dataloader_train = DataLoader(
            c4_train, sampler=c4_sampler, batch_size=per_gpu_batch
        )
        single_hop_dataloader_dev = DataLoader(
            ki_train, sampler=ki_dev_sampler, batch_size=per_gpu_batch
        )
    else:
        single_hop_dataloader_train = DataLoader(
            ki_train,
            batch_size=config.t5_model.batch_size,
            shuffle=True,
            num_workers=config.single_hop_training.num_workers,
        )
        c4_dataloader_train = DataLoader(
            c4_train,
            batch_size=config.t5_model.batch_size,
            shuffle=True,
            num_workers=config.single_hop_training.num_workers,
        )
        single_hop_dataloader_dev = DataLoader(
            ki_train,
            batch_size=config.t5_model.batch_size,
            shuffle=False,
            num_workers=config.single_hop_training.num_workers,
        )

    config.single_hop_training.additional_log_info = (
        config.single_hop_training.additional_log_info
        + f"_{additional_layer}_c{curvature}"
    )

    trainer = ModelTrainer(
        model,
        tokenizer,
        [single_hop_dataloader_train, c4_dataloader_train],
        single_hop_dataloader_dev,
        config,
        device=device,
        validation_step=1,
        checkpoint_path=config.single_hop_training.model_checkpoint_path,
        tboard_checkpoint_path=config.single_hop_training.tboard_checkpoint_path,
        method="single_hop_training",
        gpu_parallelization=gpu_parallel,
        rank=rank,
    )

    print("Knowledge Integration training with C4..")
    print(f"Base model: {config.t5_model.model_name}")
    print(f"Config: {model.config}")
    print(f"Epochs: {config.single_hop_training.epochs}")
    print(f"Global batch size: {config.t5_model.batch_size}")
    print(f"Optimizer: {config.single_hop_training.optimizer}")
    print(f"Number of Knwowledge Integration triples: {len(ki_train)}")

    trainer.train(epochs=config.single_hop_training.epochs)


# ----- DDP setup / launcher -----


def setup_ddp(rank: int, world_size: int):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(rank)
    set_seed(42)


def train_ddp(
    rank: int,
    world_size: int,
    dataset: str,
    learning_rate: float,
    epochs: int,
    checkpoint_save_path: Union[str, None],
    tboard_logs_save_path: Union[str, None],
    batch_size: int,
    additional_layer: str,
    curvature: float,
):
    setup_ddp(rank, world_size)
    try:
        _knowledge_integration_with_c4(
            dataset=dataset,
            rank=rank,
            world_size=world_size,
            learning_rate=learning_rate,
            epochs=epochs,
            checkpoint_save_path=checkpoint_save_path,
            tboard_logs_save_path=tboard_logs_save_path,
            batch_size=batch_size,
            additional_layer=additional_layer,
            curvature=curvature,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Integration Training")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--checkpoint_save_path", type=str, default="checkpoints")
    parser.add_argument(
        "--additional_layer",
        type=str,
        default="identity",
        choices=["identity", "euclidean", "hyperbolic"],
    )
    parser.add_argument("--tboard_logs_save_path", type=str, default="tboard_logs")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset = args.dataset
    learning_rate = args.learning_rate
    epochs = args.epochs
    checkpoint_save_path = args.checkpoint_save_path
    tboard_logs_save_path = args.tboard_logs_save_path
    batch_size = args.batch_size
    additional_layer = args.additional_layer
    curvature = args.curvature

    gpu_parallel = config.single_hop_training.gpu_parallelization
    world_size = torch.cuda.device_count() if gpu_parallel and torch.cuda.is_available() else 1

    if gpu_parallel and world_size > 1:
        mp.spawn(
            train_ddp,
            args=(
                world_size,
                dataset,
                learning_rate,
                epochs,
                checkpoint_save_path,
                tboard_logs_save_path,
                batch_size,
                additional_layer,
                curvature,
            ),
            nprocs=world_size,
            join=True,
        )
    else:
        _knowledge_integration_with_c4(
            dataset=dataset,
            rank=0,
            world_size=1,
            learning_rate=learning_rate,
            epochs=epochs,
            checkpoint_save_path=checkpoint_save_path,
            tboard_logs_save_path=tboard_logs_save_path,
            batch_size=batch_size,
            additional_layer=additional_layer,
            curvature=curvature,
        )
