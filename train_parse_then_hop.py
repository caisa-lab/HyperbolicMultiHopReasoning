from src.utils.util import load_dataset, load_musique_dataset
from src.train import *
from src.datasets import ParseDataset, ParseMetaQADataset
import pandas as pd
from src.knowledge_graph import create_knowledge_graph_wikimultihop, create_knowledge_graph_metaqa
from src.config import Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from src.models import *
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp
import os
from src.utils.util import set_seed

  
config = Config()
def _train_parse_then_hop(additional_layer : str, dataset : str, rank, world_size):


    if dataset in ['2wikimultihop', 'wikimultihop']:
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

        all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
        all_kg = create_knowledge_graph_wikimultihop(all_data)

        print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")
        print(f"Lenght Test Data: {len(test_dataset)}")

        parse_train = ParseDataset(train_dataset)
        parse_dev = ParseDataset(dev_dataset)
    elif dataset in ['metaqa']:
        validation_dataframe = pd.read_json('dataset/metaqa/2hops/qa_dev_evidences.json')
        train_dataframe = pd.read_json('dataset/metaqa/2hops/qa_train_evidences.json')

        parse_train = ParseMetaQADataset(train_dataframe)
        parse_dev = ParseMetaQADataset(validation_dataframe)
    else:
        print(f"Unknown Dataset")
    print(f"Number of Parse Questions Train: {len(parse_train)}")
    print(f"Number of Parse Questions Dev: {len(parse_dev)}")

    #Specify Hyperparameters via config file
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    #google/t5-large-lm-adapt
    if config.random_walk_training.gpu_parallelization:
        from torch.utils.data import DistributedSampler

        parse_sampler_train = DistributedSampler(parse_train, shuffle=True, num_replicas=world_size, rank=rank)
        parse_sampler_dev = DistributedSampler(parse_dev, shuffle=False, num_replicas=world_size, rank=rank)

        parse_dataloader_train = DataLoader(parse_train, sampler=parse_sampler_train, batch_size=config.t5_model.batch_size, num_workers=config.random_walk_training.num_workers)
        parse_dataloader_dev = DataLoader(parse_dev,  sampler=parse_sampler_dev, batch_size=config.t5_model.batch_size, num_workers=config.random_walk_training.num_workers)
    else:
        parse_dataloader_train = DataLoader(parse_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.random_walk_training.num_workers)
        parse_dataloader_dev = DataLoader(parse_dev, batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if config.parse_then_hop_training.hopping_prompt_checkpoint_path is not None:
        soft_prompt = nn.Parameter(torch.randn(100, 1024))
        soft_prompt = load_soft_prompt(soft_prompt, config.parse_then_hop_training.hopping_prompt_checkpoint_path)
    else:
        soft_prompt = None
    
    print("Loading Model...")
    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(layer_type=additional_layer, curvature=config.random_walk_training.curvature, checkpoint_hyperbolic_knit5=config.random_walk_training.model_checkpoint_path, with_model_state_dict=True, gpu_parallelization=True, soft_prompt_length=config.parse_then_hop_training.prompt_length)
    model = SoftPromptModel(curvature=config.random_walk_training.curvature, knit5=hyperbolic_knit5_model, knit5_checkpoint_path=config.random_walk_training.model_checkpoint_path, model_name='hyperbolic_hopping_prompt', soft_prompt=soft_prompt, with_model_state_dict=True, gpu_parallelization=True)
    print(f"Train with hyperbolic Soft Prompt Model with additional layer {additional_layer} and curvature {config.random_walk_training.curvature if additional_layer == 'hyperbolic' else 0.0}")

    print("Model type before passing to SoftPromptTrainer:", type(model))

    trainer = SoftPromptTrainer(model,
                      tokenizer,
                      parse_dataloader_train,
                      parse_dataloader_dev,
                      config,
                      device=device,
                      method='parse_then_hop_training',
                      checkpoint_path=config.parse_then_hop_training.hopping_prompt_checkpoint_path,
                      tboard_checkpoint_path=config.parse_then_hop_training.tboard_checkpoint_path,
                      retrain = True,
                      gpu_parallelization=config.parse_then_hop_training.gpu_parallelization,
                      rank=rank
                      )




    print(f'Parsing..')
    print(f'with model: {config.t5_model.model_name}')

    print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.parse_then_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.parse_then_hop_training.optimizer}')   

    trainer.train(epochs=config.parse_then_hop_training.epochs)
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    set_seed(42)

def train_ddp(rank, world_size, dataset, additional_layer):
    setup_ddp(rank, world_size)
    _train_parse_then_hop(additional_layer=additional_layer, dataset=dataset, rank=rank, world_size=world_size)  # Call your training method
    dist.destroy_process_group()
    
if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Parse Then Hop Training')
    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation')
    parser.add_argument('remainder', type=str, nargs='?', default=None, help='Specify the remainder (e.g., musique, 2wikimultihop)')
    parser.add_argument(
        '--additional_layer',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='identity',  # You can set a different default if needed
        help='Specify the type of additional layer to use: identity, hyperbolic, or linear'
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    dataset = args.remainder  # Pass the dataset name
    additional_layer = args.additional_layer
    mp.spawn(train_ddp, args=(world_size, dataset, additional_layer), nprocs=world_size, join=True)
