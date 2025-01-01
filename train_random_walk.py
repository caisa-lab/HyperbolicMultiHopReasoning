from src.utils.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import RandomWalkDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop, create_knowledge_graph_metaqa
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.datasets import RandomWalkMetaQADataset
import argparse
import optuna
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
config = Config()
def _train_random_walk(additional_layer : str, dataset : str, rank, world_size, lr = 0.3):

    if dataset in ['2wikimultihop', 'wikimultihop']:
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

        all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
        all_kg = create_knowledge_graph_wikimultihop(all_data)

        print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")
        print(f"Lenght Test Data: {len(test_dataset)}")

        random_walk_train = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='train')
        random_walk_dev = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev')


    elif dataset in ['metaqa']:
        df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
        df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
        df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")

        df_kg = pd.concat([df_dev, df_train, df_test])
        kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=3)

        random_walk_train = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='train', max_answers=3)
        random_walk_dev = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='dev', max_answers=3)

    else:
        print(f"Unknown Dataset")
    print(f"Number of Random Walks Train: {len(random_walk_train)}")
    print(f"Number of Random Walk Dev: {len(random_walk_dev)}")
    #Specify Hyperparameters via config file
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    if config.random_walk_training.gpu_parallelization:
        from torch.utils.data import DistributedSampler

        random_walk_sampler_train = DistributedSampler(random_walk_train, shuffle=True, num_replicas=world_size, rank=rank)
        random_walk_sampler_dev = DistributedSampler(random_walk_dev, shuffle=False, num_replicas=world_size, rank=rank)

        random_walk_dataloader_train = DataLoader(random_walk_train, sampler=random_walk_sampler_train, batch_size=config.t5_model.batch_size, num_workers=config.random_walk_training.num_workers)
        random_walk_dataloader_dev = DataLoader(random_walk_dev,  sampler=random_walk_sampler_dev, batch_size=config.t5_model.batch_size, num_workers=config.random_walk_training.num_workers)
    else:
        random_walk_dataloader_train = DataLoader(random_walk_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.random_walk_training.num_workers)
        random_walk_dataloader_dev = DataLoader(random_walk_dev, batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)
    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    longest_sequence = 0
        # # Iterate through random_walk_dev dataset
    for incomplete, complete in random_walk_dev:
        # Tokenize the sample text
        tokens = tokenizer(complete, return_tensors='pt', truncation=False).input_ids
        token_length = tokens.size(1)  # Get the token length
        # Update the longest token sequence for dev
        if token_length > longest_sequence:
            longest_sequence = token_length
    for incomplete, complete in random_walk_train:
        # Tokenize the sample text
        tokens = tokenizer(complete, return_tensors='pt', truncation=False).input_ids
        token_length = tokens.size(1)  # Get the token length
        # Update the longest token sequence for dev
        if token_length > longest_sequence:
            longest_sequence = token_length

    print(f"Longest Sequence has {longest_sequence} tokens")

    tokenizer.model_max_length = config.t5_model.tokenizer_max_length


    print("Loading Model...")
    #knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    import torch.nn as nn
    from src.utils.trainer_utils import load_soft_prompt


    if config.random_walk_training.hopping_prompt_checkpoint_path is not None:
        soft_prompt = nn.Parameter(torch.randn(config.random_walk_training.prompt_length, 1024))
        soft_prompt = load_soft_prompt(soft_prompt, config.random_walk_training.hopping_prompt_checkpoint_path)
    else:
        soft_prompt = None
    
    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(layer_type=additional_layer, curvature=config.random_walk_training.curvature, checkpoint_hyperbolic_knit5=config.random_walk_training.model_checkpoint_path, with_model_state_dict=True, gpu_parallelization=True, soft_prompt_length=config.random_walk_training.prompt_length)
    model = SoftPromptModel(curvature=config.random_walk_training.curvature, knit5=hyperbolic_knit5_model, knit5_checkpoint_path=config.random_walk_training.model_checkpoint_path, model_name='hyperbolic_hopping_prompt', soft_prompt=soft_prompt, with_model_state_dict=True, gpu_parallelization=True)
    print(f"Train with hyperbolic Soft Prompt Model with additional layer {additional_layer} and curvature {config.random_walk_training.curvature if additional_layer == 'hyperbolic' else 0.0}")


    print("Model type before passing to SoftPromptTrainer:", type(model))

    config.random_walk_training.learning_rate = lr
    trainer = SoftPromptTrainer(model,
                      tokenizer,
                      random_walk_dataloader_train,
                      random_walk_dataloader_dev,
                      config,
                      device=device,
                      method='random_walk_training',
                      checkpoint_path=config.random_walk_training.hopping_prompt_checkpoint_path,
                      tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path,
                      retrain = True,
                      gpu_parallelization=config.random_walk_training.gpu_parallelization,
                      rank = rank
                      )


    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'with lr: {config.random_walk_training.learning_rate}')

    #print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.train(epochs=config.random_walk_training.epochs)  

import torch.distributed as dist
import torch.multiprocessing as mp
import os
from src.utils.util import set_seed
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)
    set_seed(42)

def train_ddp(rank, world_size, dataset, additional_layer, lr):
    setup_ddp(rank, world_size)
    _train_random_walk(additional_layer=additional_layer, dataset=dataset, rank=rank, world_size=world_size, lr=lr)  # Call your training method
    dist.destroy_process_group()

if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('remainder', type=str, nargs='?', default=None, help='Specify the remainder (e.g., metaqa, 2wikimultihop)')
    # New argument: --additional_layer
    parser.add_argument(
        '--additional_layer',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='identity',  # You can set a different default if needed
        help='Specify the type of additional layer to use: identity, hyperbolic, or linear'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.3,  # You can set a different default if needed
        help='Specify the Learning Rate'
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    dataset = args.remainder  # Pass the dataset name
    additional_layer = args.additional_layer
    lr = args.learning_rate
    mp.spawn(train_ddp, args=(world_size, dataset, additional_layer, lr), nprocs=world_size, join=True)

