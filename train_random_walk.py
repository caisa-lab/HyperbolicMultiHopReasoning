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
def _train_random_walk(hyperbolic : bool, dataset : str, rank, world_size):

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

        print(f"Number of Random Walks Train: {len(random_walk_train)}")
        print(f"Number of Random Walk Dev: {len(random_walk_dev)}")

    elif dataset in ['metaqa']:
        df_kg = pd.read_csv("dataset/metaqa/kb.txt", sep="|")
        kg = create_knowledge_graph_metaqa(df_kg)

        validation_dataframe = pd.read_json('dataset/metaqa/2hops/qa_dev_evidences.json')
        test_dataframe = pd.read_json('dataset/metaqa/2hops/qa_test_evidences.json')

        random_walk_train = RandomWalkMetaQADataset(kg, validation_dataframe, test_dataframe, steps=3, type='train')
        random_walk_dev = RandomWalkMetaQADataset(kg, validation_dataframe, test_dataframe, steps=3, type='dev')
    else:
        print(f"Unknown Dataset")


    #Specify Hyperparameters via config file
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print(f"Process rank: {rank} using device: {device}")
    if dist.is_initialized():
        print(f"Rank: {dist.get_rank()} / World size: {dist.get_world_size()}")

    if config.random_walk_training.gpu_parallelization:
        from torch.utils.data import DistributedSampler

        random_walk_sampler_train = DistributedSampler(random_walk_train, num_replicas=world_size, rank=rank)
        random_walk_sampler_dev = DistributedSampler(random_walk_dev, num_replicas=world_size, rank=rank)

        random_walk_dataloader_train = DataLoader(random_walk_train, sampler=random_walk_sampler_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.random_walk_training.num_workers)
        random_walk_dataloader_dev = DataLoader(random_walk_dev,  sampler=random_walk_sampler_dev, batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)
    print(f'Training on device: {device}')
    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    import torch.nn as nn
    from src.utils.trainer_utils import load_soft_prompt


    if config.random_walk_training.hopping_prompt_checkpoint_path is not None:
        soft_prompt = nn.Parameter(torch.randn(config.random_walk_training.prompt_length, 1024))
        soft_prompt = load_soft_prompt(soft_prompt, config.random_walk_training.hopping_prompt_checkpoint_path)
    else:
        soft_prompt = None
    
    if hyperbolic:
        hyperbolic_knit5_model = T5ModelWithAdditionalLayer(layer_type='identity', curvature=config.random_walk_training.curvature, checkpoint_hyperbolic_knit5=config.random_walk_training.model_checkpoint_path, with_model_state_dict=True)
        model = SoftPromptModel(curvature=config.random_walk_training.curvature, knit5=hyperbolic_knit5_model, knit5_checkpoint_path=config.random_walk_training.model_checkpoint_path, model_name='hyperbolic_hopping_prompt', soft_prompt=soft_prompt, with_model_state_dict=False)
        print(f"Train with hyperbolic Soft Prompt Model with curvature {config.random_walk_training.curvature}")
    else:
        model = SoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'hopping_prompt', with_model_state_dict=False, soft_prompt=soft_prompt, curvature = config.random_walk_training.curvature)

    print("Model type before passing to SoftPromptTrainer:", type(model))

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
                      gpu_parallelization=config.single_hop_training.gpu_parallelization,
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
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def train_ddp(rank, world_size, dataset, hyperbolic):
    setup_ddp(rank, world_size)
    _train_random_walk(hyperbolic=hyperbolic, dataset=dataset, rank=rank, world_size=world_size)  # Call your training method
    dist.destroy_process_group()

if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    dataset = args.remainder  # Pass the dataset name
    hyperbolic = args.hyperbolic
    mp.spawn(train_ddp, args=(world_size, dataset, hyperbolic), nprocs=world_size, join=True)
