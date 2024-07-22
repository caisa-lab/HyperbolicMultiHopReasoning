from src.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train import Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import ParseThenHopDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph
import torch.nn as nn


if __name__ == '__main__':    
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

    print(f"Lenght Train Data: {len(train_dataset)}")
    print(f"Lenght Dev Data: {len(dev_dataset)}")
    print(f"Lenght Test Data: {len(test_dataset)}")

    parse_then_hop_train = ParseThenHopDataset(train_dataset)
    parse_then_hop_dev = ParseThenHopDataset(dev_dataset)

    print(f"Number of PaTH Train: {len(parse_then_hop_train)}")
    print(f"Number of PaTH Dev: {len(parse_then_hop_dev)}")

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    parse_then_hop_dataloader_train = DataLoader(parse_then_hop_train, batch_size=config.t5_model.batch_size, shuffle=True)
    parse_then_hop_dataloader_dev = DataLoader(parse_then_hop_dev,  batch_size=config.t5_model.batch_size, shuffle=False)


    trainer = Trainer(model,
                      tokenizer,
                      [parse_then_hop_dataloader_train],
                      parse_then_hop_dataloader_dev,
                      config,
                      device=device,
                      method='parse_then_hop_training',
                      checkpoint_path=config.parse_then_hop_training.model_checkpoint_path,
                      tboard_checkpoint_path=config.parse_then_hop_training.tboard_checkpoint_path,
                      load_optimizer=config.parse_then_hop_training.load_optimizer
                      )

    #PP Soft Prompt will be tuned
    pp_length = config.random_walk_training.prompt_length
  
    pp_embedding_size = model.config.hidden_size 
    pp_embeddings = nn.Embedding(pp_length, pp_embedding_size)
    
    #dont use random use top 100 most common tokens of tokenizer.getvocab
    top_100_token_embeddings = get_top_token_embeddings(model, tokenizer, 100)
    pp_embeddings.weight.data[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings
    
    print(f' PP Embedding num Embedding: {pp_embeddings.num_embeddings}')
    print(f' PP Embedding embedding dim: {pp_embeddings.embedding_dim}')

    #PP Soft Prompt will be tuned
    hp_length = config.random_walk_training.prompt_length
  
    hp_embedding_size = model.config.hidden_size 
    hp_embeddings = nn.Embedding(hp_length, hp_embedding_size)
    
    hp_embeddings.weight.data[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings
    
    print(f' HP Embedding num Embedding: {hp_embeddings.num_embeddings}')
    print(f' HP Embedding embedding dim: {hp_embeddings.embedding_dim}')

    optimizer = trainer.get_optimizer(pp_embeddings.parameters())

    print(f'Parse Then Hop Training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.parse_then_hop(pp_embeddings=pp_embeddings, hp_embeddings= hp_embeddings, optimizer=optimizer, epochs=config.random_walk_training.epochs)
