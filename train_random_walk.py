from src.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train import Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import RandomWalkDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph
import torch.nn as nn


if __name__ == '__main__':    
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

    all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
    all_kg = create_knowledge_graph(all_data)

    print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

    print(f"Lenght Train Data: {len(train_dataset)}")
    print(f"Lenght Dev Data: {len(dev_dataset)}")
    print(f"Lenght Test Data: {len(test_dataset)}")

    random_walk_train = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='train')
    random_walk_dev = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev')
    random_walk_test = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='test')

    print(f"Number of Random Walks Train: {len(random_walk_train)}")
    print(f"Number of Random Walk Dev: {len(random_walk_dev)}")
    print(f"Number of Random Walk Test: {len(random_walk_test)}")

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


    random_walk_dataloader_train = DataLoader(random_walk_train, batch_size=config.t5_model.batch_size, shuffle=True)
    random_walk_dataloader_dev = DataLoader(random_walk_dev,  batch_size=config.t5_model.batch_size, shuffle=False)


    trainer = Trainer(model,
                      tokenizer,
                      [random_walk_dataloader_train],
                      random_walk_dataloader_dev,
                      config,
                      device=device,
                      method='random_walk_training',
                      checkpoint_path=config.random_walk_training.model_checkpoint_path,
                      tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path
                      )

    #HP Soft Prompt will be tuned
    hp_length = config.random_walk_training.prompt_length
  
    hp_embedding_size = model.config.hidden_size 
    hp_embeddings = nn.Embedding(hp_length, hp_embedding_size)
    
    #dont use random use top 100 most common tokens of tokenizer.getvocab
    top_100_token_embeddings = get_top_token_embeddings(model, tokenizer, 100)
    hp_embeddings.weight.data[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings
    
    print(hp_embeddings.num_embeddings)
    print(hp_embeddings.embedding_dim)

    optimizer = trainer.get_optimizer(hp_embeddings.parameters(), config)

    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.train_random_walk(hopping_soft_prompt=hp_embeddings, optimizer=optimizer, epochs=config.random_walk_training.epochs)
