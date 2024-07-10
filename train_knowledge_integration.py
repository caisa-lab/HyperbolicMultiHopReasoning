from src.util import load_dataset, load_c4_dataset
import pandas as pd
from src.train import Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import KnowledgeIntegrationDataset, C4Dataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.util import correct_wrong_evidences
import argparse

def _knowledge_integration_with_c4():
    #Create Datasets
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop", do_correct_wrong_evidences=True)

    
    print("Creating Single Hop Datasets...")
    dataset_with_all_entries = pd.concat([train_dataset, dev_dataset, test_dataset])
    
    ki_dataset = KnowledgeIntegrationDataset(dataset_with_all_entries)
    
    ki_train = ki_dataset
    

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')
    
    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.model_name)
    print(f"Loading Model {config.t5_model.model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model.model_name)
    #Adjust Dropout
    model.config.dropout_rate = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1
    
    
    #TODO: Mixed Precision Using O1 Apex     https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use
    #TODO: model = amp.initialize(model, opt_level="O1")
    #TODO: Look PERT to reduce VRAM
    
    base_path = 'c4/en/c4-train.{:05d}-of-01024.json'
    c4_dataset = load_c4_dataset(base_path, number_of_files=15)
    
    C4_train = C4Dataset(c4_dataset ,tokenizer=tokenizer)
    
    c4_dataloader_train = DataLoader(C4_train, batch_size = config.t5_model.batch_size, shuffle=True)
    single_hop_dataloader_train = DataLoader(ki_train, batch_size=config.t5_model.batch_size, shuffle=True)
    single_hop_dataloader_dev = DataLoader(ki_train,  batch_size=config.t5_model.batch_size, shuffle=False)
    
    trainer = Trainer(model, tokenizer, [single_hop_dataloader_train, c4_dataloader_train], single_hop_dataloader_dev, config, device=device, validation_step=1)
    
    optimizer = trainer.get_optimizer(model.parameters(), config)
    
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.single_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.single_hop_training.optimizer}')
    print(f'With C4')
    
    trainer.train_single_hop(optimizer, epochs=config.single_hop_training.epochs)
    
def _knowledge_integration_without_c4():
    #Create Datasets
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop", do_correct_wrong_evidences=True)

    
    print("Creating Single Hop Datasets...")
    dataset_with_all_entries = pd.concat([train_dataset, dev_dataset, test_dataset])
    ki_dataset = KnowledgeIntegrationDataset(dataset_with_all_entries)
    
    ki_train = ki_dataset
    

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')
    
    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.model_name)
    print(f"Loading Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model.model_name)
    #Adjust Dropout
    model.config.dropout_rate = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1
    
    single_hop_dataloader_train = DataLoader(ki_train, batch_size=config.t5_model.batch_size, shuffle=True)
    single_hop_dataloader_dev = DataLoader(ki_train,  batch_size=config.t5_model.batch_size, shuffle=False)
    
    trainer = Trainer(model, tokenizer, [single_hop_dataloader_train], single_hop_dataloader_dev, config, device=device, validation_step=1)
    
    optimizer = trainer.get_optimizer(model.parameters(), config)
    
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.single_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.single_hop_training.optimizer}')
    print(f'Without C4')
    
    trainer.train_single_hop(optimizer, epochs=config.single_hop_training.epochs)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Integration Training')
    parser.add_argument('--c4', action='store_true', help='Include C4 dataset in training')
    args = parser.parse_args()
    
    if args.c4:
        _knowledge_integration_with_c4()
    else:
        _knowledge_integration_without_c4()
