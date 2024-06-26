from src.util import load_dataset
import pandas as pd
from src.train import Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import KnowledgeIntegrationDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split




if __name__ == '__main__':
    #Create Datasets
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset()

    
    print("Creating Single Hop Datasets...")
    dataset_with_all_entries = pd.concat([train_dataset, dev_dataset, test_dataset])
    ki_dataset = KnowledgeIntegrationDataset(dataset_with_all_entries)
    
    ki_train, ki_val = train_test_split(ki_dataset, test_size=0.1, random_state=42)
    

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')
    
    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    model_name = "google/t5-v1_1-base"
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #TODO: Mixed Precision Using O1 Apex     https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use
    #TODO: model = amp.initialize(model, opt_level="O1")
    
    
    single_hop_dataloader_train = DataLoader(ki_train, batch_size=config.t5_large_model.batch_size, shuffle=True)
    single_hop_dataloader_dev = DataLoader(ki_val,  batch_size=config.t5_large_model.batch_size, shuffle=False)
    
    trainer = Trainer(model, tokenizer, single_hop_dataloader_train, single_hop_dataloader_dev, config, device=device, validation_step=1)
    
    optimizer = trainer.get_optimizer(model.parameters(), config)
    
    trainer.train_single_hop(optimizer, epochs=3)