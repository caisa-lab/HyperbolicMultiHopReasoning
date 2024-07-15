from src.util import load_dataset, load_c4_dataset
import pandas as pd
from src.train import Trainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import OneWikiHopDataset, SingleHopDataset
import torch
from src.knowledge_graph import create_knowledge_graph
from src.config import Config
from torch.utils.data import DataLoader

def _train_finetuning_single_hop():
    #Create Datasets
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop", do_correct_wrong_evidences=True)

    
    print(f"Length Train Data: {len(train_dataset)}")
    print(f"Length Dev Data: {len(dev_dataset)}")
    #print(f"Length Test Data: {len(test_dataset)}")
    
    print("Creating One Hop Wiki Datsets...")
    
    one_hop_wiki_train = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'train')
    one_hop_wiki_dev = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'dev')
    
    
    print(f"Length Train Data One Hop Wiki: {len(one_hop_wiki_train)}")
    print(f"Length Dev Data One Hop Wiki: {len(one_hop_wiki_dev)}")
    #print(f"Lenght Test Data One Hop Wiki: {len(one_hop_wiki_test)}")
    
    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')
    
    #Define Tokenizer and Model
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.model_name)
    print(f"Loading Model {config.t5_model.model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model.model_name)
    #Adjust Dropout
    model.config.dropout_rate = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1
    
    one_hop_wiki_dataloader_train = DataLoader(one_hop_wiki_train, batch_size=config.t5_model.batch_size, shuffle=True)
    one_hop_wiki_dataloader_dev = DataLoader(one_hop_wiki_dev,  batch_size=config.t5_model.batch_size, shuffle=False)
    
    trainer = Trainer(model, tokenizer, [one_hop_wiki_dataloader_train], one_hop_wiki_dataloader_dev, config, device=device, validation_step=1, checkpoint_path='checkpoints/knowledge_integration/large_adapt_bsize256/model_epoch_29_val_loss_0.0019.pth', method='one_hop_wiki_training')
    
    optimizer = trainer.get_optimizer(model.parameters(), config, method='one_hop_wiki_training')
    
    print(f'Single Hop Finetuning...')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.one_hop_wiki_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.one_hop_wiki_training.optimizer}')
    
    trainer.train_single_hop(optimizer, epochs=config.single_hop_training.epochs)

if __name__ == '__main__':
    _train_finetuning_single_hop()
