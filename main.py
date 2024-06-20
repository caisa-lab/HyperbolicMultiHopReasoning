from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import SingleHopDataset, KnowledgeIntegrationDataset, OneWikiHopDataset
from src.config import Config
from src.train import Trainer
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.util import print_datapoint
from src.knowledge_graph import create_knowledge_graph, print_graph
import networkx as nx
from tqdm import tqdm
from torch.utils.data import DataLoader
""" 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-xxl")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-v1_1-xxl")
This is the model from the paper.


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/t5-xxl-lm-adapt")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-xxl-lm-adapt")
Could also use this model might have better perfomance since its updated


This is the Dataset where the model was pretrained with.
https://huggingface.co/datasets/allenai/c4
"""

#I only have one more Question in the Training Data the rest is the same
# They didnt have the 'has part' relation in their set.
def load_dataset():
    print(f"Loading Datasets...")
    train_dataset = pd.read_json('dataset/data/train.json')
    test_dataset = pd.read_json('dataset/data/dev.json')
    
    def contains_has_part(evidences):
        return any(r == 'has part' for e1, r, r2 in evidences)
    
    train_dataset = train_dataset[~train_dataset['evidences'].apply(contains_has_part)]
    test_dataset = test_dataset[~test_dataset['evidences'].apply(contains_has_part)]
    
    
    validation_ratio = 0.1
    train_dataset = train_dataset[(train_dataset['type'] == 'compositional') | (train_dataset['type'] == 'inference')]
    train_dataset, dev_dataset = train_test_split(train_dataset, test_size=validation_ratio, random_state=120)
    test_dataset = test_dataset[(test_dataset['type'] == 'compositional') | (test_dataset['type'] == 'inference')]

    print("Creating Knowledge Graphs...")
    kg_train = create_knowledge_graph(train_dataset)
    kg_dev = create_knowledge_graph(dev_dataset)
    kg_test = create_knowledge_graph(test_dataset)
    
    return train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test
    
def test_single_hop_training():
    #print(torch.__version__)
    #Create Datasets
    print(f"Loading Datasets...")
    train_dataset = pd.read_json('dataset/data_ids_april7/train.json')
    dev_dataset = pd.read_json('dataset/data_ids_april7/dev.json')
    #test_dataset = pd.read_json('dataset/data_ids_april7/test.json')

    print(train_dataset.head(2))

    print("Creating Knowledge Graphs...")
    kg_train = create_knowledge_graph(train_dataset)
    kg_dev = create_knowledge_graph(dev_dataset)
    #kg_test = create_knowledge_graph(test_dataset)

    
    print("Creating Single Hop Datasets...")
    single_hop_dataset_train = SingleHopDataset(kg_train)
    single_hop_dataset_dev = SingleHopDataset(kg_dev)
    #single_hop_dataset_test = SingleHopDataset(kg_test)

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')
    
    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    model_name = "google/t5-large-lm-adapt"
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    
    single_hop_dataloader_train = DataLoader(single_hop_dataset_train, batch_size=config.t5_large_model.batch_size, shuffle=True)
    single_hop_dataloader_dev = DataLoader(single_hop_dataset_dev,  batch_size=config.t5_large_model.batch_size, shuffle=False)
    
    trainer = Trainer(model, tokenizer, single_hop_dataloader_train, single_hop_dataloader_dev, config, device=device)
    
    optimizer = trainer.get_optimizer(model.parameters(), config)
    
    trainer.train_single_hop(optimizer, epochs=1)

if __name__ == '__main__':
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset()
    
    print(train_dataset.head(2))
    
    print(f"Train: Compositional + Inference = {len(train_dataset)}")
    print(f"Dev: Compositional + Inference = {len(dev_dataset)}")
    print(f"Test: Compositional + Inference = {len(test_dataset)}")
    
    one_hop_train = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'train')
    one_hop_dev = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'dev')
    one_hop_test = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'test')
    
    print(len(one_hop_train))
    print(len(one_hop_dev))
    print(len(one_hop_test))
    
    
    

