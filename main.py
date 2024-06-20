from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import SingleHopDataset
from src.config import Config
from src.train import Trainer
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.util import create_knowledge_graph, print_graph
import networkx as nx
from tqdm import tqdm
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

def load_dataset():
    print(f"Loading Datasets...")
    train_dataset = pd.read_json('dataset/data_ids_april7/train.json')
    test_dataset = pd.read_json('dataset/data_ids_april7/dev.json')
    
    
    validation_ratio = 0.1
    train_dataset = train_dataset[(train_dataset['type'] == 'compositional') | (train_dataset['type'] == 'inference')]
    train_dataset, dev_dataset = train_test_split(train_dataset, test_size=validation_ratio, random_state=1)
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

    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    model_name = "google/t5-large-lm-adapt"
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')
    
    trainer = Trainer(model, tokenizer, single_hop_dataset_train, single_hop_dataset_dev, config)
    
    optimizer = trainer.get_optimizer(model.parameters(), config)
    
    trainer.train_single_hop(optimizer, epochs=100)


if __name__ == '__main__':
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset()
    
    print(train_dataset.shape)
    print(dev_dataset.shape)
    print(test_dataset.shape)
    entities = []
    for entry in tqdm(train_dataset['evidences']):
        for triples in entry:
            #print(triples)
            e1 = triples[0]
            e2 = triples[2]
            entities.append(e1)
            entities.append(e2)
    print(len(list(set(entities))))
    
    print_graph(kg_train)
    
    entities = kg_train.nodes()
    print(f'Number of entities: {len(entities)}')
    print_graph(kg_test)
    print_graph(kg_dev)

