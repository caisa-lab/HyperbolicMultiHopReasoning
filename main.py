from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import SingleHopDataset, KnowledgeIntegrationDataset, OneWikiHopDataset, RandomWalkDataset
#from src.config import Config
#from src.train import Trainer
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.knowledge_graph import create_knowledge_graph, print_graph, visualize_knowledge_graph
import networkx as nx
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.util import load_dataset, print_datapoint, correct_wrong_evidences, check_for_wrong_2_hops
import torch.nn as nn
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


def count_all_two_hops():
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop/")
    
    correct_wrong_evidences(train_dataset)
    correct_wrong_evidences(dev_dataset)
    correct_wrong_evidences(test_dataset)
    
    print(check_for_wrong_2_hops(train_dataset))
    print(check_for_wrong_2_hops(dev_dataset))
    print(check_for_wrong_2_hops(test_dataset))
    
    all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
    all_kg = create_knowledge_graph(all_data)

    print(f"Lenght Train Data: {len(train_dataset)}")
    print(f"Lenght Dev Data: {len(dev_dataset)}")
    print(f"Lenght Test Data: {len(test_dataset)}")
    
    one_hop_wiki_train = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'train')
    one_hop_wiki_dev = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'dev')
    one_hop_wiki_test = OneWikiHopDataset(train_dataset, dev_dataset, test_dataset, 'test')
    
    print(f"Lenght Train Data One Hop Wiki: {len(one_hop_wiki_train)}")
    print(f"Lenght Dev Data One Hop Wiki: {len(one_hop_wiki_dev)}")
    print(f"Lenght Test Data One Hop Wiki: {len(one_hop_wiki_test)}")
    
    ki_data = KnowledgeIntegrationDataset(all_data)
    
    print(f"Lenght Test Data Knowledge Integration: {len(ki_data)}")
    
    
    #Count all two hops
    all_two_hops = set()
    for e1 in all_kg.nodes():
        neighbors1 = all_kg.successors(e1)
        for e2 in neighbors1:
            if all_kg.has_edge(e1, e2):
                relation1 = all_kg.get_edge_data(e1, e2).get('relation', None)
                if relation1 is not None:
                    neighbors2 = all_kg.successors(e2)
                    for e3 in neighbors2:
                        if all_kg.has_edge(e2, e3):
                            relation2 = all_kg.get_edge_data(e2, e3).get('relation', None)
                            if relation2 is not None:
                                all_two_hops.add((e1, relation1, e2, relation2, e3))
    print(f'All Two Hops: {len(all_two_hops)}')    

if __name__ == '__main__':
    
    count_all_two_hops()
    
    
    
    
    

    
    

