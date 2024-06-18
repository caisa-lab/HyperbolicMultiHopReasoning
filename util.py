from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import networkx as nx
import torch
from torch.optim import Adam
import random
import pandas as pd
import matplotlib.pyplot as plt
from knowledge_graph import create_knowledge_graph, print_graph
import numpy as np

#This Code are just some ideas of how it could work


def project_to_hyperbolic(x, c):
    norm_x = np.linalg.norm(x)
    sqrt_c = np.sqrt(c)
    tanh_term = np.tanh(sqrt_c * norm_x)
    scaled_x = x / (sqrt_c * norm_x)
    return tanh_term * scaled_x
        
def print_datapoint(dataset, idx):
    print(f"id: {train_dataset['_id'][idx]}")
    print(f"type: {train_dataset['type'][idx]}") # Type of Question: ['bridge_comparison' 'compositional' 'inference' 'comparison'] / 
                                                # Bridge_comparison: Involves comparing attributes of entities through a connecting intermediate entity.
                                                # Compositional: Requires combining multiple pieces of information to derive the answer.
                                                #   Inference: Needs logical reasoning to deduce the answer from given facts.
                                                #   Comparison: Directly compares attributes or facts of two or more entities.
    print(f"question: {train_dataset['question'][idx]}") # Original Question
    print(f"context: {train_dataset['context'][idx]}") # Context can be used to concatonate with question to form input sequence
    print(f"evidences: {train_dataset['evidences'][idx]}") # e1, r1, e2 For knowledge Graph
    print(f"answer: {train_dataset['answer'][idx]}") # Label / Answer
    print(f"evidences_id: {train_dataset['evidences_id'][idx]}")
    print(f"answer_id: {train_dataset['answer_id'][idx]}")
    
if __name__ == '__main__':
    #tokenizer = T5Tokenizer.from_pretrained('t5-base')
    #model = T5ForConditionalGeneration.from_pretrained('t5-base')
    
    train_dataset = pd.read_json('dataset/data_ids_april7/train.json')
    
    print(train_dataset.head(2))
   
    KG = create_knowledge_graph(train_dataset)

    
    from dataset_classes.datasets import RandomWalkDataset, SingleHopDataset
    import time
    start_time = time.time()
    random_walks = RandomWalkDataset(KG, 3)
    end_time = time.time()
    print(f"Runtime: {end_time - start_time}")
    print(len(random_walks.data))
    
    start_time = time.time()
    test_complete_path, test_incomplete_path = random_walks[10]
    end_time = time.time()
    print(f"Random Walk: {random_walks.data[10]}")
    print(f"Complete Path: {test_complete_path}")
    print(f"Incomplete Path: {test_incomplete_path}")
    print(f"Runtime: {end_time - start_time}")
    
    
   

    
    