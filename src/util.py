from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
import networkx as nx
import torch
from torch.optim import Adam
import random
import pandas as pd
import matplotlib.pyplot as plt
from src.knowledge_graph import create_knowledge_graph, print_graph
import numpy as np
from sklearn.model_selection import train_test_split

#TODO: There are over 1000 rows where the last entity of evidence 0 is not equal to the first of evidence 1 leading to no 2 hop in the knowledge Graph. Do we fix this?
def correct_wrong_evidences(df):
    wrong = {}
    for _, entry in df.iterrows():
        index = entry['_id']
        evidences = entry['evidences']
        
        # Ensure there are at least two evidence triples
        if len(evidences) >= 2:
            triple1 = evidences[0]
            triple2 = evidences[1]
        if triple1[2] != triple2[0]:
            wrong[index]=(triple1[2], triple2[0])
    #print(len(wrong))
    
    def update_evidences(evidences):
        if len(evidences) >= 2:
            evidences[0][-1] = evidences[1][0]
        return evidences
    df.loc[df['_id'].isin(list(wrong.keys())), 'evidences'] = df.loc[df['_id'].isin(list(wrong.keys())), 'evidences'].apply(update_evidences)

def project_to_hyperbolic(x, c):
    norm_x = np.linalg.norm(x)
    sqrt_c = np.sqrt(c)
    tanh_term = np.tanh(sqrt_c * norm_x)
    scaled_x = x / (sqrt_c * norm_x)
    return tanh_term * scaled_x
        
def print_datapoint(dataset, idx):
    print(f"id: {dataset['_id'][idx]}")
    print(f"type: {dataset['type'][idx]}") # Type of Question: ['bridge_comparison' 'compositional' 'inference' 'comparison'] / 
                                                # Bridge_comparison: Involves comparing attributes of entities through a connecting intermediate entity.
                                                # Compositional: Requires combining multiple pieces of information to derive the answer.
                                                #   Inference: Needs logical reasoning to deduce the answer from given facts.
                                                #   Comparison: Directly compares attributes or facts of two or more entities.
    print(f"question: {dataset['question'][idx]}") # Original Question
    print(f"context: {dataset['context'][idx]}") # Context can be used to concatonate with question to form input sequence
    print(f"evidences: {dataset['evidences'][idx]}") # e1, r1, e2 For knowledge Graph
    print(f"answer: {dataset['answer'][idx]}") # Label / Answer

#I only have one more Question in the Training Data the rest is the same
# They didnt have the 'has part' relation in their set.
def load_dataset():
    print(f"Loading Datasets...")
    train_dataset = pd.read_json('dataset/data_ids_april7/train.json')
    test_dataset = pd.read_json('dataset/data_ids_april7/dev.json')
    
    def contains_has_part(evidences):
        return any(r == 'has part' for e1, r, r2 in evidences)
    
    train_dataset = train_dataset[~train_dataset['evidences'].apply(contains_has_part)]
    test_dataset = test_dataset[~test_dataset['evidences'].apply(contains_has_part)]
    
    
    validation_ratio = 0.1
    train_dataset = train_dataset[(train_dataset['type'] == 'compositional') | (train_dataset['type'] == 'inference')]
    train_dataset, dev_dataset = train_test_split(train_dataset, test_size=validation_ratio, random_state=120)
    test_dataset = test_dataset[(test_dataset['type'] == 'compositional') | (test_dataset['type'] == 'inference')]
    
    #correct_wrong_evidences(train_dataset)
    #correct_wrong_evidences(dev_dataset)
    #correct_wrong_evidences(test_dataset)

    print("Creating Knowledge Graphs...")
    kg_train = create_knowledge_graph(train_dataset)
    kg_dev = create_knowledge_graph(dev_dataset)
    kg_test = create_knowledge_graph(test_dataset)
    
    return train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test