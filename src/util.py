from transformers import T5Tokenizer, T5ForConditionalGeneration
import networkx as nx
import torch
from torch.optim import Adam
import random
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the system path
sys.path.append(current_dir)

from knowledge_graph import create_knowledge_graph
import numpy as np
from sklearn.model_selection import train_test_split

#We fix it
def correct_wrong_evidences(df : pd.DataFrame):
    wrong = {}
    for _, entry in df.iterrows():
        index = entry['_id']
        evidences = entry['evidences']
        
        # Ensure there are at least two evidence triples
        if len(evidences) >= 2:
            triple1 = evidences[0]
            triple2 = evidences[1]
        else:
            print(f'Evidences {evidences}')
        if triple1[2] != triple2[0]:
            wrong[index]=(triple1[2], triple2[0])
    #print(len(wrong))
    
    def update_evidences(evidences):
        if len(evidences) >= 2:
            evidences[0][-1] = evidences[1][0]
        return evidences
    df.loc[df['_id'].isin(list(wrong.keys())), 'evidences'] = df.loc[df['_id'].isin(list(wrong.keys())), 'evidences'].apply(update_evidences)
def check_for_wrong_2_hops(df : pd.DataFrame):
    def check_e2_not_equal_e3(evidences):
        return evidences[0][2] != evidences[1][0]
    df['wrong_evidences'] = df['evidences'].apply(check_e2_not_equal_e3)
    rows_with_wrong_evidences = df[df['wrong_evidences']]
    
    return rows_with_wrong_evidences
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
def load_dataset(path: str,
                 do_correct_wrong_evidences : bool = False):
    print(f"Loading Datasets...")
    train_dataset = pd.read_json(f"{path}/train.json").dropna(how='all')
    test_dataset = pd.read_json(f"{path}/dev.json").dropna(how='all')
    
    def contains_has_part(evidences):
        return any(r == 'has part' for e1, r, r2 in evidences)
    
    train_dataset = train_dataset[~train_dataset['evidences'].apply(contains_has_part)]
    test_dataset = test_dataset[~test_dataset['evidences'].apply(contains_has_part)]
    
    
    validation_ratio = 0.1
    train_dataset = train_dataset[(train_dataset['type'] == 'compositional') | (train_dataset['type'] == 'inference')]
    train_dataset, dev_dataset = train_test_split(train_dataset, test_size=validation_ratio, random_state=120)
    test_dataset = test_dataset[(test_dataset['type'] == 'compositional') | (test_dataset['type'] == 'inference')]
    
    if do_correct_wrong_evidences:
        print(f'Correct Wrong 2 Hops...')
        correct_wrong_evidences(train_dataset)
        correct_wrong_evidences(dev_dataset)
        correct_wrong_evidences(test_dataset)

    print("Creating Knowledge Graphs...")
    kg_train = create_knowledge_graph(train_dataset)
    kg_dev = create_knowledge_graph(dev_dataset)
    kg_test = create_knowledge_graph(test_dataset)
    
    return train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test

def load_c4_dataset(base_path: str,
                    number_of_files : int,
                    chunk_size : int = 100_000):
    list_of_texts = []
    for i in tqdm(range(number_of_files), desc='Loading Dataframe with c4 Data...', file = sys.stdout):
        for chunk in pd.read_json(base_path.format(i), lines=True, chunksize=chunk_size):
            list_of_texts.extend(chunk['text'])
    return list_of_texts