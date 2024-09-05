import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset
from src.datasets import RandomWalkDataset, KnowledgeIntegrationDataset, ParseDataset
from src.utils.util import *
from src.models import SoftPromptModel
from src.config import Config
from src.train import SoftPromptTrainer
train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)
all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
all_kg = create_knowledge_graph(all_data)

print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

print(f"Lenght Train Data: {len(train_dataset)}")
print(f"Lenght Dev Data: {len(dev_dataset)}")
print(f"Lenght Test Data: {len(test_dataset)}")

ki_train = KnowledgeIntegrationDataset(all_data)

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
config.random_walk_training.num_workers = 6
config.t5_model.batch_size = 16
random_walk_dataloader_train = DataLoader(random_walk_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.random_walk_training.num_workers)
random_walk_dataloader_dev = DataLoader(random_walk_dev,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)

ki_dataloader_train = DataLoader(ki_train,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)


model_name = config.t5_model.model_name
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading Model...")
knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
import torch.nn as nn
from src.utils.trainer_utils import load_soft_prompt

if config.random_walk_training.hopping_prompt_checkpoint_path is not None:
    soft_prompt = nn.Embedding(100, 1024)
    load_soft_prompt(soft_prompt, config.random_walk_training.hopping_prompt_checkpoint_path)
else:
    soft_prompt = None


#hyperbolic_knit5_model = HyperbolicT5MapEmbeddings()
curvature = 4.0
config.random_walk_training.model_checkpoint_path = '../checkpoints/knowledge_integration/large_adapt_bsize64_c4/model_epoch_16_val_loss_0.0336.pth'
model = SoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'hyperbolic_hopping_prompt', with_model_state_dict=False, soft_prompt=soft_prompt)   
print(f"Train with hyperbolic Soft Prompt Model with curvature {curvature}")

print("Model type before passing to SoftPromptTrainer:", type(model))

trainer = SoftPromptTrainer(model,
                    tokenizer,
                    random_walk_dataloader_train,
                    random_walk_dataloader_dev,
                    config,
                    device=device,
                    method='random_walk_training',
                    checkpoint_path=config.random_walk_training.hopping_prompt_checkpoint_path,
                    tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path,
                    retrain = False
                    )


print(f'Random Walk Training..')
print(f'with model: {config.t5_model.model_name}')

print(f'Model Config: {model.knit5.config}')
print(f'for: {config.random_walk_training.epochs} epochs')
print(f'with batch size: {config.t5_model.batch_size}')
print(f'with optimizer: {config.random_walk_training.optimizer}')

def get_layer_outputs(soft_prompt_model, inputs, labels, is_soft_prompt=True, encoder = True):
    soft_prompt_model.to(device)
    activations = {}

    def hook_fn(module, input, output):
        activations[module] = output[0]

    hooks = []
    if is_soft_prompt:
        layer_list = soft_prompt_model.knit5.encoder.block if encoder else soft_prompt_model.knit5.decoder.block
        for layer in layer_list:
            hooks.append(layer.register_forward_hook(hook_fn))
    else:
        layer_list = soft_prompt_model.encoder.block if encoder else soft_prompt_model.decoder.block
        for layer in layer_list:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Run a forward pass
    with torch.no_grad():
        if is_soft_prompt:
            soft_prompt_model(inputs, labels)
        else:
            soft_prompt_model(input_ids = inputs['input_ids'], labels= labels, attention_mask=inputs.attention_mask)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations
import numpy as np
from tqdm import tqdm

layer_batch_outputs = {}

# Iterate over the batches
for idx, batch in enumerate(tqdm(ki_dataloader_train)):
    incomplete, complete = batch
    inputs = tokenizer(incomplete, truncation=True, padding=True, return_tensors='pt').to(device)
    labels = tokenizer(complete, truncation=True, padding=True, return_tensors='pt')['input_ids'].to(device)
    
    # Get the layer outputs
    outputs = get_layer_outputs(model.knit5, inputs, labels, is_soft_prompt=False, encoder=True).values() 
    
    # Store the outputs in the dictionary
    layer_batch_outputs[idx] = [torch.mean(entry, dim=1) for entry in outputs]
    
    if idx >= 10_000:  # Stop after processing 6 batches (idx starts from 0)
        break
concat_layer_outputs = {}
num_layers = len(layer_batch_outputs[0])

# Initialize a list for each layer
for layer_idx in range(num_layers):
    concat_layer_outputs[layer_idx] = []

# Iterate through each batch in layer_batch_outputs
for batch_idx in range(len(layer_batch_outputs)):
    for layer_idx in range(num_layers):
        # Extract the embeddings for the current layer and batch
        layer_embeddings = layer_batch_outputs[batch_idx][layer_idx]
        
        # Append the embeddings to the corresponding layer's list
        concat_layer_outputs[layer_idx].append(layer_embeddings)

# After all batches are processed, concatenate the embeddings for each layer
for layer_idx in range(num_layers):
    concat_layer_outputs[layer_idx] = torch.cat(concat_layer_outputs[layer_idx], dim=0)
    print(concat_layer_outputs[layer_idx].shape)

from src.utils.delta import calc_hyperbolicity, batched_delta_hyp
import torch
import math
hyperbolicity_results = {}

# Iterate over each layer's embeddings
for batch_idx, layer_embeddings in tqdm(layer_batch_outputs.items()):
    for i, emb in enumerate(layer_embeddings):
        # If layer_embedding is a list of embeddings for different inputs
        # Flatten the layer embedding to 2D if needed (e.g., from [seq_len, hidden_dim] to [seq_len*hidden_dim])
        emb = torch.tensor(emb)
        emb = emb.view(-1, emb.size(-1))
        
        # Calculate hyperbolicity
        rel_delta_mean, std, c = batched_delta_hyp(emb, batch_size=2000)
        
        # Initialize the dictionary for the layer if it doesn't exist
        if i not in hyperbolicity_results:
            hyperbolicity_results[i] = {'delta': [], 'std': [], 'c': []}
        
        # Append the results
        if math.isnan(rel_delta_mean):
            print(f"In batch {batch_idx} and layer {i} delta is nan.")
        hyperbolicity_results[i]['delta'].append(rel_delta_mean)
        hyperbolicity_results[i]['std'].append(std)
        hyperbolicity_results[i]['c'].append(c)

# Store the averaged results for each layer
results = {}
for layer, values in hyperbolicity_results.items():
    sum_delta = sum(values['delta'])
    sum_std = sum(values['std'])
    sum_c = sum(values['c'])
    
    results[layer] = {
        'delta': sum_delta / len(values['delta']),
        'std': sum_std / len(values['std']),
        'c': sum_c / len(values['c'])
    }

# `results` now contains the averaged

import matplotlib.pyplot as plt

# Print averaged results
for layer, values in results.items():
    print(f"Layer {layer}: Mean δ = {values['delta']:.3f} STD δ = {values['std']:.3f} c = {values['c']:.3f}")

# Preparing data for plotting
layers = sorted(results.keys(), key=lambda x: int(x))  # Sort layers numerically
delta_values = [results[layer]['delta'] for layer in layers]
std_values = [results[layer]['std'] for layer in layers]

# Plotting δ values with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(layers, delta_values, yerr=std_values, fmt='o', capsize=5, capthick=2, label='Mean δ with STD')

plt.xlabel("Layer")
plt.ylabel("Average δ")
plt.title("Average δ with Standard Deviation across Layers")
plt.grid(True)
plt.legend()
plt.show()
