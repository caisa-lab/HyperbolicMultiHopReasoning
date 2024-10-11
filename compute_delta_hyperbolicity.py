from src.utils.util import load_dataset
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import RandomWalkDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph
from src.models import SoftPromptModel, HyperbolicKthLayerT5Model

config = Config()
model_name = config.t5_model.model_name
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("Loading Model...")
knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
import torch.nn as nn
from src.utils.trainer_utils import load_soft_prompt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

euclidean_soft_prompt = torch.load('../checkpoints/random_walk_training/euclidean_AdaFactor_0.3_0.0/soft_prompt_epoch_34_val_loss_0.1283.pth', map_location=device)['soft_prompt_state_dict']
euclidean_model = SoftPromptModel(knit5=knit5_model, knit5_checkpoint_path=config.random_walk_training.model_checkpoint_path, model_name='', soft_prompt=euclidean_soft_prompt, with_model_state_dict=False)
euclidean_model.to(device)

train_dataset, dev_dataset, test_dataset, _, _, _ = load_dataset('../dataset/2wikimultihop', do_correct_wrong_evidences=True)

all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
all_kg = create_knowledge_graph(all_data)

print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

print(f"Lenght Train Data: {len(train_dataset)}")
print(f"Lenght Dev Data: {len(dev_dataset)}")
print(f"Lenght Test Data: {len(test_dataset)}")

#random_walk_train = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='train')
#random_walk_dev = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev')
random_walk_test = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='test')

# print(f"Number of Random Walks Train: {len(random_walk_train)}")
#print(f"Number of Random Walk Dev: {len(random_walk_dev)}")
print(f"Number of Random Walk Test: {len(random_walk_test)}")

random_walk_test_dataloader = DataLoader(random_walk_test, shuffle=False, batch_size=10, num_workers=4)


from tqdm import tqdm
from src.utils.delta import batched_delta_hyp
layer_hyperbolicity_dict = {}
for layer in range(25):
    layer_embedding_list = []
    with torch.no_grad():
        for incomplete, complete in tqdm(random_walk_test_dataloader):
            inputs = tokenizer(incomplete, padding=True, truncation=True, return_tensors='pt')
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            labels = tokenizer(complete, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)

            input_embeddings = euclidean_model.knit5.shared(input_ids)
            outputs = euclidean_model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, output_hidden_states = True)
            encoder_output = outputs.encoder_hidden_states[layer].cpu()
            #print(encoder_output_without_soft_prompt.device)
            layer_embedding_list.append(encoder_output.view(-1, encoder_output.size(-1)))
    output_embeddings = torch.cat(layer_embedding_list, dim=0).half()
    layer_embedding_list = None
    mean, std, c = batched_delta_hyp(output_embeddings)
    layer_hyperbolicity_dict[layer] = {'mean': mean, 'std': std, 'c': c}
    print(f"Layer: {layer} Done!")


    import matplotlib.pyplot as plt

def plot_layer_hyperbolicity(layer_hyperbolicity_dict, save_path = None):
    layers = list(layer_hyperbolicity_dict.keys())
    means = [layer_hyperbolicity_dict[layer]['mean'] for layer in layers]
    std_devs = [layer_hyperbolicity_dict[layer]['std'] for layer in layers]
    
    # Create the error plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, means, yerr=std_devs, fmt='o', capsize=5, label='Mean ± Std')
    
    # Adding labels and title
    plt.xlabel('Layer')
    plt.ylabel('Mean Hyperbolicity')
    plt.title('Mean and Standard Deviation of Hyperbolicity by Layer')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

plot_layer_hyperbolicity(layer_hyperbolicity_dict, save_path = "delta_hyperbolicity_encoder_random_walk_euclidean.png")
