from src.utils.util import load_dataset
from src.utils.trainer_utils import load_model_checkpoint
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import RandomWalkDataset, ParseDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop
import matplotlib.pyplot as plt
import sys
import argparse
def plot_layer_hyperbolicity(layer_hyperbolicity_dict, title, save_path = None,):
    layers = list(layer_hyperbolicity_dict.keys())
    means = [layer_hyperbolicity_dict[layer]['mean'] for layer in layers]
    std_devs = [layer_hyperbolicity_dict[layer]['std'] for layer in layers]
    
    # Create the error plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(layers, means, yerr=std_devs, fmt='o', capsize=5, label='Mean ± Std')
    
    # Adding labels and title
    plt.xlabel('Layer')
    plt.ylabel('Mean Hyperbolicity')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    
    # Show the plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
def _comput_delta_hyperbolicity(dataloader, model, tuned, parse):
    from tqdm import tqdm
    from src.utils.delta import batched_delta_hyp
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    layer_embedding_list = []
    with torch.no_grad():
        for question, answer in tqdm(dataloader, file=sys.stdout):
            inputs = tokenizer(question, padding=True, truncation=True, return_tensors='pt')
            labels = tokenizer(answer, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels = labels)

            # Extract the encoder's output at the desired layer
            encoder_output = outputs.encoder_hidden_states[24].cpu()

            # Compute the sentence embedding by averaging over the token embeddings
            sentence_embedding = torch.mean(encoder_output, dim=1)
            layer_embedding_list.append(sentence_embedding)
    output_embeddings = torch.cat(layer_embedding_list, dim=0)
    mean, std, c = batched_delta_hyp(output_embeddings)
    dataloader_string = "Parse" if parse else "Random_Walk"
    tuned_string = "Knit5" if tuned else "T5"
    print(f"Done!")
    print(f"Delta Hyperbolicity on {dataloader_string} Dataset of {tuned_string} Model")
    print(f"{mean = }")
    print(f"{std = }")
    print(f"{c = }")


    
    #plot_layer_hyperbolicity(layer_hyperbolicity_dict, save_path = f"delta_hyperbolicity_{dataloader_string}_{tuned_string}.png", title=f'Delta Hyperbolicity on {dataloader_string} Dataset of {tuned_string} Model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('--tuned', action='store_true', help='Takes Knit5')
    parser.add_argument('--parse', action='store_true', help='Does Parse Dataset')
    parser.add_argument('--random_walk', action='store_true', help='Does Random Walk Dataset')
    args = parser.parse_args()
    tuned = args.tuned
    random_walk = args.random_walk
    parse = args.parse
    dataset_string = "Parse" if parse else "Random Walk"
    tuned_string = "Tuned" if tuned else "Not Tuned"
    
    print(f"Start {tuned_string} {dataset_string}")
    model_checkpoint_path = 'checkpoints/knowledge_integration/large_adapt_bsize64_c4/model_epoch_16_val_loss_0.0336.pth'

    

    config = Config()
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    import torch.nn as nn
    from src.utils.trainer_utils import load_soft_prompt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    knit5_model.to(device)
    if tuned:
        load_model_checkpoint(knit5_model, model_checkpoint_path, with_model_state_dict=False)

    train_dataset, dev_dataset, test_dataset, _, _, _ = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

    all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
    all_kg = create_knowledge_graph_wikimultihop(all_data)

    print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

    print(f"Lenght Train Data: {len(train_dataset)}")
    print(f"Lenght Dev Data: {len(dev_dataset)}")
    print(f"Lenght Test Data: {len(test_dataset)}")
    from torch.utils.data import ConcatDataset
    if random_walk:
        random_walk_train = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='train')
        random_walk_dev = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev')
        random_walk_test = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='test')
        combined_dataset = ConcatDataset([random_walk_train, random_walk_test, random_walk_dev])
        dataloader = DataLoader(combined_dataset, shuffle=False, batch_size=64, num_workers=16)
    if parse:
        parse_train = ParseDataset(train_dataset)
        parse_dev = ParseDataset(dev_dataset)
        parse_test = ParseDataset(test_dataset)
        combined_dataset = ConcatDataset([parse_train, parse_dev, parse_test])
        dataloader = DataLoader(combined_dataset, shuffle=False, batch_size=64, num_workers=16)

    


    _comput_delta_hyperbolicity(dataloader, knit5_model, tuned, parse)