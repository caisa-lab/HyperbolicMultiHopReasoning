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
        for idx, (question, answer) in enumerate(tqdm(dataloader, file=sys.stdout)):
            inputs = tokenizer(question, padding=True, truncation=True, return_tensors='pt')
            labels = tokenizer(answer, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, labels = labels)

            # Extract the encoder's output at the desired layer
            encoder_output = outputs.encoder_hidden_states[24]

            mask = attention_mask.bool()

            token_embeddings = encoder_output[mask]
            layer_embedding_list.append(token_embeddings.detach().cpu())
            if idx <= 2:
                print(f"{token_embeddings.shape = }")
            

            # mask = attention_mask.unsqueeze(-1)
            # # Convert mask to float for multiplication
            # mask = mask.float() 
            # masked_encoder_output = encoder_output * mask
            # sum_embeddings = masked_encoder_output.sum(dim=1)
            # non_padded_tokens = mask.sum(dim=1)
            # non_padded_tokens = torch.clamp(non_padded_tokens, min=1e-9)#
            # # Compute the mean by dividing the sum by the number of non-padded tokens
            # sentence_embedding = sum_embeddings / non_padded_tokens  # Shape: (batch_size, hidden_size)

            # #Append to the list
            # layer_embedding_list.append(sentence_embedding.detach().cpu())

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
    config = Config()
    print(f"Start {tuned_string} {dataset_string}")
    model_checkpoint_path = config.random_walk_training.model_checkpoint_path#'checkpoints/metaqa/knowledge_integration/Dec15_18-47-46_AdaFactor_0.001_1.0_euclidean_t5_large_batch_size64_with_c4_prefix_language_modeling/knit5_epoch_24_val_loss_0.1764.pth'

    

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
        load_model_checkpoint(knit5_model, model_checkpoint_path, with_model_state_dict=True, gpu_parallelization=True)

    from src.knowledge_graph import create_knowledge_graph_metaqa

    dataframe_kg = pd.read_csv("dataset/metaqa/kb.txt", sep="|")
    all_kg = create_knowledge_graph_metaqa(dataframe_kg)
    dev_dataset = pd.read_json('dataset/metaqa/2hops/qa_dev_evidences.json')
    test_dataset = pd.read_json('dataset/metaqa/2hops/qa_test_evidences.json')
    train_dataset = pd.read_json('dataset/metaqa/2hops/qa_train_evidences.json')
    from src.datasets import RandomWalkMetaQADataset, ParseMetaQADataset
    from torch.utils.data import ConcatDataset
    if random_walk:
        random_walk_train = RandomWalkMetaQADataset(all_kg, dev_dataset, test_dataset, steps=3, type='train', all_paths=False)
        random_walk_dev = RandomWalkMetaQADataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev', all_paths=False)
        random_walk_test = RandomWalkMetaQADataset(all_kg, dev_dataset, test_dataset, steps=3, type='test', all_paths=False)
        combined_dataset = ConcatDataset([random_walk_train, random_walk_test, random_walk_dev])
    if parse:
        parse_train = ParseMetaQADataset(train_dataset)
        parse_dev = ParseMetaQADataset(dev_dataset)
        parse_test = ParseMetaQADataset(test_dataset)
        combined_dataset = ConcatDataset([parse_train, parse_dev, parse_test])
    if not random_walk and not parse:
        from src.datasets import KnowledgeIntegrationMetaQADataset
        print("Training on MetaQA")
        metaqa_kg_data = pd.read_csv('dataset/metaqa/kb.txt', sep="|")

        #Comment Out if we want to use the whole kb. This used only the kb for questions with a single answer
        df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
        df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
        df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
        evidences = df_test['evidences'].to_list() + df_train['evidences'].to_list() + df_dev['evidences'].to_list()
        list_of_all_entities = set()
        for evidence_list in evidences:
            if len(evidence_list) > 1:
                continue
            for evidence in evidence_list:
                entity_1 = evidence[0]
                entity_2 = evidence[2]
                entity_3 = evidence[4]
                list_of_all_entities.add(entity_1)
                list_of_all_entities.add(entity_2)
                list_of_all_entities.add(entity_3)
        list_of_all_entities = list(list_of_all_entities)
        kg = create_knowledge_graph_metaqa(metaqa_kg_data, list_of_all_entities=list_of_all_entities)

        combined_dataset = KnowledgeIntegrationMetaQADataset(kg)
    dataloader = DataLoader(combined_dataset, shuffle=False, batch_size=64, num_workers=16)

    


    _comput_delta_hyperbolicity(dataloader, knit5_model, tuned, parse)