from src.utils.util import load_dataset
from src.utils.trainer_utils import load_model_checkpoint
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import RandomWalkDataset, ParseDataset
from src.models import T5ModelWithAdditionalLayer
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
def _comput_delta_hyperbolicity(dataloader, model : T5ModelWithAdditionalLayer, tuned, parse):
    from tqdm import tqdm
    from src.utils.delta import batched_delta_hyp
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)
    layer_embedding_list = []
    with torch.no_grad():
        for idx, (question, answer) in enumerate(tqdm(dataloader, file=sys.stdout)):
            # labels = tokenizer(answer, padding=True, truncation=True, return_tensors='pt').input_ids.to(device)
            
            inputs = tokenizer(question, padding=True, truncation=True, return_tensors='pt')
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
        
            outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            # Extract the encoder's output at the desired layer
            # encoder_output = outputs.encoder_hidden_states[24]
            encoder_output = outputs['last_hidden_state']
            #mask = attention_mask.bool()
            # print(encoder_output['last_hidden_state'])
            # print(mask.shape)

            #token_embeddings = encoder_output[mask]
            #layer_embedding_list.append(token_embeddings.detach().cpu())
                
            

            mask = attention_mask.unsqueeze(-1)
            # # Convert mask to float for multiplication
            mask = mask.float() 
            masked_encoder_output = encoder_output * mask
            sum_embeddings = masked_encoder_output.sum(dim=1)
            non_padded_tokens = mask.sum(dim=1)
            non_padded_tokens = torch.clamp(non_padded_tokens, min=1e-9)#
            # # Compute the mean by dividing the sum by the number of non-padded tokens
            sentence_embedding = sum_embeddings / non_padded_tokens  # Shape: (batch_size, hidden_size)

            # #Append to the list
            layer_embedding_list.append(sentence_embedding.detach().cpu())

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
    parser.add_argument('--dataset', type=str, help='choose dataset')
    parser.add_argument('--model_checkpoint', type=str, help='model checkpoint path')

    args = parser.parse_args()
    tuned = args.tuned
    random_walk = args.random_walk
    parse = args.parse
    dataset = args.dataset
    model_checkpoint_path = args.model_checkpoint
    dataset_string = "Parse" if parse else "Random Walk"
    tuned_string = "Tuned" if tuned else "Not Tuned"
    config = Config()
    print(f"Start {tuned_string} {dataset_string} on {dataset}")

    #model_checkpoint_path = 'checkpoints/metaqa/knowledge_integration/Jan04_23-55-58_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_max_answers_1/knit5.pth'    
    #model_checkpoint_path = 'checkpoints/metaqa/knowledge_integration/Jan04_23-55-58_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_max_answers_1/knit5.pth'
    #model_checkpoint_path = 'checkpoints/mlpq/knowledge_integration/Jan05_10-42-23_AdaFactor_0.001_knowledge_integration_bsize64_lr0.001/knit5.pth'
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    from src.models import T5ModelWithAdditionalLayer
    knit5_model = T5ModelWithAdditionalLayer(layer_type='identity')
    import torch.nn as nn
    from src.utils.trainer_utils import load_soft_prompt
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    knit5_model.to(device)
    GPU_PARALLELIZATION = False if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop'] else True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
    if tuned:
        load_model_checkpoint(knit5_model, model_checkpoint_path, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION)

    from src.knowledge_graph import create_knowledge_graph_metaqa, create_knowledge_graph_wikimultihop

   
    from src.datasets import ParseDataset
    from src.datasets import RandomWalkMetaQADataset, ParseMetaQADataset
    from src.datasets import RandomWalkMLPQDataset, ParseMLPQDataset
    from src.knowledge_graph import create_knowledge_graph_mlpq
    from torch.utils.data import ConcatDataset
    if random_walk:
        if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
            train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

            all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
            all_kg = create_knowledge_graph_wikimultihop(all_data)

            print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

            print(f"Lenght Train Data: {len(train_dataset)}")
            print(f"Lenght Dev Data: {len(dev_dataset)}")
            print(f"Lenght Test Data: {len(test_dataset)}")

            random_walk_train = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='train')
            random_walk_dev = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='dev')
            random_walk_test = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='test')


        elif dataset in ['metaqa']:
            # df_kg = pd.read_csv("dataset/metaqa/kb.txt", sep="|")
            # kg = create_knowledge_graph_metaqa(df_kg, from_kb=True)

            df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
            df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
            df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
            MAX_ANSWER = 1
            df_kg = pd.concat([df_dev, df_train, df_test])
            kg = create_knowledge_graph_metaqa(df_kg, from_kb=False, max_answers=MAX_ANSWER)
            random_walk_train = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='train')
            random_walk_dev = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='dev')
            random_walk_test = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='test')
        elif dataset in ['mlpq']:
            #txt_file_paths = ['dataset/mlpq/Triples_in_questions/EN_KG', 'dataset/mlpq/Triples_in_questions/FR_KG']
            train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
            validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
            test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

            df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
            kg = create_knowledge_graph_mlpq(df_kg, from_kb = False)

            random_walk_train = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='train')
            random_walk_dev = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='dev')
            random_walk_test = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='test')
        else:
            raise ValueError(f"Unknown Dataset")
        
        combined_dataset = ConcatDataset([random_walk_train])
    if parse:
        if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
            train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

            all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
            all_kg = create_knowledge_graph_wikimultihop(all_data)

            print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

            print(f"Lenght Train Data: {len(train_dataset)}")
            print(f"Lenght Dev Data: {len(dev_dataset)}")
            print(f"Lenght Test Data: {len(test_dataset)}")

            parse_train = ParseDataset(train_dataset)
            parse_dev = ParseDataset(dev_dataset)
        elif dataset in ['metaqa']:
            df_dev = pd.read_json("dataset/metaqa/2hops/qa_dev_evidences.json")
            df_train = pd.read_json("dataset/metaqa/2hops/qa_train_evidences.json")
            #df_test = pd.read_json("dataset/metaqa/2hops/qa_test_evidences.json")
            MAX_ANSWER = 1
            parse_train = ParseMetaQADataset(df_train, max_answers=MAX_ANSWER)
            parse_dev = ParseMetaQADataset(df_dev, max_answers=MAX_ANSWER)
        elif dataset in ['mlpq']:
            validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
            train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)

            parse_train = ParseMLPQDataset(train_dataframe)
            parse_dev = ParseMLPQDataset(validation_dataframe)
        else:
            raise ValueError(f"Unknown Dataset")
        combined_dataset = ConcatDataset([parse_train])
    dataloader = DataLoader(combined_dataset, shuffle=False, batch_size=256, num_workers=1)

    


    _comput_delta_hyperbolicity(dataloader, knit5_model, tuned, parse)