from src.utils.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import ParseThenHopDataset, RandomWalkDataset, RandomWalkMetaQADataset, RandomWalkMLPQDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop, create_knowledge_graph_metaqa, create_knowledge_graph_mlpq
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.eval import evaluate_parse_then_hop_training, evaluate_random_walk_training
import argparse
import optuna
import os

def test_random_walk(dataset, additional_layer, batch_size, knit5_checkpoint_path, prompt_tuning_checkpoint_path):
    MAX_ANSWER = None
    GPU_PARALLELIZATION = False if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop'] else True
    WITH_MODEL_STATE_DICT = GPU_PARALLELIZATION
    if dataset in ['2wikimultihop', 'wikimultihop', '2wikihop', 'wikihop']:
        train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

        all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
        all_kg = create_knowledge_graph_wikimultihop(all_data)

        print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

        print(f"Lenght Train Data: {len(train_dataset)}")
        print(f"Lenght Dev Data: {len(dev_dataset)}")
        print(f"Lenght Test Data: {len(test_dataset)}")
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
        random_walk_test = RandomWalkMetaQADataset(kg, df_dev, df_test, steps=3, type='test', max_answers=MAX_ANSWER)
    elif dataset in ['mlpq']:
        #txt_file_paths = ['dataset/mlpq/Triples_in_questions/EN_KG', 'dataset/mlpq/Triples_in_questions/FR_KG']
        train_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_train_question_evidences.json', lines=True)
        validation_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_dev_question_evidences.json', lines=True)
        test_dataframe = pd.read_json('dataset/mlpq/Questions/fr-en/2-hop/2hop_test_question_evidences.json', lines=True)

        df_kg = pd.concat([train_dataframe, validation_dataframe, test_dataframe])
        kg = create_knowledge_graph_mlpq(df_kg, from_kb = False)

        random_walk_test = RandomWalkMLPQDataset(kg, validation_dataframe, test_dataframe, steps=3, type='test')
    else:
        raise ValueError(f"Unknown Dataset")

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    test_random_walk_dataloader = DataLoader(random_walk_test, batch_size=batch_size, shuffle=False)


    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(layer_type=additional_layer, curvature=config.random_walk_training.curvature, checkpoint_hyperbolic_knit5=knit5_checkpoint_path, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION, soft_prompt_length=config.random_walk_training.prompt_length)
    import torch.nn as nn

    checkpoint = torch.load(prompt_tuning_checkpoint_path, map_location=device)
    hopping_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_linear_layer = checkpoint['additional_linear_layer']

    hyperbolic_knit5_model.hyperbolic_layer.load_state_dict(additional_linear_layer)
    print("Loaded Soft Prompts and Additional Linear Layer")

    print(f"{hopping_prompt.shape = }")
    model = SoftPromptModel(knit5=hyperbolic_knit5_model, knit5_checkpoint_path=knit5_checkpoint_path, soft_prompt=hopping_prompt, with_model_state_dict=WITH_MODEL_STATE_DICT, gpu_parallelization=GPU_PARALLELIZATION)




    evaluate_random_walk_training(model, tokenizer, test_random_walk_dataloader)


  


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('--dataset', type=str, nargs='?', default=None, help='Specify the dataset (e.g., metaqa, 2wikimultihop)')
    # New argument: --additional_layer
    parser.add_argument(
        '--additional_layer',
        type=str,
        choices=['identity', 'hyperbolic', 'linear'],
        default='identity',  # You can set a different default if needed
        help='Specify the type of additional layer to use: identity, hyperbolic, or linear'
    )
    parser.add_argument(
        '--knit5_checkpoint_path',
        type=str,
        default=None,
        help='Specify Checkpoint Path of finetuned KNIT5 Model'
    )
    parser.add_argument(
        '--prompt_tuning_checkpoint_path',
        type=str,
        default=None,
        help='Specify checkpoint path of prompt tuning experiments'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Specify batch size'
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    dataset = args.dataset 
    additional_layer = args.additional_layer
    knit5_checkpoint_path = args.knit5_checkpoint_path
    prompt_tuning_checkpoint_path = args.prompt_tuning_checkpoint_path
    batch_size = args.batch_size
    test_random_walk(dataset, additional_layer, batch_size, knit5_checkpoint_path, prompt_tuning_checkpoint_path)