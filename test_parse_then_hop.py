from src.utils.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import ParseThenHopDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph
from src.models import SoftPromptModel, HyperbolicKthLayerT5Model
from src.eval import evaluate_parse_then_hop_training
import argparse
import optuna
import os

BATCH_SIZE = 6
NUM_WORKERS = 4
HOPPING_PROMPT_CHECKPOINT_PATH = "checkpoints/random_walk_training/euclidean/random_walk_euclidean/soft_prompt_epoch_28_val_loss_0.1201_em_0.405318.pth"
PARSING_PROMPT_CHECKPOINT_PATH = "checkpoints/parse_then_hop_training/euclidean/parse_euclidean/soft_prompt_epoch_23_val_loss_0.0686_em_0.883859.pth"
KNIT_MODEL_CHECKPOINT_PATH = "checkpoints/knowledge_integration/large_adapt_bsize64_c4/model_epoch_16_val_loss_0.0336.pth"
def test_path():
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

    all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
    all_kg = create_knowledge_graph(all_data)

    print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

    print(f"Lenght Train Data: {len(train_dataset)}")
    print(f"Lenght Dev Data: {len(dev_dataset)}")
    print(f"Lenght Test Data: {len(test_dataset)}")

    parse_then_hop_test = ParseThenHopDataset(test_dataset)

    print(f"Number of PATH Test: {len(parse_then_hop_test)}")

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    path_test = DataLoader(parse_then_hop_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    knit5_parsing_model = HyperbolicKthLayerT5Model(curvature=config.random_walk_training.curvature, map_encoder_layers=config.t5_model.map_encoder_layers, map_decoder_layers=config.t5_model.map_decoder_layers, checkpoint_hyperbolic_knit5=KNIT_MODEL_CHECKPOINT_PATH)
    knit5_hopping_model = HyperbolicKthLayerT5Model(curvature=config.random_walk_training.curvature, map_encoder_layers=config.t5_model.map_encoder_layers, map_decoder_layers=config.t5_model.map_decoder_layers, checkpoint_hyperbolic_knit5=KNIT_MODEL_CHECKPOINT_PATH)
    import torch.nn as nn

    checkpoint = torch.load(PARSING_PROMPT_CHECKPOINT_PATH, map_location=device)
    parsing_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_parsing_linear_layer = checkpoint['additional_linear_layer']
    checkpoint = torch.load(HOPPING_PROMPT_CHECKPOINT_PATH, map_location=device)
    hopping_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_hopping_linear_layer = checkpoint['additional_linear_layer']

    knit5_hopping_model.hyperbolic_layer.load_state_dict(additional_hopping_linear_layer)
    knit5_parsing_model.hyperbolic_layer.load_state_dict(additional_parsing_linear_layer)
    print("Loaded Soft Prompts and Additional Linear Layer")

    print(f"{parsing_prompt.shape = }")
    print(f"{hopping_prompt.shape = }")
    parsing_model = SoftPromptModel(knit5=knit5_parsing_model, knit5_checkpoint_path=KNIT_MODEL_CHECKPOINT_PATH, soft_prompt=parsing_prompt, model_name="...", with_model_state_dict=False)
    hopping_model = SoftPromptModel(knit5=knit5_hopping_model, knit5_checkpoint_path=None, model_name='hyperbolic_hopping_prompt', soft_prompt=hopping_prompt, with_model_state_dict=False)




    evaluate_parse_then_hop_training(parsing_model=parsing_model,
                                     hopping_model=hopping_model,
                                     tokenizer=tokenizer,
                                     test_dataloader=path_test)


  


if __name__ == '__main__':    
    
    

    test_path()