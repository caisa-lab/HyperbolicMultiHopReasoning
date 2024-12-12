from src.utils.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import ParseThenHopDataset, RandomWalkDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop
from src.models import SoftPromptModel, T5ModelWithAdditionalLayer
from src.eval import evaluate_parse_then_hop_training, evaluate_random_walk_training
import argparse
import optuna
import os

BATCH_SIZE = 8
NUM_WORKERS = 4
HOPPING_PROMPT_CHECKPOINT_PATH = "checkpoints/random_walk_training/euclidean/random_walk_euclidean/soft_prompt_epoch_28_val_loss_0.1201_em_0.405318.pth"
KNIT_MODEL_CHECKPOINT_PATH = "checkpoints/knowledge_integration/large_adapt_bsize64_c4/model_epoch_16_val_loss_0.0336.pth"
def test_path():
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

    all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
    all_kg = create_knowledge_graph_wikimultihop(all_data)

    print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

    print(f"Lenght Train Data: {len(train_dataset)}")
    print(f"Lenght Dev Data: {len(dev_dataset)}")
    print(f"Lenght Test Data: {len(test_dataset)}")

    random_walk_test = RandomWalkDataset(all_kg, dev_dataset, test_dataset, steps=3, type='test')

    print(f"Number of PATH Test: {len(random_walk_test)}")

    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    test_random_walk_dataloader = DataLoader(random_walk_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    hyperbolic_knit5_model = T5ModelWithAdditionalLayer(layer_type='euclidean', curvature=config.random_walk_training.curvature, checkpoint_hyperbolic_knit5=KNIT_MODEL_CHECKPOINT_PATH)
    import torch.nn as nn

    checkpoint = torch.load(HOPPING_PROMPT_CHECKPOINT_PATH, map_location=device)
    hopping_prompt = nn.Parameter(checkpoint['soft_prompt_state_dict'], requires_grad=False)
    additional_linear_layer = checkpoint['additional_linear_layer']

    hyperbolic_knit5_model.hyperbolic_layer.load_state_dict(additional_linear_layer)
    print("Loaded Soft Prompts and Additional Linear Layer")

    print(f"{hopping_prompt.shape = }")
    hopping_model = SoftPromptModel(knit5=hyperbolic_knit5_model, knit5_checkpoint_path=None, model_name='hyperbolic_hopping_prompt', soft_prompt=hopping_prompt, with_model_state_dict=False)




    evaluate_random_walk_training(hopping_model, tokenizer, test_random_walk_dataloader)


  


if __name__ == '__main__':    
    test_path()