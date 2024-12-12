from src.utils.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import RandomWalkDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph_wikimultihop
from src.models import SoftPromptModel, HyperbolicKthLayerT5Model
import argparse
import optuna
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

print(f"Number of Random Walks Train: {len(random_walk_train)}")
print(f"Number of Random Walk Dev: {len(random_walk_dev)}")
print(f"Number of Random Walk Test: {len(random_walk_test)}")

#Specify Hyperparameters via config file
config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on device: {device}')

random_walk_dataloader_train = DataLoader(random_walk_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.random_walk_training.num_workers)
random_walk_dataloader_dev = DataLoader(random_walk_dev,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.random_walk_training.num_workers)

def objective(trial):
    print("Optimizing learning_rate with optuna")
    
    learning_rate = trial.suggest_categorical('lr', [3e-5, 3e-4, 3e-3, 3e-2, 3e-1])
    c = trial.suggest_categorical('c', [-1, 0, 0.1, 0.5, 1, 2, 4])
    optimizer_choice = trial.suggest_categorical('optimizer', ['AdaFactor', 'Hyperbolic'])
    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    config.random_walk_training.learning_rate = learning_rate
    config.random_walk_training.epochs = 16
    config.random_walk_training.optimizer = optimizer_choice
    config.random_walk_training.curvature = c
 
    print("Train with hyperbolic Soft Prompt Model.")
    model = SoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'hyperbolic_hopping_prompt', with_model_state_dict=False, curvature=config.random_walk_training.curvature)   
    print("Using curvature", config.random_walk_training.curvature)
    print("Using Learning Rate", config.random_walk_training.learning_rate)

    trainer = SoftPromptTrainer(model,
                      tokenizer,
                      random_walk_dataloader_train,
                      random_walk_dataloader_dev,
                      config,
                      device=device,
                      method='random_walk_training',
                      checkpoint_path=config.random_walk_training.hopping_prompt_checkpoint_path,
                      tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path,
                      )


    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.train(epochs=config.random_walk_training.epochs)  
    
    
    
    return trainer.best_loss
    

def _train_random_walk(hyperbolic : bool):
    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    import torch.nn as nn
    from src.utils.trainer_utils import load_soft_prompt


    if config.random_walk_training.hopping_prompt_checkpoint_path is not None:
        soft_prompt = nn.Parameter(torch.randn(config.random_walk_training.prompt_length, 1024))
        soft_prompt = load_soft_prompt(soft_prompt, config.random_walk_training.hopping_prompt_checkpoint_path)
    else:
        soft_prompt = None
    
    if hyperbolic:
        hyperbolic_knit5_model = HyperbolicKthLayerT5Model(curvature=config.random_walk_training.curvature, map_encoder_layers=config.t5_model.map_encoder_layers, map_decoder_layers=config.t5_model.map_decoder_layers, checkpoint_hyperbolic_knit5=config.random_walk_training.model_checkpoint_path)
        model = SoftPromptModel(curvature=config.random_walk_training.curvature, knit5=hyperbolic_knit5_model, knit5_checkpoint_path=config.random_walk_training.model_checkpoint_path, model_name='hyperbolic_hopping_prompt', soft_prompt=soft_prompt, with_model_state_dict=False)
        print(f"Train with hyperbolic Soft Prompt Model with curvature {config.random_walk_training.curvature} and Exponential Mapping at encoder layer {config.t5_model.map_encoder_layers} and at decoder layer {config.t5_model.map_decoder_layers}")
    else:
        model = SoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'hopping_prompt', with_model_state_dict=False, soft_prompt=soft_prompt, curvature = config.random_walk_training.curvature)

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
                      retrain = True
                      )


    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'with lr: {config.random_walk_training.learning_rate}')

    #print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.train(epochs=config.random_walk_training.epochs)  
    


if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Random Walk Training Training')
    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation')
    parser.add_argument('--optuna', action='store_true', help='Optimizes learning rate of Soft Prompting for Random Walk')
    args = parser.parse_args()
    
    if args.optuna:
        sqlite_db_path = "sqlite:///optuna_study.db"
        study_name = "optimize_curvature_lr"
        study = optuna.create_study(study_name=study_name, storage=sqlite_db_path, direction='minimize', load_if_exists=True)
        study.optimize(objective, n_trials=2)
        # Print the best hyperparameters
        print("Best hyperparameters: ", study.best_params)
        print("Best accuracy: ", study.best_value)
    else:
        _train_random_walk(hyperbolic = args.hyperbolic)
