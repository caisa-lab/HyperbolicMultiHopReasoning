from src.utils.util import load_dataset, get_top_token_embeddings
import pandas as pd
from src.train.soft_prompt_trainer import SoftPromptTrainer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import RandomWalkDataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.knowledge_graph import create_knowledge_graph
from src.models import HyperbolicSoftPromptModel, SoftPromptModel, HyperbolicT5Model
import argparse
import optuna

train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
all_kg = create_knowledge_graph(all_data)

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

random_walk_dataloader_train = DataLoader(random_walk_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=4)
random_walk_dataloader_dev = DataLoader(random_walk_dev,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=4)

def objective(trial):
    print("Optimizing learning_rate with optuna")
    
    learning_rate = trial.suggest_float('lr', 0.01, 1.0)
    c = trial.suggest_float('c', 0.1, 2.0)
    #google/t5-large-lm-adapt
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    config.random_walk_training.learning_rate = learning_rate
    config.random_walk_training.epochs = 5
 
    print("Train with hyperbolic Soft Prompt Model.")
    model = HyperbolicSoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'hyperbolic_hopping_prompt', with_model_state_dict=False, curvature=c)   
    print("Using curvature", c)

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
    
    if hyperbolic:
        hyperbolic_knit5_model = HyperbolicT5Model(knit5_model, 'hyperbolic_knit5')
        print("Train with hyperbolic Soft Prompt Model.")
        model = HyperbolicSoftPromptModel(hyperbolic_knit5_model, config.random_walk_training.model_checkpoint_path, 'hyperbolic_hopping_prompt', with_model_state_dict=True)   
    else:
        model = SoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'hopping_prompt')

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
                      )


    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')

    print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.random_walk_training.optimizer}')

    trainer.train(epochs=config.random_walk_training.epochs)  
    


if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Knowledge Integration Training')
    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation')
    parser.add_argument('--optuna', action='store_true', help='Optimizes learning rate of Soft Prompting for Random Walk')
    args = parser.parse_args()
    
    if args.optuna:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)
        # Print the best hyperparameters
        print("Best hyperparameters: ", study.best_params)
        print("Best accuracy: ", study.best_value)
    else:
        _train_random_walk(hyperbolic = args.hyperbolic)
