from src.utils.util import load_dataset
from src.train import *
from src.datasets import ParseDataset
import pandas as pd
from src.knowledge_graph import create_knowledge_graph
from src.config import Config
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
from src.models import *
import argparse

  
train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset('dataset/2wikimultihop', do_correct_wrong_evidences=True)

all_data = pd.concat([train_dataset, dev_dataset, test_dataset])
all_kg = create_knowledge_graph(all_data)

print(f"Nodes in Data: {len(list(all_kg.nodes()))}")

print(f"Lenght Train Data: {len(train_dataset)}")
print(f"Lenght Dev Data: {len(dev_dataset)}")
print(f"Lenght Test Data: {len(test_dataset)}")

parse_then_hop_train = ParseDataset(train_dataset)
parse_then_hop_dev = ParseDataset(dev_dataset)

print(f"Number of Random Walks Train: {len(parse_then_hop_train)}")
print(f"Number of Random Walk Dev: {len(parse_then_hop_dev)}")

#Specify Hyperparameters via config file
config = Config()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Training on device: {device}')

#google/t5-large-lm-adapt
parse_then_hop_dataloader_train = DataLoader(parse_then_hop_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.parse_then_hop_training.num_workers)
parse_then_hop_dataloader_dev = DataLoader(parse_then_hop_dev,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.parse_then_hop_training.num_workers)

def _train_parse_then_hop(hyperbolic: bool):
    model_name = config.t5_model.model_name
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading Model...")
    knit5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    if hyperbolic:
        #hyperbolic_knit5_model = HyperbolicT5Model(knit5_model)
        print("Train with hyperbolic Soft Prompt Model.")
        model = HyperbolicSoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'hyperbolic_parsing_prompt', with_model_state_dict=False)   
    else:
        model = SoftPromptModel(knit5_model, config.random_walk_training.model_checkpoint_path, 'parsing_prompt')

    print("Model type before passing to SoftPromptTrainer:", type(model))

    trainer = SoftPromptTrainer(model,
                      tokenizer,
                      parse_then_hop_dataloader_train,
                      parse_then_hop_dataloader_dev,
                      config,
                      device=device,
                      method='parse_then_hop_training',
                      checkpoint_path=config.random_walk_training.hopping_prompt_checkpoint_path,
                      tboard_checkpoint_path=config.random_walk_training.tboard_checkpoint_path,
                      )




    print(f'Random Walk Training..')
    print(f'with model: {config.t5_model.model_name}')

    print(f'Model Config: {model.knit5.config}')
    print(f'for: {config.random_walk_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.random_walk_training.optimizer}')   

    trainer.train(epochs=config.parse_then_hop_training.epochs)
    
    
if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser(description='Parse Then Hop Training')
    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation')
    args = parser.parse_args()
    

    _train_parse_then_hop(hyperbolic = args.hyperbolic)
