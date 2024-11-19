from src.utils.util import load_dataset, load_c4_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.datasets import KnowledgeIntegrationDataset, C4Dataset
import torch
from src.config import Config
from torch.utils.data import DataLoader
from src.train import *
import argparse
from src.models import HyperbolicKthLayerT5Model

def _knowledge_integration_with_c4(hyperbolic):
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop", do_correct_wrong_evidences=True)


    print("Creating Single Hop Datasets...")
    dataset_with_all_entries = pd.concat([train_dataset, dev_dataset, test_dataset])

    ki_dataset = KnowledgeIntegrationDataset(dataset_with_all_entries)

    ki_train = ki_dataset


    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.model_name)
    print(f"Loading Model {config.t5_model.model_name}...")
    #Adjust Dropout
    if hyperbolic:
        model = HyperbolicKthLayerT5Model(curvature=config.single_hop_training.curvature, map_encoder_layers=config.t5_model.map_encoder_layers, map_decoder_layers=config.t5_model.map_decoder_layers, checkpoint_hyperbolic_knit5=config.single_hop_training.model_checkpoint_path)
        print(f"Train with hyperbolic Soft Prompt Model with curvature {config.single_hop_training.curvature} and Hyperbolic Linear Layer")

    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model.model_name)
        
    model.config.dropout_rate = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1
    model.config.classifier_dropout = 0.1

    base_path = 'c4/en/c4-train.{:05d}-of-01024.json'
    c4_dataset = load_c4_dataset(base_path, number_of_files=2)

    objective = 'prefix_language_modeling'
    C4_train = C4Dataset(c4_dataset ,tokenizer=tokenizer, objective=objective)
        

    c4_dataloader_train = DataLoader(C4_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.single_hop_training.num_workers)
    single_hop_dataloader_train = DataLoader(ki_train, batch_size=config.t5_model.batch_size, shuffle=True, num_workers=config.single_hop_training.num_workers)
    single_hop_dataloader_dev = DataLoader(ki_train,  batch_size=config.t5_model.batch_size, shuffle=False, num_workers=config.single_hop_training.num_workers)

    trainer = ModelTrainer(model,
                        tokenizer,
                        [single_hop_dataloader_train, c4_dataloader_train],
                        single_hop_dataloader_dev,
                        config,
                        device=device,
                        validation_step=1,
                        checkpoint_path=config.single_hop_training.model_checkpoint_path,
                        tboard_checkpoint_path=config.single_hop_training.tboard_checkpoint_path,
                        method = 'single_hop_training')

    print(f'Knowledge Integration training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.single_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.single_hop_training.optimizer}')
    print(f'With C4')

    print(f'Training with {len(ki_train)} Triples for Knowledge Integration.')

    trainer.train(epochs=config.single_hop_training.epochs)
    
    
def _knowledge_integration_without_c4(hyperbolic):
    train_dataset, dev_dataset, test_dataset, kg_train, kg_dev, kg_test = load_dataset("dataset/2wikimultihop", do_correct_wrong_evidences=True)


    print("Creating Single Hop Datasets...")
    dataset_with_all_entries = pd.concat([train_dataset, dev_dataset, test_dataset])
    ki_dataset = KnowledgeIntegrationDataset(dataset_with_all_entries)

    ki_train = ki_dataset


    #Specify Hyperparameters via config file
    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on device: {device}')

    #Define Tokenizer and Model
    #google/t5-large-lm-adapt
    print("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.t5_model.batch_size)
    print(f"Loading Model {config.t5_model.batch_size}...")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model.model_name)
    #Adjust Dropout
    
    if hyperbolic:
        print("Train with Hyperbolic Model.")
        model = HyperbolicKthLayerT5Model(curvature=config.single_hop_training.curvature, map_encoder_layers=config.t5_model.map_encoder_layers, map_decoder_layers=config.t5_model.map_decoder_layers)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.t5_model.model_name)
        
    model.config.dropout_rate = 0.1
    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1

    single_hop_dataloader_train = DataLoader(ki_train, batch_size=config.t5_model.batch_size, shuffle=True)
    single_hop_dataloader_dev = DataLoader(ki_train,  batch_size=config.t5_model.batch_size, shuffle=False)

    trainer = ModelTrainer(model,
                            tokenizer,
                            [single_hop_dataloader_train],
                            single_hop_dataloader_dev,
                            config,
                            device=device,
                            validation_step=1,
                            checkpoint_path=config.single_hop_training.model_checkpoint_path,
                            tboard_checkpoint_path=config.single_hop_training.tboard_checkpoint_path,
                            method = 'single_hop_training')

    print(f'Knowledge Integration training..')
    print(f'with model: {config.t5_model.model_name}')
    print(f'Model Config: {model.config}')
    print(f'for: {config.single_hop_training.epochs} epochs')
    print(f'with batch size: {config.t5_model.batch_size}')
    print(f'with optimizer: {config.single_hop_training.optimizer}')
    print(f'Without C4')
    print(f'Training with {len(ki_train)} Triples for Knowledge Integration.')

    trainer.train(epoch=config.single_hop_training.epochs)
        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Integration Training')
    parser.add_argument('--c4', action='store_true', help='Include C4 dataset in training')
    parser.add_argument('--hyperbolic', action='store_true', help='Train with hyperbolic representation in single hop dataset')
    args = parser.parse_args()
    
    if args.c4:
        _knowledge_integration_with_c4(hyperbolic = args.hyperbolic)
    else:
        _knowledge_integration_without_c4(hyperbolic = args.hyperbolic)
