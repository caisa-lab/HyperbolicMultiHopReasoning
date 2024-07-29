import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from config import Config
from eval import exact_match_score, f1_score
from utils.trainer_utils import *
from src.models import HyperbolicSoftPromptModel, SoftPromptModel
from typing import Union

"""Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""
class SoftPromptTrainer:
    def __init__(self,
                 model : Union[SoftPromptModel, HyperbolicSoftPromptModel],
                 tokenizer : AutoTokenizer,
                 train_dataloader: DataLoader,
                 val_dataloader : DataLoader,
                 config : Config,
                 device : str ='cpu',
                 checkpoint_path : str = None,
                 tboard_checkpoint_path : str = None,
                 validation_step : int = 1,
                 method : str = 'random_walk_training'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = config.t5_model.tokenizer_max_length
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.tboard_checkpoint_path = tboard_checkpoint_path
        self.validation_step = validation_step
        
        
        self.start_epoch = 0
        self.patience = 5
        self.best_loss = float('inf')
        self.early_stop_counter=0
        self.best_model_path = None
        
        self.supported_methods = ['random_walk_training', 'parse_then_hop_training']
        
        
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        else:
            self.method = method
        

        if self.method == 'random_walk_training':
            self.training_config = config.random_walk_training
        elif self.method == 'parse_then_hop_training':
            self.training_config = config.parse_then_hop_training
            
            
            
        self.optimizer = get_optimizer(self.model.soft_prompt.parameters(), self.training_config)
        if isinstance(self.model, SoftPromptModel):
            print('Model is of Type SoftPromptModel')
            for param in self.model.knit5.parameters():
                param.requires_grad = False
        elif isinstance(self.model, HyperbolicSoftPromptModel):
            print('Model is of Type HyperbolicSoftPromptModel')
            for param in self.model.hyperbolic_knit5.parameters():
                param.requires_grad = False
        else:
            print("Model class hierarchy:", self.model.__class__.mro())
            raise ValueError(f'model is not of type [SoftPromptModel, HyperbolicSoftPromptModel]')
        for param in self.model.soft_prompt.parameters():
            param.requires_grad = True
            
        self.log_dir, self.model_dir = setup_directories(self.training_config)
        if self.tboard_checkpoint_path is not None:
            self.log_dir = tboard_checkpoint_path
            print(f"Continue writing to {tboard_checkpoint_path}")
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if self.checkpoint_path is not None:
            self.model.soft_prompt = load_soft_prompt(self.model.soft_prompt, checkpoint_path)
            self.optimizer, self.start_epoch = load_optimizer_and_start_epoch(self.optimizer, checkpoint_path)
        
            
        
        
    def log_tensorboard(self, loss, idx, phase, eval_metric = 'loss'):
        if eval_metric not in ['loss', 'em', 'f1']:
            raise ValueError(f'Unsupported eval_metric: {eval_metric}. Supported are [loss, em, f1]')
        else:
            self.writer.add_scalar(f'{phase}/{eval_metric}', loss, idx)
        
    
    def train(self,
            epochs : int):
        """
        Trains soft prompts. Does either the random walk or the parsing step. Concatenating the input with the soft prompt and giving it to the knit5 model.
        """
        
        if self.start_epoch != 0:
            print(f'Starting training from epoch {self.start_epoch}')
        
        for epoch in range(self.start_epoch, epochs):
            progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - {self.method}", file=sys.stdout)
            total_loss = 0
            for batch_idx, batch in enumerate(progress_bar):
                self.optimizer.zero_grad()
                
                input_batch, label_batch = batch
                  
                inputs = self.tokenizer(input_batch, padding=True, truncation=True, return_tensors = 'pt').to(self.device)
                labels = self.tokenizer(label_batch, padding=True, truncation=True, return_tensors = 'pt')['input_ids'].to(self.device)
                
                outputs = self.model(inputs, labels=labels)
                
                loss = outputs.loss
                loss.backward()
		        
                self.optimizer.step()

                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Training - {self.method} - Loss: {loss.item():.4f}")
                self.log_tensorboard(loss.item(), epoch*len(self.train_dataloader) + batch_idx, 'Training')
                
                vram_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # Convert to MB
                vram_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # Convert to MB
                self.writer.add_scalar('VRAM/Training/Allocated', vram_allocated, epoch*len(self.train_dataloader) + batch_idx)
                self.writer.add_scalar('VRAM/Training/Reserved', vram_reserved, epoch*len(self.train_dataloader) + batch_idx)
            avg_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate(epoch=epoch):
                    break #Early Stopping
            
    def evaluate(self,
                epoch : int):
        self.model.eval()
        total_loss = 0
        total_em = 0
        total_f1 = 0
        progress_bar = tqdm(self.val_dataloader, leave=True, desc=f"Epoch {epoch} - Validation - {self.method}", file=sys.stdout)
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                input_batch, label_batch = batch
                inputs = self.tokenizer(input_batch, padding=True, truncation=True, return_tensors = 'pt').to(self.device)
                labels = self.tokenizer(label_batch, padding=True, truncation=True, return_tensors = 'pt')['input_ids'].to(self.device)
                
                
                outputs = self.model(inputs, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                decoded_predictions = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
                _f1_score = sum([f1_score(pred, truth)[0] for pred, truth, in zip(decoded_predictions, label_batch)])
                em_score = sum([1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, label_batch)])
                
                
                total_em += em_score
                total_f1 += _f1_score
                progress_bar.set_description(f"Epoch {epoch} - Validation - {self.method} - Loss: {loss.item():.4f}")
                if batch_idx <= 5: 
                    self.writer.add_text(f'Validation/Prediction_vs_Label_{epoch}', 
                                     f'Prediction: {decoded_predictions[0]}\nLabel: {label_batch[0]}', epoch)
                
            avg_loss = total_loss / len(self.val_dataloader)
            avg_em_perc = total_em / len(self.val_dataloader.dataset)
            avg_f1_perc = total_f1 / len(self.val_dataloader.dataset)
            self.log_tensorboard(avg_loss, epoch, 'Validation')
            self.log_tensorboard(avg_em_perc, epoch, 'Validation', eval_metric='em')
            self.log_tensorboard(avg_f1_perc, epoch, 'Validation', eval_metric='f1')
            print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")
        soft_prompt_path = f"{self.model_dir}/{self.model.model_name}_epoch_{epoch}_val_loss_{avg_loss:.4f}.pth"
        
        
        if avg_loss < self.best_loss:
            if self.best_model_path:
                os.remove(self.best_model_path)
            self.best_loss = avg_loss
            self.early_stop_counter = 0
            self.best_model_path = soft_prompt_path
            torch.save({
                'soft_prompt_state_dict': self.model.soft_prompt.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch}, soft_prompt_path)
        else:
            self.early_stop_counter += 1
            print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
            if self.early_stop_counter >= self.patience:
                print("Early stopping trigered. Stopping training")
                return True
        return False

   
    
            
            
            
            
            
    
    
    
    
    
