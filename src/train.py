import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np

"""Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""
#TODO: Mixed Precision Using O1 Apex     https://nvidia.github.io/apex/amp.html#o1-mixed-precision-recommended-for-typical-use
class Trainer:
    def __init__(self, model, tokenizer, list_train_dataloader, val_dataloader, config, device='cpu', checkpoint_path = None, validation_step=1):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = 1024
        
        if len(list_train_dataloader) == 1:
            self.train_dataloader = list_train_dataloader[0]
        elif len(list_train_dataloader) > 1:
            self.single_hop_train_dataloader = list_train_dataloader[0]
            self.c4_train_dataloader = list_train_dataloader[1]
        else:
            raise ValueError('Length of List for Train Dataloaders must be >= 1')
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.validation_step = validation_step
        
        self.patience = 3
        self.best_loss = float('inf')
        self.early_stop_counter=0
        
        self.grad_scaler = GradScaler()
        
        self.setup_directories()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if self.checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
    def get_optimizer(self, parameters, config, phase='single_hop_training'):
        if phase not in ['single_hop_training', 'random_walk_training']:
            raise ValueError(f"Unsupported phase: {phase} Supported Phases are: ['single_hop_training', 'random_walk_training']")
        
        if phase == 'single_hop_training':
            if config.single_hop_training.optimizer == 'Adam':
                return optim.Adam(parameters, lr=config.single_hop_training.learning_rate)
            elif config.single_hop_training.optimizer == 'AdamW':
                return optim.AdamW(parameters, lr=config.single_hop_training.learning_rate, weight_decay=config.single_hop_training.optimizer_param)
            else:
                raise ValueError(f"Unsupported optimizer: {config.single_hop_training.optimizer}")
        elif phase == 'random_walk_training':
            if config.prompt_training.optimizer == 'Adam':
                return optim.Adam(parameters, lr=config.prompt_training.learning_rate)
            elif config.prompt_training.optimizer == 'AdamW':
                return optim.AdamW(parameters, lr=config.prompt_training.learning_rate, weight_decay=config.prompt_training.optimizer_param)
            else:
                raise ValueError(f"Unsupported optimizer: {config.prompt_training.optimizer}")
        
    def setup_directories(self):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.log_dir = os.path.join(self.config.single_hop_training.log_dir, current_time)
        self.model_dir = os.path.join(self.config.single_hop_training.model_save_path, current_time)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)   
        #os.makedirs(f"knowledge_integration/{self.model_dir}", exist_ok=True)   
        #os.makedirs(f"random_walk_training/{self.model_dir}", exist_ok=True)  
        #os.makedirs(f"parse_then_hop/{self.model_dir}", exist_ok=True)    
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        
    def log_tensorboard(self, loss, idx, phase, method):
        self.writer.add_scalar(f'{method}/{phase}/loss', loss, idx)
    
    #TODO: Add C4 50:50 Mixture Training
    def train_single_hop(self, optimizer, epochs):
        """Trains the Knowledge Integration. The model becomes e1 ; r1 as an input and tries to predict e2.
            For the Dataset we will use the SingleHopDataset which gets a list of [e1, r1, e2] an turns it into input_ids and attention mask for "e1 ; r1" and the label input_ids for "e2"
            In this part the Model will be Finetuned.
        Tracks only the Loss for now.
    
        """
        self.model.to(self.device)
        self.model.train()
        c4_iter = iter(self.c4_train_dataloader)
        for epoch in range(epochs):
            single_hop_iter = iter(self.single_hop_train_dataloader)
            #c4_iter = iter(self.c4_train_dataloader)
            progress_bar = tqdm(range(min(len(self.single_hop_train_dataloader), len(self.c4_train_dataloader))), leave=True, desc=f"Epoch {epoch} - Training - Knowledge Integration")
            total_loss = 0
            for batch_idx in progress_bar:
                #Train in 50:50 Mixture
                if batch_idx % 2 == 0:
                    print(f"Single Hop Batch")
                    try:
                        batch = next(single_hop_iter)
                    except StopIteration:
                        single_hop_iter = iter(self.single_hop_train_dataloader)
                        batch = next(single_hop_iter)
                else:
                    print(f"C4 Batch")
                    try:
                        batch = next(c4_iter)
                    except StopIteration:
                        c4_iter = iter(self.c4_train_dataloader)
                        batch = next(c4_iter)
                        
                input_str, label = batch[0], batch[1]
                
                print(input_str)
                
                        
                tokenized_inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt').to(self.device)
                tokenized_labels = self.tokenizer(label, padding=True, truncation=True, return_tensors='pt').to(self.device)
                input_ids = tokenized_inputs['input_ids']
                attention_mask = tokenized_inputs['attention_mask']
                labels = tokenized_labels['input_ids']
                
                optimizer.zero_grad()
                #with autocast(enabled=False):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss   

                
                
                loss.backward()
                
                #loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Training - Knowledge Integration - Loss: {loss.item():.4f}")
                if batch_idx >= 2:
                    return
                
            avg_loss = total_loss / len(self.train_dataloader)
            self.log_tensorboard(avg_loss, epoch*len(self.train_dataloader) + batch_idx, 'Training', 'Knowledge_Integration')
            #print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate_single_hop(epoch=epoch):
                    break #Early Stopping
            
    def evaluate_single_hop(self, epoch):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_dataloader, leave=True, desc=f"Epoch {epoch} - Validation - Knowledge Integration")
        with torch.no_grad():
            for batch_idx, (input_str, label) in enumerate(progress_bar):
                input_ids = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                attention_mask = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['attention_mask'].to(self.device)
                labels = self.tokenizer(label, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Validation - Knowledge Integration - Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(self.val_dataloader)
            self.log_tensorboard(avg_loss, epoch * len(self.val_dataloader) + batch_idx, 'Validation', 'Knowledge_Integration')
        print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f}")
        model_path = f"knowledge_integration/{self.model_dir}/model_epoch_{epoch+1}_val_loss_{avg_loss:.4f}.pth"
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.early_stop_counter = 0
            torch.save(self.model.state_dict(), model_path)
        else:
            self.early_stop_counter += 1
            print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
            if self.early_stop_counter >= self.patience:
                print("Early stopping trigered. Stopping training")
                return True
        return False
        
    
    def train_random_walk(self, hopping_soft_prompt, optimizer, epochs):
        """Trains the Random Walk Part. Random Walk takes in a sequence "e1 ; r1 ; r2" and should predict "e1 ; r1 ; e2 ; r2 ; e3"
        In this part the Soft Prompt HP will be Finetuned and the Model will be frozen.
        
        For the Dataset we will use the RandomWalk Dataset which contains gives as a complete and an incomplete path
        """
        
        #Freeze Model in Random Walk Training
        for param in self.model.parameters():
            param.required_grad = False
        
        for epoch in range(epochs):
            progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - Random Walk Training")
            total_loss = 0
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                
                incomplete_sequence, complete_sequence = batch
                inputs = self.tokenizer(incomplete_sequence, padding=True, truncation=True, return_tensors = 'pt').to(self.device)
                labels = self.tokenizer(complete_sequence, padding=True, truncation=True, return_tensors = 'pt')['input_ids'].to(self.device)
                
                #Generate HP Embedding and concatenate with input IDs
                hp_input = hopping_soft_prompt.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
                input_embeddings = self.model.shared(inputs['input_ids'])  # Convert input IDs to embeddings

                concatenated_embeddings = torch.cat([hp_input, input_embeddings], dim=1)
                
                
                #Adjust attention mask (take all of the soft prompt tokens should be attented)
                hp_attention_mask = torch.ones((inputs['attention_mask'].size(0), hp_input.size(1)), device=self.device)
                concatenated_attention_mask = torch.cat((hp_attention_mask, inputs['attention_mask']), dim=1)
                             
                
                outputs = self.model(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Training - Random Walk Training - Loss: {loss.item():.4f}")
                
                
            avg_loss = total_loss / len(self.train_dataloader)
            self.log_tensorboard(avg_loss, epoch*len(self.train_dataloader) + batch_idx, 'Training', 'Random_Walk_Training')
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                self.evaluate_random_walk(hopping_soft_prompt, epoch)
            
    def evaluate_random_walk(self, hopping_soft_prompt, epoch):
        self.model.eval()
        total_loss = 0
        progress_bar = tqdm(self.val_dataloader, leave=True, desc=f"Epoch {epoch} - Validation - Random Walk Training")
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                incomplete_sequence, complete_sequence = batch
                inputs = self.tokenizer(incomplete_sequence, padding=True, truncation=True, return_tensors = 'pt').to(self.device)
                labels = self.tokenizer(complete_sequence, padding=True, truncation=True, return_tensors = 'pt')['input_ids'].to(self.device)
                
                #Generate HP Embedding and concatenate with input IDs
                hp_input = hopping_soft_prompt.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
                input_embeddings = self.model.shared(inputs['input_ids'])  # Convert input IDs to embeddings

                concatenated_embeddings = torch.cat([hp_input, input_embeddings], dim=1)
                
                
                #Adjust attention mask (take all of the soft prompt tokens should be attented)
                hp_attention_mask = torch.ones((inputs['attention_mask'].size(0), hp_input.size(1)), device=self.device)
                concatenated_attention_mask = torch.cat((hp_attention_mask, inputs['attention_mask']), dim=1)

                outputs = self.model(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Validation - Random Walk Training - Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(self.val_dataloader)
            self.log_tensorboard(avg_loss, epoch * len(self.val_dataloader) + batch_idx, 'Validation', 'Random_Walk_Training')
        model_path = f"{self.model_dir}/random_walk/model_epoch_{epoch+1}_val_loss_{avg_loss:.4f}.pth"
        #TODO: SAVE SOFT PROMPT

#TODO: How to do both steps parsing and hopping in one training loop or should it be in one training loop?
def parse_then_hop(self, hp_embeddings, pp_embeddings, optimizer, epochs):
    """
    Finetunes a Soft Prompt Parsing Prompts (PP) everything else is kept frozen. Takes in a Question and the PP gives it to the model which should output an incomplete path.
    This incomplete path should be concatenated with the Hopping Prompt (HP) and then fed into the model which should output the complete path.
    """
    
    # Freeze model parameters
    for param in self.model.parameters():
        param.requires_grad = False
    
    hp_embeddings.requires_grad = False
    pp_embeddings.requires_grad = True
    
    for epoch in range(epochs):
        progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - Random Walk Training")
        total_loss = 0
        for batch_idx, batch in enumerate(progress_bar):
            
            optimizer.zero_grad()
            
            question, complete_sequence = batch
            inputs = self.tokenizer(question, padding=True, truncation=True, return_tensors='pt').to(self.device)
            labels = self.tokenizer(complete_sequence, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
            
            # Generate PP Embedding and concatenate with input IDs
            pp_input = pp_embeddings.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
            input_embeddings = self.model.shared(inputs['input_ids'])  # Convert input IDs to embeddings
            concat_pp_question_embeddings = torch.cat([pp_input, input_embeddings], dim=1)
            
            # Adjust attention mask (ensure all soft prompt tokens are attended)
            pp_attention_mask = torch.ones((inputs['attention_mask'].size(0), pp_input.size(1)), device=self.device)
            concatenated_pp_question_attention_mask = torch.cat((pp_attention_mask, inputs['attention_mask']), dim=1)
            
            # First pass through the model with PP
            outputs = self.model(inputs_embeds=concat_pp_question_embeddings, attention_mask=concatenated_pp_question_attention_mask)
            
            # Decode incomplete path
            incomplete_path = self.tokenizer.decode(outputs.logits.argmax(dim=-1).squeeze().tolist(), skip_special_tokens=True)
            inputs = self.tokenizer(incomplete_path, padding=True, truncation=True, return_tensors='pt').to(self.device)
            
            # Generate HP Embedding and concatenate with input IDs
            input_embeddings = self.model.shared(inputs['input_ids'])  # Convert input IDs to embeddings
            hp_input = hp_embeddings.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
            concat_hp_incomplete = torch.cat([hp_input, input_embeddings], dim=1)
            
            # Adjust attention mask (ensure all soft prompt tokens are attended)
            hp_attention_mask = torch.ones((inputs['attention_mask'].size(0), hp_input.size(1)), device=self.device)
            concatenated_hp_question_attention_mask = torch.cat((hp_attention_mask, inputs['attention_mask']), dim=1)
            
            # Second pass through the model with HP and labels
            outputs = self.model(inputs_embeds=concat_hp_incomplete, attention_mask=concatenated_hp_question_attention_mask, labels=labels)
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_description(f"Epoch {epoch} - Training - Random Walk Training - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(self.train_dataloader)
        self.log_tensorboard(avg_loss, epoch * len(self.train_dataloader) + batch_idx, 'Training', 'Knowledge_Integration')
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        #if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
        #    self.evaluate_random_walk(hopping_soft_prompt, epoch)

    
            
            
            
            
            
    
    
    
    
    