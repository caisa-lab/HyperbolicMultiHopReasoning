import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
import numpy as np

"""Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""

#TODO: Maybe do mixed precision to save memory. As i tried it the loss was NAN
class Trainer:
    def __init__(self, model, tokenizer, train_dataloader, val_dataloader, config, device='cpu', checkpoint_path = None, validation_step=1):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.validation_step = validation_step
        
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
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        
    def log_tensorboard(self, loss, idx, phase, method):
        self.writer.add_scalar(f'{method}/{phase}/loss', loss, idx)
        
    def train_single_hop(self, optimizer, epochs):
        """Trains the Knowledge Integration. The model becomes e1 ; r1 as an input and tries to predict e2.
            For the Dataset we will use the SingleHopDataset which gets a list of [e1, r1, e2] an turns it into input_ids and attention mask for "e1 ; r1" and the label input_ids for "e2"
            In this part the Model will be Finetuned.
        Tracks only the Loss for now.
    
        """
        self.model.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - Knowledge Integration")
            total_loss = 0
            for batch_idx, batch in enumerate(progress_bar):
                input_str, label = batch[0], batch[1]
                input_ids = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                attention_mask = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['attention_mask'].to(self.device)
                labels = self.tokenizer(label, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                
                optimizer.zero_grad()
                #with autocast():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                    
                #if torch.isnan(loss):
                 #   print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                 #   continue
                loss.backward()
                optimizer.step()
                #self.grad_scaler.scale(loss).backward()
                
                #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                #self.grad_scaler.step(optimizer)
                #self.grad_scaler.update()
                
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Training - Knowledge Integration - Loss: {loss.item():.4f}")
                

            avg_loss = total_loss / len(self.train_dataloader)
            self.log_tensorboard(avg_loss, epoch*len(self.train_dataloader) + batch_idx, 'Training', 'Knowledge_Integration')
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                self.evaluate_single_hop(epoch)
            
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
        model_path = f"{self.model_dir}/model_epoch_{epoch+1}_val_loss_{avg_loss:.4f}.pth"
        #TODO:SAVE MODEL
        
    
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
                
                
                optimizer.zero_grad()
                
                
                outputs = self.model(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Training - Random Walk Training - Loss: {loss.item():.4f}")
                
                
            avg_loss = total_loss / len(self.train_dataloader)
            self.log_tensorboard(avg_loss, epoch*len(self.train_dataloader) + batch_idx, 'Training', 'Knowledge_Integration')
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                self.evaluate_single_hop(epoch, hopping_soft_prompt)
            
    def evaluate_single_hop(self, hopping_soft_prompt, epoch):
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
            self.log_tensorboard(avg_loss, epoch * len(self.val_dataloader) + batch_idx, 'Validation', 'Knowledge_Integration')
        model_path = f"{self.model_dir}/model_epoch_{epoch+1}_val_loss_{avg_loss:.4f}.pth"
        #TODO:SAVE MODEL

    #TODO: How to do both steps parsing and hopping in one training loop or should it be in one training loop?
    def parse_then_hop(model, hp_embeddings, pp_length, pp_embedding_size, tokenizer, optimizer, dataloader, epochs):
        """
        Finetunes a Soft Prompt Parsing Prompts (PP) everything else is kept frozen. Takes in a Question and the PP gives it to the model which should output an incomplete path.
        This incomplete path should be concatonated with the Hopping Prompt (HP) and then fed into the model which should output the complete path.
        """
        
        #Freeze model for PaTH Training
        for param in model.parameters():
            param.required_grad = False
        
        #Freeze HP For Path Training
        hp_embeddings.requires_grad = False
        
        #PP Soft Prompt will be tuned
        pp_embeddings = nn.Parameter(torch.randn(pp_length, pp_embedding_size))
        pp_embeddings.requires_grad = True
        
        for epoch in range(epochs):
            total_parsing_loss = 0
            total_hopping_loss = 0
            for question, incomplete_sequence, complete_sequence in dataloader:
                
                #Parsing Step
                
                inputs = tokenizer(question, padding=True, truncation=True, return_tensors = 'pt')
                labels = tokenizer(incomplete_sequence, padding=True, truncation=True, return_tensors = 'pt')['input_ids']
                
                #Generate HP Embedding and concatenate with input IDs
                pp_input = pp_embeddings.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1)
                concatenated_input = torch.cat([pp_input, inputs['input_ids']], dim=1)
                
                #Adjust attention mask (take all of the soft prompt tokens should be attented)
                pp_attention_mask = torch.ones((inputs['attention_mask'].size(0), pp_length), device=inputs['attention_mask'].device)
                concatenated_attention_mask = torch.cat((pp_attention_mask, inputs['attention_mask']), dim=1)
                
                
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs = model(input_ids=concatenated_input, attention_mask=concatenated_attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_parsing_loss += loss.item()
                
                
                
                #Hopping Step
                incomplete_paths = outputs.logits.argmax(dim=-1)
                
                labels = tokenizer(complete_sequence, padding=True, truncation=True, return_tensors = 'pt')['input_ids']
                
                hp_input = hp_embeddings.unsqueeze(0).expand(incomplete_paths.size(0), -1, -1)
                concatenated_input_hp = torch.cat([hp_input, incomplete_paths], dim=1)
                
                # Adjust attention mask for HP
                hp_attention_mask = torch.ones((incomplete_paths.size(0), pp_length), device=incomplete_paths.device)
                concatenated_attention_mask_hp = torch.cat((hp_attention_mask, torch.ones_like(incomplete_paths)), dim=1)
                
                optimizer.zero_grad()
                outputs = model(input_ids=concatenated_input_hp, attention_mask=concatenated_attention_mask_hp, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_hopping_loss += loss.item()
                        
            avg_parsing_loss = total_parsing_loss / len(dataloader)
            avg_hopping_loss = total_hopping_loss / len(dataloader)
            print(f"Epoch {epoch + 1}, Parsing: {avg_parsing_loss:.4f}, Hopping: {avg_hopping_loss:.4f}")
    
            
            
            
            
            
    
    
    
    
    