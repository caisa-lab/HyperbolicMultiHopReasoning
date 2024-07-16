import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from datetime import datetime
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import Adafactor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import Config
from eval import exact_match_score, f1_score

"""Triggering Multi-Hop Reasoning for Question Answering
in Language Models using Soft Prompts and Random Walks: https://arxiv.org/pdf/2306.04009
"""
class Trainer:
    def __init__(self,
                 model : nn.Module,
                 tokenizer : AutoTokenizer,
                 list_train_dataloader: list,
                 val_dataloader : DataLoader,
                 config : Config,
                 device : str ='cpu',
                 checkpoint_path : str = None,
                 validation_step : int = 1,
                 method : str = 'singe_hop_training'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = config.t5_model.tokenizer_max_length
        self.train_dataloader = None
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
        
        self.patience = 5
        self.best_loss = float('inf')
        self.early_stop_counter=0
        self.best_model_path = None
        
        self.supported_methods = ['single_hop_training', 'one_hop_wiki_training', 'random_walk_training', 'parse_then_hop_training']
        
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        else:
            self.method = method
        
        self.setup_directories(method)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        if self.checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
    def get_optimizer(self,
                      parameters,
                      config : Config,
                      method : str ='single_hop_training'):
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        
        
        if config.single_hop_training.optimizer == 'Adam':
            return optim.Adam(parameters, lr=config.single_hop_training.learning_rate)
        elif config.single_hop_training.optimizer == 'AdamW':
            return optim.AdamW(parameters, lr=config.single_hop_training.learning_rate, weight_decay=config.single_hop_training.optimizer_param)
        elif config.single_hop_training.optimizer == 'AdaFactor':
            return Adafactor(parameters, lr=config.single_hop_training.learning_rate, weight_decay=config.single_hop_training.optimizer_param, relative_step=False, scale_parameter=False)
        else:
            raise ValueError(f"Unsupported optimizer: {config.single_hop_training.optimizer}")
    
        
    def setup_directories(self, method):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported phase: {method} Supported Phases are: {self.supported_methods}")
        if method == 'single_hop_training':
            self.log_dir = os.path.join(self.config.single_hop_training.log_dir, current_time)
            self.model_dir = os.path.join(self.config.single_hop_training.model_save_path, current_time) 
        elif method == 'one_hop_wiki_training':
            self.log_dir = os.path.join(self.config.one_hop_wiki_training.log_dir, current_time)
            self.model_dir = os.path.join(self.config.one_hop_wiki_training.model_save_path, current_time) 
        elif method == 'random_walk_training':
            self.log_dir = os.path.join(self.config.random_walk_training.log_dir, current_time)
            self.model_dir = os.path.join(self.config.random_walk_training.model_save_path, current_time) 
        elif method == 'parse_then_hop_training':
            self.log_dir = os.path.join(self.config.parse_then_hop_training.log_dir, current_time)
            self.model_dir = os.path.join(self.config.parse_then_hop_training.model_save_path, current_time)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)  
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
        
        
    def log_tensorboard(self, loss, idx, phase, method, eval_metric = 'loss'):
        if eval_metric not in ['loss', 'em', 'f1']:
            raise ValueError(f'Unsupported eval_metric: {eval_metric}. Supported are [loss, em, f1]')
        else:
            self.writer.add_scalar(f'{method}/{phase}/{eval_metric}', loss, idx)

    
    def train_single_hop(self,
                         optimizer : optim.Optimizer,
                         epochs : int):
        """Trains the Knowledge Integration. The model becomes e1 ; r1 as an input and tries to predict e2.
            For the Dataset we will use the SingleHopDataset which gets a list of [e1, r1, e2] an turns it into input_ids and attention mask for "e1 ; r1" and the label input_ids for "e2"
            In this part the Model will be Finetuned.
        Tracks only the Loss for now.
    
        """
        if self.train_dataloader is None:
            print('Training with C4')
        else:
            print('Training without C4')
        self.model.to(self.device)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            if self.train_dataloader is None:
                single_hop_iter = iter(self.single_hop_train_dataloader)
                c4_iter = iter(self.c4_train_dataloader)
                progress_bar = tqdm(range(2*min(len(self.single_hop_train_dataloader), len(self.c4_train_dataloader))), leave=True, desc=f"Epoch {epoch} - Training - Knowledge Integration", file=sys.stdout)
            else:
                train_iter = iter(self.train_dataloader)
                progress_bar = tqdm(range(len(self.train_dataloader)), leave=True, desc=f"Epoch {epoch} - Training - Knowledge Integration", file=sys.stdout)
            for batch_idx in progress_bar:
                if self.train_dataloader is None:
                    #Train in 50:50 Mixture
                    if batch_idx % 2 == 0:
                        #print(f"Single Hop Batch")
                        try:
                            batch = next(single_hop_iter)
                        except StopIteration:
                            single_hop_iter = iter(self.single_hop_train_dataloader)
                            batch = next(single_hop_iter)
                    else:
                        #print(f"C4 Batch")
                        try:
                            batch = next(c4_iter)
                        except StopIteration:
                            c4_iter = iter(self.c4_train_dataloader)
                            batch = next(c4_iter)
                else:
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_dataloader)
                        batch = next(train_iter)
                        
                input_str, label = batch[0], batch[1]
                
                tokenized_inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt').to(self.device)
                tokenized_labels = self.tokenizer(label, padding=True, truncation=True, return_tensors='pt').to(self.device)
                input_ids = tokenized_inputs['input_ids']
                attention_mask = tokenized_inputs['attention_mask']
                labels = tokenized_labels['input_ids']
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss   

                
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch} - Training - Knowledge Integration - AvgLoss: {loss.item():.4f}")
                
                
                
                if self.train_dataloader is None:
                    len_trainloader = min(len(self.single_hop_train_dataloader), len(self.c4_train_dataloader))
                else:
                    len_trainloader = len(self.train_dataloader)
                self.log_tensorboard(loss.item(), epoch*len_trainloader + batch_idx, 'Training', 'Knowledge_Integration')
                
                vram_allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # Convert to MB
                vram_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # Convert to MB
                self.writer.add_scalar('Knowledge_Integration/Training/VRAM/Allocated', vram_allocated, epoch*len_trainloader + batch_idx)
                self.writer.add_scalar('Knowledge_Integration/Training/VRAM/Reserved', vram_reserved, epoch*len_trainloader + batch_idx)

            avg_loss = total_loss / len_trainloader
            print(f"Epoch {epoch} - Training - AVGLoss: {avg_loss:.4f}")
            
            if (self.val_dataloader is not None) and (epoch % self.validation_step == 0):
                if self.evaluate_single_hop(epoch=epoch):
                    break #Early Stopping
            
    def evaluate_single_hop(self,
                            epoch : int):
        self.model.eval()
        total_loss = 0
        total_em = 0
        total_f1 = 0
        progress_bar = tqdm(self.val_dataloader, leave=True, desc=f"Epoch {epoch} - Validation - Knowledge Integration", file = sys.stdout)
        with torch.no_grad():
            for batch_idx, (input_str, label) in enumerate(progress_bar):
                input_ids = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                attention_mask = self.tokenizer(input_str, padding=True, truncation=True, return_tensors='pt')['attention_mask'].to(self.device)
                labels = self.tokenizer(label, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(self.device)
                
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                decoded_predictions = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
                #print(f'Prediction: {decoded_predictions}')
                #print(f'Labels: {label}')
                _f1_score = sum([f1_score(pred, truth)[0] for pred, truth, in zip(decoded_predictions, label)])
                em_score = sum([1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, label)])
                
                #print(f'Shapes:')
                #print(f'Prediction Shape: {predictions.shape}')
                #print(f'Labels Shape: {labels.shape}')
                
                #print(f'Prediction: {predictions}')
                #print(f'Labels: {labels}')
                
                total_em += em_score
                total_f1 += _f1_score
                progress_bar.set_description(f"Epoch {epoch} - Validation - Knowledge Integration - Loss: {loss.item():.4f}")
                if batch_idx <= 5: 
                    self.writer.add_text(f'Validation/Prediction_vs_Label_{epoch}', 
                                     f'Prediction: {decoded_predictions[0]}\nLabel: {label[0]}', epoch)
            avg_loss = total_loss / len(self.val_dataloader)
            avg_em_perc = total_em / len(self.val_dataloader.dataset)
            avg_f1_perc = total_f1 / len(self.val_dataloader.dataset)
            self.log_tensorboard(avg_loss, epoch, 'Validation', 'Knowledge_Integration')
            self.log_tensorboard(avg_em_perc, epoch, 'Validation', 'Knowledge_Integration', eval_metric='em')
            self.log_tensorboard(avg_f1_perc, epoch, 'Validation', 'Knowledge_Integration', eval_metric='f1')
        print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")
        
        
        
        model_path = f"{self.model_dir}/model_epoch_{epoch}_val_loss_{avg_loss:.4f}.pth"
        
        if avg_loss < self.best_loss:
            if self.best_model_path:
                os.remove(self.best_model_path)
            self.best_loss = avg_loss
            self.early_stop_counter = 0
            self.best_model_path = model_path
            torch.save(self.model.state_dict(), model_path)
        else:
            self.early_stop_counter += 1
            print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
            if self.early_stop_counter >= self.patience:
                print("Early stopping trigered. Stopping training")
                return True
        return False
        
    
    def train_random_walk(self,
                          hopping_soft_prompt : nn.Embedding,
                          optimizer : optim.Optimizer,
                          epochs : int):
        """Trains the Random Walk Part. Random Walk takes in a sequence "e1 ; r1 ; r2" and should predict "e1 ; r1 ; e2 ; r2 ; e3"
        In this part the Soft Prompt HP will be Finetuned and the Model will be frozen.
        
        For the Dataset we will use the RandomWalk Dataset which contains gives as a complete and an incomplete path
        """
        
        #Freeze Model in Random Walk Training
        for param in self.model.parameters():
            param.required_grad = False
        
        for param in hopping_soft_prompt.parameters():
            param.requires_grad = True
        
        for epoch in range(epochs):
            progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - Random Walk Training", file=sys.stdout)
            total_loss = 0
            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()
                
                
                incomplete_sequence, complete_sequence = batch
                
                #print(f'Incomplete Sequence: {incomplete_sequence[0]}')
                #print(f'Complete Sequence: {complete_sequence[0]}')
                
                inputs = self.tokenizer(incomplete_sequence, padding=True, truncation=True, return_tensors = 'pt').to(self.device)
                labels = self.tokenizer(complete_sequence, padding=True, truncation=True, return_tensors = 'pt')['input_ids'].to(self.device)
                
                #print(f'Labels Shape: {labels.shape}')
                
                #Generate HP Embedding and concatenate with input IDs
                hp_input = hopping_soft_prompt.weight.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
                
                #print(f'Soft Prompt Shape : {hp_input.shape}')
                
                input_embeddings = self.model.shared(inputs['input_ids'])  # Convert input IDs to embeddings

                #print(f'Input Embeds shape: {input_embeddings.shape}')

                concatenated_embeddings = torch.cat([hp_input, input_embeddings], dim=1)

                #print(f'Concat Embeds Shape: {concatenated_embeddings.shape}')
                
                #Adjust attention mask (take all of the soft prompt tokens should be attented)
                hp_attention_mask = torch.ones((inputs['attention_mask'].size(0), hp_input.size(1)), device=self.device)
                
                #print(f'Soft Prompt Attention mask: {hp_attention_mask.shape}')
                
                concatenated_attention_mask = torch.cat((hp_attention_mask, inputs['attention_mask']), dim=1)
                             
                #print(f'Concat Attention mask: {concatenated_attention_mask.shape}')
                
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
            
    def evaluate_random_walk(self,
                             hopping_soft_prompt : nn.Parameter,
                             epoch : int):
        self.model.eval()
        total_loss = 0
        total_em = 0
        total_f1 = 0
        progress_bar = tqdm(self.val_dataloader, leave=True, desc=f"Epoch {epoch} - Validation - Random Walk Training", file=sys.stdout)
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                incomplete_sequence, complete_sequence = batch
                inputs = self.tokenizer(incomplete_sequence, padding=True, truncation=True, return_tensors = 'pt').to(self.device)
                labels = self.tokenizer(complete_sequence, padding=True, truncation=True, return_tensors = 'pt')['input_ids'].to(self.device)
                
                #Generate HP Embedding and concatenate with input IDs
                hp_input = hopping_soft_prompt.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
                input_embeddings = self.model.shared(inputs['input_ids'])  # Convert input IDs to embeddings

                concatenated_embeddings = torch.cat([hp_input, input_embeddings], dim=1)
                
                #TODO EM score and F1 Score
                #Adjust attention mask (take all of the soft prompt tokens should be attented)
                hp_attention_mask = torch.ones((inputs['attention_mask'].size(0), hp_input.size(1)), device=self.device)
                concatenated_attention_mask = torch.cat((hp_attention_mask, inputs['attention_mask']), dim=1)

                outputs = self.model(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                decoded_predictions = [self.tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
                #print(f'Prediction: {decoded_predictions}')
                #print(f'Labels: {label}')
                _f1_score = sum([f1_score(pred, truth)[0] for pred, truth, in zip(decoded_predictions, complete_sequence)])
                em_score = sum([1 if exact_match_score(pred, truth) else 0 for pred, truth in zip(decoded_predictions, complete_sequence)])
                
                #print(f'Shapes:')
                #print(f'Prediction Shape: {predictions.shape}')
                #print(f'Labels Shape: {labels.shape}')
                
                #print(f'Prediction: {predictions}')
                #print(f'Labels: {labels}')
                
                total_em += em_score
                total_f1 += _f1_score
                progress_bar.set_description(f"Epoch {epoch} - Validation - Knowledge Integration - Loss: {loss.item():.4f}")
                if batch_idx <= 5: 
                    self.writer.add_text(f'Validation/Prediction_vs_Label_{epoch}', 
                                     f'Prediction: {decoded_predictions[0]}\nLabel: {complete_sequence[0]}', epoch)
                
                progress_bar.set_description(f"Epoch {epoch} - Validation - Random Walk Training - Loss: {loss.item():.4f}")
            avg_loss = total_loss / len(self.val_dataloader)
            avg_em_perc = total_em / len(self.val_dataloader.dataset)
            avg_f1_perc = total_f1 / len(self.val_dataloader.dataset)
            self.log_tensorboard(avg_loss, epoch, 'Validation', 'Random_Walk_Training')
            self.log_tensorboard(avg_em_perc, epoch, 'Validation', 'Random_Walk_Training', eval_metric='em')
            self.log_tensorboard(avg_f1_perc, epoch, 'Validation', 'Random_Walk_Training', eval_metric='f1')
            print(f"Epoch {epoch} - Validation - AvgLoss: {avg_loss:.4f} | AvgEM: {avg_em_perc:.4f} | AvgF1: {avg_f1_perc:.4f}")
        model_path = f"{self.model_dir}/model_epoch_{epoch}_val_loss_{avg_loss:.4f}.pth"
        
        
        if avg_loss < self.best_loss:
            if self.best_model_path:
                os.remove(self.best_model_path)
            self.best_loss = avg_loss
            self.early_stop_counter = 0
            self.best_model_path = model_path
            torch.save(self.model.state_dict(), model_path)
        else:
            self.early_stop_counter += 1
            print(f"Early stopping counter: {self.early_stop_counter} / {self.patience}")
            if self.early_stop_counter >= self.patience:
                print("Early stopping trigered. Stopping training")
                return True
        return False

def parse_then_hop(self, hp_embeddings : nn.Parameter,
                   pp_embeddings : nn.Parameter,
                   optimizer : optim.Optimizer,
                   epochs : int):
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
        progress_bar = tqdm(self.train_dataloader, leave=True, desc=f"Epoch {epoch} - Training - Random Walk Training", file=sys.stdout)
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

    
            
            
            
            
            
    
    
    
    
    
