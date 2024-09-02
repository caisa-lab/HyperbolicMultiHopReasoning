from torch import optim
import os
import torch
from datetime import datetime
from transformers import Adafactor
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from geoopt.optim import RiemannianAdam

# Get the current directory (train)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (one level up from the current directory)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import the config module
from config import BaseTrainingConfig

train_dir = current_dir
sys.path.append(train_dir)


def get_optimizer(parameters, trainer_config : BaseTrainingConfig):
    print(f'Using {trainer_config.optimizer} with learning rate {trainer_config.learning_rate}')
    if trainer_config.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr= trainer_config.learning_rate)
    elif  trainer_config.optimizer == 'AdamW':
        optimizer = optim.AdamW(parameters, lr= trainer_config.learning_rate, weight_decay= trainer_config.optimizer_param)
    elif  trainer_config.optimizer == 'AdaFactor':
        optimizer = Adafactor(parameters, lr= trainer_config.learning_rate, weight_decay= trainer_config.optimizer_param, relative_step=False, scale_parameter=False)
    elif  trainer_config.optimizer == 'Hyperbolic':
        optimizer = RiemannianAdam(parameters, lr= trainer_config.learning_rate, weight_decay= trainer_config.optimizer_param)
    else:
        raise ValueError(f"Unsupported optimizer: {trainer_config.optimizer}")
    return optimizer

def setup_directories(trainer_config : BaseTrainingConfig):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    learning_rate = trainer_config.learning_rate
    curvature = trainer_config.curvature
    optimizer = trainer_config.optimizer
    final_string = f"{current_time}_{optimizer}_{learning_rate}_{curvature}"
    log_dir = os.path.join(trainer_config.log_dir, final_string)
    model_dir = os.path.join(trainer_config.model_save_path, final_string) 
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)  
    return log_dir, model_dir

def load_model_checkpoint(model : nn.Module,
                          checkpoint_path,
                          device = 'cuda' if torch.cuda.is_available() else 'cpu',
                          with_model_state_dict = True
                          ):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if with_model_state_dict:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model
                    
def log_tensorboard(writer : SummaryWriter,
                    value : float,
                    idx,
                    phase,
                    eval_metric = 'loss'):
    if eval_metric not in ['loss', 'em', 'f1']:
        raise ValueError(f'Unsupported eval_metric: {eval_metric}. Supported are [loss, em, f1]')
    else:
        writer.add_scalar(f'{phase}/{eval_metric}', value, idx)
        
def load_soft_prompt(soft_prompt : nn.Embedding,
                    soft_prompt_path : str,
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(soft_prompt_path, map_location=device) 
    if soft_prompt is not None:  
        soft_prompt.load_state_dict(checkpoint['soft_prompt_state_dict'])
        soft_prompt.to(device)
        print(f'Loading Soft Prompt Checkpoint from {soft_prompt_path}')
    return soft_prompt

def load_optimizer_and_start_epoch(optimizer : optim.Optimizer,
                                   checkpoint_path):
    checkpoint = torch.load(checkpoint_path) 
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Loading Optimizer Checkpoint from {checkpoint_path}')
        start_epoch = checkpoint['epoch']
    
        return optimizer, start_epoch
    else:
        return optimizer, 0
    
    
def geodesic_distance(u, v, c=1.0):
    norm_u = torch.norm(u, p=2, dim=-1)
    norm_v = torch.norm(v, p=2, dim=-1)
    numerator = torch.norm(u - v, p=2, dim=-1)**2
    denominator = (1 - c * norm_u**2) * (1 - c * norm_v**2)
    distance = torch.acosh(1 + 2 * c * numerator / denominator)
    return distance

def geodesic_regularization(soft_prompt_input, min_distance=1.0, c=1.0):
    loss = 0
    num_components = soft_prompt_input.size(1)
    for i in range(num_components):
        for j in range(i+1, num_components):
            distance = geodesic_distance(soft_prompt_input[:, i], soft_prompt_input[:, j], c)
            loss += torch.clamp(min_distance - distance, min=0).mean()
    return loss
    
    