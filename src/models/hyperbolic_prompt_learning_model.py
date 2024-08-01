from typing import Union
import torch
import torch.nn as nn
from utils.util import get_top_token_embeddings, expmap0, logmap0
from utils.trainer_utils import load_model_checkpoint
from .hyperbolic_t5_model import HyperbolicT5Model
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Model
from src.config import Config
from .hyperbolic_model_utils import HyperbolicSoftPrompts

#TODO Implement with Hyperbolic Embeddings, Try like this 

"""
import torch
import geoopt
import geoopt.manifolds.poincare as poincare
from geoopt.optim import RiemannianSGD

# Initialize the Poincaré ball manifold
manifold = poincare.PoincareBall()

# Define hyperbolic soft prompts
soft_prompt_dim = 50  # Example dimension of the soft prompt
n_prompts = 10  # Number of soft prompts
hyperbolic_soft_prompt = torch.nn.Parameter(manifold.random_normal(size=(n_prompts, soft_prompt_dim), mean=0, std=1e-3))

# Move the soft prompt to the appropriate device
hyperbolic_soft_prompt = hyperbolic_soft_prompt.to(device)

# Define an optimizer for the hyperbolic soft prompts
optimizer = RiemannianSGD([hyperbolic_soft_prompt], lr=0.01)

# Assuming 'inputs' and 'labels' are your training data and the model is already loaded and frozen
model.eval()  # Freeze the model parameters

for input_str, label in training_data:
    optimizer.zero_grad()

    # Encode the input as usual
    input_ids = tokenizer(input_str, return_tensors='pt')['input_ids'].to(device)
    input_embeddings = model.shared(input_ids)

    # Concatenate the hyperbolic soft prompts with the input embeddings
    concatenated_embeddings = torch.cat([hyperbolic_soft_prompt.expand(input_embeddings.size(0), -1, -1), input_embeddings], dim=1)

    # Perform the forward pass through the model (while keeping the model frozen)
    outputs = model(inputs_embeds=concatenated_embeddings)
    
    # Compute the loss (use a loss function suitable for your task, assuming it's compatible with the model's frozen state)
    loss = loss_function(outputs, labels)

    # Backpropagation and optimization in hyperbolic space
    loss.backward()
    optimizer.step()

    print(f'Loss: {loss.item()}')
"""
class HyperbolicSoftPromptModel(nn.Module):
    def __init__(self,
                 hyperbolic_knit5 : Union[HyperbolicT5Model, T5ForConditionalGeneration],
                 hyperbolic_knit5_checkpoint_path : str,
                 model_name : str,
                 hyperbolic_soft_prompt : HyperbolicSoftPrompts = None,
                 with_model_state_dict = True,
                 curvature : float = 1.0):
        super(HyperbolicSoftPromptModel, self).__init__()
        if isinstance(hyperbolic_knit5, HyperbolicT5Model):
            self.knit5 : T5Model = hyperbolic_knit5.t5
            print("HyperbolicT5")
        elif isinstance(hyperbolic_knit5, T5ForConditionalGeneration):
            self.knit5 : T5Model = hyperbolic_knit5
            print("T5")
        print(f"{hyperbolic_knit5.__module__ = }")
        print("Model class hierarchy:", hyperbolic_knit5.__class__.mro())
        self.curvature = curvature
        self.model_name = model_name
        self.config = Config()
        if hyperbolic_knit5_checkpoint_path is not None:
            self.hyperbolic_knit5 = load_model_checkpoint(self.hyperbolic_knit5, hyperbolic_knit5_checkpoint_path, with_model_state_dict=with_model_state_dict)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.hyperbolic_soft_prompt = hyperbolic_soft_prompt if hyperbolic_soft_prompt else self.init_soft_prompt()
        
    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.t5_model.model_name)
        #HP Soft Prompt will be tuned
        soft_prompt_length = self.config.parse_then_hop_training.prompt_length

        soft_prompt_embedding_size = self.knit5.config.hidden_size
        soft_prompt_embeddings = HyperbolicSoftPrompts(soft_prompt_length, soft_prompt_embedding_size) 
        
        #dont use random use top 100 most common tokens of tokenizer.getvocab
        top_100_token_embeddings = get_top_token_embeddings(self.knit5, tokenizer, 100)
        with torch.no_grad():
            soft_prompt_embeddings[:top_100_token_embeddings.size(0), :] = expmap0(top_100_token_embeddings, c=self.curvature)
        print(f"Initializing Soft Prompt with top 100 tokens from pretraining corpus")
        return soft_prompt_embeddings   
        
    
    
    def forward(self, inputs, labels):
        hyperbolic_soft_prompt_input = self.hyperbolic_soft_prompt.soft_prompts.expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(inputs['input_ids'])  # Convert input IDs to embeddings
        
        soft_prompt_embeddings = logmap0(hyperbolic_soft_prompt_input, self.curvature)

        concatenated_embeddings = torch.cat([soft_prompt_embeddings, input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((inputs['attention_mask'].size(0), soft_prompt_embeddings.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, inputs['attention_mask']), dim=1)
        outputs = self.knit5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
        return outputs
    
    def generate(self, inputs, max_length=50, num_beams = 5, early_stopping=True):
        soft_prompt_input = self.hyperbolic_soft_prompt.soft_prompts.weight.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(inputs['input_ids'])  # Convert input IDs to embeddings

        hyperbolic_input_embeddings = expmap0(input_embeddings, self.curvature)
        
        hyperbolic_soft_prompt_embeddings = expmap0(soft_prompt_input, self.curvature)
        

        concatenated_embeddings = torch.cat([hyperbolic_soft_prompt_embeddings, hyperbolic_input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        pp_attention_mask = torch.ones((inputs['attention_mask'].size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((pp_attention_mask, inputs['attention_mask']), dim=1)

        outputs = self.knit5.generate(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
        return outputs
    
    