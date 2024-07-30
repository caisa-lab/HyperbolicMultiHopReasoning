import torch
import torch.nn as nn
from utils.util import get_top_token_embeddings, expmap0
from utils.trainer_utils import load_model_checkpoint
from .hyperbolic_t5_model import HyperbolicT5Model
from transformers import AutoTokenizer
from src.config import Config

class HyperbolicSoftPromptModel(nn.Module):
    def __init__(self,
                 hyperbolic_knit5 : HyperbolicT5Model,
                 hyperbolic_knit5_checkpoint_path : str,
                 model_name : str,
                 soft_prompt : nn.Embedding = None,
                 with_model_state_dict = True,
                 curvature : float = 1.0):
        super(HyperbolicSoftPromptModel, self).__init__()
        self.hyperbolic_knit5 = hyperbolic_knit5
        self.curvature = curvature
        self.model_name = model_name
        self.config = Config()
        if hyperbolic_knit5_checkpoint_path is not None:
            self.hyperbolic_knit5 = load_model_checkpoint(self.hyperbolic_knit5, hyperbolic_knit5_checkpoint_path, with_model_state_dict=with_model_state_dict)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.soft_prompt = soft_prompt if soft_prompt else self.init_soft_prompt()
        
    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.t5_model.model_name)
        #HP Soft Prompt will be tuned
        pp_length = self.config.parse_then_hop_training.prompt_length

        pp_embedding_size = self.hyperbolic_knit5.t5.config.hidden_size
        pp_embeddings = nn.Embedding(pp_length, pp_embedding_size)
        
        #dont use random use top 100 most common tokens of tokenizer.getvocab
        top_100_token_embeddings = get_top_token_embeddings(self.hyperbolic_knit5.t5, tokenizer, 100)
        pp_embeddings.weight.data[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings
        print(f"Initializing Soft Prompt with top 100 tokens from pretraining corpus")
        return pp_embeddings   
        
    
    
    def forward(self, inputs, labels):
        soft_prompt_input = self.soft_prompt.weight.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
        input_embeddings = self.hyperbolic_knit5.t5.shared(inputs['input_ids'])  # Convert input IDs to embeddings
        
        
        #Map input embeddings to hyperbolic space
        hyperbolic_input_embeddings = expmap0(input_embeddings, self.curvature)
        
        hyperbolic_soft_prompt_embeddings = expmap0(soft_prompt_input, self.curvature)
        

        concatenated_embeddings = torch.cat([hyperbolic_soft_prompt_embeddings, hyperbolic_input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((inputs['attention_mask'].size(0), hyperbolic_soft_prompt_embeddings.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, inputs['attention_mask']), dim=1)

        outputs = self.hyperbolic_knit5.t5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
        return outputs
    
    def generate(self, inputs, max_length=50, num_beams = 5, early_stopping=True):
        soft_prompt_input = self.soft_prompt.weight.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
        input_embeddings = self.hyperbolic_knit5.t5.shared(inputs['input_ids'])  # Convert input IDs to embeddings

        hyperbolic_input_embeddings = expmap0(input_embeddings, self.curvature)
        
        hyperbolic_soft_prompt_embeddings = expmap0(soft_prompt_input, self.curvature)
        

        concatenated_embeddings = torch.cat([hyperbolic_soft_prompt_embeddings, hyperbolic_input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        pp_attention_mask = torch.ones((inputs['attention_mask'].size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((pp_attention_mask, inputs['attention_mask']), dim=1)

        outputs = self.hyperbolic_knit5.t5.generate(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
        return outputs
    
    