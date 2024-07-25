import torch
import torch.nn as nn
from src.util import project_to_hyperbolic, get_top_token_embeddings
from transformers import T5Model
from src.train import load_model_checkpoint
from transformers import AutoTokenizer
from src.config import Config

class HyperbolicSoftPromptModel(nn.Module):
    def __init__(self,
                 knit5 : T5Model,
                 knit5_checkpoint_path : str,
                 model_name : str,
                 soft_prompt : nn.Embedding = None,
                 curvature = 1):
        super().__init__()
        self.knit5 = knit5
        self.curvature = curvature
        self.model_name = model_name
        self.config = Config()
        self.knit5 = load_model_checkpoint(self.knit5, knit5_checkpoint_path)
        

        if soft_prompt is None:
            self.soft_prompt = self.init_soft_prompt()
        else:
            self.soft_prompt = soft_prompt
        
    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.t5_model.model_name)
        #HP Soft Prompt will be tuned
        pp_length = self.config.parse_then_hop_training.prompt_length

        pp_embedding_size = self.knit5.config.hidden_size 
        pp_embeddings = nn.Embedding(pp_length, pp_embedding_size)
        
        #dont use random use top 100 most common tokens of tokenizer.getvocab
        top_100_token_embeddings = get_top_token_embeddings(self.knit5, tokenizer, 100)
        pp_embeddings.weight.data[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings
        print(f"Initializing Soft Prompt with top 100 tokens from pretraining corpus")
        return pp_embeddings   
        
    def exp_map(self, x):
        curvature_tensor = torch.tensor(self.curvature, device=x.device)
        norm = torch.norm(x, dim=-1, keepdim=True)
        return torch.tanh(norm * torch.sqrt(curvature_tensor)) * x / (norm * torch.sqrt(curvature_tensor))

    
    
    def forward(self, inputs, labels):
        soft_prompt_input = self.soft_prompt.weight.unsqueeze(0).expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(inputs['input_ids'])  # Convert input IDs to embeddings
        
        
        #Map input embeddings to hyperbolic space
        hyperbolic_input_embeddings = self.exp_map(input_embeddings)
        
        hyperbolic_soft_prompt_embeddings = self.exp_map(soft_prompt_input)
        

        concatenated_embeddings = torch.cat([hyperbolic_soft_prompt_embeddings, hyperbolic_input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((inputs['attention_mask'].size(0), hyperbolic_soft_prompt_embeddings.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, inputs['attention_mask']), dim=1)

        outputs = self.knit5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
        return outputs
    
    