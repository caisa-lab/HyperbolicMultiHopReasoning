import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Model
from src.utils.util import get_top_token_embeddings
from src.utils.trainer_utils import load_model_checkpoint
from config import Config
from geoopt.manifolds import Lorentz


class RescaledNormalization(nn.Module):
    def __init__(self):
        super(RescaledNormalization, self).__init__()
        # Learnable scaling factor initialized to 1
        self.norm_scale = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, x):
        # Perform L2 normalization
        l2_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # L2 norm along the last dimension
        normalized_x = x / l2_norm  # Normalize the vector

        # Rescale with the learnable norm scaling factor
        scaled_x = self.norm_scale * normalized_x

        return scaled_x
class SoftPromptModel(nn.Module):
    def __init__(self,
                 knit5 : T5Model,
                 knit5_checkpoint_path : str,
                 model_name : str = '',
                 curvature = 1.0, 
                 soft_prompt = None,
                 with_model_state_dict = True):
        super(SoftPromptModel, self).__init__()
        self.knit5 = knit5
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if knit5_checkpoint_path is not None:
            load_model_checkpoint(knit5, knit5_checkpoint_path, with_model_state_dict=with_model_state_dict)
        
        self.config = Config()
        
        self.curvature = curvature
        if soft_prompt is None:
            self.soft_prompt = self.init_soft_prompt()
        else:
            self.soft_prompt = soft_prompt
            
        
        for param in self.knit5.parameters():
            param.requires_grad = False
        
        self.soft_prompt.requires_grad = True
        self.curvature = 1.0
        self.manifold = Lorentz(self.curvature)
            
    
        
        
    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.t5_model.model_name)
        #HP Soft Prompt will be tuned
        pp_length = self.config.parse_then_hop_training.prompt_length
    
        pp_embedding_size = self.knit5.config.hidden_size
        pp_embeddings = torch.randn(pp_length, pp_embedding_size)
        
        #dont use random use top 100 most common tokens of tokenizer.getvocab
        top_100_token_embeddings = get_top_token_embeddings(self.knit5, tokenizer, 100)
        pp_embeddings[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings

        #pp_embeddings = torch.cat([torch.randn(pp_length,1), pp_embeddings], dim = -1)
        print(f"Initialized Embeddings with Shape: {pp_embeddings.shape}")
        print(f"Initializing Soft Prompt with top 100 tokens from pretraining corpus")
        return nn.Parameter(pp_embeddings)
    
    def forward(self, input_ids, attention_mask, labels, **forward_kwargs):
        soft_prompt_input = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)



        

        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings

  
        #soft_prompt_input = torch.cat([torch.zeros(soft_prompt_input.size(0), soft_prompt_input.size(1), 1).to(self.device), soft_prompt_input], dim=-1)
        # soft_prompt_input = self.scaler(soft_prompt_input)
        # soft_prompt_input = self.manifold.expmap0(soft_prompt_input)

        #projected_input = self.projection_layer(soft_prompt_input)
        #soft_prompt_input = soft_prompt_input[..., 1:]

        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim = 1)






        
        
        
        
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((attention_mask.size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, attention_mask), dim=1)

        
        outputs = self.knit5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels = labels, **forward_kwargs)
        return outputs
    
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        pp_input = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings

        concatenated_embeddings = torch.cat([pp_input, input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        pp_attention_mask = torch.ones((attention_mask.size(0), pp_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((pp_attention_mask, attention_mask), dim=1)

        outputs = self.knit5.generate(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, **generate_kwargs)
        return outputs