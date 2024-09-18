import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Model
from src.utils.util import get_top_token_embeddings
from src.utils.trainer_utils import load_model_checkpoint
from config import Config

class SoftPromptModel(nn.Module):
    def __init__(self,
                 knit5 : T5Model,
                 knit5_checkpoint_path : str,
                 model_name : str,
                 soft_prompt = None,
                 with_model_state_dict = True):
        super(SoftPromptModel, self).__init__()
        self.knit5 = knit5
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if knit5_checkpoint_path is not None:
            load_model_checkpoint(knit5, knit5_checkpoint_path, with_model_state_dict=with_model_state_dict)
        
        self.config = Config()
        
        if soft_prompt is None:
            self.soft_prompt = self.init_soft_prompt()
        else:
            self.soft_prompt = soft_prompt
        
        for param in self.knit5.parameters():
            param.requires_grad = False
            
    
        
        
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
    
    def forward(self, input_ids, attention_mask, labels):
        soft_prompt_input = self.soft_prompt.weight.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings

        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((attention_mask.size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, attention_mask), dim=1)

        outputs = self.knit5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels=labels)
        return outputs
    
    def soft_prompt_state_dict(self):
        return self.soft_prompt.state_dict()
    
    def generate(self, input_ids, attention_mask, max_length=50, num_beams = 5, early_stopping=True):
        pp_input = self.soft_prompt.weight.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings

        concatenated_embeddings = torch.cat([pp_input, input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        pp_attention_mask = torch.ones((attention_mask.size(0), pp_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((pp_attention_mask, attention_mask), dim=1)

        outputs = self.knit5.generate(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
        return outputs