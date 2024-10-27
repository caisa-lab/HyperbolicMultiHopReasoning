import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5Model
from src.utils.util import get_top_token_embeddings
from src.utils.trainer_utils import load_model_checkpoint
from config import Config
from geoopt.manifolds import Lorentz, PoincareBall
from src.utils.util import expmap0, logmap0
from .hyperbolic_model_utils import HyperbolicLayer
from .hyperbolic_t5_only_kth_layer import HyperbolicKthLayerT5Model
from typing import Union

class SoftPromptModel(nn.Module):
    def __init__(self,
                 knit5 : Union[T5Model, HyperbolicKthLayerT5Model],
                 knit5_checkpoint_path : str,
                 model_name : str = '',
                 curvature = 1.0, 
                 soft_prompt = None,
                 with_model_state_dict = True):
        super(SoftPromptModel, self).__init__()
        self.knit5 = knit5
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #if knit5_checkpoint_path is not None:
        #    load_model_checkpoint(knit5, knit5_checkpoint_path, with_model_state_dict=with_model_state_dict)
        
        self.config = Config()
        
        self.curvature = curvature
        if soft_prompt is None:
            self.soft_prompt = self.init_soft_prompt()
        else:
            self.soft_prompt = soft_prompt
            
        
        for param in self.knit5.parameters():
            param.requires_grad = False

        #Remove if no learnable curvature
        for param in self.knit5.hyperbolic_layer.parameters():
           param.requires_grad = True
        
        self.soft_prompt.requires_grad = True
        
    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.t5_model.model_name)
        #HP Soft Prompt will be tuned
        pp_length = self.config.random_walk_training.prompt_length
    
        pp_embedding_size = self.knit5.config.hidden_size
        pp_embeddings = torch.randn(pp_length, pp_embedding_size)
        
        #dont use random use top 100 most common tokens of tokenizer.getvocab
        top_100_token_embeddings = get_top_token_embeddings(self.knit5, tokenizer, pp_length)
        pp_embeddings[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings

        #pp_embeddings = torch.cat([torch.randn(pp_length,1), pp_embeddings], dim = -1)
        print(f"Initialized Embeddings with Shape: {pp_embeddings.shape}")
        print(f"Initializing Soft Prompt with top 100 tokens from pretraining corpus")
        return nn.Parameter(pp_embeddings)
    

    #Using Same and using Different Soft Prompt for decoder
    def _forward_with_soft_prompt_decoder(self, input_ids, attention_mask, labels, labels_attention_mask, **forward_kwargs):
        soft_prompt_input = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1)
        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings
        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim = 1)

        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((attention_mask.size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, attention_mask), dim=1)
        

        decoder_input_ids = self.knit5._shift_right(labels)

        decoder_inputs_embeds = self.knit5.decoder.embed_tokens(decoder_input_ids)
        concat_decoder_embeds = torch.cat([soft_prompt_input, decoder_inputs_embeds], dim = 1)

        concat_decoder_attention_mask = torch.cat([soft_prompt_attention_mask, labels_attention_mask], dim = 1)

        if labels is not None:
        # We need to pad the labels with -100 at the beginning to match the sequence length
            label_padding = torch.full(
                (labels.size(0), self.soft_prompt.size(0)),
                -100,
                dtype=labels.dtype,
                device=labels.device
            )
            labels = torch.cat([label_padding, labels], dim=1)


        concatenated_embeddings = concatenated_embeddings.to(self.device)
        concatenated_attention_mask = concatenated_attention_mask.to(self.device)
        concat_decoder_embeds = concat_decoder_embeds.to(self.device)
        concat_decoder_attention_mask = concat_decoder_attention_mask.to(self.device)
        labels = labels.to(self.device)

        del soft_prompt_input
        del soft_prompt_attention_mask
        del decoder_inputs_embeds
        del decoder_input_ids

        outputs = self.knit5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, decoder_inputs_embeds=concat_decoder_embeds, decoder_attention_mask = concat_decoder_attention_mask, labels=labels, **forward_kwargs)
        return outputs
    
    def _forward_soft_prompt_only_enc(self, input_ids, attention_mask, labels, **forward_kwargs):
        soft_prompt_input = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings
        #soft_prompt_input = self.hyperbolic_layer(soft_prompt_input)


        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim = 1)
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((attention_mask.size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, attention_mask), dim=1)


        outputs = self.knit5(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, labels = labels, **forward_kwargs)
        return outputs

    def forward(self, input_ids, attention_mask, labels, **forward_kwargs):
        return self._forward_soft_prompt_only_enc(input_ids, attention_mask, labels, **forward_kwargs)
    
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        pp_input = self.soft_prompt.unsqueeze(0).expand(input_ids.size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings

        concatenated_embeddings = torch.cat([pp_input, input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        pp_attention_mask = torch.ones((attention_mask.size(0), pp_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((pp_attention_mask, attention_mask), dim=1)

        outputs = self.knit5.generate(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, **generate_kwargs)
        return outputs