from typing import Union
import torch
import torch.nn as nn
from utils.util import get_top_token_embeddings, expmap0, logmap0
from utils.trainer_utils import load_model_checkpoint, geodesic_regularization
from .hyperbolic_t5_model import HyperbolicT5Model
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Model
from src.config import Config
from .hyperbolic_model_utils import HyperbolicSoftPrompts, HybridSoftPrompt
from typing import Optional, Tuple

class HyperbolicSoftPromptModel(nn.Module):
    def __init__(self,
                 hyperbolic_knit5 : Union[T5ForConditionalGeneration, HyperbolicT5Model],
                 hyperbolic_knit5_checkpoint_path : str,
                 model_name : str,
                 soft_prompt : HyperbolicSoftPrompts = None,
                 with_model_state_dict = True,
                 curvature : float = 1.0):
        super(HyperbolicSoftPromptModel, self).__init__()
        self.knit5 = hyperbolic_knit5
        if hyperbolic_knit5_checkpoint_path is not None:
            self.knit5 = load_model_checkpoint(self.knit5, hyperbolic_knit5_checkpoint_path, with_model_state_dict=with_model_state_dict)

            
            
        self.curvature = curvature
        self.model_name = model_name
        self.config = Config()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        self.soft_prompt = soft_prompt if soft_prompt else self.init_soft_prompt()
        
        for param in self.knit5.parameters():
            param.requires_grad = False
        for param in self.soft_prompt.parameters():
            param.requires_grad = True
        
    def init_soft_prompt(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.t5_model.model_name)
        #HP Soft Prompt will be tuned
        soft_prompt_length = self.config.parse_then_hop_training.prompt_length

        soft_prompt_embedding_size = self.knit5.config.hidden_size
        #euclidean_dim = soft_prompt_embedding_size // 2
        #hyperbolic_dim = soft_prompt_embedding_size - euclidean_dim
        
        #soft_prompt_embeddings = HybridSoftPrompt(soft_prompt_length, euclidean_dim=euclidean_dim, hyperbolic_dim=hyperbolic_dim, curvature=self.curvature)
        soft_prompt_embeddings = nn.Embedding(soft_prompt_length, soft_prompt_embedding_size) 
        
        #dont use random use top 100 most common tokens of tokenizer.getvocab
        top_100_token_embeddings = get_top_token_embeddings(self.knit5, tokenizer, 100)
        
        #top_100_euclidean = top_100_token_embeddings[:, :euclidean_dim]  # (100, 512)
        #top_100_hyperbolic = top_100_token_embeddings[:, euclidean_dim:]  # (100, 512)
        
        with torch.no_grad():
            soft_prompt_embeddings.weight.data[:top_100_token_embeddings.size(0), :] = top_100_token_embeddings
            
            # Initialize Euclidean part
            #self.hyperbolic_soft_prompt.euclidean_prompt.weight.data[:top_100_euclidean.size(0), :] = top_100_euclidean
            
            # Initialize Hyperbolic part (without the exponential map for initialization)
            #self.hyperbolic_soft_prompt.hyperbolic_prompt.weight.data[:top_100_hyperbolic.size(0), :] = top_100_hyperbolic

        print(f"Initializing Soft Prompt with top 100 tokens from pretraining corpus")
        return soft_prompt_embeddings   
        
    def _forward_soft_prompt_mapped(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    decoder_input_ids: Optional[torch.LongTensor] = None,
                    decoder_attention_mask: Optional[torch.BoolTensor] = None,
                    head_mask: Optional[torch.FloatTensor] = None,
                    decoder_head_mask: Optional[torch.FloatTensor] = None,
                    cross_attn_head_mask: Optional[torch.Tensor] = None,
                    encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                    labels: Optional[torch.LongTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None):
        soft_prompt_input = self.soft_prompt.weight.expand(input_ids.size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(input_ids)  # Convert input IDs to embeddings

        soft_prompt_input = expmap0(soft_prompt_input, self.curvature)

        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim=1)
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        soft_prompt_attention_mask = torch.ones((attention_mask.size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((soft_prompt_attention_mask, attention_mask), dim=1)
        
        
        outputs = self.knit5(inputs_embeds=concatenated_embeddings,
                             attention_mask=concatenated_attention_mask,
                             labels=labels,
                             decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask,
                             head_mask=head_mask,
                             decoder_head_mask=decoder_head_mask,
                             cross_attn_head_mask=cross_attn_head_mask,
                             encoder_outputs=encoder_outputs,
                             past_key_values=past_key_values,
                             decoder_inputs_embeds=decoder_inputs_embeds,
                             use_cache=use_cache,
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states,
                             return_dict=return_dict)
        
        return outputs
    
    def forward(self, 
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.FloatTensor] = None,
                    decoder_input_ids: Optional[torch.LongTensor] = None,
                    decoder_attention_mask: Optional[torch.BoolTensor] = None,
                    head_mask: Optional[torch.FloatTensor] = None,
                    decoder_head_mask: Optional[torch.FloatTensor] = None,
                    cross_attn_head_mask: Optional[torch.Tensor] = None,
                    encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                    inputs_embeds: Optional[torch.FloatTensor] = None,
                    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                    labels: Optional[torch.LongTensor] = None,
                    use_cache: Optional[bool] = None,
                    output_attentions: Optional[bool] = None,
                    output_hidden_states: Optional[bool] = None,
                    return_dict: Optional[bool] = None):
        return self._forward_soft_prompt_mapped(input_ids,
                    attention_mask,
                    decoder_input_ids,
                    decoder_attention_mask,
                    head_mask,
                    decoder_head_mask,
                    cross_attn_head_mask,
                    encoder_outputs,
                    past_key_values,
                    inputs_embeds,
                    decoder_inputs_embeds,
                    labels,
                    use_cache,
                    output_attentions,
                    output_hidden_states,
                    return_dict)
    
    def generate(self, inputs, max_length=50, num_beams = 5, early_stopping=True):
        soft_prompt_input = self.soft_prompt.weight.expand(inputs['input_ids'].size(0), -1, -1).to(self.device)
        input_embeddings = self.knit5.shared(inputs['input_ids'])  # Convert input IDs to embeddings


        soft_prompt_input = expmap0(soft_prompt_input, self.curvature)
        concatenated_embeddings = torch.cat([soft_prompt_input, input_embeddings], dim=1)
        
        
        #Adjust attention mask (take all of the soft prompt tokens should be attented)
        pp_attention_mask = torch.ones((inputs['attention_mask'].size(0), soft_prompt_input.size(1)), device=self.device)
        concatenated_attention_mask = torch.cat((pp_attention_mask, inputs['attention_mask']), dim=1)

        outputs = self.knit5.generate(inputs_embeds=concatenated_embeddings, attention_mask=concatenated_attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
        return outputs
    
    
