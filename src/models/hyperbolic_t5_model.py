from utils.util import expmap0, logmap0
from utils.trainer_utils import load_model_checkpoint
from transformers import T5ForConditionalGeneration, T5Config
from .hyperbolic_model_utils import HyperbolicEmbedding
from torch import Tensor, FloatTensor, LongTensor, BoolTensor
from typing import Tuple


class HyperbolicT5MapEmbeddings(T5ForConditionalGeneration):
    def __init__(self,
                 model_name : str = 'google/t5-large-lm-adapt',
                 curvature : float = 1.0):
        super(HyperbolicT5MapEmbeddings, self).__init__(T5Config.from_pretrained(model_name))
        print(f"Loaded model: {model_name}")
        self.curvature = curvature
        self.model_name = model_name
        #if checkpoint_path:
         #   load_model_checkpoint(self.t5, checkpoint_path, with_model_state_dict=True)
            
    
    def forward(self, input_ids: LongTensor = None,
            attention_mask: FloatTensor = None,
            decoder_input_ids: LongTensor = None,
            decoder_attention_mask: BoolTensor = None,
            head_mask: FloatTensor = None,
            decoder_head_mask: FloatTensor = None,
            cross_attn_head_mask: Tensor = None,
            encoder_outputs: Tuple[Tuple[FloatTensor]] = None,
            past_key_values: Tuple[Tuple[FloatTensor]] = None,
            inputs_embeds: Tensor = None,
            decoder_inputs_embeds: Tensor = None,
            use_cache: bool = None,
            output_attentions: bool = None,
            output_hidden_states: bool = None,
            return_dict: bool = None,
            labels = None):
        if inputs_embeds is None and input_ids is not None:
            input_embeddings = self.shared(input_ids)  # Convert input IDs to embeddings
            
            #Map input embeddings to hyperbolic space
            input_embeddings = expmap0(input_embeddings, self.curvature)
        else:
            input_embeddings = inputs_embeds
        return super(HyperbolicT5MapEmbeddings, self).forward(input_ids=input_ids,
                                                            attention_mask=attention_mask,
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
                                                            return_dict=return_dict,
                                                            inputs_embeds=input_embeddings,  # pass inputs_embeds here
                                                            labels=labels)

class HyperbolicT5Model(T5ForConditionalGeneration):
    """
    For now the shared layer is replaced with an hyerbolic Embedding layer which is then trained. 
    We feed the input_ids through the hyperbolic embedding layer and convert them to the euclidean space and push it through the original t5 model which works in euclidean space.
    TODO We can also try to turn additional layers also in the hyperbolic space...
    """
    def __init__(self,
                 model_name : str = 'google/t5-large-lm-adapt',
                 curvature : float = 1.0):
        config = T5Config.from_pretrained(model_name)
        super(HyperbolicT5Model, self).__init__(config)
        self.curvature = curvature
        self.model_name = model_name
        num_embeddings = self.shared.num_embeddings
        embedding_dim = self.shared.embedding_dim
        self.encoder_embeddings = HyperbolicEmbedding(num_embeddings, embedding_dim, curvature = curvature) #TODO Initialize this with the exponential mapped weights of the shared layers of T5 

    
    
    def forward(self,
                input_ids: LongTensor = None,
    attention_mask: FloatTensor = None,
    decoder_input_ids: LongTensor = None,
    decoder_attention_mask: BoolTensor = None,
    head_mask: FloatTensor = None,
    decoder_head_mask: FloatTensor = None,
    cross_attn_head_mask: Tensor = None,
    encoder_outputs: Tuple[Tuple[FloatTensor]] = None,
    past_key_values: Tuple[Tuple[FloatTensor]] = None,
    inputs_embeds: Tensor = None,
    decoder_inputs_embeds: Tensor = None,
    use_cache: bool = None,
    output_attentions: bool = None,
    output_hidden_states: bool = None,
    return_dict: bool = None,
    labels = None):
        input_embeddings = self.encoder_embeddings(input_ids)  # Convert input IDs to hyperbolic embeddings
        
        input_embeddings = logmap0(input_embeddings, c = self.curvature)
            


        return super(HyperbolicT5Model, self).forward(input_ids=input_ids,
    attention_mask=attention_mask,
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
    return_dict=return_dict,
    inputs_embeds=input_embeddings,  
    labels=labels)