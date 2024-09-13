from utils.util import expmap0, logmap0
from utils.trainer_utils import load_model_checkpoint
from transformers import T5ForConditionalGeneration, T5Config
from .hyperbolic_model_utils import HyperbolicEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Tuple, Optional
import torch

class HyperbolicT5Model(T5ForConditionalGeneration):
    """
    For now the shared layer is replaced with an hyerbolic Embedding layer which is then trained. 
    We feed the input_ids through the hyperbolic embedding layer and convert them to the euclidean space and push it through the original t5 model which works in euclidean space.
    TODO We can also try to turn additional layers also in the hyperbolic space...
    """
    def __init__(self,
                 model_name : str = 'google/t5-large-lm-adapt',
                 curvature : float = 1.0):
        super(HyperbolicT5Model, self).__init__(config=T5Config.from_pretrained(model_name))
        self.load_state_dict(T5ForConditionalGeneration.from_pretrained(model_name).state_dict())

        self.curvature = curvature
        self.model_name = model_name

    
    #TODO What we try:
    # Embed after the embeddings 
    # Embed after the encoder
    def _forward_after_encoder(self,
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
        
        
        
        encoder_outputs = self.encoder( input_ids = input_ids,
                                        inputs_embeds = inputs_embeds,
                                        attention_mask=attention_mask,
                                        head_mask = head_mask,
                                        output_attentions = output_attentions,
                                        output_hidden_states = output_hidden_states,
                                        return_dict = return_dict)

        hidden_states = expmap0(encoder_outputs[0], self.curvature)
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(input_ids = decoder_input_ids,
                                             encoder_attention_mask = attention_mask,
                                             encoder_hidden_states = hidden_states,
                                             attention_mask = decoder_attention_mask,
                                             inputs_embeds = decoder_inputs_embeds,
                                             past_key_values = past_key_values,
                                             head_mask = head_mask,
                                             cross_attn_head_mask = cross_attn_head_mask,
                                             use_cache = use_cache,
                                             output_attentions=output_attentions,
                                             output_hidden_states=output_hidden_states,
                                             return_dict = return_dict)

        lm_logits = self.lm_head(decoder_outputs[0])

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        # return (lm_logits, loss) if loss is not None else lm_logits
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions
        )
    def _forward_after_embeddings(self,
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
    
        if inputs_embeds is None:
            inputs_embeds = self.shared(input_ids)

        inputs_embeds = expmap0(inputs_embeds, self.curvature)

        
        encoder_outputs = self.encoder( 
                                        inputs_embeds = inputs_embeds,
                                        attention_mask=attention_mask,
                                        head_mask = head_mask,
                                        output_attentions = output_attentions,
                                        output_hidden_states = output_hidden_states,
                                        return_dict = return_dict)

        #hidden_states = expmap0(hidden_states, self.curvature)
        
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(input_ids = decoder_input_ids,
                                             encoder_attention_mask = attention_mask,
                                             encoder_hidden_states = encoder_outputs[0],
                                             attention_mask = decoder_attention_mask,
                                             inputs_embeds = decoder_inputs_embeds,
                                             past_key_values = past_key_values,
                                             head_mask = head_mask,
                                             cross_attn_head_mask = cross_attn_head_mask,
                                             use_cache = use_cache,
                                             output_attentions=output_attentions,
                                             output_hidden_states=output_hidden_states,
                                             return_dict = return_dict)

        lm_logits = self.lm_head(decoder_outputs[0])

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        # return (lm_logits, loss) if loss is not None else lm_logits
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions
        )
    
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
    
        return self._forward_after_encoder(
                             input_ids = input_ids,
                             inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask,
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