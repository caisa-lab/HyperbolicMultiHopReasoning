import torch
import torch.nn as nn
from utils.util import expmap0
from transformers import T5Model
from transformers import AutoTokenizer
from src.config import Config

class HyperbolicT5Model(nn.Module):
    def __init__(self,
                 t5 : T5Model,
                 model_name : str,
                 curvature : float = 1.0):
        super(HyperbolicT5Model, self).__init__()
        self.t5 = t5
        self.curvature = curvature
        self.model_name = model_name
        self.config = Config()

    
    
    def forward(self, input_ids, attention_mask, labels, hyperbolic = True):
        input_embeddings = self.t5.shared(input_ids)  # Convert input IDs to embeddings
        
        if hyperbolic:
            #Map input embeddings to hyperbolic space
            input_embeddings = expmap0(input_embeddings, self.curvature)
        

        outputs = self.t5(inputs_embeds=input_embeddings, attention_mask=attention_mask, labels=labels)
        return outputs
    
    def generate(self, input_ids, attention_mask, max_length=50, num_beams = 5, early_stopping=True):
        input_embeddings = self.t5.shared(input_ids)  # Convert input IDs to embeddings

        #Map input embeddings to hyperbolic space
        hyperbolic_input_embeddings = expmap0(input_embeddings, self.curvature)

        outputs = self.t5.generate(inputs_embeds=hyperbolic_input_embeddings, attention_mask=attention_mask, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
        return outputs