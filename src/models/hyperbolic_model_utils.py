import torch
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall

class HyperbolicSoftPrompts(torch.nn.Module):
    def __init__(self, prompt_length, embedding_dim, c=1.0):
        super(HyperbolicSoftPrompts, self).__init__()
        self.manifold = PoincareBall(c=c)
        self.soft_prompts = ManifoldParameter(torch.randn(prompt_length, embedding_dim), manifold=self.manifold)

    def forward(self):
        return self.soft_prompts
