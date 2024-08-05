import torch
import geoopt
from geoopt import ManifoldParameter
from geoopt.manifolds import PoincareBall
import torch.nn as nn

class HyperbolicSoftPrompts(torch.nn.Module):
    def __init__(self, prompt_length, embedding_dim, c=1.0):
        super(HyperbolicSoftPrompts, self).__init__()
        self.manifold = PoincareBall(c=c)
        self.soft_prompts = ManifoldParameter(torch.randn(prompt_length, embedding_dim) * 1e-3, manifold=self.manifold)

    def forward(self):
        return self.soft_prompts
    
    
class HyperbolicEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, c=1.0):
        super(HyperbolicEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.c = c  # curvature
        self.manifold = geoopt.manifolds.PoincareBall(c=c)
        self.embedding = geoopt.ManifoldParameter(
            torch.randn(num_embeddings, embedding_dim) * 1e-3, # Start near origin
            manifold=self.manifold
        )

    def forward(self, indices):
        return self.embedding[indices]
    
    
    
#------------------[https://github.com/Graph-and-Geometric-Learning/hyperbolic-transformer/blob/master/large/manifolds/layer.py]------------------------------
class HypLayerNorm(nn.Module):
    """
    Hyperbolic Layer Normalization Layer

    Parameters:
        manifold (Manifold): The manifold to use for normalization.
        in_features (int): The number of input features.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, in_features, manifold_out=None):
        super(HypLayerNorm, self).__init__()
        self.in_features = in_features
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.layer = nn.LayerNorm(self.in_features)
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        self.layer.reset_parameters()

    def forward(self, x):
        """Forward pass for hyperbolic layer normalization."""
        x_space = x[..., 1:]
        x_space = self.layer(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)

        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypNormalization(nn.Module):
    """
    Hyperbolic Normalization Layer

    Parameters:
        manifold (Manifold): The manifold to use for normalization.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, manifold_out=None):
        super(HypNormalization, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out

    def forward(self, x):
        """Forward pass for hyperbolic normalization."""
        x_space = x[..., 1:]
        x_space = x_space / x_space.norm(dim=-1, keepdim=True)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypActivation(nn.Module):
    """
    Hyperbolic Activation Layer

    Parameters:
        manifold (Manifold): The manifold to use for the activation.
        activation (function): The activation function.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, activation, manifold_out=None):
        super(HypActivation, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.activation = activation

    def forward(self, x):
        """Forward pass for hyperbolic activation."""
        x_space = x[..., 1:]
        x_space = self.activation(x_space)
        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypDropout(nn.Module):
    """
    Hyperbolic Dropout Layer

    Parameters:
        manifold (Manifold): The manifold to use for the dropout.
        dropout (float): The dropout probability.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, dropout, manifold_out=None):
        super(HypDropout, self).__init__()
        self.manifold = manifold
        self.manifold_out = manifold_out
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, training=False):
        """Forward pass for hyperbolic dropout."""
        if training:
            x_space = x[..., 1:]
            x_space = self.dropout(x_space)
            x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
            x = torch.cat([x_time, x_space], dim=-1)
            if self.manifold_out is not None:
                x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x


class HypLinear(nn.Module):
    """
    Hyperbolic Linear Layer

    Parameters:
        manifold (Manifold): The manifold to use for the linear transformation.
        in_features (int): The size of each input sample.
        out_features (int): The size of each output sample.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        dropout (float, optional): The dropout probability. Default is 0.0.
        manifold_out (Manifold, optional): The output manifold. Default is None.
    """

    def __init__(self, manifold, in_features, out_features, bias=True, dropout=0.0, manifold_out=None):
        super().__init__()
        self.in_features = in_features + 1  # +1 for time dimension
        self.out_features = out_features
        self.bias = bias
        self.manifold = manifold
        self.manifold_out = manifold_out

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.dropout_rate = dropout
        self.reset_parameters()

    def reset_parameters(self):
        """Reset layer parameters."""
        init.xavier_uniform_(self.linear.weight, gain=math.sqrt(2))
        if self.bias:
            init.constant_(self.linear.bias, 0)

    def forward(self, x, x_manifold='hyp'):
        """Forward pass for hyperbolic linear layer."""
        if x_manifold != 'hyp':
            x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
            x = self.manifold.expmap0(x)
        x_space = self.linear(x)

        x_time = ((x_space ** 2).sum(dim=-1, keepdims=True) + self.manifold.k).sqrt()
        x = torch.cat([x_time, x_space], dim=-1)
        if self.manifold_out is not None:
            x = x * (self.manifold_out.k / self.manifold.k).sqrt()
        return x

class HypCLS(nn.Module):
    def __init__(self, manifold, in_channels, out_channels, bias=True):
        """
        Initializes the HypCLS class with the given parameters.

        Parameters:
            - `manifold` (Manifold): The manifold object.
            - `in_channels` (int): The number of input channels.
            - `out_channels` (int): The number of output channels.
            - `bias` (bool, optional): Whether to include a bias term. Defaults to True.

        Returns:
            None
        """
        super().__init__()
        self.manifold = manifold
        self.in_channels = in_channels
        self.out_channels = out_channels
        cls_emb = self.manifold.random_normal((self.out_channels, self.in_channels + 1), mean=0, std=1. / math.sqrt(self.in_channels + 1))
        self.cls = ManifoldParameter(cls_emb, self.manifold, requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def forward(self, x, x_manifold='hyp', return_type='neg_dist'):
        if x_manifold != 'hyp':
            x = self.manifold.expmap0(torch.cat([torch.zeros_like(x)[..., 0:1], x], dim=-1))  # project to Lorentz

        dist = -2 * self.manifold.k - 2 * self.cinner(x, self.cls) + self.bias
        dist = dist.clamp(min=0)

        if return_type == 'neg_dist':
            return - dist
        elif return_type == 'prob':
            return 1.0 / (1.0 + dist)
        elif return_type == 'neg_log_prob':
            return - 1.0*torch.log(1.0 + dist)
        else:
            raise NotImplementedError