import copy
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from offlinerl.utils.net.common import MLP
from offlinerl.utils.net.transformer import Transformer
from offlinerl.utils.function import soft_clamp
from offlinerl.utils.net.common import Swish
from offlinerl.utils.normalizer import BatchNormalizer, RunningNormalizer, StaticNormalizer
from offlinerl.utils.ensemble import ParallelLinear

def reparameterize(mu, logvar):
    assert mu.shape == logvar.shape
    std = torch.exp(logvar * 0.5)
    eps = torch.randn_like(mu)
    return eps * std + mu

class VariationalEncoder(MLP):
    def __init__(self, in_features: int, out_features: int, hidden_features: int, hidden_layers: int, norm: str = None, hidden_activation: str = 'leakyrelu', output_activation: str = 'identity'):
        super().__init__(in_features, out_features * 2, hidden_features, hidden_layers, norm, hidden_activation, output_activation)
        self.z_dim = out_features
        
    def forward(self, x):
        z = super().forward(x)
        z_mu, z_logvar = z[..., :self.z_dim], z[..., self.z_dim:]
        return z_mu, z_logvar, reparameterize(z_mu, z_logvar)

class TransformerEncoder(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_features:int, heads: int, depth: int):
        super().__init__()
        self.transformer = Transformer(in_features, hidden_features, heads, depth, hidden_features)
        self.linear = nn.Linear(hidden_features, out_features * 2)
        self.out_dim = out_features
        
    def forward(self, x):
        z = self.transformer(x)
        z = z.mean(dim=1)
        z = self.linear(z)
        z_means, z_logvar = z[...,:self.out_dim], z[...,self.out_dim:]
        z_vars = torch.exp(z_logvar)
        return z_means, z_vars, None

class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=1):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        torch.nn.init.trunc_normal_(self.weight, std=1 / (2 * in_features ** 0.5))

        self.register_parameter('saved_weight', torch.nn.Parameter(self.weight.detach().clone()))
        self.register_parameter('saved_bias', torch.nn.Parameter(self.bias.detach().clone()))

        self.select = list(range(0, self.ensemble_size))

    def forward(self, x):
        weight = self.weight[self.select]
        bias = self.bias[self.select]
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight)

        x = x + bias

        return x

    def set_select(self, indexes):
        assert len(indexes) <= self.ensemble_size and max(indexes) < self.ensemble_size
        self.select = indexes
        self.weight.data[indexes] = self.saved_weight.data[indexes]
        self.bias.data[indexes] = self.saved_bias.data[indexes]

    def update_save(self, indexes):
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]

class VariationalDecoder(nn.Module):
    def __init__(self, obs_dim, action_dim, context_dim, hidden_features, hidden_layers, ensemble_size=7, normalizer="static", **kwargs):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.ensemble_size = ensemble_size
        self.normalizer_type = normalizer
        self.kwargs = kwargs
        
        if normalizer is None:
            self.obs_normalizer = self.action_normalizer = nn.Identity()
        elif normalizer == "batch":
            self.obs_normalizer = self.action_normalizer = BatchNormalizer(self.obs_dim)
        elif normalizer == "running":
            self.obs_normalizer = self.action_normalizer = RunningNormalizer(self.obs_dim)
        elif normalizer == "static":
            self.obs_normalizer = StaticNormalizer(self.obs_dim, kwargs["obs_mean"], kwargs["obs_std"])
            self.action_normalizer = nn.Identity()        
        
        self.backbones = []
        for i in range(hidden_layers):
            if i == 0:
                self.backbones.append(ParallelLinear(obs_dim + action_dim + context_dim, hidden_features, ensemble_size))
            else:
                self.backbones.append(ParallelLinear(hidden_features, hidden_features, ensemble_size))
            self.backbones.append(Swish())
        self.backbones = nn.Sequential(*self.backbones)
        
        self.output_layer = ParallelLinear(hidden_features, 2*(obs_dim+1), ensemble_size)
        
        self.max_logstd = nn.Parameter(torch.ones([ensemble_size, 1, obs_dim+1]) / 4., requires_grad=True)
        self.min_logstd = nn.Parameter(torch.ones([ensemble_size, 1, obs_dim+1]) * -5, requires_grad=True)
        
    def forward(self, obs_action, z):
        obs = obs_action[..., :self.obs_dim]
        action = obs_action[..., self.obs_dim:]
        obs, action = self.obs_normalizer(obs), self.action_normalizer(action)
        output = self.backbones(torch.cat([obs, action, z], -1))
        output = self.output_layer(output)
        mu, logstd = torch.chunk(output, 2, dim=-1)
        
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        next_obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
        next_obs = next_obs + obs_action[..., :self.obs_dim]
        mu = torch.cat([next_obs, reward], dim=-1)

        if self.ensemble_size == 1:
            mu, logstd = mu.squeeze(0), logstd.squeeze(0)
            
        return torch.distributions.Normal(mu, torch.exp(logstd))   
