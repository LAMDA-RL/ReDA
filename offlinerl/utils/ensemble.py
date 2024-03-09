from abc import abstractclassmethod, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from offlinerl.utils.function import soft_clamp
from offlinerl.utils.net.common import Swish
from offlinerl.utils.normalizer import BatchNormalizer, RunningNormalizer, StaticNormalizer

class ParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = torch.zeros(ensemble_size, in_features, out_features)
        self.bias = torch.zeros(ensemble_size, 1, out_features)
        self.weight = torch.nn.Parameter(self.weight)
        self.bias = torch.nn.Parameter(self.bias)

        self.ln = nn.LayerNorm(out_features)
        
        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))
        
    def forward(self, x, ln=True):
        if len(x.shape) == 2:
            y = torch.einsum('ij,bjk->bik', x, self.weight)
        else:
            y = torch.einsum('bij,bjk->bik', x, self.weight)
        y += self.bias
        # if self.in_features == self.out_features:
        #     y = x + y
        # if ln:
        #     y = self.ln(y)
        
        return y
    

class ParallelRDynamics(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, normalizer="static", tanh=False, **kwargs):
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
                self.backbones.append(ParallelLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                self.backbones.append(ParallelLinear(hidden_features, hidden_features, ensemble_size))
            self.backbones.append(Swish())
        self.backbones = nn.Sequential(*self.backbones)
        
        self.output_layer = ParallelLinear(hidden_features, 2*(obs_dim+1), ensemble_size)
        
        self.max_logstd = nn.Parameter(torch.ones([ensemble_size, 1, obs_dim+1]) / 4., requires_grad=True)
        self.min_logstd = nn.Parameter(torch.ones([ensemble_size, 1, obs_dim+1]) * -5, requires_grad=True)

        self.tanh = tanh

    def forward(self, obs_action, use_res=False):
        obs = obs_action[..., :self.obs_dim]
        action = obs_action[..., self.obs_dim:]
        obs, action = self.obs_normalizer(obs), self.action_normalizer(action)
        output = self.backbones(torch.cat([obs, action], -1))
        output = self.output_layer(output)
        mu, logstd = torch.chunk(output, 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        next_obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
        # if self.tanh:
        #     next_obs = F.tanh(next_obs)
        if not use_res:
            next_obs = next_obs + obs_action[..., :self.obs_dim]
        mu = torch.cat([next_obs, reward], dim=-1)

        return torch.distributions.Normal(mu, torch.exp(logstd))

    def split_parameters(self):
        total_params = []
        weight_decay = [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]
        i = 0
        for idx, bk in enumerate(self.backbones):
            if type(bk) != ParallelLinear:
                continue
            params = {
                'params': bk.parameters(),
                'weight_decay': weight_decay[i],
            }
            total_params.append(params)
            i += 1

        total_params.append({
            'params': self.output_layer.parameters(),
            'weight_decay': weight_decay[-1],
        })
        total_params.append({
            'params': [self.max_logstd] + [self.min_logstd],
        })

        return total_params
    
    @staticmethod
    def copy_params(model1, model2, idx1, idx2):
        for i, _ in enumerate(model1.backbones):
            if isinstance(_, ParallelLinear):
                model2.backbones[i].weight.data[idx2] = model1.backbones[i].weight.data[idx1]
                model2.backbones[i].bias.data[idx2] = model1.backbones[i].bias.data[idx1]
        model2.max_logstd.data[idx2] = model1.max_logstd.data[idx1]
        model2.min_logstd.data[idx2] = model1.min_logstd.data[idx1]
        model2.output_layer.weight.data[idx2] = model1.output_layer.weight.data[idx1]
        model2.output_layer.bias.data[idx2] = model1.output_layer.bias.data[idx1]
        model2.obs_normalizer.load_state_dict(model1.obs_normalizer.state_dict())
        model2.action_normalizer.load_state_dict(model1.action_normalizer.state_dict())
    
    def get_single_transition(self, idx):
        new_ref = ParallelRDynamics(self.obs_dim, self.action_dim, self.hidden_features, self.hidden_layers, ensemble_size=1, normalizer=self.normalizer_type, tanh=self.tanh, **self.kwargs)
        ParallelRDynamics.copy_params(self, new_ref, idx, 0)
        return new_ref  
        
    @staticmethod
    def from_single_transition(models, use_tanh=False):
        tmp = models[0]
        num = len(models)
        new_ref = ParallelRDynamics(tmp.obs_dim, tmp.action_dim, tmp.hidden_features, tmp.hidden_layers, ensemble_size=num, normalizer=tmp.normalizer_type, tanh=use_tanh, **tmp.kwargs)
        for idx, model in enumerate(models):
            ParallelRDynamics.copy_params(model, new_ref, 0, idx)
        return new_ref


# A upper class which wraps EnsembleTransition and 
# provide same format of outputs for compatability
class ChunkRDynamics(torch.nn.Module):
    def __init__(self, models, chunk_size=14):
        super().__init__()
        tmp = models[0]
        self.obs_dim = tmp.obs_dim
        self.action_dim = tmp.action_dim
        self.hidden_features = tmp.hidden_features
        self.hidden_layers = tmp.hidden_layers
        
        self.obs_normalizer = deepcopy(tmp.obs_normalizer)
        self.action_normalizer = deepcopy(tmp.action_normalizer)
        self.ensemble_size = len(models)
        
        self.transition = torch.nn.ModuleList()
        self.chunk_size = chunk_size
        self.chunk_num = self.ensemble_size // self.chunk_size
        self.res_size = self.ensemble_size - self.chunk_num * self.chunk_size
        for i in range(self.chunk_num):
            new_ref = ParallelLinear.from_single_transition(models[i*self.chunk_size: (i+1)*self.chunk_size])
            self.transition.append(new_ref)
        self.transition.append(
            ParallelLinear.from_single_transition(models[self.chunk_num*self.chunk_size: ])
        )
        self.chunk_num += 1
            
    def forward(self, obs_action):
        loc = []
        scale = []
        for em in self.transition:
            dist = em.forward(obs_action)
            loc.append(dist.loc)
            scale.append(dist.scale)
            torch.cuda.empty_cache()
        loc = torch.concat(loc, dim=0)
        scale = torch.concat(scale, dim=0)
        return torch.distributions.Normal(loc, scale)


class ContextParallelRDynamics(nn.Module):
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