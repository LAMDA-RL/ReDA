import torch
import torch.nn as nn

from offlinerl.utils.function import soft_clamp
from offlinerl.utils.net.common import Swish

class EnsembleLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, ensemble_size=7):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.register_parameter('weight', torch.nn.Parameter(torch.zeros(ensemble_size, in_features, out_features)))
        self.register_parameter('bias', torch.nn.Parameter(torch.zeros(ensemble_size, 1, out_features)))

        torch.nn.init.trunc_normal_(self.weight, std=1/(2*in_features**0.5))

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

class EnsembleTransition(torch.nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_features, hidden_layers, ensemble_size=7, mode='local', with_reward=True):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.mode = mode
        self.with_reward = with_reward
        self.ensemble_size = ensemble_size
        self.select = list(range(ensemble_size))

        self.activation = Swish()

        module_list = []
        for i in range(hidden_layers):
            if i == 0:
                module_list.append(EnsembleLinear(obs_dim + action_dim, hidden_features, ensemble_size))
            else:
                module_list.append(EnsembleLinear(hidden_features, hidden_features, ensemble_size))
        self.backbones = torch.nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(hidden_features, 2 * (obs_dim + self.with_reward), ensemble_size)
        self.obs_mean = None
        self.obs_std = None
        self.register_parameter('max_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * 0.25, requires_grad=True))
        self.register_parameter('min_logstd', torch.nn.Parameter(torch.ones(obs_dim + self.with_reward) * -5, requires_grad=True))

    def update_self(self, obs):
        self.obs_mean = obs.mean(dim=0)
        self.obs_std = obs.std(dim=0)

    def forward(self, obs_action):
        # Normalization for obs. If 'normaliza', no residual. 
        # use 'dims' to make forward work both when training and evaluating
        dims = len(obs_action.shape) - 2
        if self.obs_mean is not None:
            if dims:
                obs_mean = self.obs_mean.unsqueeze(0).expand(obs_action.shape[0], -1).to(obs_action.device)
                obs_std = self.obs_std.unsqueeze(0).expand(obs_action.shape[0], -1).to(obs_action.device)
            else:
                obs_mean = self.obs_mean.to(obs_action.device)
                obs_std = self.obs_std.to(obs_action.device)
            if self.mode == 'normalize':
                batch_size = obs_action.shape[dims]
                obs, action = torch.split(obs_action, [self.obs_dim, obs_action.shape[-1] - self.obs_dim], dim=-1)
                if dims:
                    obs = obs - obs_mean.unsqueeze(dims).expand(-1, batch_size, -1)
                    obs = obs / (obs_std.unsqueeze(dims).expand(-1, batch_size, -1) + 1e-8)
                else:
                    obs = obs - obs_mean.unsqueeze(dims).expand(batch_size, -1)
                    obs = obs / (obs_std.unsqueeze(dims).expand(batch_size, -1) + 1e-8)
                output = torch.cat([obs, action], dim=-1)
            else:
                output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mu, logstd = torch.chunk(self.output_layer(output), 2, dim=-1)
        logstd = soft_clamp(logstd, self.min_logstd, self.max_logstd)
        # 'local': with residual
        if self.mode == 'local' or self.mode == 'normalize':
            if self.with_reward:
                obs, reward = torch.split(mu, [self.obs_dim, 1], dim=-1)
                obs = obs + obs_action[..., :self.obs_dim]
                mu = torch.cat([obs, reward], dim=-1)
            else:
                mu = mu + obs_action[..., :self.obs_dim]
        return torch.distributions.Normal(mu, torch.exp(logstd))

    def set_select(self, indexes):
        self.select = indexes
        for layer in self.backbones:
            layer.set_select(indexes)
        self.output_layer.set_select(indexes)

    def update_save(self, indexes):
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
        
    
    @staticmethod
    def copy_params(model1, model2, idx1, idx2):
        for i, _ in enumerate(model1.backbones):
            model2.backbones[i].weight.data[idx2] = model1.backbones[i].weight.data[idx1]
            model2.backbones[i].bias.data[idx2] = model1.backbones[i].bias.data[idx1]
            model2.backbones[i].saved_weight.data[idx2] = model1.backbones[i].saved_weight.data[idx1]
            model2.backbones[i].saved_bias.data[idx2] = model1.backbones[i].saved_bias.data[idx1]
        model2.output_layer.weight.data[idx2] = model1.output_layer.weight.data[idx1]
        model2.output_layer.bias.data[idx2] = model1.output_layer.bias.data[idx1]
        model2.output_layer.saved_weight.data[idx2] = model1.output_layer.saved_weight.data[idx1]
        model2.output_layer.saved_bias.data[idx2] = model1.output_layer.saved_bias.data[idx1]
    
    def get_single_transiton(self, idx):
        if not idx in self.select:
            raise KeyError("Current Ensemble Transition's select is {}, while idx is {}.".format(self.select, idx))
        new_ref = EnsembleTransition(self.obs_dim, self.action_dim, self.hidden_features, self.hidden_layers, ensemble_size=1, mode=self.mode, with_reward=self.with_reward)
        EnsembleTransition.copy_params(self, new_ref, idx, 0)

        new_ref.max_logstd.data = self.max_logstd.data
        new_ref.min_logstd.data = self.min_logstd.data
        new_ref.obs_mean = self.obs_mean
        new_ref.obs_std = self.obs_std
        new_ref.update_save([0])
        new_ref.set_select([0])
        return new_ref  
        
    @staticmethod
    def from_single_transition(models):
        tmp = models[0]
        num = len(models)
        new_select = list(range(num))
        new_ref = EnsembleTransition(tmp.obs_dim, tmp.action_dim, tmp.hidden_features, tmp.hidden_layers, ensemble_size=num, mode=tmp.mode, with_reward=tmp.with_reward)
        for idx, model in enumerate(models):
            EnsembleTransition.copy_params(model, new_ref, 0, idx)
            new_ref.max_logstd.data = model.max_logstd.data
            new_ref.min_logstd.data = model.min_logstd.data
            new_ref.obs_mean = model.obs_mean
            new_ref.obs_std = model.obs_std
        
        new_ref.update_save(new_select)
        new_ref.set_select(new_select)
        return new_ref

    def remove_unselected(self):
        models = [
            self.get_single_transiton(idx) for idx in self.select
        ]
        return EnsembleTransition.from_single_transition(models)

    
# A upper class which wraps EnsembleTransition and 
# provide same format of outputs for compatability
class ModuleListTransition(torch.nn.Module):
    def __init__(self, models, chunk_size=14):
        super().__init__()
        tmp = models[0]
        self.obs_dim = tmp.obs_dim
        self.action_dim = tmp.action_dim
        self.hidden_features = tmp.hidden_features
        self.hidden_layers = tmp.hidden_layers
        self.mode = tmp.mode
        self.with_reward = tmp.with_reward
        self.ensemble_size = len(models)
        
        self.transition = torch.nn.ModuleList()
        # we break the ensemble models into 5 pieces
        self.chunk_size = chunk_size
        self.chunk_num = self.ensemble_size // self.chunk_size
        self.res_size = self.ensemble_size - self.chunk_num * self.chunk_size
        for i in range(self.chunk_num):
            new_ref = EnsembleTransition.from_single_transition(models[i*self.chunk_size: (i+1)*self.chunk_size])
            self.transition.append(new_ref)
        self.transition.append(
            EnsembleTransition.from_single_transition(models[self.chunk_num*self.chunk_size: ])
        )
        self.chunk_num += 1
        
    def update_self(self, obs):
        for em in self.transition:
            em.update_self(obs)
            
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
            
    def set_select(self, indexes):
        assert False
        
    def update_save(self, indexes):
        assert False
    # def append_transitions(self, models):
    #     if len(models) == 0:
    #         return self
        
    #     old_select = self.backbones[0].select
    #     old_num = len(old_select)
    #     new_select = list(range(len(old_select) + len(models)))

    #     _model = models[0]
    #     new_ref = EnsembleTransition(_model.obs_dim, _model.action_dim, _model.hidden_features, _model.hidden_layers, ensemble_size=len(new_select), mode=_model.mode, with_reward=_model.with_reward )
    #     for i, _ in enumerate(new_ref.backbones):
    #         for j, real_model_index in enumerate(self.backbones[0].select):
    #             new_ref.backbones[i].weight.data[j] = self.backbones[i].weight.data[real_model_index]
    #             new_ref.backbones[i].bias.data[j] = self.backbones[i].bias.data[real_model_index]
    #             new_ref.backbones[i].saved_weight.data[j] = self.backbones[i].saved_weight.data[real_model_index]
    #             new_ref.backbones[i].saved_bias.data[j] = self.backbones[i].saved_bias.data[real_model_index]
    #         for j, m in enumerate(models):
    #             new_ref.backbones[i].weight.data[old_num+j] = m.backbones[i].weight.data[0]
    #             new_ref.backbones[i].bias.data[old_num+j] = m.backbones[i].bias.data[0]
    #             new_ref.backbones[i].saved_weight.data[old_num+j] = m.backbones[i].saved_weight.data[0]
    #             new_ref.backbones[i].saved_bias.data[old_num+j] = m.backbones[i].saved_bias.data[0]
    #     for j, real_model_index in enumerate(self.backbones[0].select):
    #         new_ref.output_layer.weight.data[j] = self.output_layer.weight.data[real_model_index]
    #         new_ref.output_layer.bias.data[j] = self.output_layer.bias.data[real_model_index]
    #         new_ref.output_layer.saved_weight.data[j] = self.output_layer.saved_weight.data[real_model_index]
    #         new_ref.output_layer.saved_bias.data[j] = self.output_layer.saved_bias.data[real_model_index]
    #     for j, m in enumerate(models):
    #         new_ref.output_layer.weight.data[old_num+j] = m.output_layer.weight.data[0]
    #         new_ref.output_layer.bias.data[old_num+j] = m.output_layer.bias.data[0]
    #         new_ref.output_layer.saved_weight.data[old_num+j] = m.output_layer.saved_weight.data[0]
    #         new_ref.output_layer.saved_bias.data[old_num+j] = m.output_layer.saved_bias.data[0] 
        
    #     new_ref.max_logstd.data = self.max_logstd.data
    #     new_ref.min_logstd.data = self.min_logstd.data
    #     new_ref.obs_mean = self.obs_mean
    #     new_ref.obs_std = self.obs_std
    #     new_ref.update_save(new_select)
    #     new_ref.set_select(new_select)

    #     return new_ref

        

