from json import decoder
from torch import embedding
import torch.nn
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import os
import torch.nn.functional as F
from offlinerl.utils.net.common import miniblock
from offlinerl.utils.net.common import MLP


class DeterministicPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, policy_hidden_dims=(256, 256), max_log_std=2, min_log_std=-20):
        super(DeterministicPolicyNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_hidden_dims = policy_hidden_dims
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std

        self.mlp = miniblock(obs_dim, policy_hidden_dims[0], None)
        for i in range(1, len(policy_hidden_dims)):
            self.mlp += miniblock(policy_hidden_dims[i - 1], policy_hidden_dims[i], None)
        self.output_layer = nn.Linear(policy_hidden_dims[-1], action_dim)
        self.layers = nn.Sequential(*self.mlp, self.output_layer)

    def forward(self, state):
        out = self.layers(state)
        out = torch.clamp(out, -9., 9.)
        out = torch.tanh(out)
        return out


class GaussianPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, policy_hidden_dims=(256,256), max_log_std=2, min_log_std=-20):
        super(GaussianPolicyNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.policy_hidden_dims = policy_hidden_dims
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std
        
        self.mlp = miniblock(obs_dim, policy_hidden_dims[0], None)
        for i in range(1, len(policy_hidden_dims)):
            self.mlp += miniblock(policy_hidden_dims[i-1], policy_hidden_dims[i], None)
        self.output_layer = nn.Linear(policy_hidden_dims[-1], action_dim*2)
        self.layers = nn.Sequential(*self.mlp, self.output_layer)
        
    def forward(self, state):
        out = self.layers(state)
        mu, logstd = torch.split(out, [self.action_dim, self.action_dim], dim=-1)
        logstd = torch.clip(logstd, self.min_log_std, self.max_log_std)
        return mu, logstd
    
    def sample(self, state, deterministic=False):
        mu, logstd = self.forward(state)
        dist = Normal(mu, logstd.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= torch.sum(2*(np.log(2) - action - F.softplus(-2*action)), dim=-1)
        
        if deterministic:
            return torch.tanh(mu), None, mu, logstd
        else:
            return torch.tanh(action), log_prob, mu, logstd
        
        
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, value_hidden_dims=(256, 256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.value_hidden_dims = value_hidden_dims
        
        self.mlp = miniblock(obs_dim+action_dim, value_hidden_dims[0], None)
        for i in range(1, len(value_hidden_dims)):
            self.mlp += miniblock(value_hidden_dims[i-1], value_hidden_dims[i], None)
        self.output_layer = nn.Linear(value_hidden_dims[-1], 1)
        self.layers = nn.Sequential(*self.mlp, self.output_layer)

    def forward(self, state, action):
        out = torch.cat([state, action], dim=-1)
        out = self.layers(out)
        return out


class MLPDecoder(nn.Module):
    def __init__(self, obs_dim, action_dim, embedding_dim, decoder_hidden_dims=(4,)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dims = decoder_hidden_dims

        self.decoder = miniblock(embedding_dim, decoder_hidden_dims[0], None, relu=False)
        for i in range(1, len(decoder_hidden_dims)):
            self.decoder += miniblock(decoder_hidden_dims[i - 1], decoder_hidden_dims[i], None)
        self.decoder = nn.Sequential(*self.decoder)

    def forward(self, embedding):
        embedding = self.decoder(embedding)
        return embedding


class DeterministicOutputHead(nn.Module):
    def __init__(self, obs_dim, action_dim, embedding_dim, decoder_hidden_dims=(16,), head_hidden_dims=(256, 256),
                 max_log_std=2, min_log_std=-20):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dims = decoder_hidden_dims
        self.head_hidden_dims = head_hidden_dims
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std

        # self.decoder = miniblock(embedding_dim, decoder_hidden_dims[0], None, relu=True)
        # for i in range(1, len(decoder_hidden_dims)):
        #     self.decoder += miniblock(decoder_hidden_dims[i-1], decoder_hidden_dims[i], None)
        # self.head_mlp = miniblock(decoder_hidden_dims[-1] + obs_dim, head_hidden_dims[0], None)
        # for i in range(1, len(head_hidden_dims)):
        #     self.head_mlp += miniblock(head_hidden_dims[i - 1], head_hidden_dims[i], None)
        # self.decoder = nn.Sequential(*self.decoder)
        # self.head_mlp = nn.Sequential(*self.head_mlp)

        self.head_mlp = miniblock(embedding_dim + obs_dim, head_hidden_dims[0], None)
        for i in range(1, len(head_hidden_dims)):
            self.head_mlp += miniblock(head_hidden_dims[i - 1], head_hidden_dims[i], None)
        self.head_mlp = nn.Sequential(*self.head_mlp)

        self.mu_head = nn.Linear(head_hidden_dims[-1], action_dim)

    def forward(self, state, embedding):
        # embedding = self.decoder(embedding)
        output = torch.cat([state, embedding], dim=-1)
        output = self.head_mlp(output)
        mu = self.mu_head(output)
        # mu = torch.clamp(mu, -9., 9.)
        mu = torch.tanh(mu)
        return mu


class GaussianOutputHead(nn.Module):
    def __init__(self, obs_dim, action_dim, embedding_dim, decoder_hidden_dims=(16, ), head_hidden_dims=(256, 256), 
                 max_log_std=2, min_log_std=-20):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dims = decoder_hidden_dims
        self.head_hidden_dims = head_hidden_dims
        self.max_log_std = max_log_std
        self.min_log_std = min_log_std
        
        # self.decoder = miniblock(embedding_dim, decoder_hidden_dims[0], None, relu=False)
        # for i in range(1, len(decoder_hidden_dims)):
        #     self.decoder += miniblock(decoder_hidden_dims[i-1], decoder_hidden_dims[i], None)
        self.head_mlp = miniblock(decoder_hidden_dims[-1] + obs_dim, head_hidden_dims[0], None)
        for i in range(1, len(head_hidden_dims)):
            self.head_mlp += miniblock(head_hidden_dims[i-1], head_hidden_dims[i], None)
        # self.decoder = nn.Sequential(*self.decoder)
        self.head_mlp = nn.Sequential(*self.head_mlp)
        self.mu_head = nn.Linear(head_hidden_dims[-1], action_dim)
        self.logstd_head = nn.Linear(head_hidden_dims[-1], action_dim)
        
    def forward(self, state, embedding):
        # embedding = self.decoder(embedding)
        output = torch.cat([state, embedding], dim=-1)
        output = self.head_mlp(output)
        mu = self.mu_head(output)

        logstd = self.logstd_head(output)


        logstd = torch.clip(logstd, self.min_log_std, self.max_log_std)
        return mu, logstd
    
    def sample(self, state, embedding, deterministic=False):
        mu, logstd = self.forward(state, embedding)
        dist = Normal(mu, logstd.exp())
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= torch.sum(2*(np.log(2) - action - F.softplus(-2*action)), dim=-1)
        
        if deterministic:
            return torch.tanh(mu), None, mu, logstd
        else:
            return torch.tanh(action), log_prob, mu, logstd
        

class ValueHead(nn.Module):
    def __init__(self, obs_dim, action_dim, embedding_dim, decoder_hidden_dims=(16, ), head_hidden_dims=(256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.decoder_hidden_dims = decoder_hidden_dims
        self.head_hidden_dims = head_hidden_dims
        
        # self.decoder = miniblock(embedding_dim, decoder_hidden_dims[0], None, relu=True)
        # for i in range(1, len(decoder_hidden_dims)):
        #     self.decoder += miniblock(decoder_hidden_dims[i-1], decoder_hidden_dims[i], None)
        # self.head_mlp = miniblock(decoder_hidden_dims[-1]+obs_dim+action_dim, self.head_hidden_dims[0], None)
        # for i in range(1, len(head_hidden_dims)):
        #     self.head_mlp += miniblock(head_hidden_dims[i-1], head_hidden_dims[i], None)
        # self.decoder = nn.Sequential(*self.decoder)
        # self.head_mlp = nn.Sequential(*self.head_mlp)


        self.head_mlp = miniblock(embedding_dim + obs_dim + action_dim, self.head_hidden_dims[0], None)
        for i in range(1, len(head_hidden_dims)):
            self.head_mlp += miniblock(head_hidden_dims[i - 1], head_hidden_dims[i], None)
        self.head_mlp = nn.Sequential(*self.head_mlp)

        self.head = nn.Linear(head_hidden_dims[-1], 1)
        
    def forward(self, state, action, embedding):
        # embedding = self.decoder(embedding)
        output = torch.cat([state, action, embedding], dim=-1)
        output = self.head_mlp(output)
        output = self.head(output)
        return output

class SVNetwork(nn.Module):
    def __init__(self, args, obs_dim, action_dim, value_hidden_dims=(256, 256)):
        super().__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.value_hidden_dims = value_hidden_dims

        self.mlp = miniblock(obs_dim + action_dim, value_hidden_dims[0], None)
        for i in range(1, len(value_hidden_dims)):
            self.mlp += miniblock(value_hidden_dims[i - 1], value_hidden_dims[i], None)
        self.output_layer = nn.Linear(value_hidden_dims[-1], 1)
        self.layers = nn.Sequential(*self.mlp, self.output_layer)

        self.device = self.args["device"]

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)
        out = self.layers(inputs)
        return out

EPS = 1e-5
class DiscriminatorNetwork(nn.Module):
    def __init__(self, args, obs_dim, action_dim, value_hidden_dims=(256, 256, 256)):
        super().__init__()
        self.args = args
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.value_hidden_dims = value_hidden_dims

        self.mlp = miniblock(obs_dim * 2 + action_dim, value_hidden_dims[0], None)
        for i in range(1, len(value_hidden_dims)):
            self.mlp += miniblock(value_hidden_dims[i - 1], value_hidden_dims[i], None)
        self.output_layer = nn.Linear(value_hidden_dims[-1], 1)
        self.layers = nn.Sequential(*self.mlp, self.output_layer)

        self.device = self.args["device"]

    def forward(self, state, action, next_state):
        out = torch.cat([state, action, next_state], dim=-1)
        out = self.layers(out)
        return out

    def forward_reward(self, state, action, next_state):
        out = torch.cat([state, action, next_state], dim=-1)
        out = self.layers(out)
        out = F.sigmoid(out)
        score = torch.log(out + EPS) - torch.log(1 - out + EPS)
        return score

    def compute_loss(self, batch_real, batch_fake, ratio=False):
        try:
            obs_real = torch.from_numpy(batch_real.obs).to(self.device)
            actions_real = torch.from_numpy(batch_real.act).to(self.device)
            next_obs_real = torch.from_numpy(batch_real.obs_next).to(self.device)

            obs_fake = torch.from_numpy(batch_fake.obs).to(self.device)
            actions_fake = torch.from_numpy(batch_fake.act).to(self.device)
            next_obs_fake = torch.from_numpy(batch_fake.obs_next).to(self.device)
        except AttributeError:
            obs_real = torch.from_numpy(batch_real["observations"]).to(self.device)
            actions_real = torch.from_numpy(batch_real["actions"]).to(self.device)
            next_obs_real = torch.from_numpy(batch_real["next_observations"]).to(self.device)

            obs_fake = torch.from_numpy(batch_fake["observations"]).to(self.device)
            actions_fake = torch.from_numpy(batch_fake["actions"]).to(self.device)
            next_obs_fake = torch.from_numpy(batch_fake["next_observations"]).to(self.device)

        logic_real = self.forward(obs_real, actions_real, next_obs_real)
        logic_fake = self.forward(obs_fake, actions_fake, next_obs_fake)

        ones_real = torch.ones_like(logic_real)
        zeros_fake = torch.zeros_like(logic_fake)

        inputs_pre = torch.cat([logic_real, logic_fake], dim=0)
        inputs_tar = torch.cat([ones_real, zeros_fake], dim=0)

        disc_loss = F.binary_cross_entropy_with_logits(inputs_pre, inputs_tar)

        if ratio:
            return disc_loss, (F.sigmoid(logic_fake) > 0.5).float().mean()

        return disc_loss

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, "discriminator.pt"))
        
    def load(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "discriminator.pt"), map_location=self.device))

class FMetric(nn.Module):
    def __init__(self, args):
        self.f_backbone = MLP(in_features = args["obs_shape"] + args["action_shape"], 
                     out_features = 1, 
                     hidden_features = args["hidden_features"], 
                     hidden_layers = args["hidden_features"], 
                     hidden_activation = "tanh", 
                     output_activation = "identity").to(args["device"])
        self.f_optim = torch.optim.Adam(self.f_backbone.parameters(), lr=args["f_lr"])
        self.f_bound = torch.nn.Parameter(torch.FloatTensor([10]), requires_grad=False)
        self.device = args["device"]
        
    def get_value(self, state, action):
        f_value = self.f_backbone(torch.cat([state, action], dim=-1))
        f_value = torch.tanh(f_value)
        return f_value * self.f_bound

    def train(self, batch_train, batch_target):
        target_state, target_act, target_valid = batch_target["observations"], batch_target["actions"], batch_target["valid"]
        target_valid = target_valid.squeeze().astype(bool)
        target_state = torch.from_numpy(target_state[target_valid]).to(self.device)

        train_state, train_act, train_valid = batch_train["observations"], batch_train["actions"], batch_train["valid"]
        train_valid = train_valid.squeeze().astype(bool)
        train_state = torch.from_numpy(train_state[train_valid]).to(self.device)
        train_act = torch.from_numpy(train_act[train_valid]).to(self.device)

        with torch.no_grad():
            fvalue_target = self.get_value(target_state, target_act)
        fvalue_train = self.get_value(train_state, train_act)
        f_loss = - F.mse_loss(fvalue_train, fvalue_target) #\
                 # + ((fvalue_train > 10).detach().float() * fvalue_train -
                 #    (fvalue_train < -10).detach().float() * fvalue_train).mean()
        self.f_optim.zero_grad()
        f_loss.backward()
        self.f_optim.step()

        return {
            "f_loss": f_loss.item(), 
            "f_clip_rate": ((fvalue_train >= 10) | (fvalue_train <= -10)).float().mean().item(), 
            "f_mean": fvalue_target.mean().item(), 
            "f_var": fvalue_target.std().item(), 
            "f_abs_mean": torch.abs(fvalue_target).mean().item()
        }

    def save(self, save_path):
        assert save_path, "save path cannot be None!"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.f_backbone.state_dict(), os.path.join(save_path, "f_backbone.pt"))

    def load(self, load_path):
        assert load_path, "load path cannot be None!"
        self.f_backbone.load_state_dict(torch.load(os.path.join(load_path, "f_backbone.pt")))
        