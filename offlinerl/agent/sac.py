from email import policy
from json import load
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from collections import OrderedDict
import os

from offlinerl.utils.env import get_env
from offlinerl.utils.net.recurrent import RecurrentGRU, TransformerEnc
from offlinerl.utils.net.maple_net import GaussianOutputHead, ValueHead, GaussianPolicyNetwork, ValueNetwork, MLPDecoder
from offlinerl.utils.simulator import save_env, reset_env, update_config_files
from offlinerl.utils.net.transformer import Transformer

import gym.spaces.discrete as discrete


def ce_loss(logits, targets, mask):
    mask = mask.reshape(-1)
    cls_dim = logits.shape[-1]
    loss = F.cross_entropy(logits.reshape(-1, cls_dim), targets.reshape(-1), reduction='none')
    loss = (loss * mask).sum() / mask.sum()
    return loss


class SACAgent(nn.Module):
    def __init__(self, args):
        super(SACAgent, self).__init__()
        self.args = args
        self.actor = GaussianPolicyNetwork(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            policy_hidden_dims=args["policy_hidden_dims"]
        ).to(args["device"])

        self.q1 = ValueNetwork(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            value_hidden_dims=args["value_hidden_dims"]
        ).to(args["device"])

        self.q2 = ValueNetwork(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            value_hidden_dims=args["value_hidden_dims"]
        ).to(args["device"])

        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=args["device"])
        self.log_alpha = nn.Parameter(self.log_alpha, requires_grad=True)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args["actor_lr"])
        self.critic_optim = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=args["critic_lr"])
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args["actor_lr"])

        # self.rew_max = args["rew_max"]
        # self.rew_min = args["rew_min"]
        self.discount = args["discount"]
        self.device = args["device"]
        self.logger = args["logger"]

    def evaluate_action(self, state, action):
        mu, log_std = self.actor(state)
        # action_prev_tanh = torch.atanh(action)
        # action_prev_tanh = torch.log((1+action+1e-5) / (1-action+1e-5)) / 2

        action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        action_prev_tanh = 0.5 * (action.log1p() - (-action).log1p())
        log_prob = torch.distributions.Normal(mu, log_std.exp()).log_prob(action_prev_tanh + 1e-6).sum(dim=-1)
        log_prob -= torch.sum(2 * (np.log(2) - action_prev_tanh - torch.nn.functional.softplus(-2 * action_prev_tanh)),
                              dim=-1)
        return log_prob

    def get_action(self, state, deterministic=False, out_mean_std=False):
        action, log_prob, mu, logstd = self.actor.sample(state, deterministic=deterministic)
        return [action, log_prob] if not out_mean_std else [action, log_prob, mu, logstd]

    def reset(self):
        return None

    def train_policy(self, batch, behavior_cloning=False):
        # changed data interface, yinh
        # rewards = torch.from_numpy(batch.rew).to(self.device)
        # terminals = torch.from_numpy(batch.done).to(self.device)
        # obs = torch.from_numpy(batch.obs).to(self.device)
        # actions = torch.from_numpy(batch.act).to(self.device)
        # next_obs = torch.from_numpy(batch.obs_next).to(self.device)

        try:
            rewards = torch.from_numpy(batch.rew).to(self.device)
            terminals = torch.from_numpy(batch.done).to(self.device)
            obs = torch.from_numpy(batch.obs).to(self.device)
            actions = torch.from_numpy(batch.act).to(self.device)
            next_obs = torch.from_numpy(batch.obs_next).to(self.device)
        except AttributeError:
            rewards = torch.from_numpy(batch["rewards"]).to(self.device)
            terminals = torch.from_numpy(batch["terminals"]).to(self.device)
            obs = torch.from_numpy(batch["observations"]).to(self.device)
            actions = torch.from_numpy(batch["actions"]).to(self.device)
            next_obs = torch.from_numpy(batch["next_observations"]).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob, _, __ = self.actor.sample(next_obs)
            next_q1 = self.target_q1(next_obs, next_action)
            next_q2 = self.target_q2(next_obs, next_action)
            next_q = torch.min(next_q1, next_q2) - self.log_alpha.exp() * torch.unsqueeze(next_log_prob, dim=-1)
            q_target = rewards + self.discount * (1. - terminals) * next_q
            # if self.args["q_target_clip"]:
            #     q_target = torch.clip(
            #         q_target,
            #         self.args.rew_min / (1-self.discount),
            #         self.args.rew_max / (1-self.discount)
            #     )
        q1 = self.q1(obs, actions)
        q2 = self.q2(obs, actions)
        q_loss = torch.nn.functional.mse_loss(q1, q_target) + torch.nn.functional.mse_loss(q2, q_target)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)

        new_actions, new_log_prob, _, __ = self.actor.sample(obs)
        new_log_prob = torch.unsqueeze(new_log_prob, dim=-1)
        new_q1 = self.q1(obs, new_actions)
        new_q2 = self.q2(obs, new_actions)
        new_q = torch.min(new_q1, new_q2)

        if self.args["learnable_alpha"]:
            alpha_loss = - (self.log_alpha * (new_log_prob + \
                                              self.args['target_entropy']).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        if behavior_cloning:
            sac_loss = (self.log_alpha.exp().detach() * new_log_prob - new_q).mean()
            sac_loss_scaled = 2.5 * sac_loss / new_q.detach().mean()
            bc_loss = (actions - new_actions) ** 2
            bc_loss = bc_loss.sum(-1).mean()
            policy_loss = sac_loss_scaled + bc_loss * self.args["BC"]["bc_loss_coeff"]
        else:
            sac_loss = (self.log_alpha.exp().detach() * new_log_prob - new_q).mean()
            policy_loss = sac_loss

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        ret = dict()
        ret["min_q"] = new_q.detach().cpu().mean().numpy()
        ret["q_loss"] = q_loss.detach().cpu().numpy()
        ret["sac_loss"] = sac_loss.detach().cpu().numpy()
        if self.args["learnable_alpha"]:
            ret["alpha_loss"] = alpha_loss.detach().cpu().numpy()
        if behavior_cloning:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
            ret["sac_loss"] = sac_loss.detach().cpu().numpy()
            ret["bc_loss"] = bc_loss.detach().cpu().numpy()
        else:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
        return ret

    def _soft_update(self, net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

    def eval_on_real_env(self):
        env = get_env(self.args["task"])
        results = ([self.test_one_trail() for _ in range(self.args["eval_runs"])])
        rewards = [result[0] for result in results]
        episode_lengths = [result[1] for result in results]
        rew_mean = np.mean(rewards)
        len_mean = np.mean(episode_lengths)

        res = OrderedDict()
        res["Reward_Mean_Env"] = rew_mean
        try:
            res["Score"] = env.get_normalized_score(rew_mean)
        except:
            print("no data")
        res["Length_Mean_Env"] = len_mean

        return res

    def test_one_trail(self):
        rewards = lengths = 0
        env = get_env(self.args["task"])
        with torch.no_grad():
            state, done = env.reset(), False
            while not done:
                state = torch.from_numpy(state[None, :]).float().to(self.device)
                action, _ = self.get_action(state, deterministic=True)
                action_use = torch.squeeze(action).cpu().numpy()
                if type(env.action_space) == discrete.Discrete:
                    action_use = np.argmax(action_use)
                state, reward, done, _ = env.step(action_use)
                rewards += reward
                lengths += 1

        return (rewards, lengths)

    def save(self, save_path):
        assert save_path, "save path cannot be None!"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.actor.state_dict(), os.path.join(save_path, "actor.pt"))
        torch.save(self.q1.state_dict(), os.path.join(save_path, "q1.pt"))
        torch.save(self.q2.state_dict(), os.path.join(save_path, "q2.pt"))
        torch.save(self.log_alpha.data, os.path.join(save_path, "log_alpha.pt"))
        torch.save(self.target_q1.state_dict(), os.path.join(save_path, "target_q1.pt"))
        torch.save(self.target_q2.state_dict(), os.path.join(save_path, "target_q2.pt"))
        torch.save(self.actor_optim.state_dict(), os.path.join(save_path, "actor_optim.pt"))
        torch.save(self.critic_optim.state_dict(), os.path.join(save_path, "critic_optim.pt"))
        torch.save(self.alpha_optim.state_dict(), os.path.join(save_path, "alpha_optim.pt"))

    def load(self, load_path):
        assert load_path, "load path cannot be None!"
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "actor.pt"), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(load_path, "q1.pt"), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(load_path, "q2.pt"), map_location=self.device))
        self.target_q1.load_state_dict(torch.load(os.path.join(load_path, "target_q1.pt"), map_location=self.device))
        self.target_q2.load_state_dict(torch.load(os.path.join(load_path, "target_q2.pt"), map_location=self.device))
        self.log_alpha.data = torch.load(os.path.join(load_path, "log_alpha.pt"), map_location=self.device)
        self.actor_optim.load_state_dict(
            torch.load(os.path.join(load_path, "actor_optim.pt"), map_location=self.device))
        self.critic_optim.load_state_dict(
            torch.load(os.path.join(load_path, "critic_optim.pt"), map_location=self.device))
        self.alpha_optim.load_state_dict(
            torch.load(os.path.join(load_path, "alpha_optim.pt"), map_location=self.device))

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.data,
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.target_q1.load_state_dict(state_dict["target_q1"])
        self.target_q2.load_state_dict(state_dict["target_q2"])
        self.log_alpha.data = state_dict["log_alpha"]
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.alpha_optim.load_state_dict(state_dict["alpha_optim"])


class RNNSACAgent(nn.Module):
    def __init__(self, args):
        super(RNNSACAgent, self).__init__()
        self.args = args
        self.policy_gru = RecurrentGRU(
            input_dim=args["obs_shape"] + args["action_shape"],
            device=args["device"],
            rnn_hidden_dim=args["rnn_hidden_dim"],
            rnn_layer_num=args["rnn_layer_num"]
        ).to(args["device"])
        self.value_gru = RecurrentGRU(
            input_dim=args["obs_shape"] + args["action_shape"],
            device=args["device"],
            rnn_hidden_dim=args["rnn_hidden_dim"],
            rnn_layer_num=args["rnn_layer_num"]
        ).to(args["device"])
        self.policy_decoder = MLPDecoder(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["rnn_hidden_dim"], decoder_hidden_dims=args["decoder_hidden_dims"],
        ).to(args["device"])
        self.value_decoder = MLPDecoder(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["rnn_hidden_dim"], decoder_hidden_dims=args["decoder_hidden_dims"],
        ).to(args["device"])

        self.actor = GaussianOutputHead(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["decoder_hidden_dims"][-1], decoder_hidden_dims=args["decoder_hidden_dims"],
            head_hidden_dims=args["head_hidden_dims"]
        ).to(args["device"])
        self.q1 = ValueHead(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["decoder_hidden_dims"][-1], decoder_hidden_dims=args["decoder_hidden_dims"],
            head_hidden_dims=args["head_hidden_dims"]
        ).to(args["device"])
        self.q2 = ValueHead(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["decoder_hidden_dims"][-1], decoder_hidden_dims=args["decoder_hidden_dims"],
            head_hidden_dims=args["head_hidden_dims"]
        ).to(args["device"])

        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=args["device"])
        self.log_alpha = nn.Parameter(self.log_alpha, requires_grad=True)

        self.alpha_cql = args['Meta']['coef_q_conservative']

        if self.args.coupled or not self.args['use_contrastive']:
            self.actor_optim = torch.optim.Adam(
                [*self.policy_gru.parameters(), *self.policy_decoder.parameters(), *self.actor.parameters()],
                lr=args["actor_lr"])
            self.critic_optim = torch.optim.Adam(
                [*self.value_gru.parameters(), *self.value_decoder.parameters(), *self.q1.parameters(),
                 *self.q2.parameters()], lr=args["critic_lr"])
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args["actor_lr"])
        else:
            self.actor_enc_optim = torch.optim.Adam([*self.policy_gru.parameters(), *self.policy_decoder.parameters()],
                                                    lr=args["actor_lr"])
            self.critic_enc_optim = torch.optim.Adam([*self.value_gru.parameters(), *self.value_decoder.parameters()],
                                                     lr=args["critic_lr"])
            self.actor_optim = torch.optim.Adam([*self.actor.parameters()], lr=args["actor_lr"])
            self.critic_optim = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=args["critic_lr"])
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args["actor_lr"])

        # self.rew_max = args["rew_max"]
        # self.rew_min = args["rew_min"]
        self.discount = args["discount"]

        self.device = args["device"]
        self.logger = args["logger"]

    def get_action(self, state, lst_action, hidden, deterministic=False, out_mean_std=False):
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 1)
        if len(lst_action.shape) == 2:
            lst_action = torch.unsqueeze(lst_action, 1)
        if len(hidden.shape) == 2:
            hidden = torch.unsqueeze(hidden, 0)

        # only take the first element
        lens = [1] * state.shape[0]
        rnn_input_pair = torch.cat([state, lst_action], dim=-1)
        policy_embedding, next_hidden = self.policy_gru(rnn_input_pair, lens, hidden)
        action, log_prob, mu, logstd = self.actor.sample(state, self.policy_decoder(policy_embedding),
                                                         deterministic=deterministic)

        action, log_prob, mu, logstd = torch.squeeze(action, 1), torch.squeeze(log_prob,
                                                                               1) if log_prob is not None else None, torch.squeeze(
            mu, 1), torch.squeeze(logstd, 1)

        ret = []
        ret += [action,
                log_prob]  # here we don't need to worry about deterministic, cause if deterministic, then actor.sample will return torch.tanh(mu) and None as action and log_prob
        ret += ([mu, logstd] if out_mean_std else [])
        ret += [next_hidden]
        return ret

    def reset(self):
        return None

    def get_value(self, state, action, lst_action, hidden):
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, dim=1)
        if len(action.shape) == 2:
            action = torch.unsqueeze(action, dim=1)
        if len(lst_action.shape) == 2:
            lst_action = torch.unsqueeze(lst_action, dim=1)
        if len(hidden.shape) == 2:
            hidden = torch.unsqueeze(hidden, dim=0)

        lens = [1] * state.shape[0]
        rnn_input_pair = torch.cat([state, lst_action], dim=-1)
        value_embedding, next_hidden = self.value_gru(rnn_input_pair, lens, hidden)
        value_q1 = self.q1(state, action, self.value_decoder(value_embedding))
        value_q2 = self.q2(state, action, self.value_decoder(value_embedding))
        value_min = torch.min(value_q1, value_q2)

        value_min = torch.squeeze(value_min, 1)
        return value_min, next_hidden

    def estimate_log_sum_exp_q(self, obs, next_obs, policy_embedding, value_embedding, N, q_functions):
        batch_size = obs.shape[0]
        seq_length = obs.shape[1]
        obs_rep = obs.unsqueeze(2).repeat(1, 1, N, 1)
        next_obs_rep = next_obs.unsqueeze(2).repeat(1, 1, N, 1)
        policy_embedding = policy_embedding.unsqueeze(2).repeat(1, 1, N, 1)
        value_embedding = value_embedding.unsqueeze(2).repeat(1, 1, N, 1)

        # draw actions at uniform
        random_actions_q1 = torch.FloatTensor(batch_size, seq_length, N, self.args['action_shape']).uniform_(-1, 1).to(
            self.device)
        random_density_q1 = np.log(0.5 ** random_actions_q1.shape[-1])

        # draw actions from current policy
        with torch.no_grad():
            actions_q1, log_prob_q1, _, _ = self.actor.sample(obs_rep, policy_embedding)
            next_actions_q1, next_log_prob_q1, _, _ = self.actor.sample(next_obs_rep, policy_embedding)

        random_q1 = q_functions(obs_rep, random_actions_q1, value_embedding)
        policy_q1 = q_functions(obs_rep, actions_q1, value_embedding)
        next_q1 = q_functions(next_obs_rep, next_actions_q1, value_embedding)

        assert random_q1.shape == policy_q1.shape == next_q1.shape

        cat_q1 = torch.cat([random_q1 - random_density_q1, policy_q1 - log_prob_q1.unsqueeze(-1),
                            next_q1 - next_log_prob_q1.unsqueeze(-1)], dim=2)
        log_sum_exp_q1 = torch.logsumexp(cat_q1, dim=2)

        return log_sum_exp_q1

    def map_hidden(self, policy_hidden=None, value_hidden=None):
        if policy_hidden is not None:
            policy_hidden = self.policy_decoder(policy_hidden)
        if value_hidden is not None:
            value_hidden = self.value_decoder(value_hidden)

        return policy_hidden, value_hidden

    def estimate_hidden(self, batch, sac_embedding_infer="concat", reduction="mean"):
        torch.cuda.empty_cache()
        # we use 3D slices to train the policy
        batch_size, seq_len, _ = batch["observations"].shape

        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)

        if sac_embedding_infer == "concat":
            value_hidden = batch["value_hidden"][:, 0]
            policy_hidden = batch["policy_hidden"][:, 0]
            value_embedding, value_hidden_next = self.value_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=value_hidden, sequential=True)
            policy_embedding, policy_hidden_next = self.policy_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=policy_hidden, sequential=True)

        elif sac_embedding_infer == "direct":
            _observations = torch.cat([batch["observations"], batch["next_observations"][:, -1:]], dim=1)
            _last_actions = torch.cat([batch["last_actions"], batch["actions"][:, -1:]], dim=1)
            _value_embedding, _ = self.value_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                 pre_hidden=batch["value_hidden"][:, 0], sequential=True)
            _policy_embedding, _ = self.policy_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                   pre_hidden=batch["policy_hidden"][:, 0], sequential=True)
            value_embedding = _value_embedding[:, :-1]
            policy_embedding = _policy_embedding[:, :-1]

        policy_embedding = self.policy_decoder(policy_embedding)  # (bs, seq, dim)
        value_embedding = self.value_decoder(value_embedding)  # (bs, seq, dim)

        if reduction == 'sum':
            policy_embedding = policy_embedding.reshape(batch_size * seq_len, -1)
            value_embedding = value_embedding.reshape(batch_size * seq_len, -1)
            mask = batch['valid'].reshape(batch_size * seq_len, -1)

            policy_embedding = (policy_embedding * mask).sum(dim=0, keepdim=True)
            value_embedding = (value_embedding * mask).sum(dim=0, keepdim=True)
        elif reduction == 'mean':
            policy_embedding = policy_embedding.reshape(batch_size * seq_len, -1)
            value_embedding = value_embedding.reshape(batch_size * seq_len, -1)
            mask = batch['valid'].reshape(batch_size * seq_len, -1)

            policy_embedding = (policy_embedding * mask).sum(dim=0, keepdim=True) / mask.sum()
            value_embedding = (value_embedding * mask).sum(dim=0, keepdim=True) / mask.sum()

        return policy_embedding, value_embedding

    def train_policy_meta(self, batch, behavior_cloning=False, sac_embedding_infer="concat",
                          real_data_only_with_mb=False):
        torch.cuda.empty_cache()
        # we use 3D slices to train the policy
        task_num, batch_size, seq_len, _ = batch["observations"].shape
        for k in batch:
            batch[k] = batch[k].reshape(task_num * batch_size, seq_len, -1)

        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)

        if sac_embedding_infer == "concat":
            value_hidden = batch["value_hidden"][:, 0]
            policy_hidden = batch["policy_hidden"][:, 0]
            value_embedding, value_hidden_next = self.value_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=value_hidden, sequential=True)
            policy_embedding, policy_hidden_next = self.policy_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=policy_hidden, sequential=True)

            lens_next = torch.ones(len(lens)).int()
            value_embedding_next, _ = self.value_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=value_hidden_next, sequential=True)
            policy_embedding_next, _ = self.policy_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=policy_hidden_next, sequential=True)
            value_embedding_next = torch.cat([value_embedding[:, 1:], value_embedding_next], dim=1)
            policy_embedding_next = torch.cat([policy_embedding[:, 1:], policy_embedding_next], dim=1)
        elif sac_embedding_infer == "direct":
            _observations = torch.cat([batch["observations"], batch["next_observations"][:, -1:]], dim=1)
            _last_actions = torch.cat([batch["last_actions"], batch["actions"][:, -1:]], dim=1)
            _value_embedding, _ = self.value_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                 pre_hidden=batch["value_hidden"][:, 0], sequential=True)
            _policy_embedding, _ = self.policy_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                   pre_hidden=batch["policy_hidden"][:, 0], sequential=True)
            value_embedding = _value_embedding[:, :-1]
            policy_embedding = _policy_embedding[:, :-1]
            value_embedding_next = _value_embedding[:, 1:]
            policy_embedding_next = _policy_embedding[:, 1:]

        policy_embedding = self.policy_decoder(policy_embedding)
        value_embedding = self.value_decoder(value_embedding)
        policy_embedding_meta = policy_embedding.reshape(task_num, batch_size, max_len, -1)
        # policy_embedding_meta = policy_embedding_meta / torch.sqrt((policy_embedding_meta ** 2).mean(dim=-1, keepdim=True)).detach()
        value_embedding_meta = value_embedding.reshape(task_num, batch_size, max_len, -1)
        # value_embedding_meta = value_embedding_meta / torch.sqrt((value_embedding_meta ** 2).mean(dim=-1, keepdim=True)).detach()

        contrastive_mask = batch['valid'].reshape(task_num, batch_size, max_len, -1).transpose(0, 1).transpose(1,
                                                                                                               2).reshape(
            batch_size * max_len, task_num, 1)

        # batch * len, task, task   contras policy
        policy_embedding_left = policy_embedding_meta.transpose(0, 1).transpose(1, 2).reshape(batch_size * max_len,
                                                                                              task_num, -1)
        policy_embedding_right = policy_embedding_meta.transpose(0, 1).transpose(1, 2).transpose(2, 3).reshape(
            batch_size * max_len, -1, task_num)
        index = np.arange(batch_size * max_len)
        np.random.shuffle(index)
        policy_embedding_right = policy_embedding_right[index, :, :]
        contrastive_mask_ = contrastive_mask[index, :, :]

        policy_inner_prod = torch.bmm(policy_embedding_left, policy_embedding_right)
        policy_inner_prod_ = torch.bmm(policy_embedding_right.transpose(1, 2), policy_embedding_left.transpose(1, 2))
        policy_label = torch.arange(task_num, device=policy_inner_prod.device).unsqueeze(0).repeat(max_len * batch_size,
                                                                                                   1)

        policy_contrastive_loss = ce_loss(policy_inner_prod, policy_label, contrastive_mask) + ce_loss(
            policy_inner_prod_, policy_label, contrastive_mask_)

        # batch * len, task, task   contras value
        value_embedding_left = value_embedding_meta.transpose(0, 1).transpose(1, 2).reshape(batch_size * max_len,
                                                                                            task_num, -1)
        value_embedding_right = value_embedding_meta.transpose(0, 1).transpose(1, 2).transpose(2, 3).reshape(
            batch_size * max_len, -1, task_num)
        index = np.arange(batch_size * max_len)
        np.random.shuffle(index)
        value_embedding_right = value_embedding_right[index, :, :]
        contrastive_mask_ = contrastive_mask[index, :, :]

        value_inner_prod = torch.bmm(value_embedding_left, value_embedding_right)
        value_inner_prod_ = torch.bmm(value_embedding_right.transpose(1, 2), value_embedding_left.transpose(1, 2))
        value_label = torch.arange(task_num, device=value_inner_prod.device).unsqueeze(0).repeat(max_len * batch_size,
                                                                                                 1)

        value_contrastive_loss = ce_loss(value_inner_prod, value_label, contrastive_mask) + ce_loss(value_inner_prod_,
                                                                                                    value_label,
                                                                                                    contrastive_mask_)

        if not self.args.coupled:
            self.actor_enc_optim.zero_grad()
            (self.args["Meta"]["coef_policy_contrastive"] * policy_contrastive_loss).backward()
            self.actor_enc_optim.step()

            self.critic_enc_optim.zero_grad()
            (self.args["Meta"]["coef_value_contrastive"] * value_contrastive_loss).backward()
            self.critic_enc_optim.step()

            policy_embedding = policy_embedding.detach()
            value_embedding = value_embedding.detach()

        with torch.no_grad():
            value_embedding_next = self.value_decoder(value_embedding_next)
            policy_embedding_next = self.policy_decoder(policy_embedding_next)

        if real_data_only_with_mb:
            for k in batch:
                batch[k] = batch[k].reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(task_num // 2 * batch_size,
                                                                                             seq_len, -1)

            policy_embedding = policy_embedding.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
                task_num // 2 * batch_size, seq_len, -1)
            value_embedding = value_embedding.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
                task_num // 2 * batch_size, seq_len, -1)

            policy_embedding_next = policy_embedding_next.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
                task_num // 2 * batch_size, seq_len, -1)
            value_embedding_next = value_embedding_next.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
                task_num // 2 * batch_size, seq_len, -1)

        with torch.no_grad():

            action_target, log_prob_target, mu_target, logstd_target = self.actor.sample(batch["next_observations"],
                                                                                         policy_embedding_next)
            q1_target = self.target_q1(batch["next_observations"], action_target, value_embedding_next)
            q2_target = self.target_q2(batch["next_observations"], action_target, value_embedding_next)
            q_target = torch.min(q1_target, q2_target)
            q_target = q_target - self.log_alpha.exp() * torch.unsqueeze(log_prob_target, dim=-1)
            q_target = batch["rewards"] + self.discount * (~batch["terminals"]) * (q_target)
            if self.args["q_target_clip"]:
                q_target = torch.clip(q_target,
                                      self.args.rew_min / (1 - self.discount),
                                      self.args.rew_max / (1 - self.discount)
                                      )

        # update critic
        q1 = self.q1(batch["observations"], batch["actions"], value_embedding)
        q2 = self.q2(batch["observations"], batch["actions"], value_embedding)
        valid_num = torch.sum(batch["valid"])

        q1_loss = torch.sum(((q1 - q_target) ** 2) * batch['valid']) / valid_num
        q2_loss = torch.sum(((q2 - q_target) ** 2) * batch['valid']) / valid_num

        if self.args.q_conservative:
            log_sum_exp_q1 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
                                                         policy_embedding, value_embedding, N=10, q_functions=self.q1)
            log_sum_exp_q2 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
                                                         policy_embedding, value_embedding, N=10, q_functions=self.q2)
            q1_loss += self.alpha_cql * ((log_sum_exp_q1 - q1) * batch["valid"]).sum() / batch["valid"].sum()
            q2_loss += self.alpha_cql * ((log_sum_exp_q2 - q2) * batch["valid"]).sum() / batch["valid"].sum()

        td_loss = (q1_loss + q2_loss)

        if self.args.coupled:
            q_loss = td_loss + self.args["Meta"]["coef_value_contrastive"] * value_contrastive_loss
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()
        else:
            q_loss = td_loss
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

        # q_loss = td_loss + self.args["Meta"]["coef_value_contrastive"] * value_contrastive_loss

        self._soft_update(self.target_q1, self.q1, soft_target_tau=self.args["soft_target_tau"])
        self._soft_update(self.target_q2, self.q2, soft_target_tau=self.args["soft_target_tau"])

        # update alpha and actor
        actions, log_prob, mu, logstd = self.actor.sample(batch["observations"], policy_embedding)
        log_prob = log_prob.unsqueeze(dim=-1)  # (B, T, 1)

        if self.args["learnable_alpha"]:
            alpha_loss = - torch.sum(self.log_alpha * ((log_prob + \
                                                        self.args['target_entropy']) * batch[
                                                           'valid']).detach()) / valid_num
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        q1_ = self.q1(batch["observations"], actions, value_embedding.detach())
        q2_ = self.q2(batch["observations"], actions, value_embedding.detach())
        min_q_ = torch.min(q1_, q2_)

        if behavior_cloning:
            sac_loss = self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            sac_loss_scaled = 2.5 * sac_loss / min_q_.detach().mean()
            bc_loss = (batch["actions"] - actions) ** 2
            bc_loss = torch.sum(bc_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss_scaled + bc_loss * self.args["BC"]["bc_loss_coeff"]
        else:
            sac_loss = self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss

        # policy_loss += self.args["Meta"]["coef_policy_contrastive"] * policy_contrastive_loss

        if self.args.coupled:
            policy_loss += self.args["Meta"]["coef_policy_contrastive"] * policy_contrastive_loss
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
        else:
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

        ret = dict()
        ret["min_q"] = min_q_.detach().cpu().mean().numpy()
        ret["q_loss"] = q_loss.detach().cpu().numpy()
        ret["td_loss"] = td_loss.detach().cpu().numpy()
        if self.args["learnable_alpha"]:
            ret["alpha_loss"] = alpha_loss.detach().cpu().numpy()
        if behavior_cloning:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
            ret["sac_loss"] = sac_loss.detach().cpu().numpy()
            ret["bc_loss"] = bc_loss.detach().cpu().numpy()
            ret["value_con"] = value_contrastive_loss.detach().cpu().numpy()
            ret["policy_con"] = policy_contrastive_loss.detach().cpu().numpy()
        else:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
            ret["value_con"] = value_contrastive_loss.detach().cpu().numpy()
            ret["policy_con"] = policy_contrastive_loss.detach().cpu().numpy()
        return ret

    def train_policy(self, batch, behavior_cloning=False, sac_embedding_infer="concat"):
        # we use 3D slices to train the policy
        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)

        if sac_embedding_infer == "concat":
            value_hidden = batch["value_hidden"][:, 0]
            policy_hidden = batch["policy_hidden"][:, 0]
            value_embedding, value_hidden_next = self.value_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=value_hidden, sequential=True)
            policy_embedding, policy_hidden_next = self.policy_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=policy_hidden, sequential=True)

            lens_next = torch.ones(len(lens)).int()
            value_embedding_next, _ = self.value_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=value_hidden_next, sequential=True)
            policy_embedding_next, _ = self.policy_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=policy_hidden_next, sequential=True)
            value_embedding_next = torch.cat([value_embedding[:, 1:], value_embedding_next], dim=1)
            policy_embedding_next = torch.cat([policy_embedding[:, 1:], policy_embedding_next], dim=1)
        elif sac_embedding_infer == "direct":
            _observations = torch.cat([batch["observations"], batch["next_observations"][:, -1:]], dim=1)
            _last_actions = torch.cat([batch["last_actions"], batch["actions"][:, -1:]], dim=1)
            _value_embedding, _ = self.value_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                 pre_hidden=batch["value_hidden"][:, 0], sequential=True)
            _policy_embedding, _ = self.policy_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                   pre_hidden=batch["policy_hidden"][:, 0], sequential=True)
            value_embedding = _value_embedding[:, :-1]
            policy_embedding = _policy_embedding[:, :-1]
            value_embedding_next = _value_embedding[:, 1:]
            policy_embedding_next = _policy_embedding[:, 1:]

        with torch.no_grad():
            policy_embedding_next = self.policy_decoder(policy_embedding_next)
            value_embedding_next = self.value_decoder(value_embedding_next)

            action_target, log_prob_target, mu_target, logstd_target = self.actor.sample(batch["next_observations"],
                                                                                         policy_embedding_next)
            q1_target = self.target_q1(batch["next_observations"], action_target, value_embedding_next)
            q2_target = self.target_q2(batch["next_observations"], action_target, value_embedding_next)
            q_target = torch.min(q1_target, q2_target)
            q_target = q_target - self.log_alpha.exp() * torch.unsqueeze(log_prob_target, dim=-1)
            q_target = batch["rewards"] + self.discount * (~batch["terminals"]) * (q_target)
            # if self.args["q_target_clip"]:
            #     q_target = torch.clip(q_target,
            #                         self.rew_min / (1-self.discount),
            #                         self.rew_max / (1-self.discount)
            #     )
        policy_embedding = self.policy_decoder(policy_embedding)
        value_embedding = self.value_decoder(value_embedding)

        # update critic
        q1 = self.q1(batch["observations"], batch["actions"], value_embedding)
        q2 = self.q2(batch["observations"], batch["actions"], value_embedding)
        valid_num = torch.sum(batch["valid"])

        q1_loss = torch.sum(((q1 - q_target) ** 2) * batch['valid']) / valid_num
        q2_loss = torch.sum(((q2 - q_target) ** 2) * batch['valid']) / valid_num

        if self.args.q_conservative:
            log_sum_exp_q1 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
                                                         policy_embedding, value_embedding, N=10, q_functions=self.q1)
            log_sum_exp_q2 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
                                                         policy_embedding, value_embedding, N=10, q_functions=self.q2)
            q1_loss += self.alpha_cql * ((log_sum_exp_q1 - q1) * batch["valid"]).sum() / batch["valid"].sum()
            q2_loss += self.alpha_cql * ((log_sum_exp_q2 - q2) * batch["valid"]).sum() / batch["valid"].sum()

        q_loss = (q1_loss + q2_loss)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self._soft_update(self.target_q1, self.q1, soft_target_tau=self.args["soft_target_tau"])
        self._soft_update(self.target_q2, self.q2, soft_target_tau=self.args["soft_target_tau"])

        # update alpha and actor
        actions, log_prob, mu, logstd = self.actor.sample(batch["observations"], policy_embedding)
        log_prob = log_prob.unsqueeze(dim=-1)  # (B, T, 1)

        if self.args["learnable_alpha"]:
            alpha_loss = - torch.sum(self.log_alpha * ((log_prob + \
                                                        self.args['target_entropy']) * batch[
                                                           'valid']).detach()) / valid_num
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        q1_ = self.q1(batch["observations"], actions, value_embedding.detach())
        q2_ = self.q2(batch["observations"], actions, value_embedding.detach())
        min_q_ = torch.min(q1_, q2_)

        if behavior_cloning:
            sac_loss = self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            sac_loss_scaled = 2.5 * sac_loss / min_q_.detach().mean()
            bc_loss = (batch["actions"] - actions) ** 2
            bc_loss = torch.sum(bc_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss_scaled + bc_loss * self.args["BC"]["bc_loss_coeff"]
        else:
            sac_loss = self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        ret = dict()
        ret["min_q"] = min_q_.detach().cpu().mean().numpy()
        ret["q_loss"] = q_loss.detach().cpu().numpy()
        if self.args["learnable_alpha"]:
            ret["alpha_loss"] = alpha_loss.detach().cpu().numpy()
        if behavior_cloning:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
            ret["sac_loss"] = sac_loss.detach().cpu().numpy()
            ret["bc_loss"] = bc_loss.detach().cpu().numpy()
        else:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
        return ret

    def _soft_update(self, net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

    def eval_policy(self, env):
        res = self.eval_on_real_env(env)
        return res

    def eval_on_real_env(self, env):
        # env = get_env(self.args["task"])
        results = ([self.test_one_trail(env) for _ in range(self.args["eval_runs"])])
        rewards = [result[0] for result in results]
        episode_lengths = [result[1] for result in results]
        rew_mean = np.mean(rewards)
        len_mean = np.mean(episode_lengths)

        res = OrderedDict()
        res["Reward_Mean_Env"] = rew_mean
        # try:
        #     res["Score"] = env.get_normalized_score(rew_mean)
        # except:
        #     print("no attr")
        # res["Length_Mean_Env"] = len_mean

        return res

    def test_one_trail(self, env):
        # env = get_env(self.args["task"])
        with torch.no_grad():
            state, done = env.reset(), False
            lst_action = torch.zeros((1, 1, self.args['action_shape'])).to(self.device)
            hidden_policy = torch.zeros((1, 1, self.args['rnn_hidden_dim'])).to(self.device)
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]  # 这里增加了数据的维度，当做batch为1在处理
                state = torch.from_numpy(state).float().to(self.device)
                action, _, hidden_policy = self.get_action(state, lst_action, hidden_policy, deterministic=True)
                assert _ is None
                use_action = action.cpu().numpy().reshape(-1)
                if type(env.action_space) == discrete.Discrete:
                    use_action = np.argmax(use_action)
                state_next, reward, done, _ = env.step(use_action)
                lst_action = action
                state = state_next
                # action = policy.get_action(state).reshape(-1)
                # state, reward, done, _ = env.step(action)
                rewards += reward
                lengths += 1
        return (rewards, lengths)

    def save(self, save_path):
        assert save_path, "save path cannot be None!"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.policy_decoder.state_dict(), os.path.join(save_path, "policy_dec.pt"))
        torch.save(self.value_decoder.state_dict(), os.path.join(save_path, "value_dec.pt"))

        torch.save(self.policy_gru.state_dict(), os.path.join(save_path, "policy_gru.pt"))
        torch.save(self.actor.state_dict(), os.path.join(save_path, "actor.pt"))
        torch.save(self.value_gru.state_dict(), os.path.join(save_path, "value_gru.pt"))
        torch.save(self.q1.state_dict(), os.path.join(save_path, "q1.pt"))
        torch.save(self.q2.state_dict(), os.path.join(save_path, "q2.pt"))
        torch.save(self.log_alpha.data, os.path.join(save_path, "log_alpha.pt"))
        torch.save(self.target_q1.state_dict(), os.path.join(save_path, "target_q1.pt"))
        torch.save(self.target_q2.state_dict(), os.path.join(save_path, "target_q2.pt"))
        torch.save(self.actor_optim.state_dict(), os.path.join(save_path, "actor_optim.pt"))
        torch.save(self.critic_optim.state_dict(), os.path.join(save_path, "critic_optim.pt"))
        torch.save(self.alpha_optim.state_dict(), os.path.join(save_path, "alpha_optim.pt"))
        if (not self.args.coupled) and self.args['use_contrastive']:
            torch.save(self.actor_enc_optim.state_dict(), os.path.join(save_path, "actor_enc_optim.pt"))
            torch.save(self.critic_enc_optim.state_dict(), os.path.join(save_path, "critic_enc_optim.pt"))

    def load(self, load_path):
        assert load_path, "load path cannot be None!"
        self.policy_decoder.load_state_dict(
            torch.load(os.path.join(load_path, "policy_dec.pt"), map_location=self.device))
        self.value_decoder.load_state_dict(
            torch.load(os.path.join(load_path, "value_dec.pt"), map_location=self.device))

        self.policy_gru.load_state_dict(torch.load(os.path.join(load_path, "policy_gru.pt"), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "actor.pt"), map_location=self.device))
        self.value_gru.load_state_dict(torch.load(os.path.join(load_path, "value_gru.pt"), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(load_path, "q1.pt"), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(load_path, "q2.pt"), map_location=self.device))
        self.target_q1.load_state_dict(torch.load(os.path.join(load_path, "target_q1.pt"), map_location=self.device))
        self.target_q2.load_state_dict(torch.load(os.path.join(load_path, "target_q2.pt"), map_location=self.device))
        self.log_alpha.data = torch.load(os.path.join(load_path, "log_alpha.pt"), map_location=self.device)
        self.actor_optim.load_state_dict(
            torch.load(os.path.join(load_path, "actor_optim.pt"), map_location=self.device))
        self.critic_optim.load_state_dict(
            torch.load(os.path.join(load_path, "critic_optim.pt"), map_location=self.device))
        self.alpha_optim.load_state_dict(
            torch.load(os.path.join(load_path, "alpha_optim.pt"), map_location=self.device))
        if (not self.args.coupled) and self.args['use_contrastive']:
            self.actor_enc_optim.load_state_dict(
                torch.load(os.path.join(load_path, "actor_enc_optim.pt"), map_location=self.device))
            self.critic_enc_optim.load_state_dict(
                torch.load(os.path.join(load_path, "critic_enc_optim.pt"), map_location=self.device))

    def state_dict(self):
        dicts = {
            "policy_gru": self.policy_gru.state_dict(),
            "value_gru": self.value_gru.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.data,
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }
        if (not self.args.coupled) and self.args['use_contrastive']:
            dicts['actor_enc_optim'] = self.actor_enc_optim.state_dict()
            dicts['critic_enc_optim'] = self.critic_enc_optim.state_dict()

        return dicts

    def load_state_dict(self, state_dict):
        self.policy_gru.load_state_dict(state_dict["policy_gru"])
        self.value_gru.load_state_dict(state_dict["value_gru"])
        self.actor.load_state_dict(state_dict["actor"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.target_q1.load_state_dict(state_dict["target_q1"])
        self.target_q2.load_state_dict(state_dict["target_q2"])
        self.log_alpha.data = state_dict["log_alpha"]
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
        if (not self.args.coupled) and self.args['use_contrastive']:
            self.actor_enc_optim.load_state_dict(state_dict["actor_enc_optim"])
            self.critic_enc_optim.load_state_dict(state_dict["critic_enc_optim"])


class TransformerSACAgent(nn.Module):
    def __init__(self, args):
        super(TransformerSACAgent, self).__init__()
        self.args = args
        self.policy_gru = TransformerEnc(
            input_dim=args["obs_shape"] + args["action_shape"],
            device=args["device"],
            rnn_hidden_dim=args["rnn_hidden_dim"],
            rnn_layer_num=args["rnn_layer_num"]
        ).to(args["device"])
        self.value_gru = TransformerEnc(
            input_dim=args["obs_shape"] + args["action_shape"],
            device=args["device"],
            rnn_hidden_dim=args["rnn_hidden_dim"],
            rnn_layer_num=args["rnn_layer_num"]
        ).to(args["device"])
        self.policy_decoder = MLPDecoder(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["rnn_hidden_dim"], decoder_hidden_dims=args["decoder_hidden_dims"],
        ).to(args["device"])
        self.value_decoder = MLPDecoder(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["rnn_hidden_dim"], decoder_hidden_dims=args["decoder_hidden_dims"],
        ).to(args["device"])

        self.actor = GaussianOutputHead(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["decoder_hidden_dims"][-1], decoder_hidden_dims=args["decoder_hidden_dims"],
            head_hidden_dims=args["head_hidden_dims"]
        ).to(args["device"])
        self.q1 = ValueHead(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["decoder_hidden_dims"][-1], decoder_hidden_dims=args["decoder_hidden_dims"],
            head_hidden_dims=args["head_hidden_dims"]
        ).to(args["device"])
        self.q2 = ValueHead(
            obs_dim=args["obs_shape"], action_dim=args["action_shape"],
            embedding_dim=args["decoder_hidden_dims"][-1], decoder_hidden_dims=args["decoder_hidden_dims"],
            head_hidden_dims=args["head_hidden_dims"]
        ).to(args["device"])

        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.target_q1.requires_grad_(False)
        self.target_q2.requires_grad_(False)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=args["device"])
        self.log_alpha = nn.Parameter(self.log_alpha, requires_grad=True)

        self.alpha_cql = args['Meta']['coef_q_conservative']

        if self.args.coupled or not self.args['use_contrastive']:
            self.actor_optim = torch.optim.Adam(
                [*self.policy_gru.parameters(), *self.policy_decoder.parameters(), *self.actor.parameters()],
                lr=args["actor_lr"])
            self.critic_optim = torch.optim.Adam(
                [*self.value_gru.parameters(), *self.value_decoder.parameters(), *self.q1.parameters(),
                 *self.q2.parameters()], lr=args["critic_lr"])
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args["actor_lr"])
        else:
            self.actor_enc_optim = torch.optim.Adam([*self.policy_gru.parameters(), *self.policy_decoder.parameters()],
                                                    lr=args["critic_lr"])
            self.critic_enc_optim = torch.optim.Adam([*self.value_gru.parameters(), *self.value_decoder.parameters()],
                                                     lr=args["critic_lr"])
            self.actor_optim = torch.optim.Adam([*self.actor.parameters()], lr=args["actor_lr"])
            self.critic_optim = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=args["critic_lr"])
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args["actor_lr"])

        # self.rew_max = args["rew_max"]
        # self.rew_min = args["rew_min"]
        self.discount = args["discount"]

        self.device = args["device"]
        self.logger = args["logger"]

    def get_action(self, state, lst_action, hidden, deterministic=False, out_mean_std=False):
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, 1)
        if len(lst_action.shape) == 2:
            lst_action = torch.unsqueeze(lst_action, 1)
        if len(hidden.shape) == 2:
            hidden = torch.unsqueeze(hidden, 0)

        # only take the first element
        lens = [1] * state.shape[0]
        rnn_input_pair = torch.cat([state, lst_action], dim=-1)
        # print(rnn_input_pair.shape)
        policy_embedding, next_hidden = self.policy_gru(rnn_input_pair, lens, hidden)
        action, log_prob, mu, logstd = self.actor.sample(state, self.policy_decoder(policy_embedding),
                                                         deterministic=deterministic)

        action, log_prob, mu, logstd = torch.squeeze(action, 1), torch.squeeze(log_prob,
                                                                               1) if log_prob is not None else None, torch.squeeze(
            mu, 1), torch.squeeze(logstd, 1)

        ret = []
        ret += [action,
                log_prob]  # here we don't need to worry about deterministic, cause if deterministic, then actor.sample will return torch.tanh(mu) and None as action and log_prob
        ret += ([mu, logstd] if out_mean_std else [])
        ret += [next_hidden]
        return ret

    def reset(self):
        return None

    def get_value(self, state, action, lst_action, hidden):
        if len(state.shape) == 2:
            state = torch.unsqueeze(state, dim=1)
        if len(action.shape) == 2:
            action = torch.unsqueeze(action, dim=1)
        if len(lst_action.shape) == 2:
            lst_action = torch.unsqueeze(lst_action, dim=1)
        if len(hidden.shape) == 2:
            hidden = torch.unsqueeze(hidden, dim=0)

        lens = [1] * state.shape[0]
        rnn_input_pair = torch.cat([state, lst_action], dim=-1)
        value_embedding, next_hidden = self.value_gru(rnn_input_pair, lens, hidden)
        value_q1 = self.q1(state, action, self.value_decoder(value_embedding))
        value_q2 = self.q2(state, action, self.value_decoder(value_embedding))
        value_min = torch.min(value_q1, value_q2)

        value_min = torch.squeeze(value_min, 1)
        return value_min, next_hidden

    def estimate_log_sum_exp_q(self, obs, next_obs, policy_embedding, value_embedding, N, q_functions):
        batch_size = obs.shape[0]
        seq_length = obs.shape[1]
        obs_rep = obs.unsqueeze(2).repeat(1, 1, N, 1)
        next_obs_rep = next_obs.unsqueeze(2).repeat(1, 1, N, 1)
        policy_embedding = policy_embedding.unsqueeze(2).repeat(1, 1, N, 1)
        value_embedding = value_embedding.unsqueeze(2).repeat(1, 1, N, 1)

        # draw actions at uniform
        random_actions_q1 = torch.FloatTensor(batch_size, seq_length, N, self.args['action_shape']).uniform_(-1, 1).to(
            self.device)
        random_density_q1 = np.log(0.5 ** random_actions_q1.shape[-1])

        # draw actions from current policy
        with torch.no_grad():
            actions_q1, log_prob_q1, _, _ = self.actor.sample(obs_rep, policy_embedding)
            next_actions_q1, next_log_prob_q1, _, _ = self.actor.sample(next_obs_rep, policy_embedding)

        random_q1 = q_functions(obs_rep, random_actions_q1, value_embedding)
        policy_q1 = q_functions(obs_rep, actions_q1, value_embedding)
        next_q1 = q_functions(next_obs_rep, next_actions_q1, value_embedding)

        assert random_q1.shape == policy_q1.shape == next_q1.shape

        cat_q1 = torch.cat([random_q1 - random_density_q1, policy_q1 - log_prob_q1.unsqueeze(-1),
                            next_q1 - next_log_prob_q1.unsqueeze(-1)], dim=2)
        log_sum_exp_q1 = torch.logsumexp(cat_q1, dim=2)

        return log_sum_exp_q1

    def map_hidden(self, policy_hidden=None, value_hidden=None):
        if policy_hidden is not None:
            policy_hidden = self.policy_decoder(policy_hidden)
        if value_hidden is not None:
            value_hidden = self.value_decoder(value_hidden)

        return policy_hidden, value_hidden

    def estimate_hidden(self, batch, sac_embedding_infer="concat", reduction="mean"):
        torch.cuda.empty_cache()
        # we use 3D slices to train the policy
        batch_size, seq_len, _ = batch["observations"].shape

        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)

        if sac_embedding_infer == "concat":
            value_hidden = batch["value_hidden"][:, 0]
            policy_hidden = batch["policy_hidden"][:, 0]
            value_embedding, value_hidden_next = self.value_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=value_hidden, sequential=True)
            policy_embedding, policy_hidden_next = self.policy_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=policy_hidden, sequential=True)

        elif sac_embedding_infer == "direct":
            _observations = torch.cat([batch["observations"], batch["next_observations"][:, -1:]], dim=1)
            _last_actions = torch.cat([batch["last_actions"], batch["actions"][:, -1:]], dim=1)
            _value_embedding, _ = self.value_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                 pre_hidden=batch["value_hidden"][:, 0], sequential=True)
            _policy_embedding, _ = self.policy_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                   pre_hidden=batch["policy_hidden"][:, 0], sequential=True)
            value_embedding = _value_embedding[:, :-1]
            policy_embedding = _policy_embedding[:, :-1]

        policy_embedding = self.policy_decoder(policy_embedding)  # (bs, seq, dim)
        value_embedding = self.value_decoder(value_embedding)  # (bs, seq, dim)

        if reduction == 'sum':
            policy_embedding = policy_embedding.reshape(batch_size * seq_len, -1)
            value_embedding = value_embedding.reshape(batch_size * seq_len, -1)
            mask = batch['valid'].reshape(batch_size * seq_len, -1)

            policy_embedding = (policy_embedding * mask).sum(dim=0, keepdim=True)
            value_embedding = (value_embedding * mask).sum(dim=0, keepdim=True)
        elif reduction == 'mean':
            policy_embedding = policy_embedding.reshape(batch_size * seq_len, -1)
            value_embedding = value_embedding.reshape(batch_size * seq_len, -1)
            mask = batch['valid'].reshape(batch_size * seq_len, -1)

            policy_embedding = (policy_embedding * mask).sum(dim=0, keepdim=True) / mask.sum()
            value_embedding = (value_embedding * mask).sum(dim=0, keepdim=True) / mask.sum()

        return policy_embedding, value_embedding

    def train_policy_meta(self, batch, behavior_cloning=False, sac_embedding_infer="concat",
                          real_data_only_with_mb=True, only_hidden=False, only_policy=False):
        torch.cuda.empty_cache()
        # we use 3D slices to train the policy
        task_num, batch_size, seq_len, _ = batch["observations"].shape
        for k in batch:
            batch[k] = batch[k].reshape(task_num * batch_size, seq_len, -1)


        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)

        if sac_embedding_infer == "concat":
            value_hidden = batch["value_hidden"][:, 0]
            policy_hidden = batch["policy_hidden"][:, 0]
            value_embedding, value_hidden_next = self.value_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=value_hidden, sequential=True)
            policy_embedding, policy_hidden_next = self.policy_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=policy_hidden, sequential=True)

            lens_next = torch.ones(len(lens)).int()
            value_embedding_next, _ = self.value_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=value_hidden_next, sequential=True)
            policy_embedding_next, _ = self.policy_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=policy_hidden_next, sequential=True)
            value_embedding_next = torch.cat([value_embedding[:, 1:], value_embedding_next], dim=1)
            policy_embedding_next = torch.cat([policy_embedding[:, 1:], policy_embedding_next], dim=1)
        elif sac_embedding_infer == "direct":
            _observations = torch.cat([batch["observations"], batch["next_observations"][:, -1:]], dim=1)
            _last_actions = torch.cat([batch["last_actions"], batch["actions"][:, -1:]], dim=1)
            _value_embedding, _ = self.value_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                 pre_hidden=batch["value_hidden"][:, 0], sequential=True)
            _policy_embedding, _ = self.policy_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                   pre_hidden=batch["policy_hidden"][:, 0], sequential=True)
            value_embedding = _value_embedding[:, :-1]
            policy_embedding = _policy_embedding[:, :-1]
            value_embedding_next = _value_embedding[:, 1:]
            policy_embedding_next = _policy_embedding[:, 1:]

        policy_embedding = self.policy_decoder(policy_embedding)
        value_embedding = self.value_decoder(value_embedding)

        if not only_policy:
            policy_embedding_meta = policy_embedding.reshape(task_num, batch_size, max_len, -1)
            # policy_embedding_meta = policy_embedding_meta / torch.sqrt((policy_embedding_meta ** 2).mean(dim=-1, keepdim=True)).detach()
            value_embedding_meta = value_embedding.reshape(task_num, batch_size, max_len, -1)
            # value_embedding_meta = value_embedding_meta / torch.sqrt((value_embedding_meta ** 2).mean(dim=-1, keepdim=True)).detach()

            contrastive_mask = batch['valid'].reshape(task_num, batch_size, max_len, -1).transpose(0, 1).transpose(1,
                                                                                                                   2).reshape(
                batch_size * max_len, task_num, 1)

            # batch * len, task, task   contras policy
            policy_embedding_left = policy_embedding_meta.transpose(0, 1).transpose(1, 2).reshape(batch_size * max_len,
                                                                                                  task_num, -1)
            policy_embedding_right = policy_embedding_meta.transpose(0, 1).transpose(1, 2).transpose(2, 3).reshape(
                batch_size * max_len, -1, task_num)
            index = np.arange(batch_size * max_len)
            np.random.shuffle(index)
            policy_embedding_right = policy_embedding_right[index, :, :]
            contrastive_mask_ = contrastive_mask[index, :, :]

            policy_inner_prod = torch.bmm(policy_embedding_left, policy_embedding_right)
            policy_inner_prod_ = torch.bmm(policy_embedding_right.transpose(1, 2), policy_embedding_left.transpose(1, 2))
            policy_label = torch.arange(task_num, device=policy_inner_prod.device).unsqueeze(0).repeat(max_len * batch_size,
                                                                                                       1)

            policy_contrastive_loss = ce_loss(policy_inner_prod, policy_label, contrastive_mask) + ce_loss(
                policy_inner_prod_, policy_label, contrastive_mask_)

            # batch * len, task, task   contras value
            value_embedding_left = value_embedding_meta.transpose(0, 1).transpose(1, 2).reshape(batch_size * max_len,
                                                                                                task_num, -1)
            value_embedding_right = value_embedding_meta.transpose(0, 1).transpose(1, 2).transpose(2, 3).reshape(
                batch_size * max_len, -1, task_num)
            index = np.arange(batch_size * max_len)
            np.random.shuffle(index)
            value_embedding_right = value_embedding_right[index, :, :]
            contrastive_mask_ = contrastive_mask[index, :, :]

            value_inner_prod = torch.bmm(value_embedding_left, value_embedding_right)
            value_inner_prod_ = torch.bmm(value_embedding_right.transpose(1, 2), value_embedding_left.transpose(1, 2))
            value_label = torch.arange(task_num, device=value_inner_prod.device).unsqueeze(0).repeat(max_len * batch_size,
                                                                                                     1)

            value_contrastive_loss = ce_loss(value_inner_prod, value_label, contrastive_mask) + ce_loss(value_inner_prod_,
                                                                                                        value_label,
                                                                                                        contrastive_mask_)

            if not self.args.coupled:
                self.actor_enc_optim.zero_grad()
                (self.args["Meta"]["coef_policy_contrastive"] * policy_contrastive_loss).backward()
                self.actor_enc_optim.step()

                self.critic_enc_optim.zero_grad()
                (self.args["Meta"]["coef_value_contrastive"] * value_contrastive_loss).backward()
                self.critic_enc_optim.step()

                policy_embedding = policy_embedding.detach()
                value_embedding = value_embedding.detach()
        else:
            value_contrastive_loss = torch.zeros((1, 1)).sum()
            policy_contrastive_loss = torch.zeros((1, 1)).sum()

        with torch.no_grad():
            value_embedding_next = self.value_decoder(value_embedding_next)
            policy_embedding_next = self.policy_decoder(policy_embedding_next)

        if only_hidden:
            ret = dict()
            ret["value_con"] = value_contrastive_loss.detach().cpu().numpy()
            ret["policy_con"] = policy_contrastive_loss.detach().cpu().numpy()

            return ret

        # if real_data_only_with_mb:
        #     for k in batch:
        #         batch[k] = batch[k].reshape(task_num, batch_size, seq_len, -1)[:, :batch_size // 2].reshape(task_num * batch_size // 2,
        #                                                                                      seq_len, -1)
        #
        #     policy_embedding = policy_embedding.reshape(task_num, batch_size, seq_len, -1)[:, :batch_size // 2].reshape(
        #         task_num * batch_size // 2, seq_len, -1)
        #     value_embedding = value_embedding.reshape(task_num, batch_size, seq_len, -1)[:, :batch_size // 2].reshape(
        #         task_num * batch_size // 2, seq_len, -1)
        #
        #     policy_embedding_next = policy_embedding_next.reshape(task_num, batch_size, seq_len, -1)[:, :batch_size // 2].reshape(
        #         task_num * batch_size // 2, seq_len, -1)
        #     value_embedding_next = value_embedding_next.reshape(task_num, batch_size, seq_len, -1)[:, :batch_size // 2].reshape(
        #         task_num * batch_size // 2, seq_len, -1)
        #
        #     batch_size //= 2
        #     task_num *= 2
        #
        #     # for k in batch:
        #     #     batch[k] = batch[k].reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(task_num // 2 * batch_size,
        #     #                                                                                  seq_len, -1)
        #     #
        #     # policy_embedding = policy_embedding.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
        #     #     task_num // 2 * batch_size, seq_len, -1)
        #     # value_embedding = value_embedding.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
        #     #     task_num // 2 * batch_size, seq_len, -1)
        #     #
        #     # policy_embedding_next = policy_embedding_next.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
        #     #     task_num // 2 * batch_size, seq_len, -1)
        #     # value_embedding_next = value_embedding_next.reshape(task_num, batch_size, seq_len, -1)[0::2].reshape(
        #     #     task_num // 2 * batch_size, seq_len, -1)
        no_ent = True
        coef_ent = 1 if not no_ent else 0

        with torch.no_grad():

            action_target, log_prob_target, mu_target, logstd_target = self.actor.sample(batch["next_observations"],
                                                                                         policy_embedding_next)
            q1_target = self.target_q1(batch["next_observations"], action_target, value_embedding_next)
            q2_target = self.target_q2(batch["next_observations"], action_target, value_embedding_next)
            q_target = torch.min(q1_target, q2_target)
            q_target = q_target - self.log_alpha.exp() * torch.unsqueeze(log_prob_target, dim=-1)
            q_target = batch["rewards"] + self.discount * (~batch["terminals"]) * (q_target)
            # if self.args["q_target_clip"]:
            q_target = torch.clip(q_target,
                                  -10,
                                  1000,
                                  )

        # update critic
        q1 = self.q1(batch["observations"], batch["actions"], value_embedding)
        q2 = self.q2(batch["observations"], batch["actions"], value_embedding)
        valid_num = torch.sum(batch["valid"])

        q1_loss = torch.sum(((q1 - q_target) ** 2) * batch['valid']) / valid_num
        q2_loss = torch.sum(((q2 - q_target) ** 2) * batch['valid']) / valid_num

        # if self.args.q_conservative:
        #     log_sum_exp_q1 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
        #                                                  policy_embedding, value_embedding, N=10, q_functions=self.q1)
        #     log_sum_exp_q2 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
        #                                                  policy_embedding, value_embedding, N=10, q_functions=self.q2)
        #     q1_loss += self.alpha_cql * ((log_sum_exp_q1 - q1) * batch["valid"]).sum() / batch["valid"].sum()
        #     q2_loss += self.alpha_cql * ((log_sum_exp_q2 - q2) * batch["valid"]).sum() / batch["valid"].sum()

        td_loss = (q1_loss + q2_loss)

        if self.args.coupled:
            q_loss = td_loss + self.args["Meta"]["coef_value_contrastive"] * value_contrastive_loss
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()
        else:
            q_loss = td_loss
            self.critic_optim.zero_grad()
            q_loss.backward()
            self.critic_optim.step()

        # q_loss = td_loss + self.args["Meta"]["coef_value_contrastive"] * value_contrastive_loss

        self._soft_update(self.target_q1, self.q1, soft_target_tau=self.args["soft_target_tau"])
        self._soft_update(self.target_q2, self.q2, soft_target_tau=self.args["soft_target_tau"])

        # update alpha and actor
        actions, log_prob, mu, logstd = self.actor.sample(batch["observations"], policy_embedding)
        log_prob = log_prob.unsqueeze(dim=-1)  # (B, T, 1)

        if not no_ent and self.args["learnable_alpha"]:
            alpha_loss = - torch.sum(self.log_alpha * ((log_prob + \
                                                        self.args['target_entropy']) * batch[
                                                           'valid']).detach()) / valid_num
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        q1_ = self.q1(batch["observations"], actions, value_embedding.detach())
        q2_ = self.q2(batch["observations"], actions, value_embedding.detach())
        min_q_ = torch.min(q1_, q2_)

        if behavior_cloning:
            sac_loss = coef_ent * self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            sac_loss_scaled = 2.5 * sac_loss / min_q_.detach().mean()
            bc_loss = (batch["actions"] - actions) ** 2
            bc_loss = torch.sum(bc_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss_scaled + bc_loss * self.args["BC"]["bc_loss_coeff"]
        else:
            sac_loss = coef_ent * self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss

        # policy_loss += self.args["Meta"]["coef_policy_contrastive"] * policy_contrastive_loss

        if self.args.coupled:
            policy_loss += self.args["Meta"]["coef_policy_contrastive"] * policy_contrastive_loss
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
        else:
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

        ret = dict()
        ret["min_q"] = min_q_.detach().cpu().mean().numpy()
        ret["q_loss"] = q_loss.detach().cpu().numpy()
        ret["td_loss"] = td_loss.detach().cpu().numpy()
        if not no_ent and self.args["learnable_alpha"]:
            ret["alpha_loss"] = alpha_loss.detach().cpu().numpy()
        if behavior_cloning:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
            ret["sac_loss"] = sac_loss.detach().cpu().numpy()
            ret["bc_loss"] = bc_loss.detach().cpu().numpy()
            ret["value_con"] = value_contrastive_loss.detach().cpu().numpy()
            ret["policy_con"] = policy_contrastive_loss.detach().cpu().numpy()
        else:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
            ret["value_con"] = value_contrastive_loss.detach().cpu().numpy()
            ret["policy_con"] = policy_contrastive_loss.detach().cpu().numpy()
        return ret
    
    def train_policy(self, batch, behavior_cloning=False, sac_embedding_infer="concat"):
        # we use 3D slices to train the policy
        batch['valid'] = batch['valid'].astype(int)
        lens = np.sum(batch['valid'], axis=1).squeeze(-1)
        max_len = np.max(lens)
        for k in batch:
            batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)

        if sac_embedding_infer == "concat":
            value_hidden = batch["value_hidden"][:, 0]
            policy_hidden = batch["policy_hidden"][:, 0]
            value_embedding, value_hidden_next = self.value_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=value_hidden, sequential=True)
            policy_embedding, policy_hidden_next = self.policy_gru(
                torch.cat([batch["observations"], batch["last_actions"]], dim=-1), lens, pre_hidden=policy_hidden, sequential=True)

            lens_next = torch.ones(len(lens)).int()
            value_embedding_next, _ = self.value_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=value_hidden_next, sequential=True)
            policy_embedding_next, _ = self.policy_gru(
                torch.cat([batch["next_observations"][:, -1:], batch["actions"][:, -1:]], dim=-1), lens_next,
                pre_hidden=policy_hidden_next, sequential=True)
            value_embedding_next = torch.cat([value_embedding[:, 1:], value_embedding_next], dim=1)
            policy_embedding_next = torch.cat([policy_embedding[:, 1:], policy_embedding_next], dim=1)
        elif sac_embedding_infer == "direct":
            _observations = torch.cat([batch["observations"], batch["next_observations"][:, -1:]], dim=1)
            _last_actions = torch.cat([batch["last_actions"], batch["actions"][:, -1:]], dim=1)
            _value_embedding, _ = self.value_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                 pre_hidden=batch["value_hidden"][:, 0], sequential=True)
            _policy_embedding, _ = self.policy_gru(torch.cat([_observations, _last_actions], dim=-1), lens + 1,
                                                   pre_hidden=batch["policy_hidden"][:, 0], sequential=True)
            value_embedding = _value_embedding[:, :-1]
            policy_embedding = _policy_embedding[:, :-1]
            value_embedding_next = _value_embedding[:, 1:]
            policy_embedding_next = _policy_embedding[:, 1:]

        with torch.no_grad():
            policy_embedding_next = self.policy_decoder(policy_embedding_next)
            value_embedding_next = self.value_decoder(value_embedding_next)

            action_target, log_prob_target, mu_target, logstd_target = self.actor.sample(batch["next_observations"],
                                                                                         policy_embedding_next)
            q1_target = self.target_q1(batch["next_observations"], action_target, value_embedding_next)
            q2_target = self.target_q2(batch["next_observations"], action_target, value_embedding_next)
            q_target = torch.min(q1_target, q2_target)
            q_target = q_target - self.log_alpha.exp() * torch.unsqueeze(log_prob_target, dim=-1)
            q_target = batch["rewards"] + self.discount * (~batch["terminals"]) * (q_target)
            # if self.args["q_target_clip"]:
            #     q_target = torch.clip(q_target,
            #                         self.rew_min / (1-self.discount),
            #                         self.rew_max / (1-self.discount)
            #     )
        policy_embedding = self.policy_decoder(policy_embedding)
        value_embedding = self.value_decoder(value_embedding)

        # update critic
        q1 = self.q1(batch["observations"], batch["actions"], value_embedding)
        q2 = self.q2(batch["observations"], batch["actions"], value_embedding)
        valid_num = torch.sum(batch["valid"])

        q1_loss = torch.sum(((q1 - q_target) ** 2) * batch['valid']) / valid_num
        q2_loss = torch.sum(((q2 - q_target) ** 2) * batch['valid']) / valid_num

        # if self.args.q_conservative:
        #     log_sum_exp_q1 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
        #                                                  policy_embedding, value_embedding, N=10, q_functions=self.q1)
        #     log_sum_exp_q2 = self.estimate_log_sum_exp_q(batch["observations"], batch["next_observations"],
        #                                                  policy_embedding, value_embedding, N=10, q_functions=self.q2)
        #     q1_loss += self.alpha_cql * ((log_sum_exp_q1 - q1) * batch["valid"]).sum() / batch["valid"].sum()
        #     q2_loss += self.alpha_cql * ((log_sum_exp_q2 - q2) * batch["valid"]).sum() / batch["valid"].sum()

        q_loss = (q1_loss + q2_loss)
        self.critic_optim.zero_grad()
        q_loss.backward()
        self.critic_optim.step()

        self._soft_update(self.target_q1, self.q1, soft_target_tau=self.args["soft_target_tau"])
        self._soft_update(self.target_q2, self.q2, soft_target_tau=self.args["soft_target_tau"])

        # update alpha and actor
        actions, log_prob, mu, logstd = self.actor.sample(batch["observations"], policy_embedding)
        log_prob = log_prob.unsqueeze(dim=-1)  # (B, T, 1)

        if self.args["learnable_alpha"]:
            alpha_loss = - torch.sum(self.log_alpha * ((log_prob + \
                                                        self.args['target_entropy']) * batch[
                                                           'valid']).detach()) / valid_num
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

        q1_ = self.q1(batch["observations"], actions, value_embedding.detach())
        q2_ = self.q2(batch["observations"], actions, value_embedding.detach())
        min_q_ = torch.min(q1_, q2_)

        if behavior_cloning:
            sac_loss = self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            sac_loss_scaled = 10.0 * sac_loss / min_q_.detach().mean()
            bc_loss = (batch["actions"] - actions) ** 2
            bc_loss = torch.sum(bc_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss_scaled + bc_loss * self.args["BC"]["bc_loss_coeff"]
        else:
            sac_loss = self.log_alpha.exp().detach() * log_prob - min_q_
            sac_loss = torch.sum(sac_loss * batch["valid"]) / valid_num
            policy_loss = sac_loss

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        ret = dict()
        ret["min_q"] = min_q_.detach().cpu().mean().numpy()
        ret["q_loss"] = q_loss.detach().cpu().numpy()
        if self.args["learnable_alpha"]:
            ret["alpha_loss"] = alpha_loss.detach().cpu().numpy()
        if behavior_cloning:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
            ret["sac_loss"] = sac_loss.detach().cpu().numpy()
            ret["bc_loss"] = bc_loss.detach().cpu().numpy()
        else:
            ret["policy_loss"] = policy_loss.detach().cpu().numpy()
        return ret

    def _soft_update(self, net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau)

    def eval_policy(self, env):
        res = self.eval_on_real_env(env)
        return res

    def eval_on_real_env(self, env):
        # env = get_env(self.args["task"])
        results = ([self.test_one_trail(env) for _ in range(self.args["eval_runs"])])
        rewards = [result[0] for result in results]
        episode_lengths = [result[1] for result in results]
        rew_mean = np.mean(rewards)
        len_mean = np.mean(episode_lengths)

        res = OrderedDict()
        res["Reward_Mean_Env"] = rew_mean
        # try:
        #     res["Score"] = env.get_normalized_score(rew_mean)
        # except:
        #     print("no attr")
        # res["Length_Mean_Env"] = len_mean

        return res

    def zero_hidden(self, batch_size):
        self.policy_gru.zero_hidden(batch_size=batch_size)
        self.value_gru.zero_hidden(batch_size=batch_size)

    def test_one_trail(self, env):
        # env = get_env(self.args["task"])
        with torch.no_grad():
            state, done = env.reset(), False
            self.zero_hidden(batch_size=1)
            lst_action = torch.zeros((1, 1, self.args['action_shape'])).to(self.device)
            hidden_policy = torch.zeros((1, 1, self.args['rnn_hidden_dim'])).to(self.device)
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]  # 这里增加了数据的维度，当做batch为1在处理
                state = torch.from_numpy(state).float().to(self.device)
                action, _, hidden_policy = self.get_action(state, lst_action, hidden_policy, deterministic=True)
                assert _ is None
                use_action = action.cpu().numpy().reshape(-1)
                if type(env.action_space) == discrete.Discrete:
                    use_action = np.argmax(use_action)
                state_next, reward, done, _ = env.step(use_action)
                lst_action = action
                state = state_next
                # action = policy.get_action(state).reshape(-1)
                # state, reward, done, _ = env.step(action)
                rewards += reward
                lengths += 1
        return (rewards, lengths)

    def save(self, save_path):
        assert save_path, "save path cannot be None!"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.policy_decoder.state_dict(), os.path.join(save_path, "policy_dec.pt"))
        torch.save(self.value_decoder.state_dict(), os.path.join(save_path, "value_dec.pt"))

        torch.save(self.policy_gru.state_dict(), os.path.join(save_path, "policy_gru.pt"))
        torch.save(self.actor.state_dict(), os.path.join(save_path, "actor.pt"))
        torch.save(self.value_gru.state_dict(), os.path.join(save_path, "value_gru.pt"))
        torch.save(self.q1.state_dict(), os.path.join(save_path, "q1.pt"))
        torch.save(self.q2.state_dict(), os.path.join(save_path, "q2.pt"))
        torch.save(self.log_alpha.data, os.path.join(save_path, "log_alpha.pt"))
        torch.save(self.target_q1.state_dict(), os.path.join(save_path, "target_q1.pt"))
        torch.save(self.target_q2.state_dict(), os.path.join(save_path, "target_q2.pt"))
        torch.save(self.actor_optim.state_dict(), os.path.join(save_path, "actor_optim.pt"))
        torch.save(self.critic_optim.state_dict(), os.path.join(save_path, "critic_optim.pt"))
        torch.save(self.alpha_optim.state_dict(), os.path.join(save_path, "alpha_optim.pt"))
        if (not self.args.coupled) and self.args['use_contrastive']:
            torch.save(self.actor_enc_optim.state_dict(), os.path.join(save_path, "actor_enc_optim.pt"))
            torch.save(self.critic_enc_optim.state_dict(), os.path.join(save_path, "critic_enc_optim.pt"))

    def load(self, load_path):
        assert load_path, "load path cannot be None!"
        self.policy_decoder.load_state_dict(
            torch.load(os.path.join(load_path, "policy_dec.pt"), map_location=self.device))
        self.value_decoder.load_state_dict(
            torch.load(os.path.join(load_path, "value_dec.pt"), map_location=self.device))

        self.policy_gru.load_state_dict(torch.load(os.path.join(load_path, "policy_gru.pt"), map_location=self.device))
        self.actor.load_state_dict(torch.load(os.path.join(load_path, "actor.pt"), map_location=self.device))
        self.value_gru.load_state_dict(torch.load(os.path.join(load_path, "value_gru.pt"), map_location=self.device))
        self.q1.load_state_dict(torch.load(os.path.join(load_path, "q1.pt"), map_location=self.device))
        self.q2.load_state_dict(torch.load(os.path.join(load_path, "q2.pt"), map_location=self.device))
        self.target_q1.load_state_dict(torch.load(os.path.join(load_path, "target_q1.pt"), map_location=self.device))
        self.target_q2.load_state_dict(torch.load(os.path.join(load_path, "target_q2.pt"), map_location=self.device))
        self.log_alpha.data = torch.load(os.path.join(load_path, "log_alpha.pt"), map_location=self.device)
        self.actor_optim.load_state_dict(
            torch.load(os.path.join(load_path, "actor_optim.pt"), map_location=self.device))
        self.critic_optim.load_state_dict(
            torch.load(os.path.join(load_path, "critic_optim.pt"), map_location=self.device))
        self.alpha_optim.load_state_dict(
            torch.load(os.path.join(load_path, "alpha_optim.pt"), map_location=self.device))
        if (not self.args.coupled) and self.args['use_contrastive']:
            self.actor_enc_optim.load_state_dict(
                torch.load(os.path.join(load_path, "actor_enc_optim.pt"), map_location=self.device))
            self.critic_enc_optim.load_state_dict(
                torch.load(os.path.join(load_path, "critic_enc_optim.pt"), map_location=self.device))

    def state_dict(self):
        dicts = {
            "policy_gru": self.policy_gru.state_dict(),
            "value_gru": self.value_gru.state_dict(),
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "log_alpha": self.log_alpha.data,
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }
        if (not self.args.coupled) and self.args['use_contrastive']:
            dicts['actor_enc_optim'] = self.actor_enc_optim.state_dict()
            dicts['critic_enc_optim'] = self.critic_enc_optim.state_dict()

        return dicts

    def load_state_dict(self, state_dict):
        self.policy_gru.load_state_dict(state_dict["policy_gru"])
        self.value_gru.load_state_dict(state_dict["value_gru"])
        self.actor.load_state_dict(state_dict["actor"])
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.target_q1.load_state_dict(state_dict["target_q1"])
        self.target_q2.load_state_dict(state_dict["target_q2"])
        self.log_alpha.data = state_dict["log_alpha"]
        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.alpha_optim.load_state_dict(state_dict["alpha_optim"])
        if (not self.args.coupled) and self.args['use_contrastive']:
            self.actor_enc_optim.load_state_dict(state_dict["actor_enc_optim"])
            self.critic_enc_optim.load_state_dict(state_dict["critic_enc_optim"])