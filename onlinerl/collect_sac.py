import copy
import logging

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution

import itertools
import math
import os
from collections import deque, namedtuple

from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.nn.functional import softplus

import random
from argparse import ArgumentParser
from collections import deque

import gym
from gym.wrappers import RescaleAction
import numpy as np
import torch
import logging
from onlinerl.logger import Logger

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextstate', 'done'))


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)

        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high

        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)

        return action


class MeanStdevFilter():
    def __init__(self, shape, clip=3.0):
        self.eps = 1e-4
        self.shape = shape
        self.clip = clip
        self._count = 0
        self._running_sum = np.zeros(shape)
        self._running_sum_sq = np.zeros(shape) + self.eps
        self.mean = np.zeros(shape)
        self.stdev = np.ones(shape) * self.eps

    def update(self, x):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        self._running_sum += np.sum(x, axis=0)
        self._running_sum_sq += np.sum(np.square(x), axis=0)
        # assume 2D data
        self._count += x.shape[0]
        self.mean = self._running_sum / self._count
        self.stdev = np.sqrt(
            np.maximum(
                self._running_sum_sq / self._count - self.mean ** 2,
                self.eps
            ))

    def __call__(self, x):
        return np.clip(((x - self.mean) / self.stdev), -self.clip, self.clip)

    def invert(self, x):
        return (x * self.stdev) + self.mean


class ReplayPool:

    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self._memory = deque(maxlen=int(capacity))
        self.length = 0

    def push(self, transition: Transition):
        """ Saves a transition """
        self._memory.append(transition)
        self.length = self.length + 1 if self.length < self.capacity else self.length

    def sample(self, batch_size: int) -> Transition:
        transitions = random.sample(self._memory, batch_size)
        return Transition(*zip(*transitions))

    def get(self, start_idx: int, end_idx: int) -> Transition:
        transitions = list(itertools.islice(self._memory, start_idx, end_idx))
        return Transition(*zip(*transitions))

    def get_all(self) -> Transition:
        return self.get(0, len(self._memory))

    def __len__(self) -> int:
        return len(self._memory)

    def clear_pool(self):
        self._memory.clear()
        self.length = 0


# Taken from: https://github.com/pytorch/pytorch/pull/19785/files
# The composition of affine + sigmoid + affine transforms is unstable numerically
# tanh transform is (2 * sigmoid(2x) - 1)
# Old Code Below:
# transforms = [AffineTransform(loc=0, scale=2), SigmoidTransform(), AffineTransform(loc=-1, scale=2)]
class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))


def make_checkpoint(agent, step_count, env_name):
    q_funcs, target_q_funcs, policy, log_alpha = agent.q_funcs, agent.target_q_funcs, agent.policy, agent.log_alpha

    save_path = "checkpoints/model-{}.pt".format(step_count)

    if not os.path.isdir('checkpoints'):
        os.makedirs('checkpoints')

    torch.save({
        'double_q_state_dict': q_funcs.state_dict(),
        'target_double_q_state_dict': target_q_funcs.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'log_alpha_state_dict': log_alpha
    }, save_path)


def evaluate_agent(env, agent, state_filter, n_starts=1, deterministic=True):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter, deterministic=deterministic)
            # action = agent.get_action(state, state_filter=state_filter, deterministic=deterministic)
            nextstate, reward, done, _ = env.step(action)
            reward_sum += reward
            state = nextstate
    return reward_sum / n_starts


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MLPNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size=256):
        super(MLPNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, x):
        return self.network(x)


class Policy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Policy, self).__init__()
        self.action_dim = action_dim
        self.network = MLPNetwork(state_dim, action_dim * 2, hidden_size)

    def forward(self, x, get_logprob=False):
        mu_logstd = self.network(x)
        mu, logstd = mu_logstd.chunk(2, dim=1)
        logstd = torch.clamp(logstd, -20, 2)
        std = logstd.exp()
        dist = Normal(mu, std)
        transforms = [TanhTransform(cache_size=1)]
        dist = TransformedDistribution(dist, transforms)
        action = dist.rsample()
        if get_logprob:
            logprob = dist.log_prob(action).sum(dim=1, keepdim=True)
        else:
            logprob = None
        mean = torch.tanh(mu)
        return action, logprob, mean


class DoubleQFunc(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(DoubleQFunc, self).__init__()
        self.network1 = MLPNetwork(state_dim + action_dim, 1, hidden_size)
        self.network2 = MLPNetwork(state_dim + action_dim, 1, hidden_size)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.network1(x), self.network2(x)


class SAC_Agent:

    def __init__(self, seed, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256,
                 update_interval=1):
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -action_dim
        self.batchsize = batchsize
        self.update_interval = update_interval

        torch.manual_seed(seed)

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size).to(device)
        self.target_q_funcs = copy.deepcopy(self.q_funcs)
        self.target_q_funcs.eval()
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)

        # aka temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.temp_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        #self.replay_pool = ReplayPool(capacity=int(1e6))

    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter:
            state = state_filter(state)
        with torch.no_grad():
            action, _, mean = self.policy(torch.Tensor(state).view(1, -1).to(device))
        if deterministic:
            return mean.squeeze().cpu().numpy()
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)

    def update_q_functions(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch, logprobs_batch, _ = self.policy(nextstate_batch, get_logprob=True)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * (q_target - self.alpha * logprobs_batch)
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy_and_temp(self, state_batch):
        action_batch, logprobs_batch, _ = self.policy(state_batch, get_logprob=True)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (self.alpha * logprobs_batch - qval_batch).mean()
        temp_loss = -self.log_alpha * (logprobs_batch.detach() + self.target_entropy).mean()
        return policy_loss, temp_loss

    def optimize(self, n_updates, memory, state_filter=None):
        q1_loss, q2_loss, pi_loss, a_loss = 0, 0, 0, 0
        for i in range(n_updates):
            # samples = self.replay_pool.sample(self.batchsize)
            samples = memory.sample(self.batchsize)

            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.done).to(device).unsqueeze(1)

            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_functions(state_batch, action_batch, reward_batch,
                                                                 nextstate_batch, done_batch)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            # update policy and temperature parameter
            for p in self.q_funcs.parameters():
                p.requires_grad = False
            pi_loss_step, a_loss_step = self.update_policy_and_temp(state_batch)
            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()
            self.temp_optimizer.zero_grad()
            a_loss_step.backward()
            self.temp_optimizer.step()
            for p in self.q_funcs.parameters():
                p.requires_grad = True

            self.alpha = self.log_alpha.exp()

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()
            pi_loss += pi_loss_step.detach().item()
            a_loss += a_loss_step.detach().item()
            if i % self.update_interval == 0:
                self.update_target()
        return {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'pi_loss': pi_loss,
            'a_loss': a_loss
        }

    def save(self, path, name='policy.pt'):
        path = os.path.join(path, name)
        state_dict = {
            "policy": self.policy.state_dict(),
            "q": self.q_funcs.state_dict(),
            "target_q": self.target_q_funcs.state_dict(),
            "alpha": self.log_alpha
        }
        torch.save(state_dict, path)

    def load(self, path, name='policy.pt'):
        path = os.path.join(path, name)
        state_dict = torch.load(path)
        self.policy.load_state_dict(state_dict["policy"])
        self.q_funcs.load_state_dict(state_dict["q"])
        self.target_q_funcs.load_state_dict(state_dict["target_q"])
        self.log_alpha = state_dict["alpha"]


class DataBatch():

    def __init__(self):
        self.data = {
            'observations': [],
            'actions': [],
            'next_observations': [],
            'terminals': [],
            'rewards': [],
        }

    def add_transition(self, transition):
        for k in transition.keys():
            self.data[k].append(np.array(transition[k]).reshape(-1))

    def save(self, path):
        save_data = {}
        for k in self.data.keys():
            save_data[k] = np.stack(self.data[k])

        np.save(path, save_data)
        # np.load(path, allow_pickle=True).item()


def collect_data(agent, env, state_filter=None, collect_num=10000, deterministic=True):
    data_batch = DataBatch()
    steps = 0
    max_steps = env.spec.max_episode_steps
    total_returns = []
    while steps < collect_num:
        state = env.reset()
        done = False
        time_steps = 0
        returns = 0

        while (not done):
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter, deterministic=deterministic)
            nextstate, reward, done, _ = env.step(action)
            returns += reward

            real_done = False if time_steps == max_steps else done
            data_batch.add_transition({
                'observations': state,
                'actions': action,
                'next_observations': nextstate,
                'terminals': real_done,
                'rewards': reward,
            })
            state = nextstate
            time_steps += 1
            steps += 1
        total_returns.append(returns)

    total_returns = np.array(total_returns)
    print("collect num {}, mean {}, std {}, min {}, max {}".format(len(total_returns),
                                                                   total_returns.mean(),
                                                                   total_returns.std(),
                                                                   total_returns.min(),
                                                                   total_returns.max()))

    return data_batch


def train_agent_model_free(agent, env, eval_env, params, tb_logger=None, policy_path=None, data_path=None):
    ckpt_num = 0

    interval = 20000
    env_1_idx = [500000 // interval]
    env_3_idx = [100000 // interval, 500000 // interval, 900000 // interval]
    env_5_idx = [100000 // interval, 200000 // interval, 500000 // interval, 800000 // interval, 900000 // interval]
    env_10_idx = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 48]

    replay_pool = ReplayPool(capacity=int(1e6))
    utd = params['utd']

    update_timestep = params['update_every_n_steps']
    seed = params['seed']
    log_interval = 5000
    n_random_actions = params['n_random_actions']
    n_evals = params['n_evals']
    n_collect_steps = params['n_collect_steps']
    use_statefilter = params['obs_filter']

    assert n_collect_steps > agent.batchsize, "We must initially collect as many steps as the batch size!"

    avg_length = 0
    time_step = 0
    cumulative_timestep = 0
    cumulative_log_timestep = 0
    n_updates = 0
    i_episode = 0
    log_episode = 0
    samples_number = 0
    episode_rewards = []
    episode_steps = []

    if use_statefilter:
        state_filter = MeanStdevFilter(env.env.observation_space.shape[0])
    else:
        state_filter = None

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    # env.action_space.np_random.seed(seed)

    max_steps = env.spec.max_episode_steps

    data_batch = collect_data(agent, eval_env, None, collect_num=interval)
    data_batch.save(os.path.join(data_path, 'data_{}.npy').format(ckpt_num))
    agent.save(policy_path, 'policy_{}.pt'.format(ckpt_num))
    ckpt_num += 1

    current_score = 0.0
    patience = 1000000

    while samples_number < patience:
        time_step = 0
        episode_reward = 0
        i_episode += 1
        log_episode += 1
        state = env.reset()
        if state_filter:
            state_filter.update(state)
        done = False

        while (not done):
            cumulative_log_timestep += 1
            cumulative_timestep += 1
            time_step += 1
            samples_number += 1
            if samples_number < n_random_actions:
                action = env.action_space.sample()
            else:
                action = agent.get_action(state, state_filter=state_filter)
            nextstate, reward, done, _ = env.step(action)
            episode_reward += reward

            if samples_number % interval == 0:
                if ckpt_num in env_1_idx:
                    collect_num = patience // 1
                elif ckpt_num in env_3_idx:
                    collect_num = patience // 3
                elif ckpt_num in env_5_idx:
                    collect_num = patience // 5
                elif ckpt_num in env_10_idx:
                    collect_num = patience // 10
                else:
                    collect_num = interval
                data_batch = collect_data(agent, eval_env, None, collect_num=collect_num)
                data_batch.save(os.path.join(data_path, 'data_{}.npy').format(ckpt_num))
                agent.save(policy_path, 'policy_{}.pt'.format(ckpt_num))
                ckpt_num += 1

            # if we hit the time-limit, it's not a 'real' done; we don't want to assign low value to those states
            real_done = False if time_step == max_steps else done
            replay_pool.push(Transition(state, action, reward, nextstate, real_done))

            state = nextstate
            if state_filter:
                state_filter.update(state)

            # update if it's time
            if cumulative_timestep % update_timestep == 0 and cumulative_timestep > n_collect_steps:
                results = agent.optimize(update_timestep, replay_pool, state_filter=state_filter)
                tb_logger.add_scalar('train', results)
                n_updates += 1

            # logging
            if cumulative_timestep % log_interval == 0 and cumulative_timestep > n_collect_steps:
                avg_length = np.mean(episode_steps)
                running_reward = np.mean(episode_rewards)
                eval_reward = evaluate_agent(eval_env, agent, state_filter, n_starts=n_evals)
                current_score = eval_reward

                print(
                    'Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(
                        i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
                # logging.info('Episode {} \t Samples {} \t Avg length: {} \t Test reward: {} \t Train reward: {} \t Number of Policy Updates: {}'.format(
                #         i_episode, samples_number, avg_length, eval_reward, running_reward, n_updates))
                tb_logger.add_scalar('eval', {'score': eval_reward}, iter=samples_number)
                episode_steps = []
                episode_rewards = []

        episode_steps.append(time_step)
        episode_rewards.append(episode_reward)


def reset_param(env, type, param):
    if type == 'grav':
        env.unwrapped.model.opt.gravity[2] = param * (-9.81)
    if type == 'dofd':
        for idx in range(len(env.unwrapped.model.dof_damping)):
            env.unwrapped.model.dof_damping[idx] *= param
    return env


def main():
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--use_obs_filter', dest='obs_filter', action='store_true')
    parser.add_argument('--update_every_n_steps', type=int, default=1)
    parser.add_argument('--n_random_actions', type=int, default=10000)
    parser.add_argument('--n_collect_steps', type=int, default=1000)
    parser.add_argument('--n_evals', type=int, default=10)
    parser.add_argument('--utd', type=int, default=1)
    parser.add_argument('--save_model', dest='save_model', action='store_true')
    parser.set_defaults(obs_filter=False)
    parser.set_defaults(save_model=False)
    parser.add_argument('--tb_path', type=str, default='./onlinerl/tb/')
    parser.add_argument('--policy_path', type=str, default='./onlinerl/ckpt/')
    parser.add_argument('--data_path', type=str, default='./onlinerl/data/')

    parser.add_argument('--type', type=str, default='grav')
    parser.add_argument('--degree', type=float, default=1.0)
    parser.add_argument('--recollect', action='store_true')
    parser.add_argument('--sto', action='store_true')
    parser.add_argument('--oe', action='store_true')

    parser.add_argument('--info', action='store_true')

    args = parser.parse_args()
    params = vars(args)
    recollect_flag = params['recollect']

    tb_logger = Logger(args)
    env_type = params['type']
    env_degree = params['degree']
    seed = params['seed']

    env = gym.make(params['env'])
    env = RescaleAction(env, -1, 1)
    #env = NormalizedActions(env)
    env = reset_param(env, env_type, env_degree)

    eval_env = gym.make(params['env'])
    eval_env = RescaleAction(eval_env, -1, 1)
    # env = NormalizedActions(env)
    eval_env = reset_param(eval_env, env_type, env_degree)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC_Agent(seed, state_dim, action_dim)

    policy_path = params["policy_path"]
    policy_path = os.path.join(policy_path, 'sac_{}_{}_{}_{}_{}'.format(args.env, args.type, args.degree, args.seed, args.utd))
    if not os.path.exists(policy_path):
        os.makedirs(policy_path, exist_ok=True)

    data_path = params["data_path"]
    data_path = os.path.join(data_path, 'sac_{}_{}_{}'.format(args.env, args.type, args.degree))
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    train_agent_model_free(agent=agent,
                           env=env,
                           eval_env=eval_env,
                           params=params,
                           tb_logger=tb_logger,
                           policy_path=policy_path,
                           data_path=data_path)



if __name__ == '__main__':
    main()
