import gym
import d4rl
import numpy as np

from offlinerl.utils.data import SampleBatch
from offlinerl.utils.env import get_env_shape, get_env_obs_act_spaces

def load_d4rl_buffer(task, dataset=None, name=None):
    if dataset is None:
        name = task[5:] if name is None else name
        env = gym.make(name)
        dataset = d4rl.qlearning_dataset(env)

    buffer = SampleBatch(
        obs=dataset['observations'],
        obs_next=dataset['next_observations'],
        act=dataset['actions'],
        rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
        done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
    )

    return buffer


def get_dateset_info(task, buffer=None):
    obs_shape, action_shape = get_env_shape(task)
    obs_space, action_space = get_env_obs_act_spaces(task)
    
    obs = np.concatenate([buffer["obs"], buffer["obs_next"]], axis=0)
    
    return {
        "obs_max": obs.max(axis=0), 
        "obs_min": obs.min(axis=0), 
        "obs_mean": obs.mean(axis=0), 
        "obs_std": obs.std(axis=0), 
        "rew_max": buffer["rew"].max(), 
        "rew_min": buffer["rew"].min(), 
        "rew_mean": buffer["rew"].mean(), 
        "rew_std": buffer["rew"].std(), 
        "obs_shape": obs_shape, 
        "obs_space": obs_space, 
        "action_shape": action_shape, 
        "action_space": action_space
    }


def get_acrobot_info(task):
    print(task)
    obs_shape, action_shape = get_env_shape(task)
    obs_space, action_space = get_env_obs_act_spaces(task)

    num = 1000000
    obs_size = 6
    obs = np.random.randn(num, obs_size)
    obs_next = np.random.randn(num, obs_size)
    act = np.random.randn(num, 3)
    rew = np.random.randn(num, 1)
    done = (np.ones((num, 1)) == 1)
    # buffer = {
    #     'obs': obs,
    #     'obs_next': obs_next,
    #     'act': act,
    #     'rew': rew,
    #     'done': done
    # }
    buffer = SampleBatch(
        obs=obs,
        obs_next=obs_next,
        act=act,
        rew=rew,
        done=done,
    )

    obs = np.concatenate([buffer["obs"], buffer["obs_next"]], axis=0)

    return {
        "obs_max": obs.max(axis=0),
        "obs_min": obs.min(axis=0),
        "obs_mean": obs.mean(axis=0),
        "obs_std": obs.std(axis=0),
        "rew_max": buffer["rew"].max(),
        "rew_min": buffer["rew"].min(),
        "rew_mean": buffer["rew"].mean(),
        "rew_std": buffer["rew"].std(),
        "obs_shape": obs_shape,
        "obs_space": obs_space,
        "action_shape": action_shape,
        "action_space": action_space
    }, buffer


def get_mountain_info(task):
    print(task)
    obs_shape, action_shape = get_env_shape(task)
    obs_space, action_space = get_env_obs_act_spaces(task)

    num = 1000000
    obs_size = 2
    obs = np.random.randn(num, obs_size)
    obs_next = np.random.randn(num, obs_size)
    act = np.random.randn(num, 1)
    rew = np.random.randn(num, 1)
    done = (np.ones((num, 1)) == 1)
    # buffer = {
    #     'obs': obs,
    #     'obs_next': obs_next,
    #     'act': act,
    #     'rew': rew,
    #     'done': done
    # }
    buffer = SampleBatch(
        obs=obs,
        obs_next=obs_next,
        act=act,
        rew=rew,
        done=done,
    )

    obs = np.concatenate([buffer["obs"], buffer["obs_next"]], axis=0)

    return {
        "obs_max": obs.max(axis=0),
        "obs_min": obs.min(axis=0),
        "obs_mean": obs.mean(axis=0),
        "obs_std": obs.std(axis=0),
        "rew_max": buffer["rew"].max(),
        "rew_min": buffer["rew"].min(),
        "rew_mean": buffer["rew"].mean(),
        "rew_std": buffer["rew"].std(),
        "obs_shape": obs_shape,
        "obs_space": obs_space,
        "action_shape": action_shape,
        "action_space": action_space
    }, buffer


def get_pendulum_info(task):
    print(task)
    obs_shape, action_shape = get_env_shape(task)
    obs_space, action_space = get_env_obs_act_spaces(task)

    num = 1000000
    obs_size = 4
    obs = np.random.randn(num, obs_size)
    obs_next = np.random.randn(num, obs_size)
    act = np.random.randn(num, 1)
    rew = np.random.randn(num, 1)
    done = (np.ones((num, 1)) == 1)
    # buffer = {
    #     'obs': obs,
    #     'obs_next': obs_next,
    #     'act': act,
    #     'rew': rew,
    #     'done': done
    # }
    buffer = SampleBatch(
        obs=obs,
        obs_next=obs_next,
        act=act,
        rew=rew,
        done=done,
    )

    obs = np.concatenate([buffer["obs"], buffer["obs_next"]], axis=0)

    return {
        "obs_max": obs.max(axis=0),
        "obs_min": obs.min(axis=0),
        "obs_mean": obs.mean(axis=0),
        "obs_std": obs.std(axis=0),
        "rew_max": buffer["rew"].max(),
        "rew_min": buffer["rew"].min(),
        "rew_mean": buffer["rew"].mean(),
        "rew_std": buffer["rew"].std(),
        "obs_shape": obs_shape,
        "obs_space": obs_space,
        "action_shape": action_shape,
        "action_space": action_space
    }, buffer