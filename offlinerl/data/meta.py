import gym
import numpy as np
import os

from offlinerl.utils.data import SampleBatch
from offlinerl.utils.env import get_env_shape, get_env_obs_act_spaces

def load_data(path, name):
    path = os.path.join(path, "{}.npy".format(name))
    dataset = np.load(path, allow_pickle=True).item()

    return dataset

def merge_data(data1, data2, data_limit=None):
    if data2 is None:
        return data1
    if data_limit is None:
        data_limit = data1.shape[0]
    data = {}
    for k in data1.keys():
        data[k] = np.concatenate([data1[k][:data_limit], data2[k][:data_limit]], axis=0)

    return data


# def load_task_buffer(task, level, type, degree):
#     path = './onlinerl/data/'
#     path = os.path.join(path, "sac_{}_{}_{}".format(task, type, degree))
#
#     if level == 'medium-expert':
#         dataset_medium = load_data(path, "medium")
#         dataset_expert = load_data(path, "expert")
#         dataset = merge_data(dataset_medium, dataset_expert)
#     else:
#         dataset = load_data(path, level)
#
#     # dataset["rewards"] = (dataset["rewards"] - dataset["rewards"].min()) / (dataset["rewards"].max() - dataset["rewards"].min())
#
#     # buffer = SampleBatch(
#     #     observations=dataset['observations'],
#     #     next_observations=dataset['next_observations'],
#     #     actions=dataset['actions'],
#     #     rewards=np.expand_dims(np.squeeze(dataset['rewards']), 1),
#     #     terminals=np.expand_dims(np.squeeze(dataset['terminals']), 1),
#     # )
#     buffer = dataset
#
#     return buffer

def load_task_buffer(task, level, type, degree):
    path = './onlinerl/data/'
    path = os.path.join(path, "sac_{}_{}_{}".format(task, type, degree))
    data_limit = 1000000 // len(level)

    total_dataset = None
    for l in level:
        dataset = load_data(path, "data_{}".format(str(l)))
        total_dataset = merge_data(dataset, total_dataset, data_limit=data_limit)
    buffer = total_dataset

    return buffer

def load_meta_buffer(task, data_type_list):
    meta_buffer = {}
    for data_type in data_type_list:
        # meta_buffer["{}-{}-{}".format(len(data_type[0]), data_type[1], data_type[2])] = \
        #     load_task_buffer(task, data_type[0], data_type[1], data_type[2])
        meta_buffer[data_type[2]] = load_task_buffer(task, data_type[0], data_type[1], data_type[2])

    return meta_buffer

# def load_meta_buffer(task, data_type_list):
#     meta_buffer = {}
#     for data_type in data_type_list:
#         meta_buffer["{}-{}-{}".format(data_type[0], data_type[1], data_type[2])] = \
#             load_task_buffer(task, data_type[0], data_type[1], data_type[2])
#
#     return meta_buffer


def get_dateset_info(buffer):
    obs = np.concatenate([buffer["observations"], buffer["next_observations"]], axis=0)

    return {
        "obs_max": obs.max(axis=0),
        "obs_min": obs.min(axis=0),
        "obs_mean": obs.mean(axis=0),
        "obs_std": obs.std(axis=0),
        "rew_max": buffer["rewards"].max(),
        "rew_min": buffer["rewards"].min(),
        "rew_mean": buffer["rewards"].mean(),
        "rew_std": buffer["rewards"].std()
    }

def get_env_info(task):
    env = gym.make(task)

    obs_dim = env.observation_space.shape
    action_space = env.action_space
    if len(obs_dim) == 1:
        obs_dim = obs_dim[0]
    if hasattr(env.action_space, 'n'):
        act_dim = env.action_space.n
    else:
        act_dim = action_space.shape[0]

    obs_space = env.observation_space
    act_space = env.action_space

    return {
        "obs_shape": obs_dim,
        "obs_space": obs_space,
        "action_shape": act_dim,
        "action_space": act_space
    }