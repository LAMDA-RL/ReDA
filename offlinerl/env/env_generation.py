import random

import gym
from offlinerl.env.unstable_wrapper import Unstable_Env


def set_env(env, scheme=None):
    env = Unstable_Env(env)
    env.reset_scheme(scheme)
    return env


def scheme_generation(interval_steps=100, max_steps=1000, min=0.5, max=1.5, type='grav'):
    steps = 0
    scheme = {}
    while steps <= max_steps:
        param = random.uniform(min, max)
        scheme[steps] = (type, param)
        steps += interval_steps

    return scheme


def env_generation(task):
    schemes = [
        # {0: ('grav', 0.5)},
        # {0: ('grav', 1.0)},
        # {0: ('grav', 1.5)},
        # {0: ('grav', 0.8)},
        # {0: ('grav', 1.2)},
        # {0: ('grav', 0.5), 500: ('grav', 1.2)},
        # {0: ('grav', 1.5), 500: ('grav', 0.8)},
        # {
        #  0: ('grav', 1.5), 100: ('grav', 1.2),
        #  200: ('grav', 0.8), 300: ('grav', 0.5),
        #  400: ('grav', 0.8), 500: ('grav', 1.2),
        #  600: ('grav', 1.5), 700: ('grav', 1.2),
        #  800: ('grav', 0.8), 900: ('grav', 0.5),
        #  },
        # {
        #     0: ('grav', 1.8), 100: ('grav', 1.2),
        #     200: ('grav', 0.8), 300: ('grav', 0.3),
        #     400: ('grav', 0.5), 500: ('grav', 1.0),
        #     600: ('grav', 1.5), 700: ('grav', 2.0),
        #     800: ('grav', 1.0), 900: ('grav', 0.2),
        # },
        # {0: ('grav', 0.5), 500: ('grav', 1.5)},
        # {0: ('grav', 1.5), 500: ('grav', 0.5)},
        # {0: ('grav', 0.5), 300: ('grav', 1.2), 600: ('grav', 1.5)},
        # {0: ('grav', 1.5), 300: ('grav', 0.8), 600: ('grav', 0.5)},
        # {0: ('grav', 0.5), 300: ('grav', 1.2), 600: ('grav', 0.8)},
        # {0: ('grav', 1.5), 300: ('grav', 0.8), 600: ('grav', 1.2)},
    ]
    random.seed(1000)
    for _ in range(1):
        scheme = scheme_generation()
        schemes.append(scheme)

    env_list = []
    for scheme in schemes:
        env = gym.make(task)
        env = set_env(env, scheme)
        env_list.append(env)

    return env_list


