from gym import Wrapper


"""
env = WrapEnv(gym.make("EnvName"))
"""

class Ewrapper(Wrapper):
    def __init__(self, env):
        super(Ewrapper, self).__init__(env)
        self.env = env

    def step(self, action):
        output = self.env.step(action)
        if len(output) == 5:
            output = (output[0], output[1], output[2] or output[3], output[4])
        return output

    def reset(self):
        output = self.env.reset()
        if len(output) == 2:
            output = output[0]
        return output


# import gym
# env1 = gym.make("CartPole-v0")
# obs, _ = env1.reset()
# print(obs)
# next_obs, reward, done, _, info = env1.step(0)
# env2 = Ewrapper(gym.make("CartPole-v1"))
# obs = env2.reset()
# print(obs)
# next_obs, reward, done, info = env2.step(0)
