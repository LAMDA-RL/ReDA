from gym import Wrapper


class Unstable_Env(Wrapper):
    def __init__(self, env):
        super(Unstable_Env, self).__init__(env)
        self.env = env
        self.max_steps = env.spec.max_episode_steps
        self.param = self.env.unwrapped.model.opt.gravity[2]

    def get_param(self):
        return self.env.unwrapped.model.opt.gravity[2]

    def step(self, action):
        self.steps += 1
        if self.steps in self.scheme.keys():
            self.reset_param(self.scheme[self.steps])
        return self.env.step(action)

    def reset_param(self, scheme):
        type = scheme[0]
        param = scheme[1]
        if type == 'grav':
            self.env.unwrapped.model.opt.gravity[2] = param * (-9.81)
        if type == 'dofd':
            for idx in range(len(self.env.unwrapped.model.dof_damping)):
                self.env.unwrapped.model.dof_damping[idx] *= param

    def reset(self):
        self.steps = 0
        if self.steps in self.scheme.keys():
            self.reset_param(self.scheme[self.steps])
        return self.env.reset()

    def reset_scheme(self, scheme):
        self.scheme = scheme

