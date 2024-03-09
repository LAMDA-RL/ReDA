import numpy as np


def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1))
    done = ~not_done
    done = done[:, None]
    return done

def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done =  np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) * \
                np.isfinite(next_obs).all(axis=-1) \
                * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                * (height > .7) \
                * (np.abs(angle) < .2)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_halfcheetahveljump(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_antangle(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = 	np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_ant(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    x = next_obs[:, 0]
    not_done = np.isfinite(next_obs).all(axis=-1) \
                * (x >= 0.2) \
                * (x <= 1.0)

    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = np.logical_and(np.all(next_obs > -100, axis=-1), np.all(next_obs < 100, axis=-1)) \
                * (height > 0.8) \
                * (height < 2.0) \
                * (angle > -1.0) \
                * (angle < 1.0)
    done = ~not_done
    done = done[:,None]
    return done

def termination_fn_point2denv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

def termination_fn_point2dwallenv(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    done = np.array([False]).repeat(len(obs))
    done = done[:,None]
    return done

# def termination_fn_pendulum(obs, act, next_obs):
#     assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
#
#     done = np.zeros((len(obs), 1), dtype=bool)
#     return done

def termination_fn_pendulum(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    va = next_obs[:, 1]
    not_done = np.isfinite(next_obs).all(axis=-1) \
                * (va >= -0.2) \
                * (va <= 0.2)

    done = ~not_done
    done = done[:, None]
    return done

# def termination_fn_doublependulum(obs, act, next_obs):
#     assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
#
#     not_done = np.isfinite(next_obs).all(axis=-1) \
#                * (va >= -0.2) \
#                * (va <= 0.2)
#     done = ~not_done
#     done = done[:, None]
#     return done

def termination_fn_humanoid(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    z = next_obs[:,0]
    done = (z < 1.0) + (z > 2.0)

    done = done[:,None]
    return done

def termination_fn_mountaincar(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

    pos = next_obs[:, 0]
    vel = next_obs[:, 1]

    done = (pos > 0.45) * (vel > 0.0)

    done = done[:, None]
    return done

def termination_fn_acrobot(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
    done = (-np.cos(next_obs[:, 0]) - np.cos(next_obs[:, 1]+next_obs[:, 0]) >= 1.0)
    done = done[:, None]

    return done

def termination_fn_other(obs, act, next_obs, global_min_obs, global_max_obs):
    obs_range = global_max_obs - global_min_obs
    not_done = np.logical_and(np.all(next_obs > global_min_obs - obs_range, axis=-1), np.all(next_obs < global_max_obs + obs_range, axis=-1))
    done = ~not_done
    done = done[:, None]
    return done

def is_terminal(obs,act, next_obs,task, global_min_obs=None,global_max_obs=None):
    # return termination_fn_other(obs, act, next_obs, global_min_obs, global_max_obs)
    if 'halfcheetahvel' in task:
        return termination_fn_halfcheetahveljump(obs, act, next_obs)
    elif 'halfcheetah' in task or 'HalfCheetah' in task or 'Ant' in task:
        return termination_fn_halfcheetah(obs, act, next_obs)
    elif 'hopper' in task or 'Hopper' in task:
        return termination_fn_hopper(obs,act,next_obs)
    elif 'antangle' in task:
        return termination_fn_antangle(obs,act,next_obs)
    elif 'ant' in task:
        return termination_fn_ant(obs, act, next_obs)
    elif 'walker2d' in task or 'Walker2d' in task:
        return termination_fn_walker2d(obs, act, next_obs)
    elif 'point2denv' in task:
        return termination_fn_point2denv(obs, act, next_obs)
    elif 'point2dwallenv' in task:
        return termination_fn_point2dwallenv(obs,act, next_obs)
    # elif 'doblependulum' in task:
    #     return termination_fn_doublependulum(obs,act,next_obs)
    elif 'pendulum' in task:
        return termination_fn_pendulum(obs,act,next_obs)
    elif 'acrobot' in task:
        return termination_fn_acrobot(obs,act,next_obs)
    elif 'humanoid' in task:
        return termination_fn_humanoid(obs, act, next_obs)
    elif 'mountaincar' in task:
        return termination_fn_mountaincar(obs, act, next_obs)






















