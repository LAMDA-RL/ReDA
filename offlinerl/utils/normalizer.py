import numpy as np
import torch
import torch.nn as nn

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
        
class RunningNormalizer(nn.Module):
    def __init__(self, shape, eps=1e-8, verbose=0):
        super().__init__()

        self.shape = shape
        self.verbose = verbose

        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.eps = eps
        self.count = 1e-4

    def forward(self, x: torch.Tensor, inverse=False):
        if inverse:
            return x * torch.sqrt(self.var) + self.mean
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def to(self, *args, **kwargs):
        self.mean = self.mean.to(*args, **kwargs)
        self.var = self.var.to(*args, **kwargs)

    def update(self, samples: torch.Tensor):
        sample_count = samples.shape[0]
        sample_mean = samples.mean(dim=0)
        sample_var = samples.var(dim=0, unbiased=False)
        delta = sample_mean - self.mean
        total_count = self.count + sample_count

        new_mean = self.mean + delta * sample_count / total_count
        m_a = self.var * self.count
        m_b = sample_var * sample_count
        m_2 = m_a + m_b + delta * delta * self.count * sample_count / (self.count + sample_count)
        new_var = m_2 / (self.count + sample_count)

        new_count = sample_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def state_dict(self, *args, **kwargs):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, state_dict, strict=True):
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.count = state_dict['count']

    def get_rms(self):
        rms = RunningMeanStd(self.shape)
        rms.count = self.count
        rms.mean = self.mean.cpu().numpy()
        rms.var = self.var.cpu().numpy()
        return rms
    
class StaticNormalizer(nn.Module):
    def __init__(self, shape, mean=None, std=None, eps=1e-8):
        super().__init__()
        self.shape = shape
        if mean is None:
            mean = torch.zeros(shape, dtype=torch.float32)
        if std is None:
            std = torch.ones(shape, dtype=torch.float32)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.eps = eps
        
    def forward(self, x, inverse=False):
        if inverse:
            return x*self.std + self.mean
        else:
            return (x - self.mean) / (self.std + self.eps)
        
    def update(self):
        pass

        
        
        
class BatchNormalizer(nn.Module):
    pass