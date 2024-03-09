import torch
from torch.functional import F

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

def product_of_gaussians(mus, logvars):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.exp(logvars)
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-3)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared