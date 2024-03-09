import os
import uuid
import random

# import aim
import torch
import numpy as np
from loguru import logger
# from UtilsRL.misc.device import select_free_cuda

from offlinerl.utils.logger import log_path

def setup_seed(seed=1024):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
        
def set_free_device_fn():
    device = 'cuda'+":"+str(select_free_cuda()) if torch.cuda.is_available() else 'cpu'

    return device

def select_free_cuda():
    return 0


# def init_exp_logger(repo=None, experiment_name=None, flush_frequency=1):
#     if repo is None:
#         repo = os.path.join(log_path(),"./.aim")
#         if not os.path.exists(repo):
#             logger.info('{} dir is not exist, create {}',repo, repo)
#             os.system(str("cd " + os.path.join(repo,"../") + "&& aim init"))
    
#     aim_logger = aim.Session(repo = repo, experiment=experiment_name, flush_frequency=flush_frequency)
#     aim_logger.experiment_name = experiment_name
    
#     return aim_logger