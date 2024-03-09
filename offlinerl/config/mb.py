from email.policy import default
from UtilsRL.misc.namespace import NameSpace
from offlinerl.utils.env import get_env

task = "Hopper-v2"
level_type = "e-m-r"
param_type = "grav"
seed = 370002
tb_log_path = "./offlinerl/tb_mb"
exp_name = "mb"
default_path = "./offlinerl/out_mb"


class ablation(NameSpace):
    sac_embedding_infer = "concat"
    clip_obs = True
    probe_mode = "PBT"
    probe_init = True
    sl = False


class debug(NameSpace):
    de = False


class email(NameSpace):
    to = "1@2.com"
    account = "3@4.com"
    password = None

dynamics_path = default_path + "/dynamics"
mainloop_path = default_path + "/mainloop"

mainloop_save_interval = 20

iter = None
start_epoch = 0
total_epoch = 50
test_mode = False
####### train
soft_expanding = 0.05
horizon = 3
env_pool_size = None  # will be computed in run time
# model_pool_size = 50000
ratio = 0.5
real_bc_only = True


###### SAC Agent
coupled = False
q_target_clip = True
rnn_hidden_dim = 64
rnn_layer_num = 1
decoder_hidden_dims = (2,)
head_hidden_dims = (256, 256)
actor_lr = 3e-4
critic_lr = 3e-4
discount = 0.99
soft_target_tau = 5e-3
learnable_alpha = True
eval_runs = 5
use_contrastive = True
policy_hidden_dims = (256, 256)
value_hidden_dims = (256, 256)
load_hidden = False
test_hidden = False
van = True

q_conservative = False
behavior_cloning = True
advers = True
td3bc = True
td_lambda = 0.0
# eta = 0.5
contra_type = 'prod'
rep_only = False


####### train dynamics
class Dynamics(NameSpace):
    init_num = 3
    select_num = 3
    hidden_layer_size = 200
    hidden_layer_num = 3
    batch_size = 2048
    lr = 1e-3
    l2_loss_coef = 0.000075
    normalizer = "static"
    max_epoch = 200
    min_epoch = 150
    eval_with_var_loss = False
    train_with_clip_loss = False

###### BC stage
class BC(NameSpace):
    train_epoch = 3000
    train_update = 100
    batch_size = 256
    reset_interval = 4
    bc_loss_coeff = 1.


class Meta(NameSpace):
    model_pool_size = 250000
    rollout_batch_size = 2000
    model_sample_size = 1

    lam = -1.0

    train_batch_size = 256
    init_epoch = 0
    pretrain_epoch = 5
    train_epoch = 600
    train_epoch_hidden = 250
    train_update = 1000

    eval_interval = 50

    coef_policy_contrastive = 2.0
    coef_value_contrastive = 2.0
    coef_q_conservative = 2.0

    reset_interval = 2
    log_interval = 1
    save_interval = 100


class Adv(NameSpace):
    batch_size = 256
    init_batch_size = 2000
    train_batch_size = 256
    train_update = 250
    train_interval = 50
    train_epoch = 50

    lam = 1.0  # 3.0
    lam_rew = 1.0
    lam_pen = -10.0
    horizon = 5


class Eval(NameSpace):
    num_traj = 5
    num_env = 5
