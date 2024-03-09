import os.path

from UtilsRL.exp import setup, parse_args
from UtilsRL.logger import TensorboardLogger
from offlinerl.algo.mb_sac_v2 import RNNTrainer
from offlinerl.data.task_generation import task_generation
from offlinerl.env.env_generation import env_generation

# load configs
import offlinerl.config.mb as rnn_config
args = parse_args(rnn_config, convert=True)

# initialize logger
# logger = TensorboardLogger(
#     log_path=os.path.join(args["tb_log_path"], args['task']),
#     name=args["exp_name"],
#     txt=True
# )
logger = TensorboardLogger(
    log_path=os.path.join(args["tb_log_path"], args['task']),
    name=args["exp_name"],
    txt=True
)

# setup experiments
args = setup(args, logger)
# logger.log_str(str(args))

# eval_envs = env_generation(args['task'])
train_tasks, test_tasks = task_generation(args['task'], args['level_type'], args['param_type'], with_eval=True)
trainer = RNNTrainer(args, train_tasks, test_tasks, [])

if args["test_mode"]:
    args["Meta"]["train_epoch"] = 1

train_stage = {
    "dynamics": trainer.train_dynamics,
    "meta_policy": trainer.train_mainloop,
}
load_stage = {
    "dynamics": trainer.load_dynamics,
    "meta_policy": trainer.load_mainloop,
}

load = True
for stage in ["dynamics", "meta_policy"]:
    if stage == args["from"]:
        load = False
    if load:
        load_stage[stage](args[stage+"_path"])
    else:
        train_stage[stage](args[stage+"_path"])
    if stage == args["to"]:
        break