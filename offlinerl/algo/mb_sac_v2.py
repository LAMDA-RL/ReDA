import torch
import torch.nn as nn
from torch.distributions import kl, Normal, kl_divergence
import numpy as np
import os
from copy import deepcopy
import random
import gym

# from UtilsRL.monitor import Monitor
from UtilsRL.logger import TensorboardLogger
# from UtilsRL.misc.decorator import profile

from offlinerl.data.meta import get_env_info, get_dateset_info
import offlinerl.utils.loader as loader

from offlinerl.utils.simple_replay_pool import SimpleReplayTrajPool
from offlinerl.agent.sac import SACAgent
from offlinerl.agent.sac import TransformerSACAgent as RNNSACAgent
from offlinerl.utils.ensemble import ParallelRDynamics
from offlinerl.utils.data import SampleBatch
from offlinerl.utils.rollout_sac import meta_policy_rollout, adv_rollout



class ContrastiveReward():

    def __init__(self, args, meta_pools, ref_policy):
        self.args = args
        self.meta_pools = meta_pools
        self.ref_policy = ref_policy

        self.ref_context = {
            'policy': None,
            'value': None,
        }

    def reset(self, key):
        batch = self.meta_pools[key].random_batch(self.args["Adv"]["batch_size"])
        with torch.no_grad():
            ph, vh = self.ref_policy.estimate_hidden(batch, reduction='mean')
            self.ref_context['policy'] = ph.reshape(-1, 1)
            self.ref_context['value'] = vh.reshape(-1, 1)

        batchs = {}
        for k in self.meta_pools.keys():
            batch = self.meta_pools[k].random_batch_for_initial(self.args["Adv"]["init_batch_size"])

            for bk in batch:
                if bk not in batchs.keys():
                    batchs[bk] = batch[bk]
                else:
                    batchs[bk] = np.concatenate([batchs[bk], batch[bk]], axis=0)

        self.initial_batch = batchs

        obs = torch.from_numpy(batchs["observations"]).to(self.args.device)
        lst_action = torch.from_numpy(batchs["last_actions"]).to(self.args.device)
        value_hidden = torch.from_numpy(batchs["value_hidden"]).to(self.args.device)
        policy_hidden = torch.from_numpy(batchs["policy_hidden"]).to(self.args.device)

        self.ref_policy.zero_hidden(batch_size=self.args["Adv"]["batch_size"])
        with torch.no_grad():
            action, _, _, _, policy_hidden = self.ref_policy.get_action(obs, lst_action, policy_hidden, deterministic=False, out_mean_std=True)
            _, value_hidden = self.ref_policy.get_value(obs, action, lst_action, value_hidden)
            proj_ph, proj_vh = self.ref_policy.map_hidden(policy_hidden, value_hidden)

            # proj_ph, proj_vh = self.ref_policy.estimate_hidden(batch)

            policy_inner_prod = torch.einsum("ebj,jf->ebf", proj_ph, self.ref_context['policy']).squeeze(0)
            value_inner_prod = torch.einsum("ebj,jf->ebf", proj_vh, self.ref_context['value']).squeeze(0)

        self.rew_base = policy_inner_prod + value_inner_prod

        self.policy_hidden = policy_hidden
        self.value_hidden = value_hidden

    def step_rew(self, obs, action, lst_action):
        with torch.no_grad():
            _, _, _, _, policy_hidden = self.ref_policy.get_action(obs, lst_action, self.policy_hidden, deterministic=False, out_mean_std=True)
            _, value_hidden = self.ref_policy.get_value(obs, action, lst_action, self.value_hidden)

            self.policy_hidden = policy_hidden
            self.value_hidden = value_hidden

            proj_ph, proj_vh = self.ref_policy.map_hidden(policy_hidden, value_hidden)

            policy_inner_prod = torch.einsum("ebj,jf->ebf", proj_ph, self.ref_context['policy']).squeeze(0)
            value_inner_prod = torch.einsum("ebj,jf->ebf", proj_vh, self.ref_context['value']).squeeze(0)

            # policy_inner_prod = torch.bmm(proj_ph, self.ref_context['policy']).squeeze(0)
            # value_inner_prod = torch.bmm(proj_vh, self.ref_context['value']).squeeze(0)

        rew = policy_inner_prod + value_inner_prod - self.rew_base
        self.rew_base = policy_inner_prod + value_inner_prod

        return rew.squeeze(1)


class RNNTrainer():
    def __init__(self, args, buffers, eval_buffers=None, eval_envs=None):
        self.args = args
        self.logger: TensorboardLogger = args["logger"]
        self.device = args["device"]

        self.logged_rets = []

        total_info = {}
        env_info = get_env_info(args["task"])
        for attr in ["obs_shape", "obs_space", "action_shape", "action_space"]:
            if isinstance(env_info[attr], np.ndarray):
                total_info[attr] = torch.tensor(env_info[attr], dtype=torch.float32, requires_grad=False, device=self.device)
            else:
                total_info[attr] = env_info[attr]

        for k in buffers.keys():
            info = {}
            data_info = get_dateset_info(buffers[k])
            for attr in ["obs_max", "obs_min", "obs_mean", "obs_std", "rew_max", "rew_min", "rew_mean", "rew_std"]:
                if isinstance(data_info[attr], np.ndarray) or isinstance(data_info[attr], np.float64):
                    info[attr] = torch.tensor(data_info[attr], dtype=torch.float32, requires_grad=False, device=self.device)
                else:
                    info[attr] = data_info[attr]

            for attr in ["obs_max", "obs_min", "rew_max", "rew_min"]:
                if attr in total_info.keys():
                    total_info[attr] = torch.max(total_info[attr], info[attr])
                    args[attr] = total_info[attr]
                else:
                    total_info[attr] = info[attr]
                    args[attr] = total_info[attr]

            obs_range = info["obs_max"] - info["obs_min"]
            soft_expanding = obs_range * args["soft_expanding"]
            info["obs_max"], info["obs_min"] = info["obs_max"] + soft_expanding, info["obs_min"] - soft_expanding

            total_info[k] = info

        for key, val in env_info.items():
            args[key] = val

        assert "rew_min" in args.keys()

        self.meta_info = total_info
        args["target_entropy"] = - args["action_shape"]
        self.meta_policy = RNNSACAgent(args).to(self.device)

        self.meta_pools = {}
        self.meta_pools_eval = {}
        for key in buffers.keys():
            buffer = buffers[key]
            eval_buffer = eval_buffers[key]
            args["env_pool_size"] = int((buffer["observations"].shape[0] / args["horizon"]) * 1.2)
            args["env_pool_size_step"] = int((buffer["observations"].shape[0]) * 1.2)
            traj_env_pool = SimpleReplayTrajPool(args.obs_space,
                                                args.action_space,
                                                args.horizon,
                                                args.rnn_hidden_dim,
                                                args.env_pool_size)
            traj_env_pool_eval = SimpleReplayTrajPool(args.obs_space,
                                                 args.action_space,
                                                 1000,
                                                 args.rnn_hidden_dim,
                                                 args.env_pool_size // (500 // args.horizon))

            loader.restore_pool_meta(traj_env_pool, args['task'], adapt=True, maxlen=args["horizon"], \
                                     policy_hook=self.meta_policy.policy_gru, value_hook=self.meta_policy.value_gru,
                                     device=self.device, dataset=buffer)

            # if not self.args.test_hidden:
            #     loader.restore_pool_meta(traj_env_pool_eval, args['task'], adapt=True, maxlen=1000, \
            #                              policy_hook=self.meta_policy.policy_gru, value_hook=self.meta_policy.value_gru,
            #                              device=self.device, dataset=eval_buffer)

            # else:
            loader.restore_pool_meta(traj_env_pool_eval, args['task'], adapt=True, maxlen=args["horizon"], \
                                        policy_hook=self.meta_policy.policy_gru, value_hook=self.meta_policy.value_gru,
                                        device=self.device, dataset=eval_buffer)

            self.meta_pools[key] = traj_env_pool
            self.meta_pools_eval[key] = traj_env_pool_eval

        self.buffer = buffers
        if args.advers:
            self.adv_policy = SACAgent(args).to(self.device)
            self.rew_contrastive = ContrastiveReward(args, self.meta_pools, self.meta_policy)
        torch.cuda.empty_cache()

        self.eval_envs = eval_envs

    def train_dynamics(self, path, keys_list=None):
        # self.logger.log_str("start to train dynamics ...", type="WARNING")
        args = self.args
        torch.cuda.empty_cache()

        self.dynamics = {}
        if keys_list == None:
            keys_list = [key for key in self.buffer.keys()]
        for key in keys_list:
            dataset = self.buffer[key]

            buffer = SampleBatch(
                observations=dataset['observations'],
                next_observations=dataset['next_observations'],
                actions=dataset['actions'],
                rewards=np.expand_dims(np.squeeze(dataset['rewards']), 1),
                terminals=np.expand_dims(np.squeeze(dataset['terminals']), 1),
            )
            data_size = len(buffer)
            val_size = min(int(data_size * 0.1) + 1, 1000)
            train_size = data_size - val_size
            train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
            train_buffer = buffer[train_splits.indices]
            val_buffer = buffer[val_splits.indices]

            dynamics = ParallelRDynamics(
                obs_dim=args["obs_shape"],
                action_dim=args["action_shape"],
                hidden_features=args["Dynamics"]["hidden_layer_size"],
                hidden_layers=args["Dynamics"]["hidden_layer_num"],
                ensemble_size=args["Dynamics"]["init_num"],
                normalizer=args["Dynamics"]["normalizer"],
                obs_mean=self.meta_info[key]["obs_mean"],
                obs_std=self.meta_info[key]["obs_std"],
                tanh=False,
            ).to(self.device)
            dynamics_optim = torch.optim.AdamW(dynamics.split_parameters(), lr=args["Dynamics"]["lr"],
                                               weight_decay=args["Dynamics"]["l2_loss_coef"])
            val_losses = [float('inf') for i in range(dynamics.ensemble_size)]
            from_which_epoch = [-1 for i in range(dynamics.ensemble_size)]
            best_snapshot = [dynamics.get_single_transition(i) for i in range(dynamics.ensemble_size)]

            batch_step = 0
            cnt = 0
            batch_size = args["Dynamics"]["batch_size"]
            for epoch in range(args["Dynamics"]["max_epoch"]):
                # for epoch in Monitor("train transition").listen(range(args["Dynamics"]["max_epoch"])):
                idxs = np.random.randint(train_buffer.shape[0], size=[dynamics.ensemble_size, train_buffer.shape[0]])
                for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                    batch_step += 1
                    batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                    batch = train_buffer[batch_idxs].to_torch(device=self.device)

                    dist = dynamics(torch.cat([batch["observations"], batch["actions"]], dim=-1))
                    samples = dist.rsample()
                    model_loss = ((samples - torch.cat([batch['next_observations'], batch['rewards']], dim=-1)) ** 2).mean(dim=(1, 2)).sum()

                    # model_loss = (- dist.log_prob(torch.cat([batch["next_observations"], batch["rewards"]], dim=-1))).mean(dim=1).sum()

                    clip_loss = 0.01 * (2. * dynamics.max_logstd).sum() - 0.01 * (2. * dynamics.min_logstd).sum() if \
                        args["Dynamics"]["train_with_clip_loss"] else 0
                    loss = model_loss + clip_loss
                    self.logger.log_scalars("Dynamics", {
                        "model_loss_{}".format(key): model_loss.detach().cpu().item(),
                        "all_loss_{}".format(key): loss.detach().cpu().item(),
                        "mean_logstd_{}".format(key): dist.scale.cpu().mean().item(),
                    }, step=batch_step)
                    dynamics_optim.zero_grad()
                    loss.backward()
                    dynamics_optim.step()

                new_val_losses = list(self._eval_dynamics(dynamics, val_buffer, inc_var_loss=args["Dynamics"][
                    "eval_with_var_loss"]).cpu().numpy())

                indexes = []
                for i, new_loss, old_loss in zip(range(len(val_losses)), new_val_losses, val_losses):
                    # if new_loss < old_loss:
                    if (old_loss - new_loss) / np.abs(old_loss) > 0.0:
                        indexes.append(i)
                        val_losses[i] = new_loss
                # self.logger.log_str(f"Epoch {epoch}: updated {len(indexes)} models {indexes}", type="LOG")
                # self.logger.log_str(f"model losses are {val_losses}", type="LOG")
                self.logger.log_scalar("Dynamics/val_loss_{}".format(key), np.mean(new_val_losses), batch_step)

                if len(indexes) > 0:
                    for idx in indexes:
                        best_snapshot[idx] = dynamics.get_single_transition(idx)
                        from_which_epoch[idx] = epoch
                    cnt = 0
                else:
                    cnt += 1

                if cnt >= 10 and epoch >= args["Dynamics"]['min_epoch']:
                    # self.logger.log_str(f"early stopping, final best losses are {val_losses}", type="LOG")
                    # self.logger.log_str(f"from which epoch: {from_which_epoch}", type="LOG")
                    break

            pairs = [(idx, val_loss) for idx, val_loss in enumerate(val_losses)]
            pairs = sorted(pairs, key=lambda x: x[1])
            selected_indexes = [p[0] for p in pairs[:args["Dynamics"]["select_num"]]]
            # if not os.path.exists(path):
            #     os.makedirs(path)
            for i, idx in enumerate(selected_indexes):
                if not os.path.exists(os.path.join(path, str(i))):
                    os.makedirs(os.path.join(path, str(i)), exist_ok=True)
                torch.save(best_snapshot[idx], os.path.join(path, str(i), "{}.pt".format(key)))

            self.dynamics[key] = dynamics

            # self.logger.log_str(f"dynamics training is done, saving to {path}")

        self.load_dynamics(path)

    def load_dynamics(self, path):
        # self.logger.log_str("load dynamics from {}".format(path), type="WARNING")
        self.dynamics = {}
        self.idynamics = {}

        for key in self.buffer.keys():
            self.idynamics[key] = []

        for key in self.buffer.keys():
            # try:
            models = [
                torch.load(os.path.join(path, name, "{}.pt".format(key)), map_location='cpu') \
                for name in os.listdir(path)
            ]
            # except:
                # self.train_dynamics(path, keys_list=[key])
                #
                # models = [
                #     torch.load(os.path.join(path, name, "{}.pt".format(key)), map_location='cpu') \
                #     for name in os.listdir(path)
                # ]

            for ikey in self.idynamics.keys():
                if ikey != key:
                    self.idynamics[ikey] += models

            dynamics = ParallelRDynamics.from_single_transition(models).to(self.device)
            self.dynamics[key] = dynamics

        # for key in self.idynamics.keys():
        #     self.idynamics[key] = ParallelRDynamics.from_single_transition(self.idynamics[key]).to(self.device)

        torch.cuda.empty_cache()

    def _eval_dynamics(self, dynamics, valdata, inc_var_loss=True):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = dynamics(torch.cat([valdata['observations'], valdata['actions']], dim=-1))
            # temp = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2)
            if inc_var_loss:
                mse_losses = ((dist.mean - torch.cat([valdata['next_observations'], valdata['rewards']], dim=-1)) ** 2 / (
                        dist.variance + 1e-8)).mean(dim=(1, 2))
                logvar = dist.scale.log()
                logvar = 2 * dynamics.max_logstd - torch.nn.functional.softplus(
                    2 * dynamics.max_logstd - dist.variance.log())
                logvar = 2 * dynamics.min_logstd + torch.nn.functional.softplus(logvar - 2 * dynamics.min_logstd)
                var_losses = logvar.mean(dim=(1, 2))
                loss = mse_losses + var_losses
            else:
                loss = ((dist.mean - torch.cat([valdata['next_observations'], valdata['rewards']], dim=-1)) ** 2).mean(dim=(1, 2))
            return loss

    def train_mainloop(self, path):
        torch.cuda.empty_cache()
        if self.args.start_epoch > 0:
            self.load_mainloop(path, self.args.start_epoch)

        self.train_meta_policy(path)
        # save meta policy
        self.meta_policy.save(os.path.join(path, "meta_policy"))

    def load_mainloop(self, path, start_epoch):
        epoch_path = os.path.join(path, str(start_epoch))
        if not os.path.exists(epoch_path):
            raise ValueError("load path not found")

        # load candidate
        # self.logger.log_str(f"load candidate set from {path}", type="WARNING")
        # load meta policy
        self.meta_policy.load(os.path.join(epoch_path, "meta_policy"))

    def train_meta_policy(self, path):
        load_hidden = self.args.load_hidden
        # self.logger.log_str(f"Start to train meta policy ...", type="WARNING")
        args = self.args
        torch.cuda.empty_cache()

        self.model_pools = {k:SimpleReplayTrajPool(args.obs_space, args.action_space, args.horizon, args.rnn_hidden_dim,
                                          args.Meta.model_pool_size) for k in self.buffer.keys()}

        self.adv_pools = {k:SimpleReplayTrajPool(args.obs_space, args.action_space, args.horizon, args.rnn_hidden_dim,
                                          args.Meta.model_pool_size) for k in self.buffer.keys()}

        gradient_steps = 0
        self.gradient_steps_ = 0
        task_ignore = not args["use_contrastive"]
        if not load_hidden:
            for i_epoch in range(1, 150 + 1):
                # for i_epoch in Monitor("meta loop").listen(range(1, 50 + 1)):
                if i_epoch % 25 == 0:
                    for _ in range(25):
                        self.train_adv_policy()
                self.rollout()
                torch.cuda.empty_cache()

                train_loss = dict()
                for j in range(args["Meta"]["train_update"]):
                    batch = self.get_train_policy_batch(self.meta_pools, self.model_pools, args["Meta"]["train_batch_size"],
                                                        task_ignore=task_ignore, split=False)
                    train_res = self.meta_policy.train_policy_meta(batch, behavior_cloning=self.args.behavior_cloning,
                                                                   sac_embedding_infer=args["ablation"][
                                                                       "sac_embedding_infer"], only_hidden=True)
                    for _key in train_res:
                        train_loss[_key] = train_loss.get(_key, 0) + train_res[_key]

                for _key in train_loss:
                    train_loss[_key] = train_loss[_key] / args["Meta"]["train_update"]
                # if i_epoch % args["Meta"]["log_interval"] == 0:
                self.logger.log_scalars("Pretrain", train_loss, step=i_epoch)
            self.meta_policy.save(os.path.join(path, "meta_policy"))
        else:
            self.meta_policy.load(os.path.join(path, "meta_policy"))

        for i_epoch in range(1, args["Meta"]["train_epoch"] + 1):
            # for i_epoch in Monitor("meta loop").listen(range(1, args["Meta"]["train_epoch"] + 1)):
            torch.cuda.empty_cache()

            train_loss = dict()
            for j in range(args["Meta"]["train_update"]):
                batch = self.get_train_policy_batch(self.meta_pools, self.meta_pools, args["Meta"]["train_batch_size"],
                                                    task_ignore=task_ignore, split=False)
                train_res = self.meta_policy.train_policy_meta(batch, behavior_cloning=self.args.behavior_cloning,
                                                            sac_embedding_infer=args["ablation"][
                                                                "sac_embedding_infer"], only_policy=True)
                for _key in train_res:
                    train_loss[_key] = train_loss.get(_key, 0) + train_res[_key]

                if gradient_steps % 1000 == 0:

                    eval_res = self.eval_raw_policy()
                    print(eval_res, gradient_steps)
                    self.logger.log_scalars("Meta", eval_res, step=gradient_steps)

                gradient_steps += 1

            for _key in train_loss:
                train_loss[_key] = train_loss[_key] / args["Meta"]["train_update"]
            # if i_epoch % args["Meta"]["log_interval"] == 0:
            self.logger.log_scalars("Meta", train_loss, step=i_epoch)

            # if i_epoch % args["Meta"]["reset_interval"] == 0:
            #     for key in self.meta_pools.keys():
            #         loader.reset_hidden_meta(self.meta_pools[key], args['task'], maxlen=args["horizon"], \
            #                          policy_hook=self.meta_policy.policy_gru, value_hook=self.meta_policy.value_gru,
            #                          device=self.device, dataset=self.buffer[key])

            torch.cuda.empty_cache()

        np.save(os.path.join(path, "ret.npy"), np.array(self.logged_rets))
        self.save_data(path)

    def train_adv_policy(self):
        args = self.args
        if self.args.advers:
            for k in self.dynamics.keys():
                self.rew_contrastive.reset(k)
                dynamics = ParallelRDynamics.from_single_transition(self.idynamics[k]).to(self.device)
                rollout_res = adv_rollout(args, self.adv_policy, dynamics, self.rew_contrastive,
                                          self.adv_pools[k], deterministic=False, clip_args=self.meta_info[k])

            for j in range(args["Adv"]["train_update"]):
                batch = self.get_train_policy_batch(self.adv_pools, self.adv_pools, args["Adv"]["train_batch_size"], seq=False)
                train_res = self.adv_policy.train_policy(batch, behavior_cloning=False)

                if self.gradient_steps_ % int(args["Meta"]["eval_interval"] * 100 // args['horizon']) == 0:
                    self.logger.log_scalars("Adv", train_res, step=self.gradient_steps_)

                self.gradient_steps_ += 1


    def rollout(self):
        args = self.args
        if self.args.advers:
            for k in self.dynamics.keys():
                rollout_res = meta_policy_rollout(args, self.adv_policy, self.dynamics[k], self.meta_pools,
                                                  self.model_pools[k], args["Meta"]["rollout_batch_size"], deterministic=False,
                                                  clip_args=self.meta_info[k], rnn=False)

                rollout_res = {k + "_{}_Adv".format(k): v for k, v in rollout_res.items()}
                self.logger.log_scalars("Rollout", rollout_res, self.gradient_steps_)
        else:
            for k in self.dynamics.keys():
                rollout_res = meta_policy_rollout(args, self.meta_policy, self.dynamics[k], self.meta_pools,
                                                  self.model_pools[k], args["Meta"]["rollout_batch_size"], deterministic=False,
                                                  clip_args=self.meta_info[k])
                rollout_res = {k+"_{}_Meta".format(k): v for k, v in rollout_res.items()}
                self.logger.log_scalars("Rollout", rollout_res, self.gradient_steps_)

    def eval_raw_policy(self):
        total_eval_res = dict()
        total_returns = 0.
        for param in [0.5, 0.8, 1.0, 1.2, 1.5]:
            env = gym.make(self.args.task)
            env.unwrapped.model.opt.gravity[2] = env.unwrapped.model.opt.gravity[2] * param
            eval_res = self.meta_policy.eval_policy(env)
            total_returns += eval_res["Reward_Mean_Env"]

            for k in eval_res.keys():
                total_eval_res["{}_{}".format(k, str(param))] = eval_res[k]

        self.logged_rets.append(total_returns / 5)

        for param in self.meta_pools.keys():
            env = gym.make(self.args.task)
            env.unwrapped.model.opt.gravity[2] = env.unwrapped.model.opt.gravity[2] * float(param)
            print(self.meta_pools_eval.keys())
            batch = self.meta_pools[param].random_batch(1)

            batch['valid'] = batch['valid'].astype(int)
            lens = np.sum(batch['valid'], axis=1).squeeze(-1)
            max_len = np.max(lens)
            for k in batch:
                batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)
            self.meta_policy.policy_gru.inputs = [torch.cat([batch["observations"], batch["last_actions"]], dim=-1)]
            self.meta_policy.value_gru.inputs = [torch.cat([batch["observations"], batch["last_actions"]], dim=-1)]

            eval_res = self.meta_policy.eval_policy(env)

            for k in eval_res.keys():
                total_eval_res["IID_{}_{}".format(k, str(param))] = eval_res[k]

        for param in self.meta_pools_eval.keys():
            env = gym.make(self.args.task)
            env.unwrapped.model.opt.gravity[2] = env.unwrapped.model.opt.gravity[2] * float(param)
            print(self.meta_pools_eval.keys())
            batch = self.meta_pools_eval[param].random_batch(1)

            batch['valid'] = batch['valid'].astype(int)
            lens = np.sum(batch['valid'], axis=1).squeeze(-1)
            max_len = np.max(lens)
            for k in batch:
                batch[k] = torch.from_numpy(batch[k][:, :max_len]).to(self.device)
            self.meta_policy.policy_gru.inputs = [torch.cat([batch["observations"], batch["last_actions"]], dim=-1)]
            self.meta_policy.value_gru.inputs = [torch.cat([batch["observations"], batch["last_actions"]], dim=-1)]
            self.meta_policy.policy_gru.pop_pos = 1
            self.meta_policy.value_gru.pop_pos = 1

            eval_res = self.meta_policy.eval_policy(env)

            for k in eval_res.keys():
                total_eval_res["OOD_{}_{}".format(k, str(param))] = eval_res[k]

        return total_eval_res

    def eval_meta_policy(self):
        total_eval_res = dict()
        for _ in range(self.args['Eval']['num_env']):
            env = random.choice(self.eval_envs)
            eval_res = self.meta_policy.eval_policy(env)
            total_eval_res.update(eval_res)

        for k in total_eval_res.keys():
            total_eval_res[k] /= self.args['Eval']['num_env']

        return total_eval_res

    def load_meta_policy(self, path):
        checkpoints = [int(i) for i in os.listdir(path)]
        # self.logger.log_str(f"Loading meta policy from {path}, found checkpoints {checkpoints}")
        last_checkpoint = max(checkpoints)
        self.meta_policy = RNNSACAgent(self.args)
        self.meta_policy.load(os.path.join(path, str(last_checkpoint)))

        torch.cuda.empty_cache()

    def get_train_policy_batch(self, env_pools, model_pools, batch_size, task_ignore=True, seq=True, split=False):
        batch_size = int(batch_size // len(env_pools))

        env_batch_size = int(batch_size * self.args['ratio'])
        model_batch_size = batch_size - env_batch_size

        batchs = {}
        max_len = self.args["horizon"]
        task_num = len(env_pools)
        for key in env_pools.keys():
            env_pool = env_pools[key]
            model_pool = model_pools[key]
            env_batch = env_pool.random_batch(batch_size) if seq else env_pool.random_batch_for_initial(batch_size)
            model_batch = model_pool.random_batch(batch_size) if seq else model_pool.random_batch_for_initial(batch_size)

            if not split:
                keys = set(env_batch.keys()) & set(model_batch.keys())
                batch = {k: np.concatenate((env_batch[k], model_batch[k]), axis=0) for k in keys}
                for k in batch:
                    if k not in batchs.keys():
                        batchs[k] = batch[k]
                    else:
                        batchs[k] = np.concatenate([batchs[k], batch[k]], axis=0)  # (bs, seq, dim) or (bs, dim)
                task_num = len(env_pools)
            else:
                for k in env_batch:
                    if k not in batchs.keys():
                        batchs[k] = env_batch[k]
                    else:
                        batchs[k] = np.concatenate([batchs[k], env_batch[k]], axis=0)  # (bs, seq, dim) or (bs, dim)

                for k in model_batch:
                    if k not in batchs.keys():
                        batchs[k] = model_batch[k]
                    else:
                        batchs[k] = np.concatenate([batchs[k], model_batch[k]], axis=0)  # (bs, seq, dim) or (bs, dim)

                task_num = len(env_pools) + len(model_pools)

        if not task_ignore and seq:
            if not split:
                batch_size *= 2
            for k in batchs.keys():
                batchs[k] = batchs[k].reshape(task_num, batch_size, max_len, -1)

        return batchs