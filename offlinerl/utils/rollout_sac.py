import torch
from torch.distributions import kl, Normal
import numpy as np

from offlinerl.utils.net.terminal_check import is_terminal
from offlinerl.utils.simple_replay_pool import SimpleReplayPool, SimpleReplayTrajPool


def pbt_rollout(args, agent, transition, env_pool, model_pool, batch_size, deterministic=False, population=None):
    assert isinstance(model_pool, SimpleReplayPool)
    batch = env_pool.random_batch(batch_size)
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    last_action = torch.zeros([obs.shape[0], args["action_shape"]], dtype=torch.float32).to(args.device)

    rollout_res = {
        "raw_reward": 0,
        "pbt_log_prob": 0,
        "reward": 0,
        "n_samples": 0,
        "uncertainty": 0
    }
    with torch.no_grad():
        select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
        for i in range(args["horizon"]):
            batch_size = obs.shape[0]
            action, _ = agent.get_action(obs, deterministic=deterministic, out_mean_std=False)
            # next_sample = transition(torch.cat([obs, action], dim=-1)).sample()

            next_obs_dists = transition(torch.cat([obs, action], dim=-1))  # 这里得到的是一个分布
            next_sample = next_obs_dists.sample()

            next_obses_mode = next_obs_dists.mean[:, :, :-1]
            next_obs_mean = torch.mean(next_obses_mode, dim=0)
            diff = next_obses_mode - next_obs_mean
            disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
            aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
            # uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty
            uncertainty = aleatoric_uncertainty

            reward_raw = next_sample[:, :, -1:]
            next_obs = next_sample[:, :, :-1]

            reward_raw = reward_raw[select_model, np.arange(batch_size)]
            next_obs = next_obs[select_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])

            if args["ablation"]["clip_obs"]:
                clip_num = ((next_obs < args["obs_min"]).any(dim=-1) | (next_obs > args["obs_max"]).any(
                    dim=-1)).sum().item()
                next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
                rollout_res["clip"] = rollout_res.get("clip", 0) + clip_num
            else:
                next_obs = next_obs
            reward_raw = torch.clamp(reward_raw, args["rew_min"], args["rew_max"])

            if population is not None:
                pbt_log_prob = torch.cat([pp.evaluate_action(obs, action).unsqueeze(0) for pp in population], dim=0)
                pbt_log_prob = torch.log(1e-6 + pbt_log_prob.exp().mean(dim=0)).unsqueeze(-1)
            else:
                pbt_log_prob = torch.zeros_like(reward_raw)
            reward = args["Probe"]["rew_coeff"] * reward_raw + args["Probe"]["pbt_coeff"] * pbt_log_prob \
                     + args["Probe"]["unc_coeff"] * uncertainty
            # TODO Penalty
            # log rollout results
            rollout_res["raw_reward"] += reward_raw.sum().item()
            rollout_res["pbt_log_prob"] += pbt_log_prob.sum().item()
            rollout_res["reward"] += reward.sum().item()
            rollout_res["n_samples"] += obs.shape[0]
            rollout_res["uncertainty"] += uncertainty.sum().item()

            nonterm_mask = ~term.squeeze(-1)
            samples = {
                "observations": obs.cpu().numpy(),
                "actions": action.cpu().numpy(),
                "next_observations": next_obs.cpu().numpy(),
                "rewards": reward.cpu().numpy(),
                "terminals": term,
                "valid": np.ones_like(term),
                "last_actions": last_action.cpu().numpy(),
            }
            model_pool.add_samples(samples)

            obs = next_obs[nonterm_mask]
            last_action = action[nonterm_mask]
            select_model = select_model[nonterm_mask]
    for _key in rollout_res:
        if _key != "n_samples":
            rollout_res[_key] /= rollout_res["n_samples"]
    return rollout_res


# new function by yinh
# def probe_rollout(args, agent, transition, env_pool, model_pool, batch_size, deterministic=False, probe_policy=None):
#     batch = env_pool.random_batch_for_initial(batch_size)
#     obs = torch.from_numpy(batch["observations"]).to(args.device)

#     probe_metric = {}
#     current_nonterm = np.ones([len(obs)], dtype=bool)
#     with torch.no_grad():
#         select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
#         for i in range(args["horizon"]):
#             action, _, mu, logstd = agent.get_action(obs, deterministic=deterministic, out_mean_std=True)
#             next_sample = transition(torch.cat([obs, action], dim=-1)).sample()
#             reward = next_sample[:, :, -1:]
#             next_obs = next_sample[:, :, :-1]

#             # compute transition

#             reward = reward[select_model, np.arange(batch_size)]
#             next_obs = next_obs[select_model, np.arange(batch_size)]
#             term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
#             next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
#             reward = torch.clamp(reward, args["rew_min"], args["rew_max"])

#             # compute probe policy
#             if probe_policy is not None:
#                 if args["ablation"]["probe_policy_div"] == "mse":
#                     probe_actions = [pp.get_action(obs, deterministic=True)[0] for pp in probe_policy]
#                     probe_diffs = [((a-action)**2).sum(-1) for a in probe_actions]
#                     probe_diffs = sum(probe_diffs) / len(probe_diffs)
#                     reward += args["Probe"]["div_coeff"] * probe_diffs.unsqueeze(-1)
#                     if "mse" not in probe_metric:
#                         probe_metric["mse"] = []
#                     probe_metric["mse"].append(probe_diffs.mean().item())

#             nonterm_mask = ~term.squeeze(-1)
#             samples = {
#                 "observations": obs.cpu().numpy(),
#                 "actions": action.cpu().numpy(),
#                 "next_observations": next_obs.cpu().numpy(),
#                 "rewards": reward.cpu().numpy(),
#                 "terminals": term,
#                 "valid": current_nonterm.reshape(-1, 1),
#             }
#             # samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
#             num_samples = batch_size    # check whether equals to obs.shape[0]
#             assert num_samples == obs.shape[0]
#             index = np.arange(
#                 model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
#             for k in samples:
#                 model_pool.fields[k][index] = samples[k]
#             current_nonterm = current_nonterm & nonterm_mask
#             obs = next_obs

#             model_pool._pointer += num_samples
#             model_pool._pointer %= model_pool._max_size
#             model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)

#     for key in probe_metric:
#         probe_metric[key] = np.mean(probe_metric[key])

#     return probe_metric

# def probe_rollout_rnn(args, agent, transition, env_pool, model_pool, batch_size, deterministic=False, probe_policy=None):
#     batch = env_pool.random_batch_for_initial(batch_size)
#     obs = torch.from_numpy(batch["observations"]).to(args.device)
#     lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
#     value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
#     policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)

#     probe_metric = {}
#     current_nonterm = np.ones([len(obs)], dtype=bool)
#     with torch.no_grad():
#         select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
#         for i in range(args["horizon"]):
#             action, _, mu, logstd, policy_hidden_next = agent.get_action(obs, lst_action, policy_hidden, deterministic=deterministic, out_mean_std=True)
#             next_sample = transition(torch.cat([obs, action], dim=-1)).sample()
#             reward = next_sample[:, :, -1:]
#             next_obs = next_sample[:, :, :-1]

#             # compute transition

#             reward = reward[select_model, np.arange(batch_size)]
#             next_obs = next_obs[select_model, np.arange(batch_size)]
#             term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
#             next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
#             reward = torch.clamp(reward, args["rew_min"], args["rew_max"])

#             # compute probe policy
#             if probe_policy is not None:
#                 if args["ablation"]["probe_policy_div"] == "mse":
#                     probe_actions = [pp.get_action(obs, lst_action, policy_hidden, deterministic=True)[0] for pp in probe_policy]
#                     probe_diffs = [((a-action)**2).sum(-1) for a in probe_actions]
#                     probe_diffs = sum(probe_diffs) / len(probe_diffs)
#                     reward += args["Probe"]["div_coeff"] * probe_diffs.unsqueeze(-1)
#                     if "mse" not in probe_metric:
#                         probe_metric["mse"] = []
#                     probe_metric["mse"].append(probe_diffs.mean().item())
#                 # elif args["ablation"]["probe_policy_div"] == "divergence":
#                 #     probe_dists = [pp.get_action(obs, lst_action, policy_hidden, out_mean_std=True) for pp in probe_policy]
#                 #     probe_mu = [i[2] for i in probe_dists]
#                 #     probe_logstd = [i[3] for i in probe_dists]

#                 #     this_dist = Normal(mu, logstd.exp())
#                 #     bc_dist = Normal(probe_mu[0], probe_logstd[0])
#                 #     probe_mean_dist = None
#                 #     bc_divergence = 0.5*kl.kl_divergence(this_dist, bc_dist) + 0.5*kl.kl_divergence(bc_dist, this_dist)
#                 #     probe_divergence = probe_mean_dist.entropy()
#                 #     reward += args["Probe"]["div_bc_coeff"]*bc_divergence + args["Probe"]["div_probe_coeff"]*probe_divergence
#                 #     probe_metric["bc_divergence"] = bc_divergence.cpu().mean.item()
#                 #     probe_metric["probe_divergence"] = probe_divergence.cpu().mean().item()

#             nonterm_mask = ~term.squeeze(-1)
#             samples = {
#                 "observations": obs.cpu().numpy(),
#                 "actions": action.cpu().numpy(),
#                 "next_observations": next_obs.cpu().numpy(),
#                 "rewards": reward.cpu().numpy(),
#                 "terminals": term,
#                 "last_actions": lst_action.cpu().numpy(),
#                 "valid": current_nonterm.reshape(-1, 1),
#                 "value_hidden": batch["value_hidden"],
#                 "policy_hidden": batch["policy_hidden"],
#             }
#             samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
#             num_samples = batch_size    # check whether equals to obs.shape[0]
#             assert num_samples == obs.shape[0]
#             index = np.arange(
#                 model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
#             for k in samples:
#                 model_pool.fields[k][index, i] = samples[k][:, 0]
#             current_nonterm = current_nonterm & nonterm_mask
#             obs = next_obs
#             lst_action = action
#             policy_hidden = policy_hidden_next

#         model_pool._pointer += num_samples
#         model_pool._pointer %= model_pool._max_size
#         model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)

#     for key in probe_metric:
#         probe_metric[key] = np.mean(probe_metric[key])

#     return probe_metric


def adv_rollout(args, agent, transition, env_pool, model_pool, deterministic=False, clip_args=None):
    batch = env_pool.initial_batch
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
    # value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
    # policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)
    batch_size = obs.shape[0]
    action, _ = agent.get_action(obs, deterministic=deterministic, out_mean_std=False)
    rollout_res = {
        "raw_reward": 0,
        "reward": 0,
        "n_samples": 0,
        "uncertainty": 0
    }
    current_nonterm = np.ones([len(obs)], dtype=bool)
    with torch.no_grad():
        select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
        for i in range(args["horizon"]):
            next_obs_dists = transition(torch.cat([obs, action], dim=-1))
            next_sample = next_obs_dists.sample()

            next_obses_mode = next_obs_dists.mean[:, :, :-1]
            next_obs_mean = torch.mean(next_obses_mode, dim=0)
            diff = next_obses_mode - next_obs_mean
            disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
            aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
            # uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty
            uncertainty = aleatoric_uncertainty

            reward = next_sample[:, :, -1:]
            next_obs = next_sample[:, :, :-1]
            # compute transition

            # select model and clamp
            reward = reward[select_model, np.arange(batch_size)]
            next_obs = next_obs[select_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
            next_obs = torch.clamp(next_obs, clip_args["obs_min"], clip_args["obs_max"])
            # reward_raw = torch.clamp(reward, args["rew_min"], args["rew_max"])
            reward_raw = reward

            rollout_res["raw_reward"] += reward_raw.sum().item()
            rollout_res["reward"] += reward.sum().item()
            rollout_res["n_samples"] += obs.shape[0]
            rollout_res["uncertainty"] += uncertainty.sum().item()

            nonterm_mask = ~term.squeeze(-1)

            next_action, _ = agent.get_action(obs, deterministic=deterministic, out_mean_std=False)
            reward = env_pool.step_rew(next_obs, next_action, action)

            # nonterm_mask = ~term.squeeze(-1)
            next_obs = torch.clamp(next_obs, clip_args["obs_min"], clip_args["obs_max"])

            samples = {
                "observations": obs.cpu().numpy(),
                "actions": action.cpu().numpy(),
                "next_observations": next_obs.cpu().numpy(),
                "rewards": reward.cpu().numpy(),
                "terminals": term,
                "last_actions": lst_action.cpu().numpy(),
                "valid": current_nonterm.reshape(-1, 1),
                "value_hidden": batch["value_hidden"],
                "policy_hidden": batch["policy_hidden"],
            }

            # assert len(samples["terminals"].shape) == 1
            # assert len(samples["rewards"].shape) == 1

            samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
            num_samples = batch_size  # 检查一下是不是和obs.shape[0]一样
            assert num_samples == obs.shape[0]
            index = np.arange(
                model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
            for k in samples:
                model_pool.fields[k][index, i] = samples[k][:, 0]
            current_nonterm = current_nonterm & nonterm_mask

            # rollout_res["raw_reward"] += reward_raw.sum().item()
            # rollout_res["reward"] += reward.sum().item()
            # rollout_res["n_samples"] += (~term).sum()
            # rollout_res["uncertainty"] += uncertainty.sum().item()
            # if (current_nonterm).sum() <= 0:
            #     break
            obs = next_obs
            action = next_action

        model_pool._pointer += num_samples
        model_pool._pointer %= model_pool._max_size
        model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)

    for _key in rollout_res:
        if _key != "n_samples":
            rollout_res[_key] /= rollout_res["n_samples"]
    return rollout_res


def model_rollout(args, agent, transition, env_pool, model_pool, batch_size, deterministic=False):
    batch = env_pool.random_batch_for_initial(batch_size)
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
    value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
    policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)

    uncertainty_metric = {}
    current_nonterm = np.ones([len(obs)], dtype=bool)
    with torch.no_grad():
        select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
        for i in range(args["horizon"]):
            action, _, mu, logstd, policy_hidden_next = agent.get_action(obs, lst_action, policy_hidden,
                                                                         deterministic=deterministic, out_mean_std=True)
            next_sample = transition(torch.cat([obs, action], dim=-1)).sample()
            reward = next_sample[:, :, -1:]
            next_obs = next_sample[:, :, :-1]

            # compute transition

            # select model and clamp
            reward = reward[select_model, np.arange(batch_size)]
            next_obs = next_obs[select_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
            next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
            reward = torch.clamp(reward, args["rew_min"], args["rew_max"])

            nonterm_mask = ~term.squeeze(-1)
            samples = {
                "observations": obs.cpu().numpy(),
                "actions": action.cpu().numpy(),
                "next_observations": next_obs.cpu().numpy(),
                "rewards": reward.cpu().numpy(),
                "terminals": term,
                "last_actions": lst_action.cpu().numpy(),
                "valid": current_nonterm.reshape(-1, 1),
                # "value_hidden": batch["value_hidden"],
                # "policy_hidden": batch["policy_hidden"],
            }
            model_pool.add_samples(samples)
            # samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
            # num_samples = batch_size    # 检查一下是不是和obs.shape[0]一样
            # assert num_samples == obs.shape[0]
            # index = np.arange(
            #     model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
            # for k in samples:
            #     model_pool.fields[k][index, i] = samples[k][:, 0]
            current_nonterm = current_nonterm & nonterm_mask
            obs = next_obs
            lst_action = action
            policy_hidden = policy_hidden_next

        # model_pool._pointer += num_samples
        # model_pool._pointer %= model_pool._max_size
        # model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)


def policy_rollout(args, agent, transition, env_pool, model_pool, batch_size, deterministic=False, rew_fn=None):
    batch = env_pool.random_batch_for_initial(batch_size)
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
    value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
    policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)

    uncertainty_metric = {}
    rollout_res = {
        "raw_reward": 0,
        "reward": 0,
        "n_samples": 0,
        "uncertainty": 0
    }
    current_nonterm = np.ones([len(obs)], dtype=bool)
    with torch.no_grad():
        select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
        agent.reset()
        for i in range(args["horizon"]):
            action, _, mu, logstd, policy_hidden_next = agent.get_action(obs, lst_action, policy_hidden,
                                                                         deterministic=deterministic, out_mean_std=True)
            next_obs_dists = transition(torch.cat([obs, action], dim=-1))
            next_sample = next_obs_dists.sample()

            next_obses_mode = next_obs_dists.mean[:, :, :-1]
            next_obs_mean = torch.mean(next_obses_mode, dim=0)
            diff = next_obses_mode - next_obs_mean
            disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
            aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
            # uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty
            uncertainty = aleatoric_uncertainty

            reward = next_sample[:, :, -1:]
            next_obs = next_sample[:, :, :-1]
            # compute transition

            # select model and clamp
            reward = reward[select_model, np.arange(batch_size)]
            next_obs = next_obs[select_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
            next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
            if rew_fn is not None:
                reward_raw = torch.clamp(rew_fn(obs, action, next_obs, torch.from_numpy(term).to(args.device)),
                                         args['rew_min'], args['rew_max'])
            else:
                reward_raw = torch.clamp(reward, args["rew_min"], args["rew_max"])
            # reward_raw = torch.clamp(reward, args["rew_min"], args["rew_max"])

            reward = reward_raw + args["Meta"]["lam"] * uncertainty

            rollout_res["raw_reward"] += reward_raw.sum().item()
            rollout_res["reward"] += reward.sum().item()
            rollout_res["n_samples"] += obs.shape[0]
            rollout_res["uncertainty"] += uncertainty.sum().item()

            nonterm_mask = ~term.squeeze(-1)
            samples = {
                "observations": obs.cpu().numpy(),
                "actions": action.cpu().numpy(),
                "next_observations": next_obs.cpu().numpy(),
                "rewards": reward.cpu().numpy(),
                "terminals": term,
                "last_actions": lst_action.cpu().numpy(),
                "valid": current_nonterm.reshape(-1, 1),
                "value_hidden": batch["value_hidden"],
                "policy_hidden": batch["policy_hidden"],
            }
            samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
            num_samples = batch_size  # 检查一下是不是和obs.shape[0]一样
            assert num_samples == obs.shape[0]
            index = np.arange(
                model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
            for k in samples:
                model_pool.fields[k][index, i] = samples[k][:, 0]
            current_nonterm = current_nonterm & nonterm_mask
            # if (current_nonterm).sum() <= 0:
            #     break
            obs = next_obs
            lst_action = action
            policy_hidden = policy_hidden_next

        model_pool._pointer += num_samples
        model_pool._pointer %= model_pool._max_size
        model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)

    for _key in rollout_res:
        if _key != "n_samples":
            rollout_res[_key] /= rollout_res["n_samples"]
    return rollout_res


def meta_policy_rollout(args, agent, transition, env_pools, model_pool, batch_size, deterministic=False, rew_fn=None,
                        clip_args=None, rnn=True):
    batch = {}
    total_batch_size = 0
    for k in env_pools.keys():
        bt = env_pools[k].random_batch_for_initial(batch_size)

        for k in bt:
            if k not in batch.keys():
                batch[k] = bt[k]
            else:
                batch[k] = np.concatenate([batch[k], bt[k]], axis=0)  # (bs, seq, dim) or (bs, dim)
        total_batch_size += batch_size

    batch_size = total_batch_size
    # batch = env_pool.random_batch_for_initial(batch_size)
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
    if rnn:
        agent.zero_hidden(batch_size=total_batch_size)
    value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
    policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)

    uncertainty_metric = {}
    rollout_res = {
        "raw_reward": 0,
        "reward": 0,
        "n_samples": 0,
        "uncertainty": 0
    }
    current_nonterm = np.ones([len(obs)], dtype=bool)
    with torch.no_grad():
        select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
        agent.reset()
        for i in range(args["horizon"]):
            if rnn:
                action, _, mu, logstd, policy_hidden_next = agent.get_action(obs, lst_action, policy_hidden,
                                                                             deterministic=deterministic,
                                                                             out_mean_std=True)
            else:
                action, _, mu, logstd = agent.get_action(obs, deterministic=deterministic, out_mean_std=True)
                policy_hidden_next = policy_hidden
            next_obs_dists = transition(torch.cat([obs, action], dim=-1))
            next_sample = next_obs_dists.sample()

            next_obses_mode = next_obs_dists.mean[:, :, :-1]
            next_obs_mean = torch.mean(next_obses_mode, dim=0)
            diff = next_obses_mode - next_obs_mean
            disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
            aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
            # uncertainty = disagreement_uncertainty if args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty
            # uncertainty = aleatoric_uncertainty
            uncertainty = disagreement_uncertainty

            reward = next_sample[:, :, -1:]
            next_obs = next_sample[:, :, :-1]
            # compute transition

            # select model and clamp
            reward = reward[select_model, np.arange(batch_size)]
            next_obs = next_obs[select_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
            next_obs = torch.clamp(next_obs, clip_args["obs_min"], clip_args["obs_max"])
            if rew_fn is not None:
                # if hy:
                #     reward_raw = rew_fn(obs, action, next_obs, torch.from_numpy(term).to(args.device))
                #     reward_raw += torch.clamp(reward, clip_args["rew_min"], clip_args["rew_max"])
                # else:
                reward_raw = torch.clamp(rew_fn(obs, action, next_obs, torch.from_numpy(term).to(args.device)),
                                         clip_args['rew_min'], clip_args['rew_max'])
            else:
                reward_raw = torch.clamp(reward, clip_args["rew_min"], clip_args["rew_max"])

            reward = reward_raw + args["Meta"]["lam"] * uncertainty

            rollout_res["raw_reward"] += reward_raw.sum().item()
            rollout_res["reward"] += reward.sum().item()
            rollout_res["n_samples"] += (~term).sum()
            rollout_res["uncertainty"] += uncertainty.sum().item()

            nonterm_mask = ~term.squeeze(-1)
            samples = {
                "observations": obs.cpu().numpy(),
                "actions": action.cpu().numpy(),
                "next_observations": next_obs.cpu().numpy(),
                "rewards": reward.cpu().numpy(),
                "terminals": term,
                "last_actions": lst_action.cpu().numpy(),
                "valid": current_nonterm.reshape(-1, 1),
                "value_hidden": batch["value_hidden"],
                "policy_hidden": batch["policy_hidden"],
            }
            samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
            num_samples = batch_size  # 检查一下是不是和obs.shape[0]一样
            assert num_samples == obs.shape[0]
            index = np.arange(
                model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
            for k in samples:
                model_pool.fields[k][index, i] = samples[k][:, 0]
            current_nonterm = current_nonterm & nonterm_mask
            # if (current_nonterm).sum() <= 0:
            #     break
            obs = next_obs
            lst_action = action
            policy_hidden = policy_hidden_next

        model_pool._pointer += num_samples
        model_pool._pointer %= model_pool._max_size
        model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)

    for _key in rollout_res:
        if _key != "n_samples":
            rollout_res[_key] /= rollout_res["n_samples"]
    return rollout_res


# added by yinh
def fix_model_rollout(args, agent, transition, env_pool, model_pool, batch_size, deterministic=False,
                      select_model=None):
    assert isinstance(model_pool, SimpleReplayPool)
    batch = env_pool.random_batch(batch_size)
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    last_action = torch.zeros([obs.shape[0], args["action_shape"]], dtype=torch.float32).to(args.device)

    rollout_res = dict()
    with torch.no_grad():
        select_model = np.ones([batch_size]) * select_model
        for i in range(args.horizon):
            batch_size = obs.shape[0]
            action, _ = agent.get_action(obs, deterministic=deterministic, out_mean_std=False)
            next_sample = transition(torch.cat([obs, action], dim=-1)).sample()
            reward_raw = next_sample[:, :, -1:]
            next_obs = next_sample[:, :, :-1]

            reward_raw = reward_raw[select_model, np.arange(batch_size)]
            next_obs = next_obs[select_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args.task)

            if args["ablation"]["clip_obs"]:
                clip_num = ((next_obs < args["obs_min"]).any(dim=-1) | (next_obs > args["obs_max"]).any(
                    dim=-1)).sum().item()
                next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
                rollout_res["clip"] = rollout_res.get("clip", 0) + clip_num
            else:
                next_obs = next_obs
            reward_raw = torch.clamp(reward_raw, args["rew_min"], args["rew_max"])

            reward = reward_raw

            nonterm_mask = ~term.squeeze(-1)
            samples = {
                "observations": obs.cpu().numpy(),
                "actions": action.cpu().numpy(),
                "next_observations": next_obs.cpu().numpy(),
                "rewards": reward.cpu().numpy(),
                "terminals": term,
                "valid": np.ones_like(term),
                "last_actions": last_action.cpu().numpy()
            }
            model_pool.add_samples(samples)

            obs = next_obs[nonterm_mask]
            last_action = action[nonterm_mask]
            select_model = select_model[nonterm_mask]

    return


# def adv_meta_policy_rollout(args, agent, transition, env_pools, model_pool, batch_size, deterministic=False, rew_fn=None,
#                         clip_args=None, rnn=True):
#     batch = {}
#     total_batch_size = 0
#     for k in env_pools.keys():
#         bt = env_pools[k].random_batch_for_initial(batch_size)
#
#         for k in bt:
#             if k not in batch.keys():
#                 batch[k] = bt[k]
#             else:
#                 batch[k] = np.concatenate([batch[k], bt[k]], axis=0)  # (bs, seq, dim) or (bs, dim)
#         total_batch_size += batch_size
#
#     batch_size = total_batch_size
#     # batch = env_pool.random_batch_for_initial(batch_size)
#     obs = torch.from_numpy(batch["observations"]).to(args.device)
#     lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
#     value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
#     policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)
#
#     uncertainty_metric = {}
#     rollout_res = {
#         "raw_reward": 0,
#         "reward": 0,
#         "n_samples": 0,
#         "uncertainty": 0
#     }
#     current_nonterm = np.ones([len(obs)], dtype=bool)
#     with torch.no_grad():
#         select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
#         agent.reset()
#         for i in range(args["horizon"]):
#             if rnn:
#                 action, _, mu, logstd, policy_hidden_next = agent.get_action(obs, lst_action, policy_hidden,
#                                                                              deterministic=deterministic,
#                                                                              out_mean_std=True)
#             else:
#                 action, _, mu, logstd = agent.get_action(obs, deterministic=deterministic, out_mean_std=True)
#                 policy_hidden_next = policy_hidden
#             next_obs_dists = transition(torch.cat([obs, action], dim=-1))
#             next_sample = next_obs_dists.sample()
#
#             next_obses_mode = next_obs_dists.mean[:, :, :-1]
#             next_obs_mean = torch.mean(next_obses_mode, dim=0)
#             diff = next_obses_mode - next_obs_mean
#             disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
#             aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
#             # uncertainty = disagreement_uncertainty if args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty
#             # uncertainty = aleatoric_uncertainty
#             uncertainty = disagreement_uncertainty
#
#             reward = next_sample[:, :, -1:]
#             next_obs = next_sample[:, :, :-1]
#             # compute transition
#
#             # select model and clamp
#             reward = reward[select_model, np.arange(batch_size)]
#             next_obs = next_obs[select_model, np.arange(batch_size)]
#             term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
#             next_obs = torch.clamp(next_obs, clip_args["obs_min"], clip_args["obs_max"])
#             if rew_fn is not None:
#                 # if hy:
#                 #     reward_raw = rew_fn(obs, action, next_obs, torch.from_numpy(term).to(args.device))
#                 #     reward_raw += torch.clamp(reward, clip_args["rew_min"], clip_args["rew_max"])
#                 # else:
#                 reward_raw = torch.clamp(rew_fn(obs, action, next_obs, torch.from_numpy(term).to(args.device)),
#                                          clip_args['rew_min'], clip_args['rew_max'])
#             else:
#                 reward_raw = torch.clamp(reward, clip_args["rew_min"], clip_args["rew_max"])
#
#             reward = reward_raw + args["Meta"]["lam"] * uncertainty
#
#             rollout_res["raw_reward"] += reward_raw.sum().item()
#             rollout_res["reward"] += reward.sum().item()
#             rollout_res["n_samples"] += (~term).sum()
#             rollout_res["uncertainty"] += uncertainty.sum().item()
#
#             nonterm_mask = ~term.squeeze(-1)
#             samples = {
#                 "observations": obs.cpu().numpy(),
#                 "actions": action.cpu().numpy(),
#                 "next_observations": next_obs.cpu().numpy(),
#                 "rewards": reward.cpu().numpy(),
#                 "terminals": term,
#                 "last_actions": lst_action.cpu().numpy(),
#                 "valid": current_nonterm.reshape(-1, 1),
#                 "value_hidden": batch["value_hidden"],
#                 "policy_hidden": batch["policy_hidden"],
#             }
#             samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
#             num_samples = batch_size  # 检查一下是不是和obs.shape[0]一样
#             assert num_samples == obs.shape[0]
#             index = np.arange(
#                 model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
#             for k in samples:
#                 model_pool.fields[k][index, i] = samples[k][:, 0]
#             current_nonterm = current_nonterm & nonterm_mask
#             # if (current_nonterm).sum() <= 0:
#             #     break
#             obs = next_obs
#             lst_action = action
#             policy_hidden = policy_hidden_next
#
#         model_pool._pointer += num_samples
#         model_pool._pointer %= model_pool._max_size
#         model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)
#
#     for _key in rollout_res:
#         if _key != "n_samples":
#             rollout_res[_key] /= rollout_res["n_samples"]
#     return rollout_res


# def fix_model_rollout(args, agent, transition, env_pool, model_pool, batch_size, deterministic=False, probe_policy=None, select_model = None):
#     batch = env_pool.random_batch_for_initial(batch_size)
#     obs = torch.from_numpy(batch["observations"]).to(args.device)
#     lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
#     probe_metric = {}
#     current_nonterm = np.ones([len(obs)], dtype=bool)
#     with torch.no_grad():
#         if select_model is None:
#             select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
#         for i in range(args["horizon"]):
#             action, _, mu, logstd = agent.get_action(obs, deterministic=deterministic, out_mean_std=True)
#             next_sample = transition(torch.cat([obs, action], dim=-1)).sample()
#             reward = next_sample[:, :, -1:]
#             next_obs = next_sample[:, :, :-1]

#             # compute transition

#             reward = reward[select_model, np.arange(batch_size)]
#             next_obs = next_obs[select_model, np.arange(batch_size)]
#             term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"])
#             next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
#             reward = torch.clamp(reward, args["rew_min"], args["rew_max"])

#             # compute probe policy
#             if probe_policy is not None:
#                 if args["ablation"]["probe_policy_div"] == "mse":
#                     probe_actions = [pp.get_action(obs, deterministic=True)[0] for pp in probe_policy]
#                     probe_diffs = [((a-action)**2).sum(-1) for a in probe_actions]
#                     probe_diffs = sum(probe_diffs) / len(probe_diffs)
#                     reward += args["Probe"]["div_coeff"] * probe_diffs.unsqueeze(-1)
#                     if "mse" not in probe_metric:
#                         probe_metric["mse"] = []
#                     probe_metric["mse"].append(probe_diffs.mean().item())

#             nonterm_mask = ~term.squeeze(-1)
#             samples = {
#                 "observations": obs.cpu().numpy(),
#                 "actions": action.cpu().numpy(),
#                 "next_observations": next_obs.cpu().numpy(),
#                 "rewards": reward.cpu().numpy(),
#                 "terminals": term,
#                 "last_actions": lst_action.cpu().numpy(),
#                 "valid": current_nonterm.reshape(-1, 1),
#             }
#             # samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
#             num_samples = batch_size    # check whether equals to obs.shape[0]
#             assert num_samples == obs.shape[0]

#             model_pool.add_samples(samples)
#             current_nonterm = current_nonterm & nonterm_mask
#             obs = next_obs
#             lst_action = action

#             """
#             index = np.arange(
#                 model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
#             for k in samples:
#                 model_pool.fields[k][index] = samples[k]
#             current_nonterm = current_nonterm & nonterm_mask
#             obs = next_obs

#             model_pool._pointer += num_samples
#             model_pool._pointer %= model_pool._max_size
#             model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)
#             """
#     for key in probe_metric:
#         probe_metric[key] = np.mean(probe_metric[key])

#     return probe_metric

def _aleatoric(sample, mu, std):
    return torch.max(torch.norm(std, dim=-1, keepdim=True), dim=0)[0]


def aleatoric(sample, dist):
    return torch.max(torch.norm(dist.stddev, dim=-1, keepdim=True), dim=0)[0]


def ensemble(sample, dist):
    mu, std = dist.loc, dist.stddev
    avg_mu = mu.mean(dim=0)
    return ((std.square() + mu.square()).mean(0) - avg_mu.square()).sqrt()


def meta_rollout(args, agent, meta_dynamics, env_pool, model_pool, batch_size, device, deterministic=False, z=None,
                 lam=0, uncertainty_mode="aleatoric"):
    assert z is not None, "z cannot be None!"
    assert isinstance(env_pool, SimpleReplayTrajPool)
    assert isinstance(model_pool, SimpleReplayTrajPool)
    z_size = z.shape[0]
    if uncertainty_mode == "aleatoric":
        uncertainty_fn = _aleatoric
    elif uncertainty_mode == "ensemble":
        uncertainty_fn = ensemble
    else:
        raise ValueError

    batch = env_pool.random_batch_for_initial(batch_size)
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
    value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
    policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)

    rollout_res = {
        "n_samples": 0,
        "reward": 0,
        "reward_raw": 0,
        "uncertainty": 0
    }
    current_nonterm = np.ones([len(obs)], dtype=bool)
    selected_model = np.random.randint(0, z_size, size=batch_size)
    with torch.no_grad():
        for i in range(args.horizon):
            torch.cuda.empty_cache()
            action, _, mu, logstd, policy_hidden_next = agent.get_action(obs, lst_action, policy_hidden,
                                                                         deterministic=deterministic, out_mean_std=True)
            # changed by yinh: obs_action.shape == z_size * batch_size * dim
            obs_action = torch.cat([obs, action], dim=-1).unsqueeze(0).repeat((z_size, 1, 1))

            # z_use = torch.randn((z_size, batch_size, args.MetaDynamics.latent_dim * args.Probe.num_policy)).to(device)
            # z_use = (z_use - 0.5) * 1
            z_use = z.unsqueeze(1).repeat(1, batch_size, 1)

            obs_action, z_use = obs_action.reshape((z_size * batch_size, -1)), z_use.reshape((z_size * batch_size, -1))
            next_sample_dist = meta_dynamics(obs_action, z_use)
            next_sample = next_sample_dist.sample()
            torch.cuda.empty_cache()

            _mu, _std = next_sample_dist.loc, next_sample_dist.stddev
            _mu, _std = _mu.reshape([z_size, batch_size, -1]), _std.reshape([z_size, batch_size, -1])
            uncertainty = uncertainty_fn(next_sample, _mu, _std)
            next_sample = next_sample.reshape([z_size, batch_size, -1])
            next_obs = next_sample[:, :, :-1]
            reward_raw = next_sample[:, :, -1:]

            next_obs = next_obs[selected_model, np.arange(batch_size)]
            reward_raw = reward_raw[selected_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args.task)

            if args["ablation"]["clip_obs"]:
                clip_num = ((next_obs < args["obs_min"]).any(dim=-1) | (next_obs > args["obs_max"]).any(
                    dim=-1)).sum().item()
                next_obs = torch.clamp(next_obs, args["obs_min"], args["obs_max"])
                rollout_res["clip"] = rollout_res.get("clip", 0) + clip_num
            else:
                next_obs = next_obs
            reward_raw = torch.clamp(reward_raw, args["rew_min"], args["rew_max"])

            assert reward_raw.shape == uncertainty.shape
            reward = reward_raw + lam * uncertainty

            rollout_res["n_samples"] += current_nonterm.sum()
            rollout_res["reward"] += reward[current_nonterm].sum().item()
            rollout_res["reward_raw"] += reward_raw[current_nonterm].sum().item()
            rollout_res["uncertainty"] += uncertainty[current_nonterm].sum().item()

            nonterm_mask = ~term.squeeze(-1)
            samples = {
                "observations": obs.cpu().numpy(),
                "actions": action.cpu().numpy(),
                "next_observations": next_obs.cpu().numpy(),
                "rewards": reward.cpu().numpy(),
                "terminals": term,
                "last_actions": lst_action.cpu().numpy(),
                "valid": current_nonterm.reshape(-1, 1),
                "value_hidden": batch["value_hidden"],
                "policy_hidden": batch["policy_hidden"]
            }
            samples = {k: np.expand_dims(v, 1) for k, v in samples.items()}
            num_samples = batch_size
            assert num_samples == obs.shape[0]
            index = np.arange(
                model_pool._pointer, model_pool._pointer + num_samples
            ) % model_pool._max_size
            for k in samples:
                model_pool.fields[k][index, i] = samples[k][:, 0]

            current_nonterm = current_nonterm & nonterm_mask
            obs = next_obs
            lst_action = action
            policy_hidden = policy_hidden_next

        model_pool._pointer += num_samples
        model_pool._pointer %= model_pool._max_size
        model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)

    for _key in rollout_res:
        if _key != "n_samples":
            rollout_res[_key] /= rollout_res["n_samples"]
    return rollout_res