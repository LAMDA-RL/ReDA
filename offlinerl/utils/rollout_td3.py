import torch
from torch.distributions import kl, Normal
import numpy as np

from offlinerl.utils.net.terminal_check import is_terminal
from offlinerl.utils.simple_replay_pool import SimpleReplayPool, SimpleReplayTrajPool


def adv_rollout(args, agent, transition, env_pool, model_pool, deterministic=False, clip_args=None):
    batch = env_pool.initial_batch
    obs = torch.from_numpy(batch["observations"]).to(args.device)
    lst_action = torch.from_numpy(batch["last_actions"]).to(args.device)
    # value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
    # policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)
    batch_size = obs.shape[0]
    agent.reset()
    action = agent.get_action(obs, deterministic=deterministic, exp=True)
    rollout_res = {
        "raw_reward": 0,
        "reward": 0,
        "n_samples": 0,
        "uncertainty": 0,
        "rew_min": 0,
        "rew_max": 0,
    }
    current_nonterm = np.ones([len(obs)], dtype=bool)
    with torch.no_grad():
        select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
        for i in range(args["Adv"]["horizon"]):
            next_obs_dists = transition(torch.cat([obs, action], dim=-1))
            next_sample = next_obs_dists.sample()

            next_obses_mode = next_obs_dists.mean[:, :, :-1]
            next_obs_mean = torch.mean(next_obses_mode, dim=0)
            diff = next_obses_mode - next_obs_mean
            disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
            aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
            # uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty
            # uncertainty = aleatoric_uncertainty
            uncertainty = disagreement_uncertainty

            reward = next_sample[:, :, -1:]
            next_obs = next_sample[:, :, :-1]
            # compute transition

            # select model and clamp
            reward = reward[select_model, np.arange(batch_size)]

            next_obs = next_obs[select_model, np.arange(batch_size)]
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"],
                               clip_args["obs_min"].cpu().numpy(), clip_args["obs_max"].cpu().numpy())

            next_obs = torch.clamp(next_obs, clip_args["obs_min"], clip_args["obs_max"])
            # reward_raw = torch.clamp(reward, args["rew_min"], args["rew_max"])
            reward_raw = torch.clamp(reward, clip_args["rew_min"], clip_args["rew_max"])

            rollout_res["raw_reward"] += reward_raw.mean().item() * (~term).sum()
            rollout_res["reward"] += reward.mean().item() * (~term).sum()
            rollout_res["n_samples"] += (~term).sum()
            rollout_res["uncertainty"] += uncertainty.mean().item() * (~term).sum()
            rollout_res["rew_min"] += clip_args["rew_min"].mean() * (~term).sum()
            rollout_res["rew_max"] += clip_args["rew_max"].mean() * (~term).sum()

            nonterm_mask = ~term.squeeze(-1)

            next_action = agent.get_action(obs, deterministic=deterministic, exp=True)

            reward_hidden = env_pool.step_rew(next_obs, next_action, action)
            rollout_res["hidden_reward"] = reward_hidden.sum().item()

            reward = args["Adv"]["lam"] * reward_hidden \
                     + args["Adv"]["lam_rew"] * reward_raw + args["Adv"]["lam_pen"] * uncertainty

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
            num_samples = batch_size
            assert num_samples == obs.shape[0]
            index = np.arange(
                model_pool._pointer, model_pool._pointer + num_samples) % model_pool._max_size
            for k in samples:
                model_pool.fields[k][index, i] = samples[k][:, 0]
            current_nonterm = current_nonterm & nonterm_mask

            obs = next_obs
            action = next_action

        model_pool._pointer += num_samples
        model_pool._pointer %= model_pool._max_size
        model_pool._size = min(model_pool._max_size, model_pool._size + num_samples)

    for _key in rollout_res:
        if _key != "n_samples":
            rollout_res[_key] /= rollout_res["n_samples"]
    return rollout_res



def meta_policy_rollout(args, agent, transition, env_pools, model_pool, batch_size, deterministic=False, rew_fn=None, clip_args=None, rnn=True):
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
    value_hidden = torch.from_numpy(batch["value_hidden"]).to(args.device)
    policy_hidden = torch.from_numpy(batch["policy_hidden"]).to(args.device)

    uncertainty_metric = {}
    rollout_res = {
        "raw_reward": 0,
        "reward": 0,
        "n_samples": 0,
        "uncertainty": 0,
        "rew_min": 0,
        "rew_max": 0,
    }
    current_nonterm = np.ones([len(obs)], dtype=bool)
    with torch.no_grad():
        select_model = np.random.randint(0, transition.ensemble_size, size=[batch_size])
        agent.reset()
        for i in range(args["horizon"]):
            if rnn:
                action, policy_hidden_next = agent.get_action(obs, lst_action, policy_hidden, deterministic=deterministic, exp=True)
            else:
                action = agent.get_action(obs, deterministic=deterministic, exp=True)
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
            term = is_terminal(obs.cpu().numpy(), action.cpu().numpy(), next_obs.cpu().numpy(), args["task"], clip_args["obs_min"].cpu().numpy(), clip_args["obs_max"].cpu().numpy())
            next_obs = torch.clamp(next_obs, clip_args["obs_min"], clip_args["obs_max"])
            if rew_fn is not None:
                reward_raw = torch.clamp(rew_fn(obs, action, next_obs, torch.from_numpy(term).to(args.device)),
                                         clip_args['rew_min'], clip_args['rew_max'])
            else:
                reward_raw = torch.clamp(reward, clip_args["rew_min"], clip_args["rew_max"])

            reward = reward_raw + args["Meta"]["lam"] * uncertainty

            rollout_res["raw_reward"] += reward_raw.mean().item() * (~term).sum()
            rollout_res["reward"] += reward.mean().item() * (~term).sum()
            rollout_res["n_samples"] += (~term).sum()
            rollout_res["uncertainty"] += uncertainty.mean().item() * (~term).sum()
            rollout_res["rew_min"] += clip_args["rew_min"].mean() * (~term).sum()
            rollout_res["rew_max"] += clip_args["rew_max"].mean() * (~term).sum()

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
            num_samples = batch_size
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
