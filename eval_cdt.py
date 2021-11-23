import argparse
import gym
import json
import os
import pickle
import random
import time
import torch
import wasserstein

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from decision_transformer.models.decision_transformer import GeneralizedDecisionTransformer


plt.style.use('ggplot')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

VELOCITY_DIM = {
    'halfcheetah': (8, ),
    'hopper': (5, ),
    'walker2d': (8, ),
}


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(output_dir, eval_dir, variant):
    gpu = variant.get('gpu', 0)
    device = torch.device(
        f"cuda:{gpu}" if (torch.cuda.is_available() and gpu >= 0) else "cpu"
    )

    env_name, dataset = variant['env'], variant['dataset']
    seed = variant['seed']
    dist_dim = variant['dist_dim']
    n_bins = variant['n_bins']
    distributions = variant['distributions']
    assert distributions in ['categorical', 'deterministic']
    gamma = variant['gamma']
    if distributions != 'categorical':
        assert gamma == 1.
    condition = variant['condition']
    assert condition in ['reward', 'xvel']

    if env_name == 'hopper':
        eval_env = gym.make('Hopper-v3')
    elif env_name == 'halfcheetah':
        eval_env = gym.make('HalfCheetah-v3')
    elif env_name == 'walker2d':
        eval_env = gym.make('Walker2d-v3')
    else:
        raise NotImplementedError
    vel_dim = VELOCITY_DIM[env_name]
    max_ep_len = 1000
    eval_env.seed(2 ** 32 - 1 - seed)

    state_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    if distributions == 'categorical':
        r_dists_dim = dist_dim
    elif distributions == 'deterministic':
        r_dists_dim = 1

    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    states, traj_lens, returns, rewards = [], [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        if condition == 'reward':
            rewards.extend(path['rewards'])
        elif condition == 'xvel':
            rewards.extend(path['observations'][:, vel_dim[0]])
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # for categorical distribution matching
    r_min = min(rewards)
    r_max = max(rewards)
    bins = np.linspace(r_min, r_max, n_bins)
    label = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    width = bins[1] - bins[0]

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print(f'Modality: {condition}')
    print(f'Distribution: {distributions}')
    print('=' * 50)

    K = variant['K']

    print('Preparing empirical distributions.')
    # for evaluation with best/50% trajectories
    _idxes = np.argsort([np.sum(path['rewards']) for path in trajectories]) # rank 0 is the most bad demo.
    trajs_rank = np.empty_like(_idxes)
    trajs_rank[_idxes] = np.arange(len(_idxes))
    n_evals = 5

    r_dists = []
    if condition in ('reward', 'xvel') and distributions == 'categorical':
        for path in trajectories:
            dist = np.zeros(n_bins - 1)
            distributional_rewards = []
            steps_to_go = 0
            if condition == 'reward':
                modality = path['rewards']
            elif condition == 'xvel':
                modality = path['observations'][:, vel_dim[0]]
            for t, r in enumerate(reversed(modality)):
                discretized_r = np.histogram(np.clip(r, r_min, r_max), bins=bins)[0]
                steps_to_go *= gamma
                dist *= steps_to_go
                dist = discretized_r + dist
                dist_norm = dist.sum()
                dist /= dist_norm
                steps_to_go += 1
                distributional_rewards.append(dist)
            path['r_dists'] = np.concatenate(distributional_rewards[::-1], axis=0).reshape(-1, n_bins - 1)
            r_dists.append(path['r_dists'])
    elif condition in ('reward', 'xvel') and distributions == 'deterministic':
        for path in trajectories:
            dist = 0
            distributional_rewards = []
            if condition == 'reward':
                modality = path['rewards']
            elif condition == 'xvel':
                modality = path['observations'][:, vel_dim[0]]
            for t, r in enumerate(reversed(modality)):
                dist *= max_ep_len
                dist = gamma * dist + r
                dist /= max_ep_len
                distributional_rewards.append(dist)
            path['r_dists'] = np.array(distributional_rewards[::-1]).reshape(-1, 1)
            r_dists.append(path['r_dists'])
    else:
        raise NotImplementedError
    assert len(trajs_rank) == len(r_dists)
    # train / eval split
    best_trajs = [r_dists[np.where(trajs_rank == len(trajs_rank)-idx-1)[0][0]] for idx in range(n_evals)]  # top-{n_evals}
    middle_trajs = [r_dists[np.where(trajs_rank == int(len(trajs_rank)/2)+idx-2)[0][0]] for idx in range(n_evals)]  # 50%-{n_evals}

    if condition == 'reward':
        best_trajs_all = [
            np.histogram(np.clip(trajectories[np.where(trajs_rank == len(trajs_rank)-idx-1)[0][0]]['rewards'], r_min, r_max), bins=bins)[0].astype(float) for idx in range(n_evals)]
        best_trajs_all = [t/(t.sum() + 1e-12) for t in best_trajs_all]
        middle_trajs_all = [
            np.histogram(np.clip(trajectories[np.where(trajs_rank == int(len(trajs_rank)/2)+idx-2)[0][0]]['rewards'], r_min, r_max), bins=bins)[0].astype(float) for idx in range(n_evals)]
        middle_trajs_all = [t/(t.sum() + 1e-12) for t in middle_trajs_all]
    elif condition == 'xvel':
        best_trajs_all = [
            np.histogram(np.clip(trajectories[np.where(trajs_rank == len(trajs_rank)-idx-1)[0][0]]['observations'][:, vel_dim[0]], r_min, r_max), bins=bins)[0].astype(float) for idx in range(n_evals)]
        best_trajs_all = [t/(t.sum() + 1e-12) for t in best_trajs_all]
        middle_trajs_all = [
            np.histogram(np.clip(trajectories[np.where(trajs_rank == int(len(trajs_rank)/2)+idx-2)[0][0]]['observations'][:, vel_dim[0]], r_min, r_max), bins=bins)[0].astype(float) for idx in range(n_evals)]
        middle_trajs_all = [t/(t.sum() + 1e-12) for t in middle_trajs_all]
    else:
        raise NotImplementedError

    eval_trajectories = {}
    for i in range(n_evals):
        eval_trajectories['best_traj_{}'.format(i)] = (best_trajs[i], best_trajs_all[i])
        eval_trajectories['middle_traj_{}'.format(i)] = (middle_trajs[i], middle_trajs_all[i])

    model = GeneralizedDecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=K,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        dist_dim=r_dists_dim,
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4*variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )

    model = model.to(device=device)

    print('Starting evaluation loop.')
    model.eval()

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # for itr in range(variant['max_iters']):
    for itr in [9]:
        outputs = dict()
        model.load_state_dict(torch.load(os.path.join(output_dir, f'dt_{itr}.pth'), map_location=device))
        eval_start = time.time()
        for k, v in eval_trajectories.items():
            returns, traj_rewards = [], []
            traj_rewardsy = []
            for _ in range(variant['num_eval_episodes']):

                state = eval_env.reset()
                states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
                actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
                rewards = torch.zeros(0, device=device, dtype=torch.float32)
                input_distribution = v[0]

                next_target = input_distribution[0]

                target_distributions = torch.from_numpy(
                    next_target).to(device=device, dtype=torch.float32).reshape(1, r_dists_dim)

                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

                # dummy
                target_return = torch.tensor(0, device=device, dtype=torch.float32).reshape(1, 1)

                episode_return, episode_length = 0, 0
                max_traj_len = len(input_distribution)

                for t in range(max_traj_len):
                    # add padding
                    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
                    rewards = torch.cat([rewards, torch.zeros(1, device=device)])
                    action = model.get_action(
                        (states.to(dtype=torch.float32) - state_mean) / state_std,
                        actions.to(dtype=torch.float32),
                        rewards.to(dtype=torch.float32),
                        target_return.to(dtype=torch.float32),
                        timesteps.to(dtype=torch.long),
                        target_distributions.to(dtype=torch.float32),
                    )
                    actions[-1] = action
                    action = action.detach().cpu().numpy()

                    state, reward, done, _ = eval_env.step(action)

                    if condition == 'xvel':
                        traj_rewards.append(state[vel_dim[0]])
                    elif condition == 'reward':
                        traj_rewards.append(reward)
                    cur_state = torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(1, state_dim)
                    states = torch.cat([states, cur_state], dim=0)
                    rewards[-1] = reward

                    if t < max_traj_len - 1:
                        next_target = input_distribution[t+1]
                        target = next_target

                        target_distributions = torch.cat(
                            [
                                target_distributions,
                                torch.from_numpy(target).reshape(1, r_dists_dim).to(device=device, dtype=torch.float32)
                                ],
                            dim=1
                        )

                    timesteps = torch.cat(
                        [timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)],
                        dim=1
                        )

                    episode_return += reward
                    episode_length += 1

                    if done:
                        break
                returns.append(episode_return)

            # evaluation
            target_all = v[1]
            wsd = wasserstein.EMD()
            all_reward_distribution = np.histogram(np.clip(np.array(traj_rewards), r_min, r_max), bins=bins)[0].astype(float)
            all_reward_distribution /= (all_reward_distribution.sum() + 1e-12)
            distance = wsd(
                target_all,
                np.array(label).reshape(-1, 1),
                all_reward_distribution,
                np.array(label).reshape(-1, 1)
                )
            plt.bar(label, target_all, width, color='dodgerblue', alpha=0.5, label='target')
            plt.bar(label, all_reward_distribution, width, color='tomato', alpha=0.5, label='rollout')
            plt.legend()
            if condition == 'reward':
                xlabel = 'Reward'
            elif condition == 'xvel':
                xlabel = 'x-Velocity'
            plt.xlabel(xlabel)
            plt.ylabel('Probability')
            plt.title('Distance={:.5f}'.format(distance))
            plt.savefig(os.path.join(eval_dir, f'categorical_{k}_{itr}.pdf'), dpi=300)
            plt.close()
            if condition == 'reward':
                outputs[f'target_{k}_w_dis_r'] = distance
            elif condition == 'xvel':
                outputs[f'target_{k}_w_dis_x'] = distance
            outputs[f'target_{k}_return'] = np.mean(returns)
            if (itr == variant['max_iters'] - 1) and variant['save_rollout']:
                np.save(os.path.join(eval_dir, f'rollout_{k}_{itr}.npy'), traj_rewards)
                np.save(os.path.join(eval_dir, f'target_{k}_{itr}.npy'), target_all)

        outputs['time/evaluation'] = time.time() - eval_start

        print('=' * 80)
        print(f'Iteration {itr}')
        for k, v in outputs.items():
            print(f'{k}: {v}')

        _record_values = [itr]
        _basic_columns = ['iter']
        for k, v in outputs.items():
            _basic_columns.append(k)
            _record_values.append(v)
        with open(os.path.join(eval_dir, "eval_log.txt"), "w") as f:
            print("\t".join(_basic_columns), file=f)
        with open(os.path.join(eval_dir, "eval_log.txt"), "a+") as f:
            print("\t".join(str(x) for x in _record_values), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='halfcheetah')
    parser.add_argument('--dataset', type=str, default='medium-expert')
    parser.add_argument('--condition', type=str, default='reward')  # or xvel
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dist_dim', type=int, default=30)
    parser.add_argument('--n_bins', type=int, default=31)
    parser.add_argument('--distributions', type=str, default='categorical')  # or deterministic
    parser.add_argument('--gamma', type=float, default=1.00)
    # for eval
    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--save_rollout', type=bool, default=False)

    args = parser.parse_args()

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # log dir
    save_dir = f'{args.env}-{args.dataset}-{args.distributions}-dim_{args.dist_dim}-bin_{args.n_bins}-gamma_{args.gamma}-{args.condition}-ctx_{args.K}-seed_{args.seed}'
    print(save_dir)
    output_dir = os.path.join('./results', save_dir)
    assert os.path.exists(output_dir)

    eval_dir = os.path.join(output_dir, f'eval')
    print(eval_dir)

    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, 'params_eval.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    experiment(output_dir, eval_dir, variant=vars(args))
