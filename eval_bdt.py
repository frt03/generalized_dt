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


from decision_transformer.models.decision_transformer import BidirectionalTransformer


plt.style.use('ggplot')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

VELOCITY_DIM = {
    'halfcheetah': (8, ),
    'hopper': (5, ),
    'walker2d': (8, ),
}


def experiment(output_dir, eval_dir, variant):
    gpu = variant.get('gpu', 0)
    device = torch.device(
        f"cuda:{gpu}" if (torch.cuda.is_available() and gpu >= 0) else "cpu"
    )

    env_name, dataset = variant['env'], variant['dataset']
    seed = variant['seed']
    n_bins = variant['n_bins']
    gamma = variant['gamma']
    assert gamma == 1.
    z_dim = variant['z_dim']

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
    # env.seed(seed)
    eval_env.seed(2 ** 32 - 1 - seed)

    state_dim = eval_env.observation_space.shape[0]
    act_dim = eval_env.action_space.shape[0]

    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    states, traj_lens, returns, rewards, xvel, yvel = [], [], [], [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        rewards.extend(path['rewards'])
        xvel.extend(path['observations'][:, vel_dim[0]])
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    xvel = np.array(xvel)

    r_min = min(rewards)
    r_max = max(rewards)
    bins = np.linspace(r_min, r_max, n_bins)
    label = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    width = bins[1] - bins[0]

    x_min = min(xvel)
    x_max = max(xvel)
    xbins = np.linspace(x_min, x_max, n_bins)
    xlabel = [(xbins[i]+xbins[i+1])/2 for i in range(len(xbins)-1)]
    xwidth = xbins[1] - xbins[0]

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print(f'z-dim: {z_dim}')
    print('=' * 50)

    K = variant['K']

    print('Preparing empirical distributions.')
    # for evaluation with best/50% trajectories
    _idxes = np.argsort([np.sum(path['rewards']) for path in trajectories]) # rank 0 is the most bad demo.
    trajs_rank = np.empty_like(_idxes)
    trajs_rank[_idxes] = np.arange(len(_idxes))
    n_evals = 5

    # train / eval split
    best_trajs = [np.where(trajs_rank == len(trajs_rank)-idx-1)[0][0] for idx in range(n_evals)]  # top-{n_evals}
    middle_trajs = [np.where(trajs_rank == int(len(trajs_rank)/2)+idx-2)[0][0] for idx in range(n_evals)]  # 50%-{n_evals}

    best_trajs_s = [trajectories[i]['observations'] for i in best_trajs]
    best_trajs_r = [trajectories[i]['rewards'] for i in best_trajs]
    best_trajs_x = [trajectories[i]['observations'][:, vel_dim[0]] for i in best_trajs]

    middle_trajs_s = [trajectories[i]['observations'] for i in middle_trajs]
    middle_trajs_r = [trajectories[i]['rewards'] for i in middle_trajs]
    middle_trajs_x = [trajectories[i]['observations'][:, vel_dim[0]] for i in middle_trajs]

    eval_trajectories = {}
    for i in range(n_evals):
        eval_trajectories['best_traj_{}'.format(i)] = (best_trajs_s[i], best_trajs_r[i], best_trajs_x[i])
        eval_trajectories['middle_traj_{}'.format(i)] = (middle_trajs_s[i], middle_trajs_r[i], middle_trajs_x[i])

    model = BidirectionalTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=variant['embed_dim'],
        z_dim=z_dim,
        max_length=K,
        max_ep_len=max_ep_len,
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
            returns, traj_rewards, traj_xs = [], [], []
            traj_ys = []
            sim_states = []
            for _ in range(variant['num_eval_episodes']):
                state = eval_env.reset()
                states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
                actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
                rewards = torch.zeros(0, device=device, dtype=torch.float32)
                # K-step anti-causal aggregation
                target_states = torch.from_numpy(v[0][0:K]).reshape(-1, state_dim).to(device=device, dtype=torch.float32)
                # reverse ordered
                target_states = torch.flip(target_states, dims=(1, )).to(device=device)
                target_timesteps = torch.flip(torch.arange(K), dims=(0,)).to(device=device, dtype=torch.long)
                target_attention_mask = torch.cat([torch.zeros(K-target_states.shape[1]), torch.ones(target_states.shape[1])])
                target_attention_mask = torch.flip(target_attention_mask, dims=(0,)).to(dtype=torch.long, device=device)


                timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
                target_return = torch.tensor(0, device=device, dtype=torch.float32).reshape(1, 1)  # dummy

                episode_return, episode_length = 0, 0
                max_traj_len = len(v[0])
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
                        target_states=(target_states.to(dtype=torch.float32) - state_mean) / state_std,
                        target_masks=target_attention_mask,
                        target_timesteps=target_timesteps,
                    )

                    sim_states.append((states.to(dtype=torch.float32) - state_mean) / state_std)

                    actions[-1] = action
                    action = action.detach().cpu().numpy()

                    state, reward, done, _ = eval_env.step(action)

                    traj_rewards.append(reward)
                    traj_xs.append(state[vel_dim[0]])

                    cur_state = torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(1, state_dim)
                    states = torch.cat([states, cur_state], dim=0)
                    rewards[-1] = reward

                    if t < max_traj_len - 1:
                        target_states = torch.from_numpy(v[0][t+1:t+K+1]).reshape(-1, state_dim).to(device=device)
                        target_states = torch.cat([target_states, torch.zeros((K-len(target_states), state_dim), device=device)], dim=0)
                        target_states = torch.flip(target_states, dims=(0, )).to(device=device, dtype=torch.float32)
                        target_timesteps = torch.flip(torch.clamp(torch.arange(t+1, t+K+1), max=max_traj_len-1), dims=(0, )).to(device=device, dtype=torch.long)
                        target_attention_mask = torch.cat([torch.ones(target_states.shape[1]), torch.zeros(K-target_states.shape[1])])
                        target_attention_mask = torch.flip(target_attention_mask, dims=(0, )).to(dtype=torch.long, device=device)

                    timesteps = torch.cat(
                        [timesteps,torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)],
                        dim=1
                        )

                    episode_return += reward
                    episode_length += 1

                    if done:
                        break
                returns.append(episode_return)

            outputs[f'target_{k}_return'] = np.mean(returns)
            wsd = wasserstein.EMD()
            # eval on reward dist
            rollout_reward_distribution = np.histogram(np.clip(np.array(traj_rewards), r_min, r_max), bins=bins)[0].astype(float)
            rollout_reward_distribution /= (rollout_reward_distribution.sum() + 1e-12)
            target_reward_distribution = np.histogram(np.clip(v[1], r_min, r_max), bins=bins)[0].astype(float)
            target_reward_distribution /= (target_reward_distribution.sum() + 1e-12)

            distance = wsd(
                target_reward_distribution,
                np.array(label).reshape(-1, 1),
                rollout_reward_distribution,
                np.array(label).reshape(-1, 1)
                )

            plt.bar(label, target_reward_distribution, width, color='dodgerblue', alpha=0.5, label='target')
            plt.bar(label, rollout_reward_distribution, width, color='tomato', alpha=0.5, label='rollout')
            plt.legend()
            plt.xlabel('Reward')
            plt.ylabel('Probability')
            plt.title('Distance={:.5f}'.format(distance))
            plt.savefig(os.path.join(eval_dir, f'reward_{k}_{itr}.pdf'), dpi=300)
            plt.close()
            outputs[f'target_{k}_w_dis_r'] = distance

            # eval on xdist
            rollout_xvel_distribution = np.histogram(np.clip(np.array(traj_xs), x_min, x_max), bins=xbins)[0].astype(float)
            rollout_xvel_distribution /= (rollout_xvel_distribution.sum() + 1e-12)
            target_xvel_distribution = np.histogram(np.clip(v[2], x_min, x_max), bins=xbins)[0].astype(float)
            target_xvel_distribution /= (target_xvel_distribution.sum() + 1e-12)

            distance = wsd(
                target_xvel_distribution,
                np.array(xlabel).reshape(-1, 1),
                rollout_xvel_distribution,
                np.array(xlabel).reshape(-1, 1)
                )

            plt.bar(xlabel, target_xvel_distribution, xwidth, color='dodgerblue', alpha=0.5, label='target')
            plt.bar(xlabel, rollout_xvel_distribution, xwidth, color='tomato', alpha=0.5, label='rollout')
            plt.legend()
            plt.xlabel('x-Velocity')
            plt.ylabel('Probability')
            plt.title('Distance={:.5f}'.format(distance))
            plt.savefig(os.path.join(eval_dir, f'xvel_{k}_{itr}.pdf'), dpi=300)
            plt.close()
            outputs[f'target_{k}_w_dis_x'] = distance

            if (itr == variant['max_iters'] - 1) and variant['save_rollout']:
                np.save(os.path.join(eval_dir, f'rollout_r_{k}_{itr}.npy'), np.array(traj_rewards).reshape(-1, 1))
                np.save(os.path.join(eval_dir, f'target_r_{k}_{itr}.npy'), v[1].reshape(-1, 1))
                np.save(os.path.join(eval_dir, f'rollout_x_{k}_{itr}.npy'), np.array(traj_xs).reshape(-1, 1))
                np.save(os.path.join(eval_dir, f'target_x_{k}_{itr}.npy'), v[2].reshape(-1, 1))
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
    parser.add_argument('--gamma', type=float, default=1.00)
    parser.add_argument('--z_dim', type=int, default=1)
    # for eval
    parser.add_argument('--num_eval_episodes', type=int, default=20)
    parser.add_argument('--save_rollout', type=bool, default=False)

    args = parser.parse_args()

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # log dir
    save_dir = f'{args.env}-{args.dataset}-bdt-z_{args.z_dim}-dim_{args.dist_dim}-bin_{args.n_bins}-gamma_{args.gamma}-ctx_{args.K}-seed_{args.seed}'
    output_dir = os.path.join('./results', save_dir)
    os.makedirs(output_dir, exist_ok=True)

    eval_dir = os.path.join(output_dir, f'eval')
    os.makedirs(eval_dir, exist_ok=True)

    with open(os.path.join(eval_dir, 'params_eval.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    experiment(output_dir, eval_dir, variant=vars(args))
