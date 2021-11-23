import argparse
import gym
import json
import os
import pickle
import random
import torch

import numpy as np

from decision_transformer.models.decision_transformer import BidirectionalTransformer
from decision_transformer.training.seq_trainer import SequenceTrainer


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(output_dir, variant):
    gpu = variant.get('gpu', 0)
    device = torch.device(
        f"cuda:{gpu}" if (torch.cuda.is_available() and gpu >= 0) else "cpu"
    )

    env_name, dataset = variant['env'], variant['dataset']
    seed = variant['seed']
    # n_bins = variant['n_bins']
    gamma = variant['gamma']
    assert gamma == 1.
    z_dim = variant['z_dim']

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
    else:
        raise NotImplementedError
    scale = 1000.
    max_ep_len = 1000
    env.seed(seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    states, traj_lens, returns, rewards = [], [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
        rewards.extend(path['rewards'])
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # for categorical distribution matching
    # r_min = min(rewards)
    # r_max = max(rewards)
    # bins = np.linspace(r_min, r_max, n_bins)
    # label = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]

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
    batch_size = variant['batch_size']

    # for evaluation with best/50% trajectories
    _idxes = np.argsort([np.sum(path['rewards']) for path in trajectories]) # rank 0 is the most bad demo.
    trajs_rank = np.empty_like(_idxes)
    trajs_rank[_idxes] = np.arange(len(_idxes))
    n_evals = 5
    # train / eval split
    eval_indices = [np.where(trajs_rank == len(trajs_rank)-idx-1)[0][0] for idx in range(n_evals)] + [np.where(trajs_rank == int(len(trajs_rank)/2)+idx-2)[0][0] for idx in range(n_evals)]
    # remove eval trajectories
    train_indices =  [i for i in range(len(trajs_rank))]
    for i in eval_indices:
        train_indices.remove(i)

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.array(train_indices),
            size=batch_size,
            replace=True,
        )
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device) / scale
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

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

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=None,
    )

    print('Starting training loop.')
    for itr in range(variant['max_iters']):
        outputs = trainer.train_only_iteration(num_steps=variant['num_steps_per_iter'], iter_num=itr+1, print_logs=True)
        if variant['save_model']:
            torch.save(model.state_dict(), os.path.join(output_dir, f'dt_{itr}.pth'))
        # record training loss, etc...
        if itr == 0:
            _basic_columns = ['iter']
            _record_values = [itr]
            for k, v in outputs.items():
                _basic_columns.append(k)
                _record_values.append(v)
            with open(os.path.join(output_dir, "train_log.txt"), "w") as f:
                print("\t".join(_basic_columns), file=f)
            with open(os.path.join(output_dir, "train_log.txt"), "a+") as f:
                print("\t".join(str(x) for x in _record_values), file=f)
        else:
            _record_values = [itr]
            for v in outputs.values():
                _record_values.append(v)
            with open(os.path.join(output_dir, "train_log.txt"), "a+") as f:
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
    parser.add_argument('--save_model', type=bool, default=False)
    parser.add_argument('--z_dim', type=int, default=1)

    args = parser.parse_args()

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # log dir
    save_dir = f'{args.env}-{args.dataset}-bdt-z_{args.z_dim}-dim_{args.dist_dim}-bin_{args.n_bins}-gamma_{args.gamma}-ctx_{args.K}-seed_{args.seed}'
    output_dir = os.path.join('./results', save_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'params.json'), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    experiment(output_dir, variant=vars(args))
