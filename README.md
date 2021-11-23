# Generalized Decision Transformer for Offline Hindsight Information Matching

[[arxiv]](https://arxiv.org/abs/2111.10364)

If you use this codebase for your research, please cite the paper:
```
@article{furuta2021generalized,
  title={Generalized Decision Transformer for Offline Hindsight Information Matching},
  author={Hiroki Furuta and Yutaka Matsuo and Shixiang Shane Gu},
  journal={arXiv preprint arXiv:2111.10364},
  year={2021}
}
```


## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.
Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.
Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Run experiments

Run train_cdt.py to train Categorical DT:
```
python train_cdt.py --env halfcheetah --dataset medium-expert --gpu 0 --seed 0 --dist_dim 30 --n_bins 31 --distributions 'categorical' --condition 'reward' --save_model True

python train_cdt.py --env halfcheetah --dataset medium-expert --gpu 0 --seed 0 --dist_dim 30 --n_bins 31 --distributions 'categorical' --condition 'xvel' --save_model True
```

Run eval_cdt.py to eval CDT using saved weights:
```
python eval_cdt.py --env halfcheetah --dataset medium-expert --gpu 0 --seed 0 --dist_dim 30 --n_bins 31 --distributions 'categorical' --condition 'reward' --eval_target 'fix' --save_rollout True
```


For Bi-directional DT, run train_bdt.py & eval_bdtf.py
```
python train_bdt.py --env halfcheetah --dataset medium-expert --gpu 0 --seed 0 --dist_dim 30 --n_bins 31 --z_dim 16 --save_model True
python eval_bdt.py --env halfcheetah --dataset medium-expert --gpu 0 --seed 0 --dist_dim 30 --n_bins 31 --eval_target 'fix' --z_dim 16 --save_rollout True
```

## Reference
This repository is developed on top of original [Decision Transformer](https://github.com/kzl/decision-transformer).
