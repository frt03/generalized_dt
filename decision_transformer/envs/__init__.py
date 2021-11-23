import numpy as np
from gym.envs.registration import register


for vel in np.arange(0.0, 3.1, 0.1):
    register(
        id=f'CheetahVel{vel:.1f}-v0',
        entry_point='decision_transformer.envs.cheetah_vel:HalfCheetahVelEnv',
        max_episode_steps=200,
        kwargs={'goal_vel': vel},
    )

register(
        id=f'BackflipCheetah-v0',
        entry_point='decision_transformer.envs.backflip_cheetah:BackflipCheetahEnv',
        max_episode_steps=1000,
    )
