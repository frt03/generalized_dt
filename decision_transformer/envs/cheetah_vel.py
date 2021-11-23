import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahVelEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, goal_vel=0.0):
        self.goal_vel = goal_vel
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            goal_vel=self.goal_vel,
            forward_vel=forward_vel,
            xposbefore=xposbefore,
        )
        return (observation, reward, done, infos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human', width=500, height=500, **kwargs):
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width=width, height=height)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)[::-1, :, :]
            return data
        elif mode == 'human':
            self._get_viewer(mode).render()
