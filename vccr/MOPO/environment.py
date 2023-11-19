import mujoco
import numpy as np
from utils import xml_string


class HalfCheetahEnv():
  def __init__(self, frame_skip=5, forward_reward_weight=1.0, ctrl_cost_weight=0.1, reset_noise_scale=0.1):
    self.frame_skip = frame_skip
    self.forward_reward_weight = forward_reward_weight
    self.ctrl_cost_weight = ctrl_cost_weight
    self.reset_noise_scale = reset_noise_scale

    self.initialize_simulation()
    self.init_qpos = self.data.qpos.ravel().copy()
    self.init_qvel = self.data.qvel.ravel().copy()
    self.dt = self.model.opt.timestep * self.frame_skip

    self.observation_dim = 17
    self.action_dim = 6
    self.action_limit = 1.

  def initialize_simulation(self):
    self.model = mujoco.MjModel.from_xml_string(xml_string)
    self.data = mujoco.MjData(self.model)
    mujoco.mj_resetData(self.model, self.data)
    self.renderer = mujoco.Renderer(self.model)

  def reset_simulation(self):
    mujoco.mj_resetData(self.model, self.data)

  def step_mujoco_simulation(self, ctrl, n_frames):
    self.data.ctrl[:] = ctrl
    mujoco.mj_step(self.model, self.data, nstep=n_frames)
    self.renderer.update_scene(self.data,0)

  def set_state(self, qpos, qvel):
    self.data.qpos[:] = np.copy(qpos)
    self.data.qvel[:] = np.copy(qvel)
    if self.model.na == 0:
      self.data.act[:] = None
    mujoco.mj_forward(self.model, self.data)

  def sample_action(self):
    return (2.*np.random.uniform(size=(self.action_dim,)) - 1)*self.action_limit

  def step(self, action):
    x_position_before = self.data.qpos[0]
    self.step_mujoco_simulation(action, self.frame_skip)
    x_position_after = self.data.qpos[0]
    x_velocity = (x_position_after - x_position_before) / self.dt

    # Rewards
    ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))
    forward_reward = self.forward_reward_weight * x_velocity
    observation = self.get_obs()
    reward = forward_reward - ctrl_cost
    terminated = False
    info = {
        "x_position": x_position_after,
        "x_velocity": x_velocity,
        "reward_run": forward_reward,
        "reward_ctrl": -ctrl_cost,
    }
    return observation, reward, terminated, info

  def get_obs(self):
    position = self.data.qpos.flat.copy()
    velocity = self.data.qvel.flat.copy()
    position = position[1:]

    observation = np.concatenate((position, velocity)).ravel()
    return observation

  def render(self):
    return self.renderer.render()

  def reset(self):
    self.reset_simulation()
    noise_low = -self.reset_noise_scale
    noise_high = self.reset_noise_scale
    qpos = self.init_qpos + np.random.uniform(
        low=noise_low, high=noise_high, size=self.model.nq
    )
    qvel = (self.init_qvel + self.reset_noise_scale * np.random.standard_normal(self.model.nv))
    self.set_state(qpos, qvel)
    observation = self.get_obs()
    return observation