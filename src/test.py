import gymnasium as gym
import mani_skill.envs
import time
import sapien
from mani_skill.agents.utils import *
from envs.tasks.surgery_robot.base_env import SurgeryRobotMinimalEnv
env = gym.make(
    "SurgeryRobotMinimal-v1",
    obs_mode="state",
    num_envs=11,
    # parallel_in_single_scene=True,
    # robot_init_qpos_noise=0.1,
)
env.reset()
while True:
    env.step(env.action_space.sample())
    env.render_human()