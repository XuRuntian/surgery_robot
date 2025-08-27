import gymnasium as gym
import mani_skill.envs
import time
from envs.tasks.surgery_robot.base_env import SurgeryRobotEnv
env = gym.make("SurgeryRobot-v1")