import gymnasium as gym
import mani_skill.envs
import time
import sapien
from mani_skill.agents.utils import *
from envs.tasks.surgery_robot.base_env import SurgeryRobotMinimalEnv
env = gym.make("SurgeryRobotMinimal-v1", render_mode="human")
obs, info = env.reset()
for _ in range(300000):
    env.render()  # 刷新可视化窗口
    time.sleep(0.01)  # 控制帧率，避免画面过快
env.close()  # 关闭环境和窗口