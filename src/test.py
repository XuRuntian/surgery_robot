import gymnasium as gym
import mani_skill.envs
import time
from envs.tasks.surgery_robot.base_env import SurgeryRobotMinimalEnv
env = gym.make("SurgeryRobotMinimal-v1", render_mode="human")
obs, info = env.reset()

# 简单交互循环（示例）
for _ in range(1000):
    # action = env.action_space.sample()  # 随机动作（可替换为你的控制逻辑）
    # obs, reward, terminated, truncated, info = env.step(action)
    env.render()  # 刷新可视化窗口
    time.sleep(0.01)  # 控制帧率，避免画面过快

    # if terminated or truncated:
    #     obs, info = env.reset()  # 重置环境

env.close()  # 关闭环境和窗口