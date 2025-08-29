from typing import Any, Dict, Union
import numpy as np
import sapien
from typing import Union, List
import torch

from agents.surgical_continuum_robot import SurgicalContinuumRobot
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("SurgeryRobotMinimal-v1",  max_episode_steps=800)  
class SurgeryRobotMinimalEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["surgical_continuum_robot"]
    agent: Union[SurgicalContinuumRobot]

    # 核心参数（仅保留必要项）
    lesion_radius = 0.015          # 病灶半径
    waterjet_effective_dist = 0.03  # 水刀有效距离

    def __init__(
        self,
        *args,
        robot_uids="surgical_continuum_robot",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # 简化相机配置（仅基础观察相机）
        self.sensor_cam_config = {
            "eye": [0.6, 0, 1.5],
            "target": [0.25, 0, 0.85],
            "fov": np.pi / 3,
            "width": 256,
            "height": 256,
        }
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------- 1. 简化传感器配置 ---------------- #
    @property
    def _default_sensor_configs(self):
        main_cam_pose = sapien_utils.look_at(
            eye=self.sensor_cam_config["eye"],
            target=self.sensor_cam_config["target"]
        )
        return [
            CameraConfig(
                uid="main_cam",
                pose=main_cam_pose,
                width=self.sensor_cam_config["width"],
                height=self.sensor_cam_config["height"],
                fov=self.sensor_cam_config["fov"],
                near=0.01,
                far=2.0,
            )
        ]

    # ---------------- 2. 修复核心：ActorBuilder 传 scene + 位姿参数调整 ---------------- #
    def _load_agent(self, options: dict):
        # 机器人初始位姿（固定，简化）
        robot_init_pose = sapien.Pose(p=[0.0, 0.0, 0.0],
                            q=[1, 0, 0, 0]  # 直接使用绕x轴旋转180度的四元数，无需调用工具函数
                    )  
        super()._load_agent(options, robot_init_pose)
        # 添加水刀可视化（关键：传入 self.scene 给 ActorBuilder）
   

    # # ---------------- 3. 简化场景加载 ---------------- #
    # def _load_scene(self, options: dict):
    #     # 加载手术台（复用现有工具类，避免自定义复杂逻辑）        
    #     self.surgery_table = TableSceneBuilder(
    #     self, robot_init_qpos_noise=self.robot_init_qpos_noise
    #     )
    #     self.surgery_table.build()
    #     # 初始化病灶列表
    #     self.lesions: List[sapien.Actor] = []

    # ---------------- 4. 简化episode初始化 ---------------- #
    def _initialize_episode(self, env_idx, options: dict):
        # self.surgery_table.initialize(env_idx)
        # 机器人复位（无噪声，使用默认复位位姿）
        self.agent.reset(init_qpos=self.agent.keyframes["reset"].qpos)




    # ---------------- 5. 最小观察空间 ---------------- #
    def _get_obs_extra(self, info: dict):
        # 仅返回必要信息：末端位置、病灶位置、水刀激活状态
        ee_pos = self.agent.tcp_pose.p
        return {
            "ee_pos": ee_pos,
        }
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        return 0.0
    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
    # ---------------- 6. 简化step逻辑 ---------------- #
    def step(self, action: np.ndarray):

        # 调用父类step执行物理仿真
        obs, reward, terminated, truncated, info = super().step(action)

        # 简化奖励计算：靠近病灶+激活水刀即给奖励
        if self.lesions:
            ee_pos = self.agent.tcp_pose.p
            lesion_pos = self.lesions[0].pose.p
            dist = np.linalg.norm(lesion_pos - ee_pos)
            # 距离奖励（0~0.5）：越近奖励越高
            reward = np.clip(1 - (dist / self.waterjet_effective_dist), 0.0, 1.0) * 0.5
            # 成功条件：激活水刀且距离≤有效距离（3cm）

        return obs, reward, terminated, truncated, info