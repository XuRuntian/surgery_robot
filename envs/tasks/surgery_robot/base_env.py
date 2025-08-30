from typing import Any, Dict, Union, List
import torch
import numpy as np
import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building import actors
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sapien_env import BaseEnv
from agents.surgical_continuum_robot import SurgicalContinuumRobot


@register_env("SurgeryRobotMinimal-v1", max_episode_steps=800)
class SurgeryRobotMinimalEnv(BaseEnv):
    _sample_video_link = ""
    SUPPORTED_ROBOTS = ["surgical_continuum_robot"]
    agent: Union[SurgicalContinuumRobot]

    # 核心参数
    lesion_radius = 0.015
    lesion_color = np.array([1.0, 0.2, 0.2, 1.0])
    waterjet_effective_dist = 0.03
    robot_init_pose = sapien.Pose(p=[0.0, 0.0, 0.1], q=[0, 1, 0, 0])
    lesion_rand_range = {"x": [0.1, 0.4], "y": [-0.2, 0.2], "z": [0.8, 1.0]}
    lin_vel_thresh = 1e-2
    ang_vel_thresh = 0.5

    def __init__(
        self,
        *args,
        robot_uids: str = "surgical_continuum_robot",
        robot_init_qpos_noise: float = 0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.lesion: Union[sapien.Actor, None] = None
        self.surgery_table: Union[TableSceneBuilder, None] = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18))

    @property
    def _default_sensor_configs(self):
        cam_pose = sapien_utils.look_at(eye=[0.6, 0.0, 1.5], target=[0.25, 0.0, 0.85])
        return [CameraConfig(uid="obs_cam", pose=cam_pose, width=256, height=256, fov=np.pi/3, near=0.01, far=2.0)]

    @property
    def _default_human_render_camera_configs(self):
        render_cam_pose = sapien_utils.look_at(eye=[0.8, -0.3, 1.2], target=[0.25, 0.0, 0.9])
        return CameraConfig(uid="render_cam", pose=render_cam_pose, width=512, height=512, fov=np.pi/3, near=0.01, far=10.0)

    def _load_scene(self, options: dict):
        self.surgery_table = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.surgery_table.build()
        self.lesion = actors.build_sphere(scene=self.scene, radius=self.lesion_radius, color=self.lesion_color, name="lesion_sphere", body_type="kinematic")

    def _load_agent(self, options: dict):
        super()._load_agent(options=options, initial_agent_poses=self.robot_init_pose)
        assert hasattr(self.agent, "tcp"), "SurgicalContinuumRobot must have 'tcp' attribute"
        print(f"[调试] 机器人加载后 self.agent 是否为 None: {self.agent is None}")
        if self.agent is not None and hasattr(self.agent, "robot"):
            print(f"[调试] 环境中 agent.robot 是否为 None: {self.agent.robot is None}")
        else:
            print(f"[调试] 错误！环境加载后 self.agent 为空或无 robot 属性")

        assert hasattr(self.agent, "tcp"), "SurgicalContinuumRobot must have 'tcp' attribute"
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.surgery_table.initialize(env_idx)

            # 关节角统一为 [b, N] 2维
            reset_qpos = self.agent.keyframes["reset"].qpos
            reset_qpos = torch.tensor(reset_qpos, device=self.device).unsqueeze(0).repeat(b, 1)  # [b, N]
            if self.robot_init_qpos_noise > 0:
                qpos_noise = torch.normal(0.0, self.robot_init_qpos_noise, size=(b, len(reset_qpos)), device=self.device)
                qlimits = torch.tensor(self.agent.robot.get_qlimits(), device=self.device)
                reset_qpos = torch.clamp(reset_qpos + qpos_noise, qlimits[0, :, 0], qlimits[0, :, 1])
            self.agent.reset(init_qpos=reset_qpos)
            self.agent.robot.set_root_pose(self.robot_init_pose)

            # 病灶位置统一为 [b, 3] 2维
            lesion_pos = torch.zeros((b, 3), device=self.device)
            lesion_pos[:, 0] = torch.rand(b, device=self.device)*(0.3) + 0.1
            lesion_pos[:, 1] = torch.rand(b, device=self.device)*(0.4) - 0.2
            lesion_pos[:, 2] = torch.rand(b, device=self.device)*(0.2) + 0.8
            lesion_pose = Pose.create_from_pq(p=lesion_pos, q=torch.tensor([1,0,0,0], device=self.device).repeat(b, 1))  # q: [b,4] 2维
            self.lesion.set_pose(lesion_pose)

    def evaluate(self) -> Dict[str, torch.Tensor]:
        with torch.device(self.device):
            # 所有物理量统一为 [1, x] 2维（单环境b=1）
            ee_pos = torch.tensor(self.agent.tcp.pose.p, device=self.device).squeeze().unsqueeze(0)  # [1, 3]
            lesion_pos = torch.tensor(self.lesion.pose.p, device=self.device).squeeze().unsqueeze(0)  # [1, 3]
            ee_lin_vel = torch.tensor(self.agent.tcp.linear_velocity, device=self.device).squeeze().unsqueeze(0)  # [1, 3]
            ee_ang_vel = torch.tensor(self.agent.tcp.angular_velocity, device=self.device).squeeze().unsqueeze(0)  # [1, 3]

            # 距离统一为 [1, 1] 2维；标志统一为 [1] 2维
            ee_to_lesion_dist = torch.linalg.norm(ee_pos - lesion_pos, dim=-1).unsqueeze(1)  # [1, 1]
            lin_vel_norm = torch.linalg.norm(ee_lin_vel, dim=-1)  # [1]
            ang_vel_norm = torch.linalg.norm(ee_ang_vel, dim=-1)  # [1]

            is_ee_in_effective_dist = (ee_to_lesion_dist.squeeze(1) <= 0.03)  # [1]
            is_ee_lin_static = (lin_vel_norm <= 1e-2)  # [1]
            is_ee_ang_static = (ang_vel_norm <= 0.5)  # [1]
            is_ee_static = torch.logical_and(is_ee_lin_static, is_ee_ang_static)  # [1]
            success = torch.logical_and(is_ee_in_effective_dist, is_ee_static)  # [1]

            return {
                "ee_to_lesion_dist": ee_to_lesion_dist,  # [1,1]
                "is_ee_in_effective_dist": is_ee_in_effective_dist,  # [1]
                "is_ee_static": is_ee_static,  # [1]
                "success": success  # [1]
            }

    def _get_obs_extra(self, info: Dict) -> Dict:
        with torch.device(self.device):
            # 所有观察值统一为 [1, x] 2维
            obs = {
                "ee_pos": torch.tensor(self.agent.tcp.pose.p, device=self.device).squeeze().unsqueeze(0),  # [1,3]
                "lesion_pos": torch.tensor(self.lesion.pose.p, device=self.device).squeeze().unsqueeze(0),  # [1,3]
                "ee_to_lesion_dist": info["ee_to_lesion_dist"],  # [1,1]（继承evaluate的2维格式）
                "is_ee_in_effective_dist": info["is_ee_in_effective_dist"].float().unsqueeze(1)  # [1,1]
            }

            if "state" in self.obs_mode:
                obs.update({
                    "ee_lin_vel": torch.tensor(self.agent.tcp.linear_velocity, device=self.device).squeeze().unsqueeze(0),  # [1,3]
                    "ee_ang_vel": torch.tensor(self.agent.tcp.angular_velocity, device=self.device).squeeze().unsqueeze(0),  # [1,3]
                    "ee_to_lesion_vec": obs["lesion_pos"] - obs["ee_pos"],  # [1,3]（2维相减保持2维）
                    "robot_qpos": torch.tensor(self.agent.robot.get_qpos(), device=self.device).squeeze().unsqueeze(0),  # [1,N]
                    "robot_qvel": torch.tensor(self.agent.robot.get_qvel(), device=self.device).squeeze().unsqueeze(0),  # [1,N]
                    "is_ee_static": info["is_ee_static"].float().unsqueeze(1)  # [1,1]
                })
            return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        with torch.device(self.device):
            b = self.num_envs
            reward = torch.zeros((b,), device=self.device)  # 初始 [b,] 2维

            ee_to_lesion_dist = info["ee_to_lesion_dist"].squeeze(1)  # [b]
            near_reward = 2.0 * (1.0 - torch.tanh(5.0 * ee_to_lesion_dist))  # [b]
            reward += near_reward

            in_range_mask = info["is_ee_in_effective_dist"]  # [b]
            reward += torch.where(in_range_mask, torch.tensor(3.0, device=self.device), torch.tensor(0.0, device=self.device))

            static_mask = torch.logical_and(in_range_mask, info["is_ee_static"])  # [b]
            reward += torch.where(static_mask, torch.tensor(5.0, device=self.device), torch.tensor(0.0, device=self.device))

            success_mask = info["success"]  # [b]
            reward += torch.where(success_mask, torch.tensor(5.0, device=self.device), torch.tensor(0.0, device=self.device))

            return reward.unsqueeze(1)  # 统一转为 [b,1] 2维

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info) / 15.0

    def step(self, action: np.ndarray) -> tuple:
        obs, reward, terminated, truncated, info = super().step(action)
        eval_info = self.evaluate()
        info.update(eval_info)

        # 动作转2维，奖励转标量（符合Gym输出）
        action_tensor = torch.tensor(action, device=self.device).unsqueeze(0)  # [1, action_dim]
        reward = self.compute_dense_reward(obs, action_tensor, info).squeeze().item()  # [1,1]→标量
        terminated = eval_info["success"].squeeze().item()  # [1]→标量

        return obs, reward, terminated, truncated, info