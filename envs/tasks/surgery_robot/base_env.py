from typing import Any, Dict, Union, List
import torch
import numpy as np
import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building import actors  # 复用ManiSkill标准物体构建工具
from mani_skill.sensors.camera import CameraConfig
from mani_skill.envs.sapien_env import BaseEnv

# 导入你的手术机器人类
from agents.surgical_continuum_robot import SurgicalContinuumRobot


@register_env("SurgeryRobotMinimal-v1", max_episode_steps=800)  # 规范环境名，添加版本号
class SurgeryRobotMinimalEnv(BaseEnv):
    """
    **Task Description:**
    **Randomizations:**

    **Success Conditions:**

    """
    # 1. 标准化任务元信息（参考PlaceSphereEnv）
    _sample_video_link = ""  # 可选：后续可添加任务演示视频链接
    SUPPORTED_ROBOTS = ["surgical_continuum_robot"]  # 明确支持的机器人
    agent: Union[SurgicalContinuumRobot]  # 类型注解，确保类型安全

    # 2. 核心参数集中定义（便于维护与修改）
    # 病灶（小球）参数
    lesion_radius = 0.015          # 病灶半径（m）
    lesion_color = np.array([1.0, 0.2, 0.2, 1.0])  # 红色病灶（RGBA）
    # 水刀参数
    waterjet_effective_dist = 0.03  # 水刀有效作用距离（m）
    # 机器人参数
    robot_init_pose = sapien.Pose(p=[0.0, 0.0, 0.1], q=[0, 1, 0, 0])  # 机器人初始位姿
    # 随机化范围（病灶位置）
    lesion_rand_range = {
        "x": [0.1, 0.4],
        "y": [-0.2, 0.2],
        "z": [0.8, 1.0]
    }
    # 静态判断阈值（参考PlaceSphereEnv）
    lin_vel_thresh = 1e-2  # 线性速度阈值（m/s）
    ang_vel_thresh = 0.5    # 角速度阈值（rad/s）

    def __init__(
        self,
        *args,
        robot_uids: str = "surgical_continuum_robot",
        robot_init_qpos_noise: float = 0.02,
        **kwargs
    ):
        """初始化环境，统一参数传递逻辑（参考PlaceSphereEnv）"""
        self.robot_init_qpos_noise = robot_init_qpos_noise  # 机器人初始关节噪声
        self.lesion: Union[sapien.Actor, None] = None       # 病灶物体（后续初始化）
        self.surgery_table: Union[TableSceneBuilder, None] = None  # 手术台场景
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # 3. 仿真配置（可选，如需GPU加速可配置）
    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,  # 适配复杂碰撞检测
                max_rigid_patch_count=2**18
            )
        )

    # 4. 传感器配置（分观察相机与人类渲染相机，参考PlaceSphereEnv）
    @property
    def _default_sensor_configs(self):
        """用于算法观察的相机（低分辨率，高效）"""
        cam_pose = sapien_utils.look_at(
            eye=self.sensor_cam_config["eye"],
            target=self.sensor_cam_config["target"]
        )
        return [
            CameraConfig(
                uid="obs_cam",
                pose=cam_pose,
                width=256,
                height=256,
                fov=np.pi / 3,
                near=0.01,
                far=2.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        """用于人类可视化的相机（高分辨率，视角友好）"""
        # 手术场景视角：从侧上方观察机器人与病灶
        render_cam_pose = sapien_utils.look_at(
            eye=[0.8, -0.3, 1.2],  # 相机位置
            target=[0.25, 0.0, 0.9]  # 观察目标（病灶区域）
        )
        return CameraConfig(
            uid="render_cam",
            pose=render_cam_pose,
            width=512,
            height=512,
            fov=np.pi / 3,
            near=0.01,
            far=10.0,
        )

    @property
    def sensor_cam_config(self):
        """观察相机参数（集中定义，便于修改）"""
        return {
            "eye": [0.6, 0.0, 1.5],
            "target": [0.25, 0.0, 0.85]
        }

    # 5. 场景加载（复用ManiSkill工具，避免重复造轮子）
    def _load_scene(self, options: dict):
        """加载手术台、病灶（小球），参考PlaceSphereEnv的actors工具"""
        # 加载手术台（复用TableSceneBuilder，标准化场景构建）
        self.surgery_table = TableSceneBuilder(
            env=self,
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.surgery_table.build()

        # 加载病灶（小球）：复用actors.build_sphere，标准化动态/静态物体创建
        self.lesion = actors.build_sphere(
            scene=self.scene,
            radius=self.lesion_radius,
            color=self.lesion_color,
            name="lesion_sphere",
            body_type="kinematic"  # 病灶固定不动（手术场景中病灶位置固定）
        )

    # 6. 机器人加载（标准化agent加载逻辑）
    def _load_agent(self, options: dict):
        """加载手术机器人，传递初始位姿（参考PlaceSphereEnv）"""
        super()._load_agent(
            options=options,
            initial_agent_poses=self.robot_init_pose  # 统一传递初始位姿
        )
        # 确保机器人TCP已初始化（手术机器人特有校验）
        assert hasattr(self.agent, "tcp"), "SurgicalContinuumRobot must have 'tcp' attribute"

    # 7. Episode初始化（支持批量环境，用torch操作实现随机化）
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """初始化每一轮episode：重置机器人、随机化病灶位置（参考PlaceSphereEnv批量逻辑）"""
        with torch.device(self.device):  # 适配GPU/CPU环境
            b = len(env_idx)  # 批量环境数量（单环境时b=1）

            # 1. 初始化手术台（复用TableSceneBuilder的标准化初始化）
            self.surgery_table.initialize(env_idx)

            # 2. 重置机器人关节（无噪声或带噪声）
            reset_qpos = self.agent.keyframes["reset"].qpos  # 机器人默认复位关节角
            if self.robot_init_qpos_noise > 0:
                # 添加关节噪声
                qpos_noise = torch.normal(
                    mean=0.0,
                    std=self.robot_init_qpos_noise,
                    size=(b, len(reset_qpos)),
                    device=self.device
                )
                # 确保噪声后的关节角在限位内
                qlimits = torch.tensor(self.agent.robot.get_qlimits(), device=self.device)
                reset_qpos = torch.clamp(
                    torch.tensor(reset_qpos, device=self.device).unsqueeze(0).repeat(b, 1) + qpos_noise,
                    qlimits[0, :, 0],  # 关节下限
                    qlimits[0, :, 1]   # 关节上限
                )
            self.agent.reset(init_qpos=reset_qpos)  # 机器人复位
            self.agent.robot.set_root_pose(self.robot_init_pose)  # 重置机器人根节点位姿

            # 3. 随机化病灶位置（批量生成随机位置，参考PlaceSphereEnv）
            lesion_pos = torch.zeros((b, 3), device=self.device)
            # x轴随机：[0.1, 0.4]
            lesion_pos[:, 0] = torch.rand((b,), device=self.device) * (self.lesion_rand_range["x"][1] - self.lesion_rand_range["x"][0]) + self.lesion_rand_range["x"][0]
            # y轴随机：[-0.2, 0.2]
            lesion_pos[:, 1] = torch.rand((b,), device=self.device) * (self.lesion_rand_range["y"][1] - self.lesion_rand_range["y"][0]) + self.lesion_rand_range["y"][0]
            # z轴随机：[0.8, 1.0]
            lesion_pos[:, 2] = torch.rand((b,), device=self.device) * (self.lesion_rand_range["z"][1] - self.lesion_rand_range["z"][0]) + self.lesion_rand_range["z"][0]

            # 设置病灶位姿（标准化Pose创建）
            lesion_pose = Pose.create_from_pq(
                p=lesion_pos,
                q=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(b, 1)  # 无旋转
            )
            self.lesion.set_pose(lesion_pose)

    # 8. 任务评估（标准化评估逻辑，返回关键状态标志）
    def evaluate(self) -> Dict[str, torch.Tensor]:
        """评估当前环境状态：是否靠近病灶、是否静态、是否成功（参考PlaceSphereEnv）"""
        with torch.device(self.device):
            # 1. 获取关键位置与速度
            ee_pos = torch.tensor(self.agent.tcp.pose.p, device=self.device).unsqueeze(0)  # 末端位置
            lesion_pos = torch.tensor(self.lesion.pose.p, device=self.device).unsqueeze(0)  # 病灶位置
            ee_lin_vel = torch.tensor(self.agent.tcp.linear_velocity, device=self.device).unsqueeze(0)  # 末端线速度
            ee_ang_vel = torch.tensor(self.agent.tcp.angular_velocity, device=self.device).unsqueeze(0)  # 末端角速度

            # 2. 计算状态标志
            # 末端是否在水刀有效范围内（距离 ≤ 有效距离）
            ee_to_lesion_dist = torch.linalg.norm(ee_pos - lesion_pos, dim=1)
            is_ee_in_effective_dist = (ee_to_lesion_dist <= self.waterjet_effective_dist)

            # 末端是否静态（线速度、角速度低于阈值）
            is_ee_lin_static = (torch.linalg.norm(ee_lin_vel, dim=1) <= self.lin_vel_thresh)
            is_ee_ang_static = (torch.linalg.norm(ee_ang_vel, dim=1) <= self.ang_vel_thresh)
            is_ee_static = torch.logical_and(is_ee_lin_static, is_ee_ang_static)

            # 任务成功：有效范围内 + 末端静态
            success = torch.logical_and(is_ee_in_effective_dist, is_ee_static)

            return {
                "ee_to_lesion_dist": ee_to_lesion_dist,
                "is_ee_in_effective_dist": is_ee_in_effective_dist,
                "is_ee_static": is_ee_static,
                "success": success
            }

    # 9. 观察空间（标准化观察信息，分模式返回）
    def _get_obs_extra(self, info: Dict) -> Dict:
        """返回额外观察信息（参考PlaceSphereEnv，按obs_mode适配）"""
        with torch.device(self.device):
            # 基础观察：末端位姿、病灶位置、末端到病灶的向量
            obs = {
                "ee_pos": torch.tensor(self.agent.tcp.pose.p, device=self.device),
                "lesion_pos": torch.tensor(self.lesion.pose.p, device=self.device),
                "ee_to_lesion_dist": info["ee_to_lesion_dist"],
                "is_ee_in_effective_dist": info["is_ee_in_effective_dist"],
            }

            # 若为state模式，返回更详细信息（参考PlaceSphereEnv的state模式逻辑）
            if "state" in self.obs_mode:
                obs.update({
                    "ee_lin_vel": torch.tensor(self.agent.tcp.linear_velocity, device=self.device),
                    "ee_ang_vel": torch.tensor(self.agent.tcp.angular_velocity, device=self.device),
                    "ee_to_lesion_vec": obs["lesion_pos"] - obs["ee_pos"],  # 末端到病灶的向量
                    "robot_qpos": torch.tensor(self.agent.robot.get_qpos(), device=self.device),  # 机器人关节角
                    "robot_qvel": torch.tensor(self.agent.robot.get_qvel(), device=self.device)   # 机器人关节速度
                })

            return obs

    # 10. 分阶段奖励计算（参考PlaceSphereEnv的层次化奖励）
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict) -> torch.Tensor:
        """计算稠密奖励：分「靠近奖励」→「有效范围奖励」→「成功奖励」（层次化设计）"""
        with torch.device(self.device):
            b = action.shape[0] if action.ndim > 0 else 1  # 批量大小
            reward = torch.zeros((b,), device=self.device)

            # 1. 基础奖励：靠近病灶（距离越近奖励越高）
            ee_to_lesion_dist = info["ee_to_lesion_dist"]
            near_reward = 2.0 * (1.0 - torch.tanh(5.0 * ee_to_lesion_dist))  # 0~2分
            reward += near_reward

            # 2. 进阶奖励：末端进入水刀有效范围（额外加奖励）
            in_range_mask = info["is_ee_in_effective_dist"]
            range_reward = torch.where(in_range_mask, 3.0, 0.0)  # 有效范围内加3分（累计0~5分）
            reward += range_reward

            # 3. 静态奖励：末端在有效范围内且静态（鼓励稳定瞄准）
            static_mask = torch.logical_and(in_range_mask, info["is_ee_static"])
            static_reward = torch.where(static_mask, 5.0, 0.0)  # 静态加5分（累计0~10分）
            reward += static_reward

            # 4. 成功奖励：满足所有成功条件（额外加奖励，总奖励上限15分）
            success_mask = info["success"]
            success_reward = torch.where(success_mask, 5.0, 0.0)
            reward += success_reward

            return reward

    # 11. 标准化奖励（参考PlaceSphereEnv，除以最大奖励）
    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict) -> torch.Tensor:
        """归一化奖励（最大奖励15分，确保奖励在[0,1]区间）"""
        max_reward = 15.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    # 12. 环境step逻辑（标准化step流程）
    def step(self, action: np.ndarray) -> tuple:
        """执行动作，返回环境反馈（参考PlaceSphereEnv，整合评估与奖励）"""
        # 1. 调用父类step执行物理仿真
        obs, reward, terminated, truncated, info = super().step(action)

        # 2. 执行任务评估（补充评估信息到info）
        eval_info = self.evaluate()
        info.update(eval_info)

        # 3. 计算稠密奖励（覆盖父类默认奖励）
        reward = self.compute_dense_reward(obs=obs, action=torch.tensor(action, device=self.device), info=info)
        reward = reward.item() if reward.ndim == 0 else reward.cpu().numpy()  # 适配单环境返回标量

        # 4. 判断终止条件（成功则终止）
        terminated = eval_info["success"].item() if isinstance(eval_info["success"], torch.Tensor) else eval_info["success"]

        return obs, reward, terminated, truncated, info