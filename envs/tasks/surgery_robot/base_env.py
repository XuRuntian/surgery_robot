import os.path as osp
from typing import Any, Dict, List, Union
import torch
import numpy as np
import sapien
from transforms3d.euler import euler2quat

from agents.surgical_continuum_robot import SurgicalContinuumRobot 
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# 需要重写
@register_env("SurgeryRobotRemoveLesion-v1", max_episode_steps=800)  
class SurgeryRobotRemoveLesionEnv(BaseEnv):
    """
    **Task Description:**
    医疗水刀机器人任务：在模拟手术平台上，随机生成1-3个病灶（红色球体），机器人需控制水刀末端（linkc_12_body）
    移动到病灶附近，调整末端朝向病灶，启动水刀喷射（通过动作维度控制），满足「有效距离+角度阈值」条件即判定病灶清除，
    所有病灶清除则任务成功。

    **Randomizations:**
    1. 病灶数量随机：1-3个（模拟单/多病灶场景）；
    2. 病灶位置随机：在手术平台的有效区域内（x: 0.1~0.4, y: -0.2~0.2, z: 0.85~0.95），避免超出机器人工作空间；
    3. 病灶朝向随机：每个病灶的表面法线方向随机（增加末端角度对准难度）；
    4. 机器人初始位姿微扰：关节角度添加±0.02rad噪声（模拟实际手术前的位姿偏差）。

    **Success Conditions:**
    1. 所有病灶均满足「清除条件」：
       - 水刀末端与病灶的距离 ≤ 0.03m（水刀有效喷射范围）；
       - 水刀末端朝向与病灶表面法线的夹角 ≤ 15°（π/12 rad，确保喷射方向对准病灶）；
       - 水刀处于「喷射激活」状态（动作维度中喷射指令≥0.5）；
    2. 机器人处于静态：关节速度的L2范数 ≤ 0.1 rad/m/s（避免末端晃动导致误操作）。
    """
    # 环境核心配置
    SUPPORTED_ROBOTS = ["surgical_continuum_robot"]  # 仅支持你的手术机器人
    agent: Union[SurgicalContinuumRobot]  # 绑定机器人类型

    # 病灶与水刀参数（可根据实际需求调整）
    lesion_radius = 0.015  # 病灶球体半径（模拟1.5cm大小的病灶）
    waterjet_effective_dist = 0.03  # 水刀有效作用距离（3cm）
    waterjet_angle_thresh = np.pi / 12  # 水刀对准角度阈值（15°）
    max_lesion_count = 3  # 最大病灶数量
    min_lesion_count = 1  # 最小病灶数量

    # 手术平台配置（替代普通桌子，更贴合医疗场景）
    table_height = 0.8  # 手术平台高度（80cm，符合人体工学）
    table_half_size = (0.3, 0.4, 0.02)  # 平台尺寸：长60cm、宽80cm、厚2cm

    def __init__(
        self,
        *args,
        robot_uids="surgical_continuum_robot",
        robot_init_qpos_noise=0.02,  # 机器人初始位姿噪声
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        # 预定义相机参数（手术视角：主视角+末端视角）
        self.sensor_cam_config = {
            "eye": [0.6, 0, 1.2],  # 主相机位置（手术台侧上方）
            "target": [0.25, 0, 0.85],  # 主相机聚焦点（手术平台中心）
            "fov": np.pi / 3,
            "width": 256,
            "height": 256,
        }
        self.ee_cam_config = {
            "eye": [0, 0, 0.05],  # 末端相机相对位置（水刀喷射方向前方5cm）
            "target": [0, 0, 0.1],  # 末端相机聚焦点（喷射方向10cm处）
            "fov": np.pi / 4,
            "width": 128,
            "height": 128,
        }
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------- 1. 传感器配置（手术场景可视化与观察） ---------------- #
    @property
    def _default_sensor_configs(self):
        """默认传感器：手术台主相机（用于全局观察）"""
        main_cam_pose = sapien_utils.look_at(
            eye=self.sensor_cam_config["eye"],
            target=self.sensor_cam_config["target"]
        )
        return [
            CameraConfig(
                uid="main_surgery_cam",
                pose=main_cam_pose,
                width=self.sensor_cam_config["width"],
                height=self.sensor_cam_config["height"],
                fov=self.sensor_cam_config["fov"],
                near=0.01,
                far=2.0,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        """人类可视化相机：更清晰的手术视角（用于调试）"""
        render_cam_pose = sapien_utils.look_at(
            eye=[0.8, -0.3, 1.0],  # 侧后方视角，方便观察机器人与病灶互动
            target=[0.25, 0, 0.85]
        )
        return CameraConfig(
            uid="human_surgery_cam",
            pose=render_cam_pose,
            width=800,
            height=600,
            fov=np.pi / 3,
            near=0.01,
            far=2.0,
        )

    # ---------------- 2. 机器人加载（手术台坐标系适配） ---------------- #
    def _load_agent(self, options: dict):
        """加载机器人到手术场景的初始位置（确保工作空间覆盖手术平台）"""
        # 机器人初始位姿：x=0.0, y=0.0, z=0.0（底座对准手术台一侧）
        robot_init_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        super()._load_agent(options, robot_init_pose)
        # 为末端添加水刀喷射可视化（蓝色圆柱体，模拟喷射轨迹）
        self._add_waterjet_visualization()

    def _add_waterjet_visualization(self):
        """添加水刀喷射的视觉标记（非物理碰撞，仅可视化）"""
        # 从机器人末端link获取挂载点
        ee_link = self.agent.tcp
        # 喷射轨迹：从末端向前延伸waterjet_effective_dist（3cm）
        self.waterjet_visual = ActorBuilder(self.scene) \
            .add_visual(
                type="cylinder",
                radius=0.003,  # 喷射束半径3mm
                length=self.waterjet_effective_dist,  # 喷射长度=有效距离
                color=[0, 0.8, 1.0, 0.6]  # 半透明蓝色
            ) \
            .build(
                name="waterjet_visual",
                pose=sapien.Pose(p=[0, 0, self.waterjet_effective_dist/2]),  # 居中对齐末端
                parent_link=ee_link,  # 挂载到末端
                add_collision=False  # 无碰撞（仅视觉）
            )
        # 初始隐藏喷射效果
        self.waterjet_visual.set_visibility(False)

    # ---------------- 3. 场景加载（手术平台+病灶+目标标记） ---------------- #
    def _load_scene(self, options: dict):
        """构建手术场景：手术平台+病灶生成区域+目标标记"""
        # 1. 加载手术平台（替代普通桌子，参数适配医疗场景）
        self.surgery_table = TableSceneBuilder(
            self,
            table_height=self.table_height,
            table_half_size=self.table_half_size,
            table_color=[0.9, 0.9, 0.9, 1.0],  # 浅灰色手术台
            robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.surgery_table.build()

        # 2. 初始化病灶存储列表（支持多病灶）
        self.lesions: List[ActorBuilder] = []
        # 3. 病灶目标标记（半透明绿色球体，标记需要对准的区域）
        self.lesion_target_marks: List[ActorBuilder] = []

    # ---------------- 4. episode初始化（随机生成病灶） ---------------- #
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """每轮episode重置：随机生成病灶+重置机器人位姿"""
        with torch.device(self.device):
            b = len(env_idx)  # 批量环境数量（单环境时b=1）
            # 1. 重置手术平台状态（若有碰撞则重置）
            self.surgery_table.initialize(env_idx)
            # 2. 清除上一轮的病灶和标记
            self._clear_prev_lesions()
            # 3. 随机生成病灶数量（1~3个）
            lesion_counts = torch.randint(
                low=self.min_lesion_count,
                high=self.max_lesion_count + 1,
                size=(b,)
            )
            # 4. 为每个环境生成病灶（单环境时取第一个count）
            lesion_count = lesion_counts[0].item() if b > 0 else self.min_lesion_count
            for i in range(lesion_count):
                self._spawn_single_lesion()
            # 5. 重置机器人到初始位姿（添加微扰）
            self.agent.reset(
                qpos=self.agent.keyframes["reset"].qpos + 
                torch.normal(0, self.robot_init_qpos_noise, size=self.agent.keyframes["reset"].qpos.shape)
            )
            # 6. 隐藏水刀喷射效果
            self.waterjet_visual.set_visibility(False)

    def _clear_prev_lesions(self):
        """清除上一轮的病灶和目标标记"""
        for lesion in self.lesions:
            self.scene.remove_actor(lesion)
        for mark in self.lesion_target_marks:
            self.scene.remove_actor(mark)
        self.lesions = []
        self.lesion_target_marks = []

    def _spawn_single_lesion(self):
        """生成单个病灶（红色球体）和对应的目标标记（半透明绿色球体）"""
        # 随机病灶位置（手术平台上方5~15mm处，避免与平台碰撞）
        lesion_x = np.random.uniform(0.1, 0.4)  # x范围：10~40cm（手术台右侧）
        lesion_y = np.random.uniform(-0.2, 0.2)  # y范围：-20~20cm（左右对称）
        lesion_z = np.random.uniform(
            self.table_height + 0.005, 
            self.table_height + 0.015
        )
        lesion_pos = sapien.Pose(p=[lesion_x, lesion_y, lesion_z])

        # 随机病灶表面法线（模拟病灶朝向，增加对准难度）
        norm_theta = np.random.uniform(0, np.pi/4)  # 与z轴夹角0~45°
        norm_phi = np.random.uniform(0, 2*np.pi)    # 绕z轴旋转0~360°
        norm_dir = np.array([
            np.sin(norm_theta)*np.cos(norm_phi),
            np.sin(norm_theta)*np.sin(norm_phi),
            np.cos(norm_theta)
        ])
        # 病灶朝向：法线方向即为病灶“正面”，用四元数表示
        lesion_quat = sapien_utils.dir_to_quat(norm_dir)
        lesion_pose = Pose(p=lesion_pos.p, q=lesion_quat)

        # 1. 生成病灶（红色不透明球体，有碰撞）
        lesion = ActorBuilder(self.scene) \
            .add_visual(
                type="sphere",
                radius=self.lesion_radius,
                color=[1.0, 0.2, 0.2, 1.0]  # 红色病灶
            ) \
            .add_collision(
                type="sphere",
                radius=self.lesion_radius
            ) \
            .build(
                name=f"lesion_{len(self.lesions)}",
                pose=lesion_pose,
                body_type="dynamic"  # 动态（可被碰撞，但质量小避免被机器人推动）
            )
        lesion.set_mass(0.01)  # 病灶质量10g（轻量化）
        self.lesions.append(lesion)

        # 2. 生成病灶目标标记（半透明绿色球体，标记水刀需要对准的区域）
        target_mark = ActorBuilder(self.scene) \
            .add_visual(
                type="sphere",
                radius=self.waterjet_effective_dist,  # 标记范围=水刀有效距离
                color=[0.2, 1.0, 0.2, 0.3]  # 半透明绿色
            ) \
            .build(
                name=f"lesion_target_{len(self.lesion_target_marks)}",
                pose=lesion_pose,
                add_collision=False  # 无碰撞（仅标记）
            )
        self.lesion_target_marks.append(target_mark)

    # ---------------- 5. 观察空间（agent需要的任务信息） ---------------- #
    def _get_obs_extra(self, info: Dict):
        """补充额外观察信息：病灶位置、末端与病灶的相对姿态、喷射状态"""
        # 1. 获取末端位姿（位置+四元数）
        ee_pose = self.agent.tcp_pose
        ee_pos = ee_pose.p
        ee_quat = ee_pose.q
        # 2. 获取所有病灶的位置和法线方向
        lesion_info = []
        for lesion in self.lesions:
            lesion_pos = lesion.pose.p
            # 计算末端到病灶的距离
            dist_to_lesion = np.linalg.norm(lesion_pos - ee_pos)
            # 计算末端朝向与病灶法线的夹角
            ee_dir = sapien_utils.quat_to_dir(ee_quat)  # 末端朝向（水刀喷射方向）
            lesion_norm = sapien_utils.quat_to_dir(lesion.pose.q)  # 病灶法线方向
            angle_between = np.arccos(np.clip(np.dot(ee_dir, lesion_norm), -1.0, 1.0))
            # 存储单个病灶信息
            lesion_info.append([
                *lesion_pos,    # 病灶位置 (x,y,z)
                dist_to_lesion, # 末端到病灶距离
                angle_between   # 末端与病灶法线夹角
            ])
        # 3. 水刀喷射状态（0=未喷射，1=喷射）
        waterjet_active = 1.0 if self.waterjet_visual.get_visibility() else 0.0

        # 整理观察字典
        obs = {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "waterjet_active": waterjet_active,
            "lesion_count": len(lesion_info),
            "lesion_info": np.array(lesion_info) if lesion_info else np.zeros((0, 5))
        }
        # 若为状态观察模式，补充更多细节
        if "state" in self.obs_mode:
            obs.update({
                "ee_vel": self.agent.robot.get_qvel()[-6:],  # 末端速度（简化取后6维）
                "lesion_poses": np.array([lesion.pose.raw_pose for lesion in self.lesions])
            })
        return obs

    # ---------------- 6. 任务评估（判断是否成功清除病灶） ---------------- #
    def evaluate(self):
        """评估当前episode的成功状态（补充每个病灶的清除状态）"""
        cleared_lesions = []  # 存储每个病灶的清除状态（True/False）
        ee_pose = self.agent.tcp_pose
        ee_pos = ee_pose.p
        ee_dir = sapien_utils.quat_to_dir(ee_pose.q)
        waterjet_active = self.waterjet_visual.get_visibility()

        # 逐个判断每个病灶是否被清除
        for lesion in self.lesions:
            lesion_pos = lesion.pose.p
            lesion_norm = sapien_utils.quat_to_dir(lesion.pose.q)
            # 清除三条件：距离≤有效距离 + 角度≤阈值 + 水刀喷射
            dist_ok = np.linalg.norm(lesion_pos - ee_pos) <= self.waterjet_effective_dist
            dot_product = np.clip(np.dot(ee_dir, lesion_norm), -1.0, 1.0)
            angle_ok = np.arccos(dot_product) <= self.waterjet_angle_thresh
            jet_ok = waterjet_active
            # 记录当前病灶是否清除
            cleared_lesions.append(dist_ok and angle_ok and jet_ok)

        # 整体任务成功条件：所有病灶清除 + 机器人静态
        all_cleared = len(cleared_lesions) > 0 and all(cleared_lesions)
        robot_qvel_norm = np.linalg.norm(self.agent.robot.get_qvel())
        is_static = robot_qvel_norm <= 0.1

        # 补充返回每个病灶的清除状态（供奖励函数使用）
        return {
            "success": all_cleared and is_static,
            "all_lesions_cleared": all_cleared,
            "is_robot_static": is_static,
            "cleared_lesions": cleared_lesions,  # 新增：每个病灶的清除状态
            "cleared_lesion_count": sum(cleared_lesions),
            "total_lesion_count": len(cleared_lesions)
        }

    # ---------------- 7. 奖励函数（引导agent学习正确行为） ---------------- #
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """密集奖励：分阶段引导agent完成任务"""
        reward = 0.0
        b = action.shape[0]  # 批量大小（单环境时b=1）
        ee_pose = self.agent.tcp_pose
        ee_pos = ee_pose.p
        ee_quat = ee_pose.q
        # 关键：在当前代码块中重新计算末端朝向（确保变量作用域正确）
        ee_dir = sapien_utils.quat_to_dir(ee_quat)  # 水刀喷射方向（末端朝向）
        waterjet_active = self.waterjet_visual.get_visibility()

        # 1. 距离奖励：鼓励末端靠近未清除的病灶（取最近未清除病灶的距离）
        if len(self.lesions) > 0:
            # 过滤出未被清除的病灶（从info中获取已清除状态，避免重复计算）
            uncleared_lesions = [
                lesion for idx, lesion in enumerate(self.lesions)
                if not info.get("cleared_lesions", [False]*len(self.lesions))[idx]
            ]
            if uncleared_lesions:
                min_dist = min([np.linalg.norm(lesion.pose.p - ee_pos) for lesion in uncleared_lesions])
                # 距离奖励：有效距离内奖励1.0，超出则随距离衰减（0~1）
                dist_reward = np.clip(1 - (min_dist / self.waterjet_effective_dist), 0.0, 1.0)
                reward += dist_reward * 0.3  # 权重0.3

        # 2. 角度奖励：鼓励末端对准未清除的病灶（取最近未清除病灶的角度）
        if len(self.lesions) > 0:
            uncleared_lesions = [
                lesion for idx, lesion in enumerate(self.lesions)
                if not info.get("cleared_lesions", [False]*len(self.lesions))[idx]
            ]
            if uncleared_lesions:
                # 计算每个未清除病灶的角度偏差
                angle_deviations = []
                for lesion in uncleared_lesions:
                    lesion_norm = sapien_utils.quat_to_dir(lesion.pose.q)  # 病灶表面法线（目标朝向）
                    # 计算两向量夹角（用点积，避免NaN）
                    dot_product = np.clip(np.dot(ee_dir, lesion_norm), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    angle_deviations.append(angle)
                min_angle = min(angle_deviations)
                # 角度奖励：阈值内奖励1.0，超出则随角度衰减（0~1）
                angle_reward = np.clip(1 - (min_angle / self.waterjet_angle_thresh), 0.0, 1.0)
                reward += angle_reward * 0.4  # 权重0.4

        # 3. 喷射奖励：仅对“未清除且对准”的病灶喷射才给奖励（避免无效喷射）
        if waterjet_active and len(self.lesions) > 0:
            # 先获取当前各病灶的清除状态（从info中读取，与evaluate逻辑保持一致）
            cleared_lesions = info.get("cleared_lesions", [False]*len(self.lesions))
            valid_jet_count = 0  # 有效喷射的病灶数量
            for idx, lesion in enumerate(self.lesions):
                # 仅对未清除的病灶判断喷射有效性
                if not cleared_lesions[idx]:
                    # 条件1：距离在有效范围内
                    dist_ok = np.linalg.norm(lesion.pose.p - ee_pos) <= self.waterjet_effective_dist
                    # 条件2：角度在阈值范围内
                    lesion_norm = sapien_utils.quat_to_dir(lesion.pose.q)
                    dot_product = np.clip(np.dot(ee_dir, lesion_norm), -1.0, 1.0)
                    angle_ok = np.arccos(dot_product) <= self.waterjet_angle_thresh
                    # 有效喷射：两个条件都满足
                    if dist_ok and angle_ok:
                        valid_jet_count += 1
            # 喷射奖励：每个有效喷射的病灶给0.1/步（避免一次性奖励过高）
            reward += valid_jet_count * 0.1

        # 4. 清除奖励：每新清除一个病灶，给予一次性奖励（避免重复奖励）
        current_cleared = sum(info.get("cleared_lesions", [False]*len(self.lesions)))
        # 初始化上一轮清除数量（首次调用时设为0）
        if not hasattr(self, "prev_cleared_count"):
            self.prev_cleared_count = 0
        new_cleared = max(0, current_cleared - self.prev_cleared_count)
        if new_cleared > 0:
            reward += new_cleared * 2.0  # 每个新清除的病灶奖励2.0
        # 更新上一轮清除数量（用于下一次计算）
        self.prev_cleared_count = current_cleared

        # 5. 任务成功奖励：所有病灶清除且机器人静态，给予最终奖励
        if info.get("success", False):
            reward += 5.0  # 最终奖励5.0，鼓励快速完成任务

        # 6. 惩罚项：避免机器人长时间不动（防止agent卡住）
        robot_qvel = self.agent.robot.get_qvel()
        qvel_norm = np.linalg.norm(robot_qvel)
        if qvel_norm < 0.05 and not info.get("success", False):
            reward -= 0.01  # 每步惩罚0.01，轻微督促agent移动

        # 批量环境适配（单环境返回标量，多环境返回张量）
        return torch.tensor([reward]*b, device=self.device, dtype=torch.float32)

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """归一化奖励：将奖励映射到[0,1]范围（方便训练收敛）"""
        dense_reward = self.compute_dense_reward(obs, action, info)
        return dense_reward / 10.0  # 最大奖励约10（5+2*2+1），归一化到[0,1]

    # ---------------- 8. 动作后处理（控制水刀喷射状态） ---------------- #
    def step(self, action: torch.Tensor):
        """重写step方法：从动作中解析水刀喷射指令"""
        # 假设动作最后1维是水刀喷射指令（0~1）：≥0.5则激活喷射
        if action.shape[-1] > 0:
            jet_cmd = action[0, -1].item() if action.ndim > 1 else action[-1].item()
            self.waterjet_visual.set_visibility(jet_cmd >= 0.5)
            # 截断动作：去掉喷射指令维度，仅保留机器人控制维度（6维：linear/revolute/ax/ay/cx/cy）
            action = action[..., :-1] if action.shape[-1] > 6 else action
        # 调用父类step执行物理仿真
        return super().step(action)