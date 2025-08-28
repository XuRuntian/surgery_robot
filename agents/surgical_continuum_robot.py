from __future__ import annotations
from copy import deepcopy
from typing import Dict, Tuple
import os
import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.link import Link
from mani_skill.utils.building import MJCFLoader 

# 可选：用于设置 base 的碰撞层
MY_BASE_COLLISION_BIT = 31


# ---------------- 1. 自定义控制器类（保留校验逻辑，确保无空关节） ---------------- #
class TendonPDJointPosController(PDJointPosController):
    def __init__(self, config: PDJointPosControllerConfig, agent: "SurgicalContinuumRobot"):
        # 初始化前先校验 agent 的主动关节列表无空值
        assert "" not in agent.all_joint_names, \
            f"Agent 主动关节列表含空字符串：{agent.all_joint_names}"
        super().__init__(config, agent)
        self.tendon_groups = agent.tendon_groups
        self.control_joint_mapping = self._build_control_mapping()
        self._validate_mapping()  # 校验映射关节无空值

    def _build_control_mapping(self):
        return {
            0: ["linear_joint"],
            1: ["j_revolute"],
            2: self.tendon_groups["ax"],
            3: self.tendon_groups["ay"],
            4: self.tendon_groups["cx"],
            5: self.tendon_groups["cy"]
        }

    def _validate_mapping(self):
        """校验映射关节无空值且都在主动关节列表中"""
        all_mapped_joints = []
        for joints in self.control_joint_mapping.values():
            # 先校验映射关节本身无空值
            assert "" not in joints, f"映射关节含空字符串：{joints}"
            all_mapped_joints.extend(joints)
        # 再校验映射关节都在主动关节列表中
        missing_joints = [j for j in all_mapped_joints if j not in self.agent.all_joint_names]
        if missing_joints:
            raise ValueError(f"映射关节不在主动关节列表中：{missing_joints}")

    def compute_torque(self, action: np.ndarray) -> np.ndarray:
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if action.shape[-1] != 6:
            raise ValueError(f"控制器需6维动作，实际输入{action.shape[-1]}维！")

        # 获取主动关节当前位置（确保无空关节）
        current_joint_pos = self.get_current_joint_positions()
        joint_names = self.agent.all_joint_names
        assert len(current_joint_pos) == len(joint_names), \
            f"关节位置长度与关节名长度不匹配：{len(current_joint_pos)} vs {len(joint_names)}"

        # 分配动作（关节名必存在，无空值）
        for ctrl_dim, target_joints in self.control_joint_mapping.items():
            ctrl_value = action[0, ctrl_dim]
            for joint_name in target_joints:
                joint_idx = joint_names.index(joint_name)
                if self.config.use_delta:
                    current_joint_pos[joint_idx] += ctrl_value
                else:
                    current_joint_pos[joint_idx] = ctrl_value

        return super().compute_torque(current_joint_pos)


@register_agent(asset_download_ids=["surgical_continuum_robot"])
class SurgicalContinuumRobot(BaseAgent):
    uid = "surgical_continuum_robot"
    xml_path = os.path.join(
        os.path.dirname(__file__),
        "../assets/robots/surgical_continuum_robot/surgery_robot.xml"
    )
    # 初始关键帧（空数组，_load_articulation 中更新）
    keyframes = dict(
        reset=Keyframe(pose=sapien.Pose(), qpos=np.array([], dtype=np.float32))
    )
    # 主动关节列表（_load_articulation 中初始化，无空值）
    all_joint_names = []
    # 肌腱组配置（无空值，与 MJCF 一致）
    tendon_groups = {
        "ax": ["ja_1", "ja_3", "ja_5", "ja_7", "ja_9", "ja_11", 
               "ja_13", "ja_15", "ja_17", "ja_19", "ja_21", "ja_23", "jb"],
        "ay": ["ja_2", "ja_4", "ja_6", "ja_8", "ja_10", "ja_12", 
               "ja_14", "ja_16", "ja_18", "ja_20", "ja_22", "ja_24"],
        "cx": ["jc_1", "jc_3", "jc_5", "jc_7", "jc_9", "jc_11"],
        "cy": ["jc_2", "jc_4", "jc_6", "jc_8", "jc_10", "jc_12"],
    }
    # 控制器参数
    ee_link_name = "linkc_12_body"
    base_stiffness = 5000    # linear_joint
    base_damping = 300       
    revolute_stiffness = 3000# j_revolute
    revolute_damping = 100   
    tendon_stiffness = 2000  # 肌腱关节
    tendon_damping = 500     
    force_limit = {"linear": 20000, "revolute": 100, "tendon": 500}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ---------------- 2. 加载模型：确保主动关节列表无空值，且与机器人一致 ---------------- #
    def _load_articulation(self, initial_pose: sapien.Pose):
        loader = MJCFLoader(ignore_classes=['motor'], visual_groups=[0, 2])
        loader.scene = self.scene

        # 加载 MJCF 模型
        self.robot = loader.load(
            mjcf_file=self.xml_path,
            package_dir=os.path.dirname(self.xml_path),
            name=self.uid
        )
        self.robot.set_pose(initial_pose)

        # ---------------- 关键修改：过滤空名称关节 ---------------- #
        # 1. 获取原生关节列表，过滤掉空字符串名称的关节（肌腱误解析的虚拟关节）
        native_joints = [joint for joint in self.robot.joints if joint.name.strip() != ""]
        # 2. 提取有效关节名称，排序确保顺序固定
        self.all_joint_names = sorted([joint.name for joint in native_joints])
        active_joint_count = len(self.all_joint_names)

        # 校验：确保无空字符串，且关节数量符合预期（39个物理关节）
        assert "" not in self.all_joint_names, f"主动关节列表仍含空字符串：{self.all_joint_names}"
        assert active_joint_count == 39, f"有效关节数量异常（预期39，实际{active_joint_count}），请检查过滤逻辑"

        # 更新关键帧 qpos（与有效关节数量一致）
        self.keyframes["reset"].qpos = np.zeros(active_joint_count, dtype=np.float32)

        # 校验肌腱组关节是否都在有效列表中
        all_tendon_joints = []
        for group_joints in self.tendon_groups.values():
            all_tendon_joints.extend(group_joints)
        missing_tendon_joints = [j for j in all_tendon_joints if j not in self.all_joint_names]
        if missing_tendon_joints:
            raise ValueError(f"肌腱组关节不在有效关节列表中：{missing_tendon_joints}")

        # 打印日志确认
        print(f"✅ 过滤后有效关节数量：{active_joint_count}")
        print(f"✅ 有效关节列表：{self.all_joint_names}")
        return self.robot

    # ---------------- 3. 传感器配置（无空值风险，保留） ---------------- #
    @property
    def _sensor_configs(self):
        assert self.ee_link_name in [link.name for link in self.robot.links], \
            f"末端Link {self.ee_link_name} 未找到（检查 MJCF）"
        return [
            CameraConfig(
                uid="end_camera",
                pose=sapien.Pose(p=[0, -0.03, 0.008], q=[0.7071, 0, 0, -0.7071]),
                width=1920, height=1080, fov=np.pi/2, near=0.01, far=100,
                mount=self.robot.links_map[self.ee_link_name],
            ),
        ]

    # ---------------- 4. 控制器配置：彻底移除空值处理，强校验无空值 ---------------- #
    @property
    def _controller_configs(self):
        assert self.all_joint_names, "主动关节列表未初始化"
        assert "" not in self.all_joint_names, "主动关节列表含空字符串"
        joint_names = self.all_joint_names

        # 工具函数：为有效关节分配参数（无空关节分支）
        def get_param(joint_name, param_type):
            if param_type == "lower":
                if joint_name == "linear_joint":
                    return 0.0
                elif joint_name == "j_revolute":
                    return -6.28
                else:  # 肌腱关节（ja_1~ja_24、jb、jc_1~jc_12）
                    return -0.26
            elif param_type == "upper":
                if joint_name == "linear_joint":
                    return 0.255
                elif joint_name == "j_revolute":
                    return 6.28
                else:
                    return 0.26
            elif param_type == "stiffness":
                if joint_name == "linear_joint":
                    return self.base_stiffness
                elif joint_name == "j_revolute":
                    return self.revolute_stiffness
                else:
                    return self.tendon_stiffness
            elif param_type == "damping":
                if joint_name == "linear_joint":
                    return self.base_damping
                elif joint_name == "j_revolute":
                    return self.revolute_damping
                else:
                    return self.tendon_damping
            elif param_type == "force_limit":
                if joint_name == "linear_joint":
                    return self.force_limit["linear"]
                elif joint_name == "j_revolute":
                    return self.force_limit["revolute"]
                else:
                    return self.force_limit["tendon"]
            else:
                raise ValueError(f"未知参数类型：{param_type}")

        # 生成绝对控制参数
        lower = [get_param(j, "lower") for j in joint_names]
        upper = [get_param(j, "upper") for j in joint_names]
        stiffness = [get_param(j, "stiffness") for j in joint_names]
        damping = [get_param(j, "damping") for j in joint_names]
        force_limit = [get_param(j, "force_limit") for j in joint_names]

        # 绝对位置控制器
        tendon_pd_pos_config = PDJointPosControllerConfig(
            joint_names=joint_names,
            lower=lower, upper=upper, stiffness=stiffness, damping=damping, force_limit=force_limit,
            normalize_action=False, use_delta=False
        )

        # 生成相对控制参数（微调范围）
        delta_lower = [
            -0.01 if j == "linear_joint" else
            -0.1 if j == "j_revolute" else
            -0.05 for j in joint_names
        ]
        delta_upper = [
            0.01 if j == "linear_joint" else
            0.1 if j == "j_revolute" else
            0.05 for j in joint_names
        ]
        delta_force_limit = [f // 2 for f in force_limit]

        # 相对位置控制器
        tendon_pd_delta_pos_config = PDJointPosControllerConfig(
            joint_names=joint_names,
            lower=delta_lower, upper=delta_upper, stiffness=stiffness, damping=damping, force_limit=delta_force_limit,
            normalize_action=False, use_delta=True
        )

        return deepcopy({
            "tendon_pd_pos": dict(arm=tendon_pd_pos_config),
            "tendon_pd_delta_pos": dict(arm=tendon_pd_delta_pos_config)
        })

    # ---------------- 5. 构建控制器：确保配置无空值 ---------------- #
    def _build_controllers(self) -> Dict[str, Dict[str, BaseController]]:
        # 校验控制器配置前的主动关节列表状态
        assert self.all_joint_names, "构建控制器前，主动关节列表未初始化"
        assert "" not in self.all_joint_names, "主动关节列表含空值，无法构建控制器"

        controllers = {}
        controller_configs = self._controller_configs  # 此时配置已无空值

        for mode_name, group_configs in controller_configs.items():
            controllers[mode_name] = {}
            for group_name, config in group_configs.items():
                # 校验控制器配置的关节列表无空值
                assert "" not in config.joint_names, \
                    f"控制器 {mode_name}.{group_name} 的关节列表含空值：{config.joint_names}"
                # 实例化自定义控制器
                controllers[mode_name][group_name] = TendonPDJointPosController(
                    config=config, agent=self
                )
        return controllers

    # ---------------- 6. 初始化后处理（无空值风险，保留） ---------------- #
    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        if not self.tcp:
            raise ValueError(f"未找到末端Link：{self.ee_link_name}（检查 MJCF 中<body>的name）")

        self.base_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "base_body")
        if not self.base_link:
            raise ValueError("未找到底座Link：base_body（检查 MJCF 中<body>的name）")

        self.base_link.set_collision_group_bit(group=2, bit_idx=31, bit=1)

    # ---------------- 7. 末端位姿工具函数 ---------------- #
    @property
    def tcp_pose(self) -> sapien.Pose:
        return self.tcp.pose

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.tcp.pose.p