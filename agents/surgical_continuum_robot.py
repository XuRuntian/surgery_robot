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

# 可选：用于设置 base 的碰撞层
MY_BASE_COLLISION_BIT = 31


@register_agent()
class surgical_continuum_robot(BaseAgent):
    uid = "surgical_continuum_robot"
    urdf_path = os.path.join(
        os.path.dirname(__file__),  # 比如：/home/key/surgery_robot/agents/
        "../assets/robots/surgical_continuum_robot/surgery_robot.xml"  # 从 agents 文件夹向上两级到项目根目录，再进入 assets
    )
    keyframes = dict(
        reset=Keyframe(
            pose=sapien.Pose(),
            qpos=np.zeros(25, dtype=np.float32),
        )
    )

    arm_joint_names = [
        "r_joint1", "l_joint1", "r_joint2", "l_joint2",
        "r_joint3", "l_joint3", "r_joint4", "l_joint4",
        "r_joint5", "l_joint5", "r_joint6", "l_joint6",
    ]

    wheel_joint_names = [
        "joint_right_wheel", "joint_left_wheel",
        *[f"joint_swivel_wheel_{i}_{j}" for i in range(1, 5) for j in range(1, 3)]
    ]

    platform_joint_names = ["platform_joint"]
    head_joint_names = ["head_joint1", "head_joint2"]

    # 末端执行器 link 名称（请根据 URDF 修改）
    ee_link_name = "l_link6"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ---------------- 传感器 ---------------- #
    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="head_camera",
                pose=sapien.Pose(p=[0, -0.03, 0.008], q=[ 0.7071, 0, 0, -0.7071]),
                width=1920,
                height=1080,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["camera_link"],
            ), 
            CameraConfig(
                uid="left_hand_camera",
                pose=sapien.Pose(p=[0.08, 0, 0], q=[0, 0.7071, 0, 0.7071]),
                width=1920,
                height=1080,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["l_hand_base_link"],
            ), 
            CameraConfig(
                uid="right_hand_camera",
                pose=sapien.Pose(p=[-0.08, 0, 0], q=[0.7071, 0, -0.7071, 0]),
                width=1920,
                height=1080,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["r_hand_base_link"],
            ), 
        ]

    # ---------------- 控制器 ---------------- #
    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=None,
            upper=None,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )

        controller_configs = dict(
            pd_joint_pos=dict(arm=arm_pd_joint_pos),
            pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),
        )
        return deepcopy(controller_configs)

    # ---------------- 初始化后钩子 ---------------- #
    def _after_init(self):
        # 左臂末端
        self.tcp_left: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "l_link6"  
        )

        # 右臂末端
        self.tcp_right: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "r_link6"  
        )

        # 底盘
        self.base_link: Link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "base_link_underpan"
        )
        self.base_link.set_collision_group_bit(
            group=2, bit_idx=MY_BASE_COLLISION_BIT, bit=1
        )

    # ---------------- 工具函数 ---------------- #
    @property
    def tcp_pose_left(self) -> Pose:
        """末端 TCP pose"""
        return self.tcp_left.pose

    @property
    def tcp_pose_right(self) -> Pose:
        """末端 TCP pose"""
        return self.tcp_right.pose
    @property
    def tcp_pos_left(self) -> Pose:
        """末端 TCP位置"""
        return self.tcp_left.pose.p

    @property
    def tcp_pos_right(self) -> Pose:
        """末端 TCP位置"""
        return self.tcp_right.pose.p 