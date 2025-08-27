import os.path as osp
from typing import Dict

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from agents.surgical_continuum_robot import surgical_continuum_robot
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("SurgeryRobot-v1", max_episode_steps=500)
class SurgeryRobotEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["surgical_continuum_robot", "none"]
    SUPPORTED_ROBOTS = ["surgical_continuum_robot"]
    agent: surgical_continuum_robot
    
    def __init__(self, *args, robot_uids="surgical_continuum_robot", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    