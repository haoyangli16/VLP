from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import sapien
import sapien.physx as physx

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


@register_agent()
class XArm7(BaseAgent):
    uid = "xarm7"
    urdf_path = "/home/haoyang/project/haoyang/UniSoft/unisoft/assets/robot/sapien_xarm7/xarm7_d435.urdf"

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0,
                    0,
                    0,
                    1.0466666666666667,
                    0,
                    1.0466666666666667,
                    -1.57,
                    0.85,
                    0.85,
                    0.85,
                    0.85,
                    0.85,
                    0.85,
                ]
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        self.arm_stiffness = 2e3
        self.arm_damping = 2e2
        self.arm_force_limit = 42

        self.finger_joint_names = ["drive_joint", "right_outer_knuckle_joint"]
        self.finger_stiffness = 15
        self.finger_damping = 8.5
        self.finger_force_limit = 10

        self.ee_link_name = "base"

        self.ignore_collision_groups = [
            "left_outer_knuckle",
            "left_finger",
            "right_inner_knuckle",
            "right_outer_knuckle",
            "right_finger",
            "link_eef",
            "link7",
            "link6",
            "xarm_gripper_base_link",
        ]
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # -------------------------------------------------------------------------- #
        # Gripper
        # -------------------------------------------------------------------------- #
        gripper_target_pos = PDJointPosMimicControllerConfig(
            self.finger_joint_names,
            lower=0.0,
            upper=0.85,
            stiffness=self.finger_stiffness,
            damping=self.finger_damping,
            force_limit=self.finger_force_limit,
            normalize_action=False,
            use_delta=False,
            drive_mode="force",
        )
        # gripper_target_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_target_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_target_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_target_pos),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=gripper_target_pos
            ),
        )

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

        links = [self.robot.find_link_by_name(i) for i in self.ignore_collision_groups]
        for link in links:
            for obj in link._objs:  # type: ignore
                for s in obj.collision_shapes:
                    s.set_collision_groups([1, 1, 1 << 29, 0])

        self.create_drive(
            links=["left_inner_knuckle", "left_finger"],
            poses=[
                sapien.Pose(p=[-2.9802322e-08, 3.5464998e-02, 4.2039029e-02]),
                sapien.Pose(p=[0.0, -0.01499999, 0.01500002]),
            ],
            limit_x=[0.0, 0.0],
            limit_y=[0.0, 0.0],
            limit_z=[0.0, 0.0],
        )

        self.create_drive(
            links=["right_inner_knuckle", "right_finger"],
            poses=[
                sapien.Pose(p=[2.9802322e-08, -3.5464998e-02, 4.2039029e-02]),
                sapien.Pose(p=[1.4901161e-08, 1.5000006e-02, 1.4999989e-02]),
            ],
            limit_x=[0.0, 0.0],
            limit_y=[0.0, 0.0],
            limit_z=[0.0, 0.0],
        )

        self.robot.set_qpos(self.keyframes["rest"].qpos)

    def create_drive(self, links, poses, limit_x, limit_y, limit_z):
        a = self.robot.find_link_by_name(links[0])
        b = self.robot.find_link_by_name(links[1])
        drive = self.scene.create_drive(
            a,  # type: ignore
            poses[0],
            b,  # type: ignore
            poses[1],
        )
        if limit_x is not None:
            drive.set_limit_x(0, limit_x[0], limit_x[1])
        if limit_y is not None:
            drive.set_limit_y(0, limit_y[0], limit_y[1])
        if limit_z is not None:
            drive.set_limit_z(0, limit_z[0], limit_z[1])
