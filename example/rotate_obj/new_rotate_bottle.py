import os
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from openvlp.agent.planner import (
    GraspMotionPlanner,
    GraspState,
    RotateMotionPlanner,
    RotateState,
)
from openvlp.agent.utils import create_action_dict, get_tcp_pose
from openvlp.env.scene import RotateBottleEnv
from sapien import Pose


def main():
    env = RotateBottleEnv(
        render_mode="human", control_mode="pd_ee_delta_pose", robot_uids="panda"
    )
    env.reset()
    viewer = env.render_human()
    env.viewer.paused = True

    env.agent.robot.set_root_pose(Pose([-0.23, -0.23, 0.4], [1.0, 0, 0, 0.0]))
    env.agent.robot.set_qpos(
        np.array(
            [
                0.0007757204,
                -0.051,
                -0.011210264,
                -2.302,
                0.03720392,
                2.301,
                -0.612,
                0.019999994,
                0.019999994,
            ]
        )
    )

    action = np.zeros(7)
    idx = 0
    ee_action_scale = 0.1
    ee_rot_action_scale = 0.6
    rotate_planner = RotateMotionPlanner(
        ee_action_scale=ee_action_scale,
        ee_rot_action_scale=ee_rot_action_scale,
        dh=0.03,
        dH=0.25,
        is_google_robot=False,
        target_angle=np.pi / 2,  # 90 degrees rotation
        dSH=0.6,
    )

    while True:
        tcp_pose = env.agent.tcp.pose
        tcp_pose = Pose(
            p=tcp_pose.raw_pose[0].numpy()[:3], q=tcp_pose.raw_pose[0].numpy()[3:]
        )
        object_position = env.cup.pose.p[0].numpy()

        ee_action, gripper_action = rotate_planner.plan_motion(
            object_position, env.agent.robot.get_qpos()[0], tcp_pose
        )

        action_dict = create_action_dict(ee_action, gripper_action)
        print("action_dict", action_dict)
        action = env.agent.controller.from_action_dict(action_dict)

        print(f"Current state: {rotate_planner.get_current_state()}")
        print(f"Action: {action}")

        # env.step(action)
        env.render_human()

        if rotate_planner.get_current_state() == RotateState.FINISHED:
            print("Rotation completed!")
            break

        if viewer.window.key_press("q"):
            break

        idx += 1


if __name__ == "__main__":
    main()
