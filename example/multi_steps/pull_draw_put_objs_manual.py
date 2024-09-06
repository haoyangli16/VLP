import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from typing import Any
import os
from sapien import Pose

from openvlp.env.scene import SimplePullPushEnv, PullDrawerPutObjectsEnv
from openvlp.agent.planner import PullPushDrawerPlanner
from openvlp.agent.planner import key_board_control


def main():
    env = PullDrawerPutObjectsEnv(
        render_mode="human", control_mode="pd_ee_delta_pose", robot_uids="panda"
    )
    env.reset()

    # Initialize the PullPushDrawerPlanner
    planner = PullPushDrawerPlanner(
        ee_action_scale=0.05,
        ee_rot_action_scale=0.1,
        pull_distance=0.2,
        push_distance=0.2,
    )

    env.agent.robot.set_root_pose(Pose([-0.23, 0.333, 0.4], [1.0, 0, 0, 0.0]))
    env.agent.robot.set_qpos(
        np.array(
            [
                -0.040970627,
                -0.45182794,
                -0.04281288,
                -2.582755,
                -0.019438026,
                2.1446106,
                -2.4188137,
                0.015500386,
                0.015500386,
            ]
        )
    )

    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02])
    idx = 0

    ee_pos_scale = 0.1
    ee_rot_scale = 0.5

    env.agent.set_control_mode("pd_joint_pos")
    # env.agent._after_init()
    # env.reset()

    action_old = np.array(
        [
            [
                -0.3820436,
                -0.022775656,
                0.33796027,
                -2.4351766,
                -0.06547067,
                2.4152293,
                -2.3808963,
                0.039999135,
            ]
        ]
    )

    for _ in range(20):
        env.step(action_old)

    viewer = env.render_human()
    env.viewer.paused = True
    while True:
        # Get the current poses
        env.agent.set_control_mode("pd_ee_delta_pose")
        action = key_board_control(viewer, ee_pos_scale, ee_rot_scale, action)

        env.step(action)
        env.render_human()
        if viewer.window.key_press("q"):
            break

        # print(f"Current state: {planner.get_current_state()}")
        # print(f"EE Action: {ee_action}")
        # print(f"Gripper Action: {gripper_action}")

        idx += 1


def get_tcp_pose(env):
    for link in env.agent.robot.get_links():
        if link.get_name() == env.agent.ee_link_name:
            return link.get_cmass_local_pose()
    return None


def create_action_dict(ee_action, gripper_action):
    return {
        "arm": ee_action,
        "gripper": gripper_action,
    }


if __name__ == "__main__":
    main()
