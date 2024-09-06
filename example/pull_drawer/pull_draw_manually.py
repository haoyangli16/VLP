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
from openvlp.agent.planner import key_board_control
from openvlp.env.scene import SimplePullPushEnv


def main():
    env = SimplePullPushEnv(
        render_mode="human", control_mode="pd_ee_delta_pose", robot_uids="panda"
    )
    env.reset()
    viewer = env.render_human()
    env.viewer.paused = True

    env.agent.robot.set_root_pose(Pose([-0.23, 0.01, 0.4], [1.0, 0, 0, 0.0]))
    env.agent.robot.set_qpos(
        np.array(
            [
                0.0007757204,
                -0.63097596,
                -0.011210264,
                -2.8390987,
                0.03720392,
                3.6851797,
                -0.7222085,
                0.019999994,
                0.019999994,
            ]
        )
    )

    idx = 0
    ee_pos_scale = 0.1
    ee_rot_scale = 0.5

    action = np.zeros(7)
    while True:
        action = key_board_control(viewer, ee_pos_scale, ee_rot_scale, action)
        env.step(action)
        env.render_human()
        if viewer.window.key_press("q"):
            break
        idx += 1


if __name__ == "__main__":
    main()
