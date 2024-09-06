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
from openvlp.env.scene.rotate_laptop import RotateLaptopEnv
from openvlp.agent.planner import key_board_control


def main():
    env = RotateLaptopEnv(
        render_mode="human", control_mode="pd_ee_delta_pose", robot_uids="panda"
    )
    env.reset()
    viewer = env.render_human()
    env.viewer.paused = True

    env.agent.robot.set_root_pose(Pose([-0.23, 0.01, 0.4], [1.0, 0, 0, 0.0]))
    env.agent.robot.set_qpos(
        np.array(
            [
                -0.17332372,
                -0.8839507,
                0.1550096,
                -2.544321,
                0.09055639,
                1.7011282,
                -0.9058096,
                0.015,
                0.015,
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
