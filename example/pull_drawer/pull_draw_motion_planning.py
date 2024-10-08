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

from openvlp.env.scene import SimplePullPushEnv
from openvlp.agent.planner import PullPushDrawerPlanner
import cv2


def main():
    env = SimplePullPushEnv(
        render_mode="human", control_mode="pd_ee_delta_pose", robot_uids="panda"
    )
    env.reset()
    viewer = env.render_human()
    env.viewer.paused = True

    # Initialize the PullPushDrawerPlanner
    planner = PullPushDrawerPlanner(
        ee_action_scale=0.05,
        ee_rot_action_scale=0.1,
        pull_distance=0.2,
        push_distance=0.2,
    )

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

    action = np.array(
        [
            0.0,
            0.798,
            0.013,
            -1.79,
            0.051,
            3.719,
            -0.847,
            0.02,
        ]
    )
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02])
    idx = 0
    frames = []

    while True:
        # Get the current poses
        handle_pose = env.get_drawer_handle_pose()
        handle_pose.p[0][1] += 0.05  # A hack to move the handle to the correct position

        tcp_pose = env.agent.tcp.pose
        print(f"tcp_pose: {tcp_pose}")

        # Plan the motion
        ee_action, gripper_action = planner.plan_motion(handle_pose, tcp_pose)

        # Create the action dictionary
        action_dict = create_action_dict(ee_action, gripper_action)

        # Convert action dict to environment action
        import torch

        action = env.agent.controller.from_action_dict(action_dict)

        env.step(action)
        env.render_human()

        # Get the camera image and add it to frames
        rgba = env.get_camera_image(shader_dir="rt")
        bgr = 255 * cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        frames.append(bgr)

        if viewer.window.key_press("q"):
            break

        # print(f"Current state: {planner.get_current_state()}")
        # print(f"EE Action: {ee_action}")
        # print(f"Gripper Action: {gripper_action}")

        idx += 1

    # Create video from frames
    create_video(frames, "output_vide_pull_drawer.mp4", fps=30)


from moviepy.editor import VideoClip


def create_video(frames, output_file, fps=30):
    def make_frame(t):
        return frames[int(t * fps)]

    clip = VideoClip(make_frame, duration=len(frames) / fps)
    clip.write_videofile(output_file, fps=fps)
    print(f"Video saved as {output_file}")


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
