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
    PullPushDrawerPlanner,
    key_board_control,
)
from openvlp.agent.utils import create_action_dict, get_tcp_pose
from openvlp.env.scene import RotateBottleEnv
from sapien import Pose

import cv2


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

    # action = np.array(
    #     [
    #         0.0,
    #         0.798,
    #         0.013,
    #         -1.79,
    #         0.051,
    #         3.719,
    #         -0.847,
    #         0.02,
    #     ]
    # )
    action = np.zeros(7)
    idx = 0
    ee_action_scale = 0.05
    ee_rot_action_scale = 0.5
    planner = GraspMotionPlanner(
        ee_action_scale=ee_action_scale,
        ee_rot_action_scale=ee_rot_action_scale,
        dh=0.03,
        dH=0.25,
        is_google_robot=False,
        dSH=0.6,
    )

    frames = []
    while True:
        # Get the current poses
        # handle_pose = env.get_drawer_handle_pose()
        # handle_pose.p[0][1] += 0.05  # A hack to move the handle to the correct position

        tcp_pose = env.agent.tcp.pose
        tcp_pose = Pose(
            p=tcp_pose.raw_pose[0].numpy()[:3], q=tcp_pose.raw_pose[0].numpy()[3:]
        )
        object_position = env.cup.pose.p[0].numpy()
        # print(f"tcp_pose: {tcp_pose}")
        ee_action, gripper_action = planner.plan_motion(
            object_position, env.agent.robot.get_qpos()[0], tcp_pose
        )
        action_copy = action.copy()
        # Create the action dictionary
        # if idx < 300:
        action_dict = create_action_dict(ee_action, gripper_action)
        action = env.agent.controller.from_action_dict(action_dict)

        if planner.get_current_state() == GraspState.FINISHED:
            # TODO: replace to a rotate motion planner (rotate target angle (like rotate 90 degree), and then stop)
            action[5] = 0.5
            print("finished")
        # else:
        #     action_rot = key_board_control(
        #         viewer, ee_action_scale, ee_rot_action_scale, action_copy
        #     )
        #     action = action_rot
        print("action", action)
        env.step(action)
        env.render_human()

        # Get the camera image and add it to frames
        rgba = env.get_camera_image(shader_dir="rt")
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        frames.append(bgr)

        if viewer.window.key_press("q"):
            break

        print(idx)
        idx += 1

    # Create video from frames
    create_video(frames, "output_video_rotate_bottle.mp4", fps=30)


from moviepy.editor import VideoClip


def create_video(frames, output_file, fps=30):
    def make_frame(t):
        return frames[int(t * fps)]

    clip = VideoClip(make_frame, duration=len(frames) / fps)
    clip.write_videofile(output_file, fps=fps)
    print(f"Video saved as {output_file}")


if __name__ == "__main__":
    main()
