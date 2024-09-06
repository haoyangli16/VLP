import numpy as np
import sapien
from sapien import Pose
from openvlp.env.scene import FruitEnv
from openvlp.agent.planner.grasp import GraspMotionPlanner, GraspState
from openvlp.agent.planner.moveto import MoveToMotionPlanner, MoveToState
import cv2
from moviepy.editor import VideoClip
import os


def main():
    env = FruitEnv(
        render_mode="human",
        control_mode="pd_ee_delta_pose",
        robot_uids="panda",
        # shader_dir="rt",
    )
    env.reset()
    # viewer = env.render_human()
    # env.viewer.paused = True

    # Initialize video frames list
    frames = []

    # Initialize the GraspMotionPlanner and MoveToMotionPlanner
    grasp_planner = GraspMotionPlanner(
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.1,
        dh=0.002,
        dSH=0.61,
    )

    moveto_planner = MoveToMotionPlanner(
        ee_action_scale=0.05,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.2,
        dh=0.01,
    )

    env.agent.robot.set_root_pose(Pose([-0.06, 0.25, 0.3], [1.0, 0, 0, 0.0]))
    env.agent.robot.set_qpos(
        np.array(
            [
                -0.37908772,
                -0.37868154,
                0.26875466,
                -2.0439303,
                0.10214411,
                1.6774969,
                -2.370688,
                0.015881803,
                0.015881803,
            ]
        )
    )

    # Choose the target can and set the target position
    target_can = env.pears[9]
    target_position = np.array(
        [-0.0653981, 0.881098, 0.552293]
    )  # Example target position on the tray
    current_planner = grasp_planner
    grasping_complete = False

    # Create a directory to store frames
    output_dir = "output_frames"
    os.makedirs(output_dir, exist_ok=True)

    # warm up
    for _ in range(10):
        env.agent.set_control_mode("pd_joint_pos")
        action = np.array(
            [
                [
                    -0.37908772,
                    -0.37868154,
                    0.26875466,
                    -2.0439303,
                    0.10214411,
                    1.6774969,
                    -2.370688,
                    0.015881803,
                ]
            ]
        )
        env.step(action)
        env.scene.update_render()
        # env.render_human()

    env.agent.set_control_mode("pd_ee_delta_pose")
    frame_count = 0
    while True:
        # Get the current poses
        object_position = target_can.pose.p[0].cpu().numpy()
        tcp_pose = env.agent.tcp.pose
        tcp_pose = Pose(
            p=tcp_pose.raw_pose[0].numpy()[:3], q=tcp_pose.raw_pose[0].numpy()[3:]
        )
        robot_qpos = env.agent.robot.get_qpos()[0]

        # Plan the motion based on the current planner
        if not grasping_complete:
            ee_action, gripper_action = current_planner.plan_motion(
                object_position, robot_qpos, tcp_pose
            )
            if current_planner.get_current_state() == GraspState.FINISHED:
                grasping_complete = True
                current_planner = moveto_planner
                print("Grasping complete! Switching to MoveTo planner.")
        else:
            ee_action, gripper_action = current_planner.plan_motion(
                target_position, robot_qpos, tcp_pose
            )

        # Create the action dictionary and apply it
        if not (
            grasping_complete
            and current_planner.get_current_state() == MoveToState.RETREAT
        ):
            action_dict = create_action_dict(ee_action, gripper_action)
            action = env.agent.controller.from_action_dict(action_dict)

        env.step(action)
        env.scene.update_render()

        # Get the camera image and add it to frames
        rgba = env.get_camera_image(shader_dir="rt")
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB)
        frames.append(bgr)

        # env.render_human()
        env.scene.update_render()

        print(f"Current planner: {'Grasp' if not grasping_complete else 'MoveTo'}")
        print(f"Current state: {current_planner.get_current_state()}")
        print(f"EE Action: {ee_action}")
        print(f"Gripper Action: {gripper_action}")

        # Check if the entire process is complete
        if (
            grasping_complete
            and current_planner.get_current_state() == MoveToState.RETREAT
        ):
            current_planner.current_state = MoveToState.APPROACH
            target_position = np.array([0.242069, 0.303048, 0.703647])
            print(
                f"distance: {np.linalg.norm(env.agent.tcp.pose.p[0].cpu().numpy() - target_position)}"
            )
            if (
                np.linalg.norm(env.agent.tcp.pose.p[0].cpu().numpy() - target_position)
                < 0.025
            ):
                print("Object picked and placed successfully!")
                break

        # if frame_count == 10:
        #     break
        frame_count += 1
    # Create video from frames
    create_video(frames, "output_video.mp4", fps=30)


def create_action_dict(ee_action, gripper_action):
    return {
        "arm": ee_action,
        "gripper": np.array([gripper_action]),
    }


def create_video(frames, output_file, fps=30):
    def make_frame(t):
        return frames[int(t * fps)]

    clip = VideoClip(make_frame, duration=len(frames) / fps)
    clip.write_videofile(output_file, fps=fps)
    print(f"Video saved as {output_file}")


if __name__ == "__main__":
    main()
