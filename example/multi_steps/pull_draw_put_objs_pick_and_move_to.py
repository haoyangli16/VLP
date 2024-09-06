import numpy as np
import sapien
from sapien import Pose
from openvlp.env.scene import PullDrawerPutObjectsEnv
from openvlp.agent.planner.grasp import GraspMotionPlanner, GraspState
from openvlp.agent.planner.moveto import MoveToMotionPlanner, MoveToState


def main():
    env = PullDrawerPutObjectsEnv(
        render_mode="human", control_mode="pd_ee_delta_pose", robot_uids="panda"
    )
    env.reset()
    viewer = env.render_human()
    env.viewer.paused = True

    # Initialize the GraspMotionPlanner and MoveToMotionPlanner
    grasp_planner = GraspMotionPlanner(
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.1,
        dh=-0.008,
        dSH=0.61,
    )

    moveto_planner = MoveToMotionPlanner(
        ee_action_scale=0.05,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.2,
        dh=0.01,
    )

    env.agent.robot.set_root_pose(Pose([-0.23, 0.333, 0.4], [1.0, 0, 0, 0.0]))
    env.agent.robot.set_qpos(
        np.array(
            [
                -0.38221017,
                -0.022669552,
                0.33722252,
                -2.4347792,
                0.016076842,
                2.4140005,
                -2.2883248,
                0.015881803,
                0.015881803,
            ]
        )
    )

    # Choose the target can and set the target position
    target_can = env.fanta_cans[3]
    target_position = np.array(
        [0.0925518, 0.876873, 0.552293]
    )  # Example target position on the tray
    current_planner = grasp_planner
    grasping_complete = False

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
        env.render_human()

        if viewer.window.key_press("q"):
            break

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


def create_action_dict(ee_action, gripper_action):
    return {
        "arm": ee_action,
        "gripper": np.array([gripper_action]),
    }


if __name__ == "__main__":
    main()
