import numpy as np
import sapien
from sapien import Pose
from openvlp.env.scene import PullDrawerPutObjectsEnv
from openvlp.agent.planner import PullPushDrawerPlanner
from openvlp.agent.planner.grasp import GraspMotionPlanner, GraspState


def main():
    env = PullDrawerPutObjectsEnv(
        render_mode="human", control_mode="pd_ee_delta_pose", robot_uids="panda"
    )
    env.reset()
    viewer = env.render_human()
    env.viewer.paused = True

    # Initialize the GraspMotionPlanner
    grasp_planner = GraspMotionPlanner(
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.1,
        dh=-0.008,
        dSH=0.61,
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

    # Choose the first can as the target object
    target_can = env.fanta_cans[3]

    while True:
        # Get the current poses
        object_position = target_can.pose.p[0].cpu().numpy()
        tcp_pose = env.agent.tcp.pose
        tcp_pose = Pose(
            p=tcp_pose.raw_pose[0].numpy()[:3], q=tcp_pose.raw_pose[0].numpy()[3:]
        )
        robot_qpos = env.agent.robot.get_qpos()[0]

        # Plan the motion
        ee_action, gripper_action = grasp_planner.plan_motion(
            object_position, robot_qpos, tcp_pose
        )

        # Create the action dictionary
        action_dict = create_action_dict(ee_action, gripper_action)

        # Convert action dict to environment action
        action = env.agent.controller.from_action_dict(action_dict)
        env.step(action)
        env.render_human()

        if viewer.window.key_press("q"):
            break

        print(f"Current state: {grasp_planner.get_current_state()}")
        print(f"EE Action: {ee_action}")
        print(f"Gripper Action: {gripper_action}")


def create_action_dict(ee_action, gripper_action):
    return {
        "arm": ee_action,
        "gripper": np.array([gripper_action]),
    }


if __name__ == "__main__":
    main()
