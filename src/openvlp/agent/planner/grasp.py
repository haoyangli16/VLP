import numpy as np
from enum import Enum


class GraspState(Enum):
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4


class GraspMotionPlanner:
    def __init__(
        self,
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.1,
        dh=-0.008,
        dSH=1.05,
        is_google_robot=False,
    ):
        self.ee_action_scale = ee_action_scale
        self.ee_rot_action_scale = ee_rot_action_scale
        self.dr = dr  # tolerance in x and y
        self.dH = dH  # height above object for approach
        self.dSH = dSH  # height above object for lifting
        self.dh = dh  # distance from object for grasping
        self.current_state = GraspState.APPROACH
        self.rotate_idx = 0
        self.is_google_robot = is_google_robot

    def plan_motion(self, object_position, robot_qpos, tcp_pose):
        x0, y0, z0 = object_position
        p, q = tcp_pose.p, tcp_pose.q  # (x, y, z), (w, x, y, z)
        q = np.array([q[1], q[2], q[3], q[0]])  # turn into [x, y, z, w]
        x1, y1, z1 = p
        self.gripper_pose = robot_qpos[-3]

        # Target orientation (pointing downwards)

        target_q = np.array(
            [0, -0.707, -0.707, 0]
        )  # [w, x, y, z]  -0.399 -0.625 -0.544 0.392
        # turn into [x, y, z, w]
        target_q = np.array([target_q[1], target_q[2], target_q[3], target_q[0]])

        # Initialize ee_action
        ee_action = np.zeros(6)

        # Calculate position error
        position_error = np.array([x0 - x1, y0 - y1, z0 - z1])
        print("Position error:", position_error)

        orientation_error = np.zeros(3)

        # Determine target z based on current state
        if self.current_state == GraspState.APPROACH:
            target_z = z0 + self.dH
        elif self.current_state == GraspState.DESCEND:
            target_z = z0 + self.dh
        elif self.current_state == GraspState.LIFT:
            target_z = self.dSH
        else:  # GRASP state
            target_z = z1  # Maintain current z position

        # Calculate ee_action for position
        if self.current_state in [
            GraspState.APPROACH,
            GraspState.DESCEND,
            GraspState.LIFT,
        ]:
            for i in range(3):
                if i == 2:  # z-axis
                    error = z1 - target_z
                else:
                    if self.current_state != GraspState.LIFT:
                        error = position_error[i]
                    else:
                        error = 0

                ee_action[i] = np.clip(
                    -error, -self.ee_action_scale, self.ee_action_scale
                )

        # Determine gripper action
        if self.current_state in [GraspState.GRASP, GraspState.LIFT]:
            gripper_action = 1 if self.is_google_robot else -1
        else:
            gripper_action = 0  # Keep gripper open

        # Determine next state
        self._update_state(position_error, orientation_error, z1, z0)

        print("Current state:", self.current_state)
        return ee_action, gripper_action

    def _update_state(self, position_error, orientation_error, z1, z0):
        GRASP_THRESHOLD = 0.027 if not self.is_google_robot else 0.58
        if self.current_state == GraspState.APPROACH:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dH)) < 0.01
            ):
                self.current_state = GraspState.DESCEND
        elif self.current_state == GraspState.DESCEND:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dh)) < 0.01
            ):
                self.current_state = GraspState.GRASP
        elif self.current_state == GraspState.GRASP:
            if self.gripper_pose >= GRASP_THRESHOLD:
                self.current_state = GraspState.LIFT
        elif self.current_state == GraspState.LIFT:
            if z1 > z0 + self.dH - 0.01:
                pass  # Maintain lift state

    def get_current_state(self):
        return self.current_state

    def reset(self):
        self.current_state = GraspState.APPROACH
