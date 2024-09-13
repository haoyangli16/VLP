import numpy as np
from enum import Enum
import sapien
from scipy.spatial.transform import Rotation as R


class RotateState(Enum):
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    ROTATE = 5
    DESCEND_AFTER_ROTATE = 6
    RELEASE = 7
    FINISHED = 8


class RotateMotionPlanner:
    def __init__(
        self,
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.1,
        dh=-0.008,
        dSH=0.6,
        target_angle=np.pi / 2,  # Default to 90 degrees rotation
        is_google_robot=False,
    ):
        self.ee_action_scale = ee_action_scale
        self.ee_rot_action_scale = ee_rot_action_scale
        self.dr = dr  # tolerance in x and y
        self.dH = dH  # height above object for approach
        self.dSH = dSH  # height above object for lifting
        self.dh = dh  # distance from object for grasping
        self.target_angle = target_angle
        self.current_state = RotateState.APPROACH
        self.is_google_robot = is_google_robot
        self.initial_orientation = None
        self.rotated_angle = 0
        self.original_grasp_position = None

    def plan_motion(
        self,
        object_position: np.ndarray,
        robot_qpos: np.ndarray,
        tcp_pose: sapien.Pose,
        gripper_index: int = -2,
    ):
        x0, y0, z0 = object_position
        assert isinstance(tcp_pose, sapien.Pose)
        p, q = tcp_pose.p, tcp_pose.q  # (x, y, z), (w, x, y, z)
        q = np.array([q[1], q[2], q[3], q[0]])  # turn into [x, y, z, w]
        x1, y1, z1 = p
        self.gripper_pose = robot_qpos[gripper_index]

        ee_action = np.zeros(6)
        position_error = np.array([x0 - x1, y0 - y1, z0 - z1])

        print(100 * "-")
        print(f"tcp_pose: {tcp_pose}")
        print(f"object_position: {object_position}")
        print("Position error:", position_error)

        # Determine target z based on current state
        if self.current_state == RotateState.APPROACH:
            target_z = z0 + self.dH
        elif self.current_state in [
            RotateState.DESCEND,
            RotateState.DESCEND_AFTER_ROTATE,
        ]:
            target_z = z0 + self.dh
        elif self.current_state == RotateState.LIFT:
            target_z = z0 + self.dSH
        else:  # Other states
            target_z = z1  # Maintain current z position

        # Calculate ee_action for position
        if self.current_state in [
            RotateState.APPROACH,
            RotateState.DESCEND,
            RotateState.LIFT,
            RotateState.DESCEND_AFTER_ROTATE,
        ]:
            for i in range(3):
                if i == 2:  # z-axis
                    error = target_z - z1
                else:
                    error = (
                        position_error[i]
                        if self.current_state
                        not in [RotateState.LIFT, RotateState.DESCEND_AFTER_ROTATE]
                        else 0
                    )
                ee_action[i] = np.clip(
                    error, -self.ee_action_scale, self.ee_action_scale
                )

        # Calculate ee_action for rotation
        if self.current_state == RotateState.ROTATE:
            ee_action[3:] = self._rotate(q)

        # Determine gripper action
        if self.current_state in [
            RotateState.GRASP,
            RotateState.LIFT,
            RotateState.ROTATE,
            RotateState.DESCEND_AFTER_ROTATE,
        ]:  # CLOSE GRIPPER
            gripper_action = 1 if self.is_google_robot else -1
        elif self.current_state in [
            RotateState.RELEASE,
            RotateState.FINISHED,
            RotateState.DESCEND,
        ]:  # OPEN GRIPPER
            gripper_action = -1 if self.is_google_robot else 1
        else:
            gripper_action = 0  # Keep gripper open

        # Determine next state
        self._update_state(position_error, z1, z0)

        print("Current state:", self.current_state)
        print(f"ee_action: {ee_action}")
        print(f"gripper_action: {gripper_action}")
        return ee_action, gripper_action

    def _rotate(self, current_orientation):
        if self.initial_orientation is None:
            self.initial_orientation = current_orientation

        target_rotation = R.from_quat(self.initial_orientation) * R.from_euler(
            "z", self.target_angle
        )
        current_rotation = R.from_quat(current_orientation)
        rotation_error = (target_rotation * current_rotation.inv()).as_rotvec()

        self.rotated_angle = abs(
            R.from_quat(current_orientation).as_euler("xyz")[2]
            - R.from_quat(self.initial_orientation).as_euler("xyz")[2]
        )

        return np.clip(
            rotation_error, -self.ee_rot_action_scale, self.ee_rot_action_scale
        )

    def _update_state(self, position_error, z1, z0):
        GRASP_THRESHOLD = 0.001
        if self.current_state == RotateState.APPROACH:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dH)) < 0.01
            ):
                self.current_state = RotateState.DESCEND
        elif self.current_state == RotateState.DESCEND:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dh)) < 0.01
            ):
                self.current_state = RotateState.GRASP
                self.original_grasp_position = np.array([z0 + self.dh, z0, z0])
        elif self.current_state == RotateState.GRASP:
            if self.gripper_pose >= GRASP_THRESHOLD:
                self.current_state = RotateState.LIFT
        elif self.current_state == RotateState.LIFT:
            if z1 >= self.dSH - 0.01:
                self.current_state = RotateState.ROTATE
                self.initial_orientation = None
        elif self.current_state == RotateState.ROTATE:
            if abs(self.rotated_angle - self.target_angle) < 0.1:  # Allow small error
                self.current_state = RotateState.DESCEND_AFTER_ROTATE
        elif self.current_state == RotateState.DESCEND_AFTER_ROTATE:
            print(f"z0: {z0}")
            print(f"self.original_grasp_position[2]: {self.original_grasp_position[2]}")
            print(
                f"z0 - self.original_grasp_position[2]: {z0 - self.original_grasp_position[2]}"
            )
            if np.linalg.norm(z0 - self.original_grasp_position[2]) < 0.01:
                self.current_state = RotateState.RELEASE
        elif self.current_state == RotateState.RELEASE:
            if self.gripper_pose <= GRASP_THRESHOLD:
                self.current_state = RotateState.FINISHED

    def get_current_state(self):
        return self.current_state

    def reset(self):
        self.current_state = RotateState.APPROACH
        self.initial_orientation = None
        self.rotated_angle = 0
        self.original_grasp_position = None
