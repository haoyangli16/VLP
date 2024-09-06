import numpy as np
from enum import Enum
import sapien


class MoveToState(Enum):
    APPROACH = 1
    DESCEND = 2
    RELEASE = 3
    RETREAT = 4


class MoveToMotionPlanner:
    def __init__(
        self,
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.005,
        dH=0.1,
        dh=0.01,
        is_google_robot=False,
    ):
        self.ee_action_scale = ee_action_scale
        self.ee_rot_action_scale = ee_rot_action_scale
        self.dr = dr  # tolerance in x and y
        self.dH = dH  # height above target for approach
        self.dh = dh  # distance from target for release
        self.current_state = MoveToState.APPROACH
        self.is_google_robot = is_google_robot

    def plan_motion(
        self,
        target_position: np.ndarray,
        robot_qpos: np.ndarray,
        tcp_pose: sapien.Pose,
        gripper_index: int = -2,
    ):
        x0, y0, z0 = target_position
        assert isinstance(tcp_pose, sapien.Pose)
        p, q = tcp_pose.p, tcp_pose.q
        x1, y1, z1 = p
        self.gripper_pose = robot_qpos[gripper_index]

        ee_action = np.zeros(6)
        position_error = np.array([x0 - x1, y0 - y1, z0 - z1])

        if self.current_state == MoveToState.APPROACH:
            target_z = z0 + self.dH
        elif self.current_state == MoveToState.DESCEND:
            target_z = z0 + self.dh
        elif self.current_state == MoveToState.RETREAT:
            target_z = z0 + self.dH
        else:  # RELEASE state
            target_z = z1

        if self.current_state in [
            MoveToState.APPROACH,
            MoveToState.DESCEND,
            MoveToState.RETREAT,
        ]:
            for i in range(3):
                if i == 2:  # z-axis
                    error = target_z - z1
                else:
                    error = (
                        position_error[i]
                        if self.current_state != MoveToState.RETREAT
                        else 0.0
                    )

                ee_action[i] = np.clip(
                    error, -self.ee_action_scale, self.ee_action_scale
                )

        if self.current_state in [MoveToState.APPROACH, MoveToState.DESCEND]:
            gripper_action = 1 if self.is_google_robot else -1  # Keep gripper closed
        else:
            gripper_action = -1 if self.is_google_robot else 1  # Open gripper

        self._update_state(position_error, z1, z0)

        print("Current state:", self.current_state)
        return ee_action, gripper_action

    def _update_state(self, position_error, z1, z0):
        if self.current_state == MoveToState.APPROACH:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dH)) < 0.01
            ):
                self.current_state = MoveToState.DESCEND
        elif self.current_state == MoveToState.DESCEND:
            if (
                np.linalg.norm(position_error[:2]) < self.dr
                and abs(z1 - (z0 + self.dh)) < 0.01
            ):
                self.current_state = MoveToState.RELEASE
        elif self.current_state == MoveToState.RELEASE:
            if self.gripper_pose > 0.038:  # Assuming gripper is fully open
                self.current_state = MoveToState.RETREAT
        elif self.current_state == MoveToState.RETREAT:
            if z1 > z0 + self.dH - 0.01:
                pass  # Maintain retreat state

    def get_current_state(self):
        return self.current_state

    def reset(self):
        self.current_state = MoveToState.APPROACH
