import os
from enum import Enum
from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from sapien import Pose
from scipy.spatial.transform import Rotation as R


class DrawerState(Enum):
    APPROACH = 1
    GRASP = 2
    PULL = 3
    PUSH = 4


class PullPushDrawerPlanner:
    def __init__(
        self,
        ee_action_scale=0.02,
        ee_rot_action_scale=0.1,
        dr=0.025,
        dH=0.022,
        dh=0.015,
        pull_distance=0.2,
        push_distance=0.2,
    ):
        self.ee_action_scale = ee_action_scale
        self.ee_rot_action_scale = ee_rot_action_scale
        self.dr = dr  # tolerance in x and y
        self.dH = dH  # distance from handle for approach
        self.dh = dh  # distance from handle for grasping
        self.pull_distance = pull_distance
        self.push_distance = push_distance
        self.current_state = DrawerState.APPROACH
        self.initial_handle_position = None
        self.pull_start_position = None
        self.push_start_position = None

    def plan_motion(self, handle_pose, tcp_pose):
        handle_position = handle_pose.p
        handle_orientation = handle_pose.q
        tcp_position = tcp_pose.p
        tcp_orientation = tcp_pose.q

        ee_action = np.zeros(6)
        gripper_action = 0

        if self.current_state == DrawerState.APPROACH:
            ee_action = self._approach_handle(handle_pose, tcp_pose)
            gripper_action = 1  # Open gripper
        elif self.current_state == DrawerState.GRASP:
            ee_action = self._grasp_handle(handle_pose, tcp_pose)
            gripper_action = -1  # Close gripper
        elif self.current_state == DrawerState.PULL:
            ee_action = self._pull_drawer(handle_pose, tcp_pose)
            gripper_action = -1  # Keep gripper closed
        elif self.current_state == DrawerState.PUSH:
            ee_action = self._push_drawer(handle_pose, tcp_pose)
            gripper_action = -1  # Keep gripper closed

        self._update_state(handle_position, tcp_position)
        print(f"current_state: {self.current_state}")

        return ee_action, gripper_action

    def _approach_handle(self, handle_pose, tcp_pose):
        handle_position = handle_pose.p
        tcp_position = tcp_pose.p
        target_position = handle_position - np.array([self.dH, 0, 0])
        print(f"target_position: {target_position}")
        print(f"tcp_position: {tcp_position}")
        position_error = target_position - tcp_position

        position_action = np.clip(
            position_error, -self.ee_action_scale, self.ee_action_scale
        )

        target_orientation = self._get_target_orientation(handle_pose.q)
        orientation_error = self._get_orientation_error(target_orientation, tcp_pose.q)
        rotation_action = np.clip(
            orientation_error, -self.ee_rot_action_scale, self.ee_rot_action_scale
        )

        return np.concatenate([position_action, rotation_action], axis=1)[0]

    def _grasp_handle(self, handle_pose, tcp_pose):
        handle_position = handle_pose.p
        tcp_position = tcp_pose.p
        position_error = handle_position - tcp_position
        position_action = np.clip(
            position_error, -self.ee_action_scale, self.ee_action_scale
        )

        target_orientation = self._get_target_orientation(handle_pose.q)
        orientation_error = self._get_orientation_error(target_orientation, tcp_pose.q)
        rotation_action = np.clip(
            orientation_error, -self.ee_rot_action_scale, self.ee_rot_action_scale
        )

        return np.concatenate([position_action, rotation_action], axis=1)[0]

    def _pull_drawer(self, handle_pose, tcp_pose):
        if self.pull_start_position is None:
            self.pull_start_position = handle_pose.p.clone()

        pull_direction = np.array(
            [-1, 0, 0]
        )  # Assuming drawer pulls along negative x-axis
        target_position = self.pull_start_position + pull_direction * self.pull_distance
        position_error = target_position - tcp_pose.p
        position_action = np.clip(
            position_error, -self.ee_action_scale, self.ee_action_scale
        )

        # Maintain the orientation during pulling
        target_orientation = self._get_target_orientation(handle_pose.q)
        orientation_error = self._get_orientation_error(target_orientation, tcp_pose.q)
        rotation_action = np.clip(
            orientation_error, -self.ee_rot_action_scale, self.ee_rot_action_scale
        )

        return np.concatenate([position_action, rotation_action], axis=1)[0]

    def _push_drawer(self, handle_pose, tcp_pose):
        if self.push_start_position is None:
            self.push_start_position = handle_pose.p.clone()

        push_direction = np.array(
            [1, 0, 0]
        )  # Assuming drawer pushes along positive x-axis
        target_position = self.push_start_position + push_direction * self.push_distance
        position_error = target_position - tcp_pose.p
        position_action = np.clip(
            position_error, -self.ee_action_scale, self.ee_action_scale
        )

        # Maintain the orientation during pushing
        target_orientation = self._get_target_orientation(handle_pose.q)
        orientation_error = self._get_orientation_error(target_orientation, tcp_pose.q)
        rotation_action = np.clip(
            orientation_error, -self.ee_rot_action_scale, self.ee_rot_action_scale
        )

        return np.concatenate([position_action, rotation_action], axis=1)[0]

    def _get_target_orientation(self, handle_orientation):
        # Convert handle orientation to rotation matrix
        handle_rotation = R.from_quat(handle_orientation).as_matrix()
        # Invert the rotation matrix to get the target gripper orientation
        target_rotation = -handle_rotation
        # Convert back to quaternion
        target_orientation = R.from_matrix(target_rotation).as_quat()
        return target_orientation

    def _get_orientation_error(self, target_orientation, current_orientation):
        target_rot = R.from_quat(target_orientation)
        current_rot = R.from_quat(current_orientation)
        error_rot = target_rot * current_rot.inv()
        error = error_rot.as_rotvec()
        # error[0][2] = 0.0
        error[0] = np.array([0.0, 0.0, 0.0])
        return error

    def _update_state(self, handle_position, tcp_position):
        if self.current_state == DrawerState.APPROACH:
            distance = np.linalg.norm(handle_position - tcp_position)
            print(f"distance: {distance}")
            if distance < self.dr:
                self.current_state = DrawerState.GRASP
                self.initial_handle_position = handle_position.clone()
        elif self.current_state == DrawerState.GRASP:
            distance = np.linalg.norm(handle_position - tcp_position)
            print(f"distance: {distance}")
            if distance < self.dh:
                self.current_state = DrawerState.PULL
        elif self.current_state == DrawerState.PULL:
            if (
                np.linalg.norm(handle_position - self.initial_handle_position)
                >= self.pull_distance
            ):
                self.current_state = DrawerState.PUSH
        elif self.current_state == DrawerState.PUSH:
            if (
                np.linalg.norm(handle_position - self.initial_handle_position)
                <= self.dh
            ):
                self.current_state = DrawerState.APPROACH
                self.reset()

    def reset(self):
        self.current_state = DrawerState.APPROACH
        self.initial_handle_position = None
        self.pull_start_position = None
        self.push_start_position = None

    def get_current_state(self):
        return self.current_state
