import numpy as np
import sapien.utils
import gymnasium as gym


def key_board_control(
    viewer: sapien.utils.Viewer,
    ee_pos_scale: float,
    ee_rot_scale: float,
    action: np.ndarray,
) -> np.ndarray:
    """
    Generate control action based on keyboard input.

    This function maps keyboard inputs to robot end-effector actions, including position control,
    rotation control, and gripper control.

    Args:
        viewer (sapien.utils.Viewer): The Sapien viewer object.
        ee_pos_scale (float): Scaling factor for end-effector position control.
        ee_rot_scale (float): Scaling factor for end-effector rotation control.

    Returns:
        np.ndarray: A 7-dimensional action array where:
            - action[0:3]: End-effector position change (x, y, z)
            - action[3:6]: End-effector rotation change (rx, ry, rz)
            - action[6]: Gripper action (positive to open, negative to close)

    Keyboard controls:
        - 'f': Open gripper
        - 'g': Close gripper
        - 'i'/'k': Move end-effector forward/backward (x-axis)
        - 'j'/'l': Move end-effector left/right (y-axis)
        - 'u'/'o': Move end-effector up/down (z-axis)
        - '1'/'2': Rotate end-effector around x-axis (positive/negative)
        - '3'/'4': Rotate end-effector around y-axis (positive/negative)
        - '5'/'6': Rotate end-effector around z-axis (positive/negative)
    """
    action[:6] = 0.0

    # finger open/close
    if viewer.window.key_press("f"):
        action[6] = 1
    elif viewer.window.key_press("g"):
        action[6] = -1

    # ee position control
    if viewer.window.key_press("i"):
        action[0] = ee_pos_scale
    elif viewer.window.key_press("k"):
        action[0] = -ee_pos_scale
    elif viewer.window.key_press("j"):
        action[1] = ee_pos_scale
    elif viewer.window.key_press("l"):
        action[1] = -ee_pos_scale
    elif viewer.window.key_press("u"):
        action[2] = ee_pos_scale
    elif viewer.window.key_press("o"):
        action[2] = -ee_pos_scale

    # ee rotation control
    if viewer.window.key_press("1"):
        action[3:6] = [ee_rot_scale, 0, 0]
    elif viewer.window.key_press("2"):
        action[3:6] = [-ee_rot_scale, 0, 0]
    elif viewer.window.key_press("3"):
        action[3:6] = [0, ee_rot_scale, 0]
    elif viewer.window.key_press("4"):
        action[3:6] = [0, -ee_rot_scale, 0]
    elif viewer.window.key_press("5"):
        action[3:6] = [0, 0, ee_rot_scale]
    elif viewer.window.key_press("6"):
        action[3:6] = [0, 0, -ee_rot_scale]

    return action
