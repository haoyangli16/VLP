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
