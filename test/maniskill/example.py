import gymnasium as gym
import mani_skill.envs

env = gym.make(
    "PickCube-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    num_envs=16,
    parallel_in_single_scene=True,
    viewer_camera_configs=dict(shader_pack="rt-fast"),
    render_mode="human",
)


print("Observation space", env.observation_space)
print("Action space", env.action_space)

obs, _ = env.reset(seed=0)  # reset with a seed for determinism
done = False
while 1:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    # done = terminated or truncated
    env.render()  # a display is required to render
env.close()
