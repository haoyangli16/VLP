import numpy as np
import sapien
import torch
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from typing import Any
import os
from sapien import Pose


@register_env("SimplePickEnv-v0")
class SimplePickEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                "base_camera",
                look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1]),
                128,
                128,
                np.pi / 2,
                0.01,
                100,
            )
        ]

    def __init__(
        self,
        *args,
        robot_uids="panda",
        **kwargs,
    ):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        from mani_skill.utils.building.ground import build_ground

        self.ground = build_ground(self.scene)

    def _load_scene(self, options: dict):
        b = self.scene.create_actor_builder()
        b.add_nonconvex_collision_from_file(
            "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/cse_table_with_top.glb"
        )
        b.add_visual_from_file(
            "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/cse_table_with_top.glb"
        )
        table = b.build_static(name="table")
        table.set_pose(sapien.Pose(p=[0.1, 0, 0.0]))

        urdf_loader = self.scene.create_urdf_loader()

        furniture = urdf_loader.load(
            "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/furniture/mobility.urdf"
        )
        furniture.set_pose(sapien.Pose(p=[-1.8, 0, 0.5], q=[0.0, 0, 0, 1.0]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)


def main():
    env = SimplePickEnv(
        render_mode="human", control_mode="pd_joint_pos", robot_uids="panda"
    )
    env.reset()
    viewer = env.render_human()
    env.viewer.paused = True

    env.agent.robot.set_root_pose(Pose([-0.3, 0.01, 0.77], [0, 0, 0, 1.0]))
    env.agent.robot.set_qpos(
        np.array([0.0, 0.798, 0.013, -1.79, 0.051, 3.719, -0.847, 0.04, 0.04])
    )

    action = np.array(
        [
            0.0,
            0.798,
            0.013,
            -1.79,
            0.051,
            3.719,
            -0.847,
            0.02,
        ]
    )
    idx = 0
    while True:
        env.step(action)
        env.render_human()
        if viewer.window.key_press("q"):
            break
        idx += 1


if __name__ == "__main__":
    main()
