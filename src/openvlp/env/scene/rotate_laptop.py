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


@register_env("RotateLaptopEnv-v0")
class RotateLaptopEnv(BaseEnv):
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
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/cse_table_with_top.glb"
            ),
            scale=[1.0, 1.0, 0.5],
        )
        b.add_visual_from_file(
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/cse_table_with_top.glb"
            ),
            scale=[1.0, 1.0, 0.5],
        )
        table = b.build_static(name="table")
        table.set_pose(sapien.Pose(p=[0.1, 0, 0.0]))

        urdf_loader = self.scene.create_urdf_loader()

        self.laptop = urdf_loader.load(
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/laptop/scaled_output.urdf"
            )
        )
        self.laptop.set_pose(sapien.Pose(p=[0.35, 0, 0.48], q=[1, 0, 0, 0]))
        # self.furniture.set_qpos(np.array([0.02, 0.0, 0.0]))
        for link in self.laptop.get_links():
            link.set_mass(0.5)
        for joint in self.laptop.get_active_joints():
            joint.set_drive_properties(stiffness=0.2, damping=0.1, force_limit=20)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    # def get_drawer_handle_pose(self):
    #     # Assuming the drawer handle is the second link of the furniture
    #     for i, link in enumerate(self.safe.get_links()):
    #         if link.get_name() == "link_0":
    #             return link.pose
    #     return None
