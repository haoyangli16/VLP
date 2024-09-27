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


@register_env("PullDrawerPutObjectsEnv-v0")
class PullDrawerPutObjectsEnv(BaseEnv):
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
        self._setup_camera(shader_dir="default")
        # self.scene.set_ambient_light([5.0, 5.0, 5.0])

    def _setup_camera(self, shader_dir="default"):
        from mani_skill.envs.sapien_env import Camera, CameraConfig
        from sapien import Pose
        from mani_skill.render.shaders import ShaderConfig

        if shader_dir == "rt":
            pass
            # shader_config = ShaderConfig(
            #     shader_pack="rt",
            #     texture_names={
            #         "Color": ["rgb"],
            #     },
            #     shader_pack_config={
            #         "ray_tracing_samples_per_pixel": 128,
            #         "ray_tracing_path_depth": 16,
            #         "ray_tracing_denoiser": "optix",
            #         # "ray_tracing_exposure": 5.6,
            #     },
            #     texture_transforms={
            #         "Color": lambda data: {
            #             "rgb": (data[..., :3] * 255).to(torch.uint8)
            #         },
            #     },
            # )

            # camera = Camera(
            #     CameraConfig(
            #         "cam",
            #         Pose(),
            #         width=1920,
            #         height=1080,
            #         fov=1.17,
            #         near=0.1,
            #         far=1e03,
            #         # shader_pack="rt",
            #         shader_config=shader_config,
            #     ),  # type: ignore
            #     self.scene,
            # )
            # camera.camera.set_property("exposure", 6.2)

        elif shader_dir == "default":
            camera = Camera(
                CameraConfig(
                    "cam",
                    Pose(),
                    width=1920,
                    height=1080,
                    fov=1.17,
                    near=0.1,
                    far=1e03,
                ),  # type: ignore
                self.scene,
            )

        camera.camera.set_local_pose(
            Pose(
                [-0.434732, -0.80838, 1.41493], [0.875055, -0.12391, 0.307327, 0.35281]
            )
        )

        self.scene.sensors = {"cam": camera}

    def _setup_scene(self):
        super()._setup_scene()
        from mani_skill.utils.building.ground import build_ground

        self.ground = build_ground(self.scene)

    def _load_scene(self, options: dict):
        # --------------TABLE--------------
        b = self.scene.create_actor_builder()
        b.add_nonconvex_collision_from_file(
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/cse_table_with_top.glb"
            ),
            scale=[0.8, 1.0, 0.5],
        )
        b.add_visual_from_file(
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/marbel_coffee_table.glb"
            ),
            scale=[0.8, 1.0, 0.5],
        )
        table = b.build_static(name="table")
        table.set_pose(sapien.Pose(p=[-0.139509, 0, 0]))

        # # --------------FURNITURE--------------
        # urdf_loader = self.scene.create_urdf_loader()
        # self.cart = urdf_loader.load(
        #     # os.path.join(
        #     #     os.path.dirname(__file__),
        #     #     "../../assets/env/furniture/scaled_output.urdf",
        #     # )
        #     "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/cart/scaled_output.urdf"
        # )
        # self.cart.set_pose(
        #     sapien.Pose(
        #         p=[0.0, 1.018 - 0.06, 0.31],  # 0.06 is the height of the cart
        #         q=[1.0, 0, 0, 0.0],
        #     )
        # )
        # for link in self.cart.get_links():
        #     link.set_mass(0.5)

        # --------------fanta_can--------------
        b = self.scene.create_actor_builder()
        b.add_visual_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/Coffee_Cup.glb"),
            scale=[1.5, 1.5, 1.5],
        )
        b.add_convex_collision_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/Coffee_Cup.glb"),
            scale=[1.5, 1.5, 1.5],
        )
        self.fanta_cans = []
        for i in range(10):
            self.fanta_cans.append(b.build(name=f"fanta_can_{i}"))
            if i < 5:
                self.fanta_cans[-1].set_pose(
                    sapien.Pose(
                        p=[0.203, -0.185883 + 0.095 * i, 0.516],
                        q=[0.707, 0.707, 0, 0],
                    )
                )
            else:
                self.fanta_cans[-1].set_pose(
                    sapien.Pose(
                        p=[0.095, -0.185883 + 0.095 * (i - 5), 0.516],
                        q=[0.707, 0.707, 0, 0],
                    )
                )
            self.fanta_cans[-1].set_mass(0.02)

        b = self.scene.create_actor_builder()
        b.add_visual_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/wood_tray.glb"),
            scale=[1.0, 1.0, 1.0],
        )
        b.add_nonconvex_collision_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/wood_tray.glb"),
            scale=[1.0, 1.0, 1.0],
        )
        self.tray = b.build_static(name="tray")
        self.tray.set_pose(sapien.Pose(p=[0.14, 0, 0.385], q=[0.5, 0.5, 0.5, 0.5]))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

    def get_drawer_handle_pose(self):
        # Assuming the drawer handle is the second link of the furniture
        for i, link in enumerate(self.furniture.get_links()):
            if link.get_name() == "link_0":
                return link.pose
        return None

    def get_camera_image(self, shader_dir="default"):
        self.scene.sensors["cam"].camera.take_picture()
        rgba = self.scene.sensors["cam"].camera.get_picture("Color")[0]  # [H, W, 4]

        rgba = rgba.cpu().numpy()
        if shader_dir == "rt":
            rgba = 255 * rgba
        rgba = rgba.astype(np.uint8)
        if len(rgba.shape) != 3 or rgba.shape[2] != 4:
            rgba = rgba.reshape(
                self.scene.sensors["cam"].camera.height,
                self.scene.sensors["cam"].camera.width,
                4,
            )

        return rgba
