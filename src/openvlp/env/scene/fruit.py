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


@register_env("FruitEnv-v0")
class FruitEnv(BaseEnv):
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
        self._setup_camera()
        self.scene.set_ambient_light([5.0, 5.0, 5.0])

    def _setup_camera(self, shader_dir="rt"):
        from mani_skill.envs.sapien_env import Camera, CameraConfig
        from sapien import Pose
        from mani_skill.render.shaders import ShaderConfig

        if shader_dir == "rt":
            shader_config = ShaderConfig(
                shader_pack="rt",
                texture_names={
                    "Color": ["rgb"],
                },
                shader_pack_config={
                    "ray_tracing_samples_per_pixel": 128,
                    "ray_tracing_path_depth": 16,
                    "ray_tracing_denoiser": "optix",
                    # "ray_tracing_exposure": 5.6,
                },
                texture_transforms={
                    "Color": lambda data: {
                        "rgb": (data[..., :3] * 255).to(torch.uint8)
                    },
                },
            )

            camera = Camera(
                CameraConfig(
                    "cam",
                    Pose(),
                    width=1920,
                    height=1080,
                    fov=1.17,
                    near=0.1,
                    far=1e03,
                    # shader_pack="rt",
                    shader_config=shader_config,
                ),  # type: ignore
                self.scene,
            )
            camera.camera.set_property("exposure", 6.2)

        else:
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
            Pose([-0.603969, 1.385, 1.46288], [0.82045, 0.167445, 0.301117, -0.456237])
        )

        self.scene.sensors = {"cam": camera}

    def _setup_scene(self):
        super()._setup_scene()
        from mani_skill.utils.building.ground import build_ground

        self.ground = build_ground(self.scene)

    def _load_scene(self, options: dict):
        # --------------MARKET SCENE--------------
        b = self.scene.create_actor_builder()
        b.add_nonconvex_collision_from_file(
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/convience_store.glb"
            ),
            scale=[0.5, 0.5, 0.5],
        )
        b.add_visual_from_file(
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/convience_store.glb"
            ),
            scale=[0.5, 0.5, 0.5],
        )
        self.convience_store = b.build_static(name="convience_store")
        self.convience_store.set_pose(
            sapien.Pose(p=[-4.445, 0, 0.14], q=[0.707, 0.707, 0, 0.0])
        )

        # --------------FRUIT STAND--------------
        b = self.scene.create_actor_builder()
        b.add_nonconvex_collision_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/fruit_stand.glb"),
            # scale=[0.8, 1.0, 0.5],
        )
        b.add_visual_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/fruit_stand.glb"),
            # scale=[0.8, 1.0, 0.5],
        )
        self.fruit_stand = b.build_static(name="fruit_stand")
        self.fruit_stand.set_pose(
            Pose(
                p=[0.727613, 0, -0.160522],
                q=[0.5, 0.5, -0.5, -0.5],
            )
        )

        # --------------TABLE--------------
        b = self.scene.create_actor_builder()
        b.add_nonconvex_collision_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/AGV.glb"),
            scale=[1.0, 1.0, 1.0],
        )
        b.add_visual_from_file(
            os.path.join(os.path.dirname(__file__), "../../assets/env/AGV.glb"),
            scale=[1.0, 1.0, 1.0],
        )
        self.table = b.build_static(name="table")
        self.table.set_pose(
            Pose(p=[-0.299086, 0.243053, 0], q=[0.707107, 0.707107, 0, 0])
        )

        # --------------CART--------------
        b = self.scene.create_actor_builder()
        b.add_nonconvex_collision_from_file(
            "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/cart.glb",
            scale=[0.5, 0.5, 0.5],
        )
        b.add_visual_from_file(
            "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/cart.glb",
            scale=[0.5, 0.5, 0.5],
        )
        self.cart = b.build_static(name="cart")
        self.cart.set_pose(
            Pose([-0.206406, 0.836774, 8.14562e-05], [0.707107, 0.707107, 0, 0])
        )

        # --------------PEARs--------------
        b = self.scene.create_actor_builder()
        b.add_visual_from_file(
            "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/pear.glb",
            scale=[0.8, 0.8, 0.8],
        )
        b.add_convex_collision_from_file(
            "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/env/pear.glb",
            scale=[0.7, 0.7, 0.7],
        )
        self.pears = []
        for i in range(10):
            self.pears.append(b.build(name=f"pear_{i}"))
            if i < 5:
                self.pears[-1].set_pose(
                    sapien.Pose(
                        p=[0.564441, 0.234742 + 0.1 * i, 0.75],
                        q=[0.5, 0.5, 0.5, 0.5],
                    )
                )
            else:
                self.pears[-1].set_pose(
                    sapien.Pose(
                        p=[0.494441, 0.234742 + 0.05 * (i - 5), 0.75],
                        q=[0.5, 0.5, 0.5, 0.5],
                    )
                )
            self.pears[-1].set_mass(0.02)

        # b = self.scene.create_actor_builder()
        # b.add_visual_from_file(
        #     "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/models/tray/tray.glb",
        #     scale=[1.5, 1.0, 1.0],
        # )
        # # b.add_convex_collision_from_file(
        # #     "/home/haoyang/project/haoyang/openvlp/src/openvlp/assets/models/tray/tray.glb",
        # # )
        # self.tray = b.build_static(name="tray")
        # self.tray.set_pose(sapien.Pose(p=[0.1, 0, 0.385], q=[0.5, 0.5, 0.5, 0.5]))

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
