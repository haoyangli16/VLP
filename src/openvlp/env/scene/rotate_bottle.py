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


@register_env("RotateBottleEnv-v0")
class RotateBottleEnv(BaseEnv):
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
                [-0.330264, -0.914049, 1.18514],
                [0.855213, -0.123984, 0.239695, 0.442476],
            )
        )

        self.scene.sensors = {"cam": camera}

    def _setup_scene(self):
        super()._setup_scene()
        from mani_skill.utils.building.ground import build_ground

        self.ground = build_ground(self.scene)

    def _load_scene(self, options: dict):
        b = self.scene.create_actor_builder()
        b.add_box_collision(
            half_size=[1.44 / 2, 1.44 / 2, 0.79 / 4],
            pose=sapien.Pose(p=[0.0, 0, 0.79 / 4]),
        )
        b.add_visual_from_file(
            os.path.join(
                os.path.dirname(__file__), "../../assets/env/marbel_coffee_table.glb"
            ),
            scale=[1.0, 1.0, 0.5],
        )
        table = b.build_static(name="table")
        # table.set_pose(sapien.Pose(p=[-0.354662, -0.198988, 0], q=[1, 0, 0, 0]))
        table.set_pose(sapien.Pose(p=[0.1, 0, 0.0]))

        # b = self.scene.create_actor_builder()
        # b.add_visual_from_file(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         "../../assets/models/blue_plastic_bottle/textured.dae",
        #     ),
        #     scale=[1.0, 1.0, 1.0],
        # )
        # b.add_convex_collision_from_file(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         "../../assets/models/blue_plastic_bottle/collision.obj",
        #     ),
        #     scale=[1.0, 1.0, 1.0],
        # )
        # self.bottle = b.build(name="bottle")
        # self.bottle.set_pose(sapien.Pose(p=[0.36, 0, 0.520111], q=[1.0, 0, 0, 0]))
        # self.bottle.set_mass(0.01)

        b = self.scene.create_actor_builder()
        b.add_visual_from_file(
            os.path.join(
                os.path.dirname(__file__),
                "../../assets/env/COFFEE.glb",
            ),
            scale=[0.5, 0.5, 0.5],
        )

        self.coffee = b.build_static(name="coffee")
        self.coffee.set_pose(
            sapien.Pose(
                p=[0.11319, -0.117989, 0.392],
                q=[0.707107, 0.707107, 0, 0],
            )
        )

        b = self.scene.create_actor_builder()
        b.add_visual_from_file(
            os.path.join(
                os.path.dirname(__file__),
                "../../assets/env/cup.glb",
            ),
            scale=[0.48, 0.48, 0.48],
        )
        b.add_convex_collision_from_file(
            os.path.join(
                os.path.dirname(__file__),
                "../../assets/env/cup.glb",
            ),
            scale=[0.5, 0.5, 0.5],
        )
        self.cup = b.build(name="cup")
        self.cup.set_pose(
            sapien.Pose(
                p=[0.154908, -0.288168, 0.396001],
                q=[0.707, 0.707, 0.00, 0.00],
            )
        )
        self.cup.set_mass(0.005)

        b = self.scene.create_actor_builder()
        b.add_visual_from_file(
            os.path.join(
                os.path.dirname(__file__),
                "../../assets/env/coffee_maker.glb",
            ),
            scale=[0.048, 0.048, 0.048],
        )
        self.coffee_maker = b.build_static(name="coffee_maker")
        self.coffee_maker.set_pose(
            sapien.Pose(p=[-0.0449604, 0.0771398, 0.392], q=[0.707107, 0.707107, 0, 0])
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        pass

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info)

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
