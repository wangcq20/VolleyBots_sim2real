from typing import List, Tuple

import functorch
import omni.isaac.core.materials as materials
import omni.isaac.core.objects as objects
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.torch as torch_utils
import omni.kit.commands
import torch
import torch.distributions as D
import torch.nn.functional as NNF
from carb import Float3
from omegaconf import DictConfig
from omni.isaac.debug_draw import _debug_draw
from pxr import PhysxSchema, UsdShade
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)

import volley_bots.utils.kit as kit_utils
from volley_bots.envs.isaac_env import AgentSpec, IsaacEnv
from volley_bots.envs.volleyball.common import (
    _carb_float3_add,
    rectangular_cuboid_edges,
)
from volley_bots.robots.drone import MultirotorBase
from volley_bots.utils.torch import euler_to_quaternion, normalize, quaternion_to_euler
from volley_bots.views import RigidPrimView


_COLOR_T = Tuple[float, float, float, float]

import os
import pdb

from omni.isaac.orbit.sensors import ContactSensor, ContactSensorCfg


def _draw_net(
    W: float,
    H_NET: float,
    W_NET: float,
    color_mesh: _COLOR_T = (1.0, 1.0, 1.0, 1.0),
    color_post: _COLOR_T = (1.0, 0.729, 0, 1.0),
    size_mesh_line: float = 3.0,
    size_post: float = 10.0,
):
    n = 30

    point_list_1 = [Float3(0, -W / 2, i * W_NET / n + H_NET - W_NET) for i in range(n)]
    point_list_2 = [Float3(0, W / 2, i * W_NET / n + H_NET - W_NET) for i in range(n)]

    point_list_1.append(Float3(0, W / 2, 0))
    point_list_1.append(Float3(0, -W / 2, 0))

    point_list_2.append(Float3(0, W / 2, H_NET))
    point_list_2.append(Float3(0, -W / 2, H_NET))

    colors = [color_mesh for _ in range(n)]
    sizes = [size_mesh_line for _ in range(n)]
    colors.append(color_post)
    colors.append(color_post)
    sizes.append(size_post)
    sizes.append(size_post)

    return point_list_1, point_list_2, colors, sizes


def _draw_board(
    W: float, L: float, color: _COLOR_T = (1.0, 1.0, 1.0, 1.0), line_size: float = 10.0
):
    point_list_1 = [
        Float3(-L / 2, -W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(-L / 2, -W / 2, 0),
        Float3(L / 2, -W / 2, 0),
        Float3(-L / 6, -W / 2, 0),
        Float3(L / 6, -W / 2, 0),
        Float3(0, -W / 2, 0),
    ]
    point_list_2 = [
        Float3(L / 2, -W / 2, 0),
        Float3(L / 2, W / 2, 0),
        Float3(-L / 2, W / 2, 0),
        Float3(L / 2, W / 2, 0),
        Float3(-L / 6, W / 2, 0),
        Float3(L / 6, W / 2, 0),
        Float3(0, W / 2, 0),
    ]

    colors = [color for _ in range(len(point_list_1))]
    sizes = [line_size for _ in range(len(point_list_1))]

    return point_list_1, point_list_2, colors, sizes


def _draw_lines_args_merger(*args):
    buf = [[] for _ in range(4)]
    for arg in args:
        buf[0].extend(arg[0])
        buf[1].extend(arg[1])
        buf[2].extend(arg[2])
        buf[3].extend(arg[3])

    return (
        buf[0],
        buf[1],
        buf[2],
        buf[3],
    )


def quat_rotate(q, v):
    """
    Rotate vector v by quaternion q
    q: quaternion [w, x, y, z]
    v: vector [x, y, z]
    Returns the rotated vector
    """

    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Quaternion rotation formula
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    # Construct the rotation matrix from the quaternion
    rot_matrix = torch.stack(
        [
            ww + xx - yy - zz,
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            ww - xx + yy - zz,
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)

    v_expanded = v.expand(*q.shape[:-1], 3)

    # Rotate the vector using the rotation matrix
    return torch.matmul(rot_matrix, v_expanded.unsqueeze(-1)).squeeze(-1)


def draw_court(W: float, L: float, H_NET: float, W_NET: float):
    return _draw_lines_args_merger(_draw_net(W, H_NET, W_NET), _draw_board(W, L))


def turn_to_mask(turn: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        turn (torch.Tensor): (*, 1)

    Returns:
        torch.Tensor: (*, 2)
    """
    table = torch.tensor([[True, False], [False, True]], device=turn.device)
    return table[turn[..., 0]]


def turn_to_obs(turn: torch.Tensor):
    """convert representation of drone turn to one-hot vector

    Args:
        turn (torch.Tensor): (n_env, 1)

    Returns:
        torch.Tensor: (n_env, 2, 2)
    """
    table = torch.tensor(
        [[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]],
        device=turn.device,
    )
    return table[turn[..., 0]]


def turn_shift(t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        t (torch.Tensor): (n_env,) int64
        h (torch.Tensor): (n_env,) bool

    Returns:
        torch.Tensor: (n_env,) int64
    """
    return (t + h.long()) % 2


def out_of_bounds(
    pos: torch.Tensor, env_boundary_x: float, env_boundary_y: float
) -> torch.Tensor:
    """_summary_

    Args:
        pos (torch.Tensor): (*,3)
        env_boundary_x (float): _description_
        env_boundary_y (float): _description_

    Returns:
        torch.Tensor: (*,)
    """
    return (pos[..., 0].abs() > env_boundary_x) | (pos[..., 1].abs() > env_boundary_y)


class MultiJuggleVolleyball(IsaacEnv):
    def __init__(self, cfg, headless):
        self.L: float = cfg.task.court.L
        self.W: float = cfg.task.court.W
        self.H_NET: float = cfg.task.court.H_NET  # height of the net
        self.W_NET: float = (
            cfg.task.court.W_NET
        )  # not width of the net, but the width of the net's frame
        self.ball_mass: float = cfg.task.ball_mass
        self.ball_radius: float = cfg.task.ball_radius
        self.min_height: float = cfg.task.min_height
        self.anchor_radius = cfg.task.anchor_radius
        self.racket_radius = 0.2
        self.reward_shaping = cfg.task.reward_shaping
        self.num_drones = 2

        super().__init__(cfg, headless)

        # x, y, z boundary for drone
        self.env_boundary_x = self.L / 2
        self.env_boundary_y = self.W / 2

        # env paras
        self.time_encoding = self.cfg.task.time_encoding
        self.central_env_pos = Float3(
            *self.envs_positions[self.central_env_idx].tolist()
        )

        # drone paras
        self.drone.initialize()
        randomization = self.cfg.task.get("randomization", None)
        if randomization and "drone" in randomization:
            self.drone.setup_randomization(self.cfg.task.randomization["drone"])
        # contact sensor
        contact_sensor_cfg = ContactSensorCfg(prim_path="/World/envs/env_.*/ball")
        self.contact_sensor: ContactSensor = contact_sensor_cfg.class_type(
            contact_sensor_cfg
        )
        self.contact_sensor._initialize_impl()

        # ball paras
        self.ball = RigidPrimView(
            "/World/envs/env_*/ball",
            reset_xform_properties=False,
            track_contact_forces=False,
            shape=(-1, 1),
        )
        self.ball.initialize()

        # drone and ball init
        # (2,3) original positions of two drones without any offset
        self.anchor = torch.tensor(cfg.task.anchor, device=self.device)
        # (2,3) 2 drones' initial positions with offsets
        self.init_drone_pos_dist = D.Uniform(
            torch.tensor(cfg.task.init_drone_pos_dist.low, device=self.device)
            + self.anchor,
            torch.tensor(cfg.task.init_drone_pos_dist.high, device=self.device)
            + self.anchor,
        )
        self.init_drone_rpy_dist = D.Uniform(
            torch.tensor([-0.1, -0.1, 0.3], device=self.device) * torch.pi,
            torch.tensor([0.1, 0.1, -0.3], device=self.device) * torch.pi,
        )
        self.init_ball_offset = torch.tensor(cfg.task.ball_offset, device=self.device)

        # utils
        self.turn = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int64)
        self.last_hit_step = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_cross_step = torch.zeros(self.num_envs, 1, device=self.device)
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.ball_traj_vis = []

        # one-hot id [E,2,2]
        self.id = torch.zeros((cfg.task.env.num_envs, 2, 2), device=self.device)
        self.id[:, 0, 0] = 1
        self.id[:, 1, 1] = 1
        
        self.last_linear_v = torch.zeros(self.num_envs, self.num_drones, device=self.device)
        self.last_angular_v = torch.zeros(self.num_envs, self.num_drones, device=self.device)
        self.last_linear_a = torch.zeros(self.num_envs, self.num_drones, device=self.device)
        self.last_angular_a = torch.zeros(self.num_envs, self.num_drones, device=self.device)
        self.last_linear_jerk = torch.zeros(self.num_envs, self.num_drones, device=self.device)
        self.last_angular_jerk = torch.zeros(self.num_envs, self.num_drones, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.num_drones, 4, device=self.device)
        self.reward_action_smoothness_weight: float = cfg.task.reward_action_smoothness_weight

    def _design_scene(self):
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        material = materials.PhysicsMaterial(
            prim_path="/World/Physics_Materials/physics_material_0",
            restitution=0.8,
        )

        ball = objects.DynamicSphere(
            prim_path="/World/envs/env_0/ball",
            radius=self.ball_radius,
            mass=self.ball_mass,
            color=torch.tensor([1.0, 0.2, 0.2]),
            physics_material=material,
        )
        cr_api = PhysxSchema.PhysxContactReportAPI.Apply(ball.prim)
        cr_api.CreateThresholdAttr().Set(0.0)

        if self.use_local_usd:
            # use local usd resources
            usd_path = os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "assets",
                "default_environment.usd",
            )
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                usd_path=usd_path,
            )
        else:
            # use online usd resources
            kit_utils.create_ground_plane(
                "/World/defaultGroundPlane",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            )
        drone_prims = self.drone.spawn(translations=[(0.0, 0.0, 2.0), (0.0, 0.0, 2.0)])

        material = UsdShade.Material(material.prim)
        for drone_prim in drone_prims:
            collision_prim = drone_prim.GetPrimAtPath("bat/collisions")
            binding_api = UsdShade.MaterialBindingAPI(collision_prim)
            binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

            cr_api_drone = PhysxSchema.PhysxContactReportAPI.Apply(collision_prim)
            cr_api_drone.CreateThresholdAttr().Set(0.0)

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        drone_state_dim = self.drone.state_spec.shape[-1]
        observation_dim = (
            drone_state_dim + 3 + 3 + 3 + 3 + 2 + 2
        )  # specified in function _compute_state_and_obs
        
        self.time_encoding_dim = 4
        if self.cfg.task.time_encoding:
            self.time_encoding_dim = 4
            observation_dim += self.time_encoding_dim
        
        state_dim = observation_dim + self.time_encoding_dim

        self.observation_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "observation": UnboundedContinuousTensorSpec(
                                (2, observation_dim)  # 2 drones
                            ),
                            "state": UnboundedContinuousTensorSpec(
                                (state_dim)
                            ),
                        }
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.action_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {
                            "action": torch.stack([self.drone.action_spec] * 2, dim=0),
                        }
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.reward_spec = (
            CompositeSpec(
                {
                    "agents": CompositeSpec(
                        {"reward": UnboundedContinuousTensorSpec((2, 1))}
                    )
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.done_spec = (
            CompositeSpec(
                {
                    "done": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    "terminated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                    "truncated": DiscreteTensorSpec(2, (1,), dtype=torch.bool),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            2,
            observation_key=("agents", "observation"),
            state_key=("agents", "state"),
            action_key=("agents", "action"),
            reward_key=("agents", "reward"),
        )

        _stats_spec = CompositeSpec(
            {
                "return": UnboundedContinuousTensorSpec(1),
                "episode_len": UnboundedContinuousTensorSpec(1),
                "done": UnboundedContinuousTensorSpec(1),
                "truncated": UnboundedContinuousTensorSpec(1),
                "terminated": UnboundedContinuousTensorSpec(1),
                "ball_misbehave": UnboundedContinuousTensorSpec(1),
                "ball_too_low": UnboundedContinuousTensorSpec(1),
                "ball_too_high": UnboundedContinuousTensorSpec(1),
                "ball_hit_net": UnboundedContinuousTensorSpec(1),
                "ball_out_of_court": UnboundedContinuousTensorSpec(1),
                "drone_misbehave": UnboundedContinuousTensorSpec(1),
                "drone0_misbehave": UnboundedContinuousTensorSpec(1),
                "drone1_misbehave": UnboundedContinuousTensorSpec(1),
                "drone_too_low": UnboundedContinuousTensorSpec(1),
                "drone0_too_low": UnboundedContinuousTensorSpec(1),
                "drone1_too_low": UnboundedContinuousTensorSpec(1),
                "drone_hit_net": UnboundedContinuousTensorSpec(1),
                "drone0_hit_net": UnboundedContinuousTensorSpec(1),
                "drone1_hit_net": UnboundedContinuousTensorSpec(1),
                "wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit": UnboundedContinuousTensorSpec(1),
                "wrong_hit_turn": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit_turn": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit_turn": UnboundedContinuousTensorSpec(1),
                "wrong_hit_racket": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit_racket": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit_racket": UnboundedContinuousTensorSpec(1),
                "misbehave_penalty": UnboundedContinuousTensorSpec(1),
                "drone0_misbehave_penalty": UnboundedContinuousTensorSpec(1),
                "drone1_misbehave_penalty": UnboundedContinuousTensorSpec(1),
                "penalty_ball_misbehave": UnboundedContinuousTensorSpec(1),
                "penalty_drone_misbehave": UnboundedContinuousTensorSpec(1),
                "drone0_penalty_drone_misbehave": UnboundedContinuousTensorSpec(1),
                "drone1_penalty_drone_misbehave": UnboundedContinuousTensorSpec(1),
                "penalty_wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone0_penalty_wrong_hit": UnboundedContinuousTensorSpec(1),
                "drone1_penalty_wrong_hit": UnboundedContinuousTensorSpec(1),
                "task_reward": UnboundedContinuousTensorSpec(1),
                "reward_success_hit": UnboundedContinuousTensorSpec(1),
                "reward_success_cross": UnboundedContinuousTensorSpec(1),
                "penalty_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "penalty_yaw": UnboundedContinuousTensorSpec(1),

                "action_error_order1_mean": UnboundedContinuousTensorSpec(1),
                "action_error_order1_max": UnboundedContinuousTensorSpec(1),
                "smoothness_mean": UnboundedContinuousTensorSpec(1),
                "smoothness_max": UnboundedContinuousTensorSpec(1),
                "linear_v_max": UnboundedContinuousTensorSpec(1),
                "angular_v_max": UnboundedContinuousTensorSpec(1),
                "linear_a_max": UnboundedContinuousTensorSpec(1),
                "angular_a_max": UnboundedContinuousTensorSpec(1),
                "linear_jerk_max": UnboundedContinuousTensorSpec(1),
                "angular_jerk_max": UnboundedContinuousTensorSpec(1),
                "linear_v_mean": UnboundedContinuousTensorSpec(1),
                "angular_v_mean": UnboundedContinuousTensorSpec(1),
                "linear_a_mean": UnboundedContinuousTensorSpec(1),
                "angular_a_mean": UnboundedContinuousTensorSpec(1),
                "linear_jerk_mean": UnboundedContinuousTensorSpec(1),
                "angular_jerk_mean": UnboundedContinuousTensorSpec(1),
                "reward_action_smoothness": UnboundedContinuousTensorSpec(1),

                "num_sim_hits": UnboundedContinuousTensorSpec(1),
                "drone0_num_sim_hits": UnboundedContinuousTensorSpec(1),
                "drone1_num_sim_hits": UnboundedContinuousTensorSpec(1),
                "num_true_hits": UnboundedContinuousTensorSpec(1),
                "drone0_num_true_hits": UnboundedContinuousTensorSpec(1),
                "drone1_num_true_hits": UnboundedContinuousTensorSpec(1),
                "num_success_hits": UnboundedContinuousTensorSpec(1),
                "drone0_num_success_hits": UnboundedContinuousTensorSpec(1),
                "drone1_num_success_hits": UnboundedContinuousTensorSpec(1),
                "wrong_hit_sim": UnboundedContinuousTensorSpec(1),
                "drone0_wrong_hit_sim": UnboundedContinuousTensorSpec(1),
                "drone1_wrong_hit_sim": UnboundedContinuousTensorSpec(1),
                "num_ball_cross": UnboundedContinuousTensorSpec(1),
                "num_true_cross": UnboundedContinuousTensorSpec(1),
                "num_success_cross": UnboundedContinuousTensorSpec(1),
                "cross_height": UnboundedContinuousTensorSpec(1),
                "drone0_x": UnboundedContinuousTensorSpec(1),
                "drone0_y": UnboundedContinuousTensorSpec(1),
                "drone0_z": UnboundedContinuousTensorSpec(1),
                "drone0_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "drone1_x": UnboundedContinuousTensorSpec(1),
                "drone1_y": UnboundedContinuousTensorSpec(1),
                "drone1_z": UnboundedContinuousTensorSpec(1),
                "drone1_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "drone0_hit_x": UnboundedContinuousTensorSpec(1),
                "drone0_hit_y": UnboundedContinuousTensorSpec(1),
                "drone0_hit_z": UnboundedContinuousTensorSpec(1),
                "drone0_hit_dist_to_anchor": UnboundedContinuousTensorSpec(1),
                "drone1_hit_x": UnboundedContinuousTensorSpec(1),
                "drone1_hit_y": UnboundedContinuousTensorSpec(1),
                "drone1_hit_z": UnboundedContinuousTensorSpec(1),
                "drone1_hit_dist_to_anchor": UnboundedContinuousTensorSpec(1),
            }
        )
        if self.reward_shaping:
            _stats_spec.set("shaping_reward", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone0_shaping_reward", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("drone1_shaping_reward", UnboundedContinuousTensorSpec(1))
            _stats_spec.set("reward_hit_direction", UnboundedContinuousTensorSpec(1))
            _stats_spec.set(
                "drone0_reward_hit_direction", UnboundedContinuousTensorSpec(1)
            )
            _stats_spec.set(
                "drone1_reward_hit_direction", UnboundedContinuousTensorSpec(1)
            )
            _stats_spec.set("reward_dist_to_ball", UnboundedContinuousTensorSpec(1))
        stats_spec = _stats_spec.expand(self.num_envs).to(self.device)

        info_spec = (
            CompositeSpec(
                {
                    "drone_state": UnboundedContinuousTensorSpec((self.drone.n, 13)),
                    "prev_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
                    "policy_action": torch.stack([self.drone.action_spec] * self.drone.n, 0).to(self.device),
                }
            )
            .expand(self.num_envs)
            .to(self.device)
        )

        self.observation_spec["stats"] = stats_spec
        self.observation_spec["info"] = info_spec
        self.stats = stats_spec.zero()
        self.info = info_spec.zero()

    def check_ball_near_racket(self):
        z_direction_local = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        z_direction_world = quat_rotate(self.drone_rot, z_direction_local)  # (E, 2, 3)

        normal_vector_world = z_direction_world / torch.norm(
            z_direction_world, dim=-1
        ).unsqueeze(
            -1
        )  # (E, 2, 3)

        cylinder_bottom_center = self.drone.pos + normal_vector_world * 0.055  # (E, 2, 3) cylinder bottom center
        cylinder_axis = 2.0 * self.ball_radius * normal_vector_world

        ball_to_bottom = self.ball_pos - cylinder_bottom_center  # (E, 2, 3)
        projection_ratio = torch.sum(
            ball_to_bottom * cylinder_axis, dim=-1
        ) / torch.sum(
            cylinder_axis * cylinder_axis, dim=-1
        )  # (E, 2) projection of ball_to_bottom on cylinder_axis / cylinder_axis
        within_height = (projection_ratio >= 0) & (projection_ratio <= 1)  # (E, 2)

        projection_point = (
            cylinder_bottom_center + projection_ratio.unsqueeze(-1) * cylinder_axis
        )  # (E, 2, 3)
        distance_to_axis = torch.norm(
            self.ball_pos - projection_point, dim=-1
        )  # (E, 2)
        within_radius = distance_to_axis <= self.racket_radius  # (E, 2)

        return within_height & within_radius  # (E, 2)

    def check_out_of_court(self, pos):
        out_of_x = pos[..., 0].abs() > self.L / 2
        out_of_y = pos[..., 1].abs() > self.W / 2
        return out_of_x | out_of_y

    def check_hit_net(self, pos, radius):
        close_to_net = pos[..., 0].abs() < radius
        within_net_width = pos[..., 1].abs() < self.W / 2
        below_net_height = pos[..., 2] < self.H_NET
        return close_to_net & within_net_width & below_net_height

    def debug_draw_region(self):
        b_x = self.env_boundary_x
        b_y = self.env_boundary_y
        b_z_top = self.env_boundary_z_top
        b_z_bot = self.env_boundary_z_bot
        height = b_z_top - b_z_bot
        color = [(0.95, 0.43, 0.2, 1.0)]
        # [topleft, topright, botleft, botright]

        points_start, points_end = rectangular_cuboid_edges(
            2 * b_x, 2 * b_y, b_z_bot, height
        )
        points_start = [_carb_float3_add(p, self.central_env_pos) for p in points_start]
        points_end = [_carb_float3_add(p, self.central_env_pos) for p in points_end]

        colors_line = color * len(points_start)
        sizes_line = [1.0] * len(points_start)
        self.draw.draw_lines(points_start, points_end, colors_line, sizes_line)

    def debug_draw_turn(self):
        ori = self.envs_positions[self.central_env_idx].detach()
        points = self.anchor.clone()
        points[:, -1] = 0
        points = (ori + points).tolist()
        if self.turn[self.central_env_idx, 0] == 0:
            colors = [(0, 1, 0, 1), (1, 0, 0, 1)]  # green, red
        else:
            colors = [(1, 0, 0, 1), (0, 1, 0, 1)]  # red, green
        sizes = [30.0, 30.0]
        self.draw.draw_points(points, colors, sizes)

    def debug_draw_hit_racket(self, true_hit, ball_near_racket):
        ori = self.envs_positions[self.central_env_idx].detach()
        points = self.anchor.clone() + ori
        # points[:, -1] = 1.5
        points = points.tolist()
        colors = []
        if ball_near_racket[0] == True:
            colors.append((0, 1, 0, 1))  # green
        elif true_hit[0] == True:
            colors.append((1, 0, 0, 1))  # red
        else:
            colors.append((1, 1, 0, 1))  # yellow
        if ball_near_racket[1] == True:
            colors.append((0, 1, 0, 1))  # green
        elif true_hit[1] == True:
            colors.append((1, 0, 0, 1))  # red
        else:
            colors.append((1, 1, 0, 1))  # yellow
        sizes = [30.0, 30.0]
        # self.draw.clear_points()
        self.draw.draw_points(points, colors, sizes)

    def debug_draw_min_height(self, ball_cross):
        ori = self.envs_positions[self.central_env_idx].detach()
        points = self.anchor.mean(dim=0, keepdim=True)
        points[:, 2] = self.min_height
        points = (points + ori).tolist()
        if not ball_cross:
            colors = [(1, 1, 0, 1)]  # yellow
        elif self.ball_pos[self.central_env_idx, 0, 2] > self.min_height:
            colors = [(0, 1, 0, 1)]  # green
        else:
            colors = [(1, 0, 0, 1)]  # red
        sizes = [30]
        self.draw.draw_points(points, colors, sizes)

    def update_mean_stats(self, key, value, cnt_key, idx=None):
        idx = (
            torch.ones_like(value, device=self.device, dtype=torch.bool)
            if idx is None
            else idx
        )
        n = self.stats[cnt_key][idx]
        new_value = value[idx]
        old_value = self.stats[key][idx]
        self.stats[key][idx] = ((n - 1) * old_value + new_value) / n

    def _reset_idx(self, env_ids: torch.Tensor):
        # drone
        self.drone._reset_idx(env_ids, self.training)
        drone_pos = self.init_drone_pos_dist.sample(env_ids.shape)
        drone_rpy = self.init_drone_rpy_dist.sample((*env_ids.shape, 2))
        drone_rot = euler_to_quaternion(drone_rpy)
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), drone_rot, env_ids
        )
        self.drone.set_velocities(
            torch.zeros(len(env_ids), 2, 6, device=self.device), env_ids
        )

        # ball and turn
        turn = torch.zeros(len(env_ids), 1, device=self.device, dtype=torch.int64)
        self.turn[env_ids] = turn

        ball_pos = (
            drone_pos[turn_to_mask(turn)] + self.init_ball_offset
        )  # ball initial position is on the top of the drone
        ball_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(
            len(env_ids), 1
        )
        self.ball.set_world_poses(
            ball_pos + self.envs_positions[env_ids], ball_rot, env_ids
        )
        self.ball.set_velocities(
            torch.zeros(len(env_ids), 6, device=self.device), env_ids
        )
        # fix the mass now
        ball_masses = torch.ones_like(env_ids) * self.ball_mass
        self.ball.set_masses(ball_masses, env_ids)

        # env stats
        self.last_hit_step[env_ids] = -100.0
        self.last_cross_step[env_ids] = -100.0
        self.stats[env_ids] = 0.0

        self.last_linear_v[env_ids] = torch.zeros_like(self.last_linear_v[env_ids])
        self.last_angular_v[env_ids] = torch.zeros_like(self.last_angular_v[env_ids])
        self.last_linear_a[env_ids] = torch.zeros_like(self.last_linear_a[env_ids])
        self.last_angular_a[env_ids] = torch.zeros_like(self.last_angular_a[env_ids])
        self.last_linear_jerk[env_ids] = torch.zeros_like(self.last_linear_jerk[env_ids])
        self.last_angular_jerk[env_ids] = torch.zeros_like(self.last_angular_jerk[env_ids])

        # CTBR
        cmd_init = 2.0 * (self.drone.throttle[env_ids]) ** 2 - 1.0
        self.info['prev_action'][env_ids, :, 3] = cmd_init.mean(dim=-1)
        self.prev_actions[env_ids] = self.info['prev_action'][env_ids].clone()

        # draw
        if (env_ids == self.central_env_idx).any() and self._should_render(0):
            self.ball_traj_vis.clear()
            self.draw.clear_lines()
            # self.debug_draw_region()
            self.debug_draw_turn()
            self.debug_draw_hit_racket([False, False], [False, False])
            self.debug_draw_min_height(False)

            point_list_1, point_list_2, colors, sizes = draw_court(
                self.W, self.L, self.H_NET, self.W_NET
            )
            point_list_1 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_1
            ]
            point_list_2 = [
                _carb_float3_add(p, self.central_env_pos) for p in point_list_2
            ]
            self.draw.draw_lines(point_list_1, point_list_2, colors, sizes)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("agents", "action")].clone()
        # CTBR
        self.info["prev_action"] = tensordict[("info", "prev_action")]
        self.prev_actions = self.info["prev_action"].clone()

        self.info["policy_action"] = tensordict[("info", "policy_action")]
        self.policy_actions = tensordict[("info", "policy_action")].clone()

        self.action_error_order1 = tensordict[("stats", "action_error_order1")].clone()
        self.stats["action_error_order1_mean"].add_(self.action_error_order1.mean(dim=-1).unsqueeze(-1))
        self.stats["action_error_order1_max"].set_(torch.max(self.stats["action_error_order1_max"], self.action_error_order1.mean(dim=-1).unsqueeze(-1)))

        self.effort = self.drone.apply_action(actions)

    def _post_sim_step(self, tensordict: TensorDictBase):
        self.contact_sensor.update(self.dt)

    def _compute_state_and_obs(self):
        # clone here
        self.root_state = self.drone.get_state()
        # pos, quat(4), vel, omega
        self.info["drone_state"][:] = self.root_state[..., :13]
        self.ball_pos, _ = self.get_env_poses(self.ball.get_world_poses())
        self.ball_vel = self.ball.get_velocities()[..., :3]

        # relative position and heading
        self.rpos_ball = self.drone.pos - self.ball_pos

        pos, rot, vel, angular_vel, heading, up, throttle = torch.split(
            self.root_state, split_size_or_sections=[3, 4, 3, 3, 3, 3, 4], dim=-1
        )
        rot = torch.where((rot[..., 0] < 0).unsqueeze(-1), -rot, rot)
        self.drone_rot = rot

        rpy = quaternion_to_euler(rot)
        
        self.drone_pos = pos.squeeze(1)
        self.drone_rpy = rpy.squeeze(1)
        self.drone_vel = vel.squeeze(1)
        self.drone_angular_vel = angular_vel.squeeze(1)
        self.roll = rpy[..., 0]
        self.pitch = rpy[..., 1]
        self.yaw = rpy[..., 2]
        
        self.info["roll"] = self.roll
        self.info["pitch"] = self.pitch
        self.info["yaw"] = self.yaw
        
        self.rpos_drone = torch.stack(
            [
                # [..., drone_id, [x, y, z]]
                self.drone.pos[..., 1, :] - self.drone.pos[..., 0, :],
                self.drone.pos[..., 0, :] - self.drone.pos[..., 1, :],
            ],
            dim=1,
        )  # (E,2,3)

        rpos_anchor = self.drone.pos - self.anchor  # (E,2,3)

        obs = [
            self.root_state,  # (E,2,23)
            rpos_anchor,  # (E,2,3)
            self.rpos_drone[..., :3],  # (E,2,3)
            self.rpos_ball,  # (E,2,3)
            self.ball_vel.expand(-1, 2, 3),  # (E,2,3)
            turn_to_obs(self.turn),  # (E,2,2)
            self.id,  # (E,2,2)
        ]
        # [drone_num(2),
        # each_obs_dim: root_state(rpos_anchor)+rpos_drone(3)+rpos_ball(3)+ball_vel(3)+turn(1)]

        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).reshape(-1, 1, 1)
            obs.append(t.expand(-1, 2, self.time_encoding_dim))

        obs = torch.cat(obs, dim=-1)
        t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
        state = torch.concat([obs.mean(dim=1, keepdim=True), t.expand(-1, self.time_encoding_dim).unsqueeze(1)], dim=-1).squeeze(1)

        self.stats["smoothness_mean"].add_(self.drone.throttle_difference.mean(dim=-1, keepdim=True))
        self.stats["smoothness_max"].set_(torch.max(self.drone.throttle_difference.max(dim=-1, keepdim=True)[0], self.stats["smoothness_max"]))
        # linear_v, angular_v
        self.linear_v = torch.norm(self.root_state[..., 7:10], dim=-1)
        self.angular_v = torch.norm(self.root_state[..., 10:13], dim=-1)
        self.stats["linear_v_max"].set_(torch.max(self.stats["linear_v_max"], torch.abs(self.linear_v).max(dim=-1, keepdim=True)[0]))
        self.stats["linear_v_mean"].add_(self.linear_v.mean(dim=-1, keepdim=True))
        self.stats["angular_v_max"].set_(torch.max(self.stats["angular_v_max"], torch.abs(self.angular_v).max(dim=-1, keepdim=True)[0]))
        self.stats["angular_v_mean"].add_(self.angular_v.mean(dim=-1, keepdim=True))
        # linear_a, angular_a
        self.linear_a = torch.abs(self.linear_v - self.last_linear_v) / self.dt
        self.angular_a = torch.abs(self.angular_v - self.last_angular_v) / self.dt
        self.stats["linear_a_max"].set_(torch.max(self.stats["linear_a_max"], torch.abs(self.linear_a).max(dim=-1, keepdim=True)[0]))
        self.stats["linear_a_mean"].add_(self.linear_a.mean(dim=-1, keepdim=True))
        self.stats["angular_a_max"].set_(torch.max(self.stats["angular_a_max"], torch.abs(self.angular_a).max(dim=-1, keepdim=True)[0]))
        self.stats["angular_a_mean"].add_(self.angular_a.mean(dim=-1, keepdim=True))
        # linear_jerk, angular_jerk
        self.linear_jerk = torch.abs(self.linear_a - self.last_linear_a) / self.dt
        self.angular_jerk = torch.abs(self.angular_a - self.last_angular_a) / self.dt
        self.stats["linear_jerk_max"].set_(torch.max(self.stats["linear_jerk_max"], torch.abs(self.linear_jerk).max(dim=-1, keepdim=True)[0]))
        self.stats["linear_jerk_mean"].add_(self.linear_jerk.mean(dim=-1, keepdim=True))
        self.stats["angular_jerk_max"].set_(torch.max(self.stats["angular_jerk_max"], torch.abs(self.angular_jerk).max(dim=-1, keepdim=True)[0]))
        self.stats["angular_jerk_mean"].add_(self.angular_jerk.mean(dim=-1, keepdim=True))

        self.last_linear_v = self.linear_v.clone()
        self.last_angular_v = self.angular_v.clone()
        self.last_linear_a = self.linear_a.clone()
        self.last_angular_a = self.angular_a.clone()
        self.last_linear_jerk = self.linear_jerk.clone()
        self.last_angular_jerk = self.angular_jerk.clone()

        if self._should_render(0):
            central_env_pos = self.envs_positions[self.central_env_idx]
            ball_plot_pos = (
                self.ball_pos[self.central_env_idx] + central_env_pos
            ).tolist()  # [2, 3]
            if len(self.ball_traj_vis) > 1:
                point_list_0 = self.ball_traj_vis[-1]
                point_list_1 = ball_plot_pos
                colors = [(0.1, 1.0, 0.1, 1.0)]
                sizes = [1.5]
                self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
            self.ball_traj_vis.append(ball_plot_pos)

        return TensorDict(
            {
                "agents": {
                    "observation": obs,
                    "state":state, 
                },
                "stats": self.stats,
                "info": self.info,
            },
            self.num_envs,
        )

    def _compute_reward_and_done(self):
        # ball misbehave # 检查球的违规行为
        ball_too_low = self.ball_pos[..., 2] < 2 * self.ball_radius  # (E, 1) # 检查球是否过低（低于两倍球半径）
        ball_too_high = self.ball_pos[..., 2] > 16  # (E, 1) # 检查球是否过高（高于16米）
        ball_hit_net = self.check_hit_net(self.ball_pos, self.ball_radius)  # (E, 1) # 检查球是否撞网
        ball_out_of_court = self.check_out_of_court(self.ball_pos)  # (E, 1) # 检查球是否出界
        ball_misbehave = ( # 球的违规行为（过低、过高、撞网、出界中任意一项）
            ball_too_low | ball_too_high | ball_hit_net | ball_out_of_court
        )  # (E, 1)

        # drone misbehave # 检查无人机的违规行为
        drone_too_low = self.drone.pos[..., 2] < 2 * self.racket_radius  # (E, 2) # 检查无人机是否过低（低于两倍球拍半径）
        drone_hit_net = self.check_hit_net(self.drone.pos, self.racket_radius)  # (E, 2) # 检查无人机是否撞网
        drone_misbehave = drone_too_low | drone_hit_net  # (E, 2) # 无人机的违规行为（过低或撞网）

        # drone hit ball # 检查无人机击球
        ball_contact_forces = self.contact_sensor.data.net_forces_w  # (E, 1, 3) # 获取球的接触力（世界坐标系）
        hit_drone: torch.Tensor = self.rpos_ball.norm(p=2, dim=-1).argmin( # (E, 1) # 计算哪个无人机离球更近（argmin返回索引0或1）
            dim=1, keepdim=True
        )  
        sim_hit = torch.zeros( # (E, 2) # 初始化一个布尔张量，用于标记模拟中发生的碰撞
            self.num_envs, 2, device=self.device, dtype=torch.bool
        )  
        sim_hit[turn_to_mask(hit_drone)] = ball_contact_forces.any(-1).squeeze(-1) # 如果球受到了接触力，标记离球最近的那个无人机为“模拟击球”

        # 判断击球是否为有效（非连续）
        true_hit_step_gap = 3 # 定义两次“真实击球”之间的最小时间步间隔
        true_hit = sim_hit & ( # “真实击球”= 模拟击球 并且 距离上次击球时间 > 间隔
            (self.progress_buf.unsqueeze(-1) - self.last_hit_step) > true_hit_step_gap
        )
        wrong_hit_sim = sim_hit & ( # “错误模拟击球”= 模拟击球 但是 距离上次击球时间 <= 间隔（即连续碰撞）
            (self.progress_buf.unsqueeze(-1) - self.last_hit_step) <= true_hit_step_gap
        )
        # self.last_hit_step[sim_hit] = self.progress_buf[sim_hit.any(-1)] # 更新发生了模拟击球的环境的“最后击球时间”
        self.last_hit_step[true_hit] = self.progress_buf[true_hit.any(-1)] # 更新发生了模拟击球的环境的“最后击球时间”

        # 检查是否由正确回合的无人机击球
        wrong_hit_turn: torch.Tensor = true_hit & ( # “错误回合击球”= 真实击球 并且 击球的无人机不是当前回合的无人机
            self.turn != hit_drone
        )
    
        # 检测是否拍打到球 # 检查球是否在球拍范围内
        ball_near_racket = self.check_ball_near_racket() # (E, 2) # 检查球是否在无人机的球拍（圆柱体）范围内
        wrong_hit_racket = true_hit & torch.logical_not(ball_near_racket) # “错误球拍击球”= 真实击球 并且 球不在球拍范围内
        wrong_hit = wrong_hit_turn | wrong_hit_racket # “错误击球”= 错误回合 或 错误球拍
        success_hit = true_hit & torch.logical_not(wrong_hit) # “成功击球”= 真实击球 并且 不是错误击球

        self.turn = (self.turn + success_hit.any(dim=-1, keepdim=True)) % 2 # 如果有成功击球，则切换回合（0变1，1变0）

        # ball cross middle # 检查球是否过网
        true_cross_step_gap = 3 # 定义两次“真实过网”之间的最小时间步间隔
        ball_cross = self.ball_pos[..., 1].abs() <= self.ball_radius  # (E, 1) # 检查球是否在球网（y=0）附近（一个球半径内）
        true_cross = ball_cross & ( # “真实过网”= 球在球网附近 并且 距离上次过网时间 > 间隔
            self.progress_buf.unsqueeze(-1) - self.last_cross_step > true_cross_step_gap
        )  # (E, 1)
        above_min_height = self.ball_pos[..., 2] > self.min_height  # (E, 1) # 检查球的高度是否高于要求的最小高度
        success_cross = true_cross & above_min_height  # (E, 1) # “成功过网”= 真实过网 并且 高于最小高度
        self.last_cross_step[ball_cross] = self.progress_buf[ball_cross.squeeze(-1)] # 更新发生了球过网的环境的“最后过网时间”

        if self._should_render(0): # 如果需要渲染（通常是中心环境）
            self.debug_draw_hit_racket( # 绘制击球和球拍检测的可视化标记
                true_hit[self.central_env_idx], ball_near_racket[self.central_env_idx]
            )
            self.debug_draw_min_height(ball_cross[self.central_env_idx, 0]) # 绘制最小过网高度的可视化标记
            if success_hit[self.central_env_idx].any(): # 如果中心环境发生了成功击球
                self.debug_draw_turn() # 绘制当前回合的可视化标记

        # misbehave penalty # 计算违规行为惩罚
        _misbehave_penalty_coeff = 10.0 # 违规行为惩罚系数
        penalty_ball_misbehave = ( # 球违规的惩罚（共享，稀疏）
            _misbehave_penalty_coeff * ball_misbehave
        )  # share, sparse, (E, 1)
        penalty_drone_misbehave = ( # 无人机违规的惩罚（独立，稀疏）
            _misbehave_penalty_coeff * drone_misbehave
        )  # individual, sparse, (E, 2)
        penalty_wrong_hit = ( # 错误击球的惩罚（独立，稀疏）
            _misbehave_penalty_coeff * wrong_hit
        )  # individual, sparse, (E, 2)

        misbehave_penalty = ( # 总的违规惩罚（(E, 2)，通过广播相加）
            penalty_ball_misbehave + penalty_drone_misbehave + penalty_wrong_hit
        )  # (E, 2)

        # task reward # 计算任务奖励
        _task_reward_coeff = 1.0  # 1.0,10.0 # 任务奖励系数
        reward_success_hit = _task_reward_coeff * success_hit.any( # 成功击球的奖励（共享，稀疏）
            -1, keepdim=True
        )  # share, sparse, (E, 1)
        reward_success_cross = ( # 成功过网的奖励（共享，稀疏）
            _task_reward_coeff * success_cross
        )  # share, sparse, (E, 1)

        _dist_coeff = 0.05  # 0.05,0.03 # 距离惩罚系数
        dist_to_anchor = torch.norm(self.drone.pos - self.anchor, p=2, dim=-1)  # (E, 2) # 计算无人机到其锚点（anchor）的距离
        penalty_dist_to_anchor = _dist_coeff * ( # 距离锚点过远（超过半径）的惩罚
            dist_to_anchor - self.anchor_radius
        ).clamp(
            min=0
        )  # individual, sparse, (E, 2)
        penalty_dist_to_anchor = penalty_dist_to_anchor.mean( # 将两个无人机的惩罚平均（共享，稀疏）
            -1, keepdim=True
        )  # share, sparse, (E, 1)

        task_reward = reward_success_hit + reward_success_cross - penalty_dist_to_anchor # 总的任务奖励

        # shaping reward # 计算塑形奖励
        dist_to_ball_xy = torch.norm( # 计算无人机到球的XY平面距离
            self.drone.pos[..., :2] - self.ball_pos[..., :2], p=2, dim=-1
        )  # (E, 2)
        reward_dist_to_ball = ( # 靠近球的奖励（塑形）
            _dist_coeff * (2 * turn_to_mask(self.turn) - 1) / (1 + dist_to_ball_xy) # (2*mask-1)使得当前回合无人机为正奖励，另一无人机为负奖励
        )  # individual, dense, (E, 2)
        reward_dist_to_ball = reward_dist_to_ball.mean( # 将两个无人机的奖励平均（共享，稠密）
            -1, keepdim=True
        )  # share, dense, (E, 1)

        _direction_reward_coeff = 1.0 # 击球方向奖励系数
        target_dir_xy = ( # 目标方向（从当前回合无人机指向对方无人机）的XY向量
            self.drone.pos[turn_to_mask(~self.turn)]
            - self.drone.pos[turn_to_mask(self.turn)]
        )[
            ..., :2
        ]  # (E, 2)
        ball_dir_xy = self.ball_vel[:, 0, :2]  # (E, 2) # 球的速度向量（XY平面）
        cosine_similarity = NNF.cosine_similarity( # 计算目标方向和球速度方向的余弦相似度
            target_dir_xy, ball_dir_xy, dim=-1
        ).unsqueeze(
            -1
        )  # (E, 1)
        reward_hit_direction = ( # 击球方向奖励 = 系数 * 成功击球 * 相似度
            _direction_reward_coeff * success_hit * cosine_similarity
        )  # individual, sparse, (E, 2)

        shaping_reward = reward_dist_to_ball + reward_hit_direction # 总的塑形奖励

        not_begin_flag = (self.progress_buf > 1).unsqueeze(1) # 标记是否非初始步骤（>1），避免在第一步计算
        reward_action_smoothness = self.reward_action_smoothness_weight * torch.exp(-self.action_error_order1) * not_begin_flag.float() # 动作平滑度奖励

        _penalty_yaw_coeff = 0.03
        penalty_yaw = _penalty_yaw_coeff * self.yaw.abs()

        reward = -misbehave_penalty + task_reward + self.reward_shaping * shaping_reward + 0.8 * reward_action_smoothness - penalty_yaw # 总奖励

        # done # 计算回合是否结束
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze( # 检查是否达到最大回合长度（截断）
            -1
        )  # (E, 1)
        terminated = ( # 检查是否触发终止条件（球违规、无人机违规、错误击球）
            ball_misbehave
            | drone_misbehave.any(-1, keepdim=True)
            | wrong_hit.any(-1, keepdim=True)
        )  # (E, 1)
        done: torch.Tensor = truncated | terminated  # (E, 1) # 最终的 done 标记（截断或终止）

        # log stats # 记录统计数据
        self.stats["return"].add_(reward.mean(dim=-1, keepdim=True)) # 累加平均奖励到回报
        self.stats["episode_len"] = self.progress_buf.unsqueeze(1) # 记录当前回合长度

        self.stats["done"].add_(done.float()) # 累加 done 次数
        self.stats["truncated"].add_(truncated.float()) # 累加 truncated (截断) 次数
        self.stats["terminated"].add_(terminated.float()) # 累加 terminated (终止) 次数

        self.stats["ball_misbehave"] = ball_misbehave.float() # 记录球违规
        self.stats["ball_too_low"] = ball_too_low.float() # 记录球过低
        self.stats["ball_too_high"] = ball_too_high.float() # 记录球过高
        self.stats["ball_hit_net"] = ball_hit_net.float() # 记录球撞网
        self.stats["ball_out_of_court"] = ball_out_of_court.float() # 记录球出界

        self.stats["drone_misbehave"] = drone_misbehave.any(-1, keepdim=True).float() # 记录无人机违规（任意一个）
        self.stats["drone0_misbehave"] = drone_misbehave[..., 0].unsqueeze(-1).float() # 记录无人机0违规
        self.stats["drone1_misbehave"] = drone_misbehave[..., 1].unsqueeze(-1).float() # 记录无人机1违规
        self.stats["drone_too_low"] = drone_too_low.any(-1, keepdim=True).float() # 记录无人机过低（任意一个）
        self.stats["drone0_too_low"] = drone_too_low[..., 0].unsqueeze(-1).float() # 记录无人机0过低
        self.stats["drone1_too_low"] = drone_too_low[..., 1].unsqueeze(-1).float() # 记录无人机1过低
        self.stats["drone_hit_net"] = drone_hit_net.any(-1, keepdim=True).float() # 记录无人机撞网（任意一个）
        self.stats["drone0_hit_net"] = drone_hit_net[..., 0].unsqueeze(-1).float() # 记录无人机0撞网
        self.stats["drone1_hit_net"] = drone_hit_net[..., 1].unsqueeze(-1).float() # 记录无人机1撞网

        self.stats["wrong_hit"] = wrong_hit.any(-1, keepdim=True).float() # 记录错误击球（任意一个）
        self.stats["drone0_wrong_hit"] = wrong_hit[..., 0].unsqueeze(-1).float() # 记录无人机0错误击球
        self.stats["drone1_wrong_hit"] = wrong_hit[..., 1].unsqueeze(-1).float() # 记录无人机1错误击球
        self.stats["wrong_hit_turn"] = wrong_hit_turn.any(-1, keepdim=True).float() # 记录错误回合击球（任意一个）
        self.stats["drone0_wrong_hit_turn"] = ( # 记录无人机0错误回合击球
            wrong_hit_turn[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_wrong_hit_turn"] = ( # 记录无人机1错误回合击球
            wrong_hit_turn[..., 1].unsqueeze(-1).float()
        )
        self.stats["wrong_hit_racket"] = wrong_hit_racket.any(-1, keepdim=True).float() # 记录错误球拍击球（任意一个）
        self.stats["drone0_wrong_hit_racket"] = ( # 记录无人机0错误球拍击球
            wrong_hit_racket[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_wrong_hit_racket"] = ( # 记录无人机1错误球拍击球
            wrong_hit_racket[..., 1].unsqueeze(-1).float()
        )

        self.stats["misbehave_penalty"].add_( # 累加平均违规惩罚
            misbehave_penalty.mean(dim=-1, keepdim=True)
        )
        self.stats["drone0_misbehave_penalty"].add_( # 累加无人机0违规惩罚
            misbehave_penalty[..., 0].unsqueeze(-1)
        )
        self.stats["drone1_misbehave_penalty"].add_( # 累加无人机1违规惩罚
            misbehave_penalty[..., 1].unsqueeze(-1)
        )
        self.stats["penalty_ball_misbehave"].add_(penalty_ball_misbehave) # 累加球违规惩罚
        self.stats["penalty_drone_misbehave"].add_( # 累加平均无人机违规惩罚
            penalty_drone_misbehave.mean(dim=-1, keepdim=True)
        )
        self.stats["drone0_penalty_drone_misbehave"].add_( # 累加无人机0违规惩罚
            penalty_drone_misbehave[..., 0].unsqueeze(-1)
        )
        self.stats["drone1_penalty_drone_misbehave"].add_( # 累加无人机1违规惩罚
            penalty_drone_misbehave[..., 1].unsqueeze(-1)
        )
        self.stats["penalty_wrong_hit"].add_( # 累加平均错误击球惩罚
            penalty_wrong_hit.mean(dim=-1, keepdim=True)
        )
        self.stats["drone0_penalty_wrong_hit"].add_( # 累加无人机0错误击球惩罚
            penalty_wrong_hit[..., 0].unsqueeze(-1)
        )
        self.stats["drone1_penalty_wrong_hit"].add_( # 累加无人机1错误击球惩罚
            penalty_wrong_hit[..., 1].unsqueeze(-1)
        )

        self.stats["task_reward"].add_(task_reward) # 累加任务奖励
        self.stats["reward_success_hit"].add_(reward_success_hit) # 累加成功击球奖励
        self.stats["reward_success_cross"].add_(reward_success_cross) # 累加成功过网奖励
        self.stats["penalty_dist_to_anchor"].add_(penalty_dist_to_anchor) # 累加锚点距离惩罚
        self.stats["reward_action_smoothness"].add_(reward_action_smoothness.mean(dim=-1, keepdim=True)) # 累加动作平滑度奖励
        self.stats["penalty_yaw"].add_(penalty_yaw.mean(dim=-1, keepdim=True)) # 累加动作平滑度奖励


        if self.reward_shaping: # 如果启用了奖励塑形
            self.stats["shaping_reward"].add_(shaping_reward.mean(dim=-1, keepdim=True)) # 累加平均塑形奖励
            self.stats["drone0_shaping_reward"].add_( # 累加无人机0塑形奖励
                shaping_reward[..., 0].unsqueeze(-1)
            )
            self.stats["drone1_shaping_reward"].add_( # 累加无人机1塑形奖励
                shaping_reward[..., 1].unsqueeze(-1)
            )
            self.stats["reward_hit_direction"].add_( # 累加平均击球方向奖励
                reward_hit_direction.mean(dim=-1, keepdim=True)
            )
            self.stats["drone0_reward_hit_direction"].add_( # 累加无人机0击球方向奖励
                reward_hit_direction[..., 0].unsqueeze(-1)
            )
            self.stats["drone1_reward_hit_direction"].add_( # 累加无人机1击球方向奖励
                reward_hit_direction[..., 1].unsqueeze(-1)
            )
            self.stats["reward_dist_to_ball"].add_(reward_dist_to_ball) # 累加到球距离奖励

        self.stats["num_sim_hits"].add_(sim_hit.any(-1, keepdim=True).float()) # 累加模拟击球次数
        self.stats["drone0_num_sim_hits"].add_(sim_hit[..., 0].unsqueeze(-1).float()) # 累加无人机0模拟击球次数
        self.stats["drone1_num_sim_hits"].add_(sim_hit[..., 1].unsqueeze(-1).float()) # 累加无人机1模拟击球次数
        self.stats["num_true_hits"].add_(true_hit.any(-1, keepdim=True).float()) # 累加真实击球次数
        self.stats["drone0_num_true_hits"].add_(true_hit[..., 0].unsqueeze(-1).float()) # 累加无人机0真实击球次数
        self.stats["drone1_num_true_hits"].add_(true_hit[..., 1].unsqueeze(-1).float()) # 累加无人机1真实击球次数
        self.stats["num_success_hits"].add_(success_hit.any(-1, keepdim=True).float()) # 累加成功击球次数
        self.stats["drone0_num_success_hits"].add_( # 累加无人机0成功击球次数
            success_hit[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_num_success_hits"].add_( # 累加无人机1成功击球次数
            success_hit[..., 1].unsqueeze(-1).float()
        )
        self.stats["wrong_hit_sim"].add_(wrong_hit_sim.any(-1, keepdim=True).float()) # 累加错误模拟击球次数（连续击球）
        self.stats["drone0_wrong_hit_sim"].add_( # 累加无人机0错误模拟击球次数
            wrong_hit_sim[..., 0].unsqueeze(-1).float()
        )
        self.stats["drone1_wrong_hit_sim"].add_( # 累加无人机1错误模拟击球次数
            wrong_hit_sim[..., 1].unsqueeze(-1).float()
        )

        self.stats["num_ball_cross"].add_(ball_cross.float()) # 累加球过网次数
        self.stats["num_true_cross"].add_(true_cross.float()) # 累加球真实过网次数
        self.stats["num_success_cross"].add_(success_cross.float()) # 累加球成功过网次数
        if success_cross.any(): # 如果有成功过网
            self.update_mean_stats( # 更新平均过网高度
                "cross_height",
                self.ball_pos[..., 2],
                "num_success_cross",
                success_cross,
            )

        self.update_mean_stats( # 更新无人机0的平均x坐标
            "drone0_x", self.drone.pos[:, 0, 0].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats( # 更新无人机0的平均y坐标
            "drone0_y", self.drone.pos[:, 0, 1].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats( # 更新无人机0的平均z坐标
            "drone0_z", self.drone.pos[:, 0, 2].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats( # 更新无人机0到锚点的平均距离
            "drone0_dist_to_anchor", dist_to_anchor[:, 0].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats( # 更新无人机1的平均x坐标
            "drone1_x", self.drone.pos[:, 1, 0].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats( # 更新无人机1的平均y坐标
            "drone1_y", self.drone.pos[:, 1, 1].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats( # 更新无人机1的平均z坐标
            "drone1_z", self.drone.pos[:, 1, 2].unsqueeze(-1), "episode_len"
        )
        self.update_mean_stats( # 更新无人机1到锚点的平均距离
            "drone1_dist_to_anchor", dist_to_anchor[:, 1].unsqueeze(-1), "episode_len"
        )

        if success_hit[..., 0].any(): # 如果无人机0成功击球
            self.update_mean_stats( # 更新无人机0成功击球时的平均x坐标
                "drone0_hit_x",
                self.drone.pos[:, 0, 0].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
            self.update_mean_stats( # 更新无人机0成功击球时的平均y坐标
                "drone0_hit_y",
                self.drone.pos[:, 0, 1].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
            self.update_mean_stats( # 更新无人机0成功击球时的平均z坐标
                "drone0_hit_z",
                self.drone.pos[:, 0, 2].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
            self.update_mean_stats( # 更新无人机0成功击球时到锚点的平均距离
                "drone0_hit_dist_to_anchor",
                dist_to_anchor[:, 0].unsqueeze(-1),
                "drone0_num_success_hits",
                success_hit[..., 0].unsqueeze(-1),
            )
        if success_hit[..., 1].any(): # 如果无人机1成功击球
            self.update_mean_stats( # 更新无人机1成功击球时的平均x坐标
                "drone1_hit_x",
                self.drone.pos[:, 1, 0].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )
            self.update_mean_stats( # 更新无人机1成功击球时的平均y坐标
                "drone1_hit_y",
                self.drone.pos[:, 1, 1].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )
            self.update_mean_stats( # 更新无人机1成功击球时的平均z坐标
                "drone1_hit_z",
                self.drone.pos[:, 1, 2].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )
            self.update_mean_stats( # 更新无人机1成功击球时到锚点的平均距离
                "drone1_hit_dist_to_anchor",
                dist_to_anchor[:, 0].unsqueeze(-1),
                "drone1_num_success_hits",
                success_hit[..., 1].unsqueeze(-1),
            )
            

        # 这里的逻辑是：在回合结束时（done=True），将累加的统计量（如v, a, jerk）转换为平均值
        ep_len = self.progress_buf.unsqueeze(-1) # 获取当前回合长度
        self.stats['action_error_order1_mean'].div_( # 计算动作误差的平均值（除以回合长度）
            torch.where(done, ep_len, torch.ones_like(ep_len)) # 仅在done时除以ep_len，否则除以1
        )
        self.stats['smoothness_mean'].div_( # 计算平滑度的平均值
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_v_mean"].div_( # 计算线速度的平均值
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_v_mean"].div_( # 计算角速度的平均值
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_a_mean"].div_( # 计算线加速度的平均值
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_a_mean"].div_( # 计算角加速度的平均值
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["linear_jerk_mean"].div_( # 计算线加加速度（jerk）的平均值
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )
        self.stats["angular_jerk_mean"].div_(  # 计算角加加速度（jerk）的平均值
            torch.where(done, ep_len, torch.ones_like(ep_len))
        )

        return TensorDict( # 返回包含奖励和done信息的TensorDict
            {
                "agents": {"reward": reward.unsqueeze(-1)}, # 奖励（(E, 2, 1)）
                "done": done, # done 标记 (E, 1)
                "terminated": terminated, # 终止标记 (E, 1)
                "truncated": truncated, # 截断标记 (E, 1)
            },
            self.num_envs, # 批次大小 (E)
        )