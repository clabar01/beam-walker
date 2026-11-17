"""mjlab task "BeamWalker-DriveStraight": teach the beam walker to walk in a straight line.

Structure copied from mjlab 1.6.0's cartpole task and the course's 09_onshape_robot/mjlab_task/drive_task.py:
an MJCF entity, a velocity action, custom observation / reward functions, and register_mjlab_task().

Every robot starts at its env origin facing +x (yaw 0), so "the starting heading" is +x and the heading
error is simply the chassis yaw. On the real robot that matches calling dm.imu_reset_yaw_axis() before
it starts walking.

The policy only sees what the real robot can sense:
  leg angles (absolute encoders), leg speeds (motor speed), its own last command, and the built-in IMU
  (gyro z and yaw from legoeducation's dm.imu_device). Nothing about position or world velocity.

Importing this file registers the task.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import (
  action_rate_l2,
  bad_orientation,
  dr,
  last_action,
  reset_joints_by_offset,
  reset_root_state_uniform,
  time_out,
)
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TASK_ID = "BeamWalker-DriveStraight"

# ---- where the model is -----------------------------------------------------------------------------------
# In the repo this file is train/walker_task.py and the model is models/walker.xml (one folder up).
# On the cluster (runIt.py --with models --with train) it is <job>/train/walker_task.py and <job>/models/.
_HERE = Path(__file__).resolve().parent
_CANDIDATES = (_HERE.parent / "models" / "walker.xml", _HERE / "models" / "walker.xml")
MODEL_XML = next((p for p in _CANDIDATES if p.exists()), None)
if MODEL_XML is None:
  raise FileNotFoundError("walker.xml not found; looked in: " + ", ".join(str(p) for p in _CANDIDATES))

# ---- constants -----------------------------------------------------------------------------------------------
JOINTS = ("hinge_left", "hinge_right")          # order of actions and leg observations: LEFT, RIGHT
ACTUATORS = ("motor_left", "motor_right")       # same order
FEET = ("foot_left", "foot_right")
MAX_SPEED = 17.42      # rad/s, measured no-load speed at 100% (= walker.xml ctrlrange)
MAX_CMD_FRAC = 0.6     # policy commands are capped at 60% of MAX_SPEED: action 1.0 = 60% = 10.45 rad/s.
                       # The sim matches the real robot at 25-50% but goes 2x too far at 100%
                       # (docs/sim_vs_real.md), so keep the policy in the range that was checked.
V_CAP = 0.08           # m/s; forward speed above this earns nothing more. Sim speeds (10 s runs):
                       # 25% 0.054, 50% 0.064, 75% 0.095, 100% 0.116 m/s, so the cap sits near 60-65%
                       # throttle and leaves headroom to steer by speeding one side up.
SIM_DT = 0.002         # s, same as walker.xml
DECIMATION = 31        # policy runs every 31 physics steps = 62 ms = 16.1 Hz. Measured on the real robot
                       # (scripts/imu_rate_check.py, 2026-11-16): a confirmed motor command over BLE takes a
                       # median 60 ms round trip, so a blocking policy loop runs at 16 Hz.
ACTION_DELAY_STEPS = DECIMATION  # command delay sampled from 0..1 control step (in physics steps), redrawn
                                 # every control step: round trips vary 60-118 ms
EPISODE_S = 10.0
TILT_LIMIT_DEG = 70.0  # flipped; normal walking reaches about 35-41 deg in sim_check.py
START_Z = 0.010        # m; a leg turned +10 deg reaches ~7 mm below the chassis bottom, so start higher and drop

# Domain randomization, sampled every episode, each side independently.
FOOT_FRICTION = (0.5, 1.1)
SPEED_GAIN = (0.85, 1.15)     # motor speed = gain * command
TORQUE_SCALE = (0.80, 1.20)   # x the force limit in walker.xml (0.032 N*m, fitted 2026-11-16)
MASS_SCALE = (0.85, 1.15)     # chassis mass (and inertia, consistently)
LEG_START_DEG = 10.0

_ROBOT = SceneEntityCfg("robot")
_LEGS = SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True)


# ---- entity --------------------------------------------------------------------------------------------------


def _get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MODEL_XML))
  # mjlab adds its own ground plane (the terrain), so drop the walker's floor and lights.
  spec.delete(spec.geom("floor"))
  for light in list(spec.worldbody.lights):
    spec.delete(light)
  # walker.xml's floor has ~0 friction so the foot's value decides every contact (MuJoCo takes the max of the
  # pair). mjlab's plane has friction 1.0, which would win. A higher priority makes MuJoCo use the robot geom's
  # friction instead, so foot-friction randomization means what it says.
  for g in spec.geoms:
    g.priority = 1
  # <option> is not carried over by MjSpec.attach(); the same settings are set in MujocoCfg below.
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
  return spec


def _robot_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, START_Z), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}
    ),
    # Use the velocity actuators already in walker.xml (kv 0.05, force limit, ctrl +-17.42 rad/s).
    # Each command reaches the motor 0 to 1 control steps late (uniform, redrawn every control step).
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=JOINTS,
          delay_min_lag=0,
          delay_max_lag=ACTION_DELAY_STEPS,
          delay_update_period=DECIMATION,
          delay_per_env_phase=False,   # keep the redraw aligned with the policy's control steps
        ),
      ),
    ),
  )


# ---- helpers -------------------------------------------------------------------------------------------------


def _yaw(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Chassis yaw = heading error from the starting direction (+x), wrapped to [-pi, pi]."""
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World-frame (x, y) velocity = (along, across) the starting heading."""
  return env.scene["robot"].data.root_link_lin_vel_w[:, :2]


# ---- observations (real-robot sensors only) ----------------------------------------------------------------


def leg_angle_sin_cos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _LEGS) -> torch.Tensor:
  """[sin L, sin R, cos L, cos R] of the leg angles (0 = leg straight forward). Shape [N, 4]."""
  q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


def leg_vel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _LEGS) -> torch.Tensor:
  """Leg speeds [L, R] in rad/s. Scaled by 1/MAX_SPEED in the term config."""
  return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def gyro_z(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Yaw rate in the chassis frame (rad/s), what the IMU gyro measures. Shape [N, 1]."""
  return env.scene["robot"].data.root_link_ang_vel_b[:, 2:3]


def heading_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  """IMU yaw relative to the start (rad). Shape [N, 1]."""
  return _yaw(env).unsqueeze(-1)


def world_vel_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged (critic only): true velocity along / across the start heading."""
  return _vel_xy(env)


# ---- rewards (each is logged separately as Episode_Reward/<name>) -----------------------------------------


def forward_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Velocity along the starting heading / V_CAP, capped at 1 (backwards is negative)."""
  return torch.clamp(_vel_xy(env)[:, 0], max=V_CAP) / V_CAP


def sideways_vel_sq(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.square(_vel_xy(env)[:, 1] / V_CAP)


def heading_error_sq(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.square(_yaw(env))


def yaw_rate_sq(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.square(env.scene["robot"].data.root_link_ang_vel_b[:, 2])


# ---- domain randomization: motors -------------------------------------------------------------------------


def _uniform(lo: float, hi: float, n: int, device) -> torch.Tensor:
  return lo + (hi - lo) * torch.rand(n, device=device)


@requires_model_fields("actuator_gainprm", "actuator_forcerange")
def randomize_motors(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  speed_gain: tuple[tuple[float, float], tuple[float, float]],
  torque_scale: tuple[tuple[float, float], tuple[float, float]],
) -> None:
  """Per-side motor speed gain and torque limit, as (left_range, right_range).

  A MuJoCo <velocity> actuator pushes with kv * (ctrl - qvel): gainprm[0] = kv multiplies ctrl and biasprm[2] = -kv
  multiplies qvel. Scaling only gainprm[0] by g gives kv * (g*ctrl - qvel), i.e. the motor settles at g x the
  commanded speed. Scaling forcerange scales the torque limit. (mjlab's dr.effort_limits / dr.pd_gains do not
  accept XML velocity actuators, hence this function.)
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  env_ids = env_ids.to(env.device, dtype=torch.long)
  n = len(env_ids)
  model = env.sim.model
  gain0 = env.sim.get_default_field("actuator_gainprm")
  force0 = env.sim.get_default_field("actuator_forcerange")
  for side, name in enumerate(ACTUATORS):
    aid = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot/{name}")
    if aid < 0:
      raise ValueError(f"actuator robot/{name} not found")
    g = _uniform(*speed_gain[side], n, env.device)
    t = _uniform(*torque_scale[side], n, env.device)
    model.actuator_gainprm[env_ids, aid, 0] = gain0[aid, 0] * g
    model.actuator_forcerange[env_ids, aid, 0] = force0[aid, 0] * t
    model.actuator_forcerange[env_ids, aid, 1] = force0[aid, 1] * t


def _mass_alpha(scale: tuple[float, float]) -> tuple[float, float]:
  # dr.pseudo_inertia scales mass and inertia by exp(2 * alpha).
  return (0.5 * math.log(scale[0]), 0.5 * math.log(scale[1]))


def make_events(
  foot_friction: dict[str, tuple[float, float]] | None = None,
  speed_gain=(SPEED_GAIN, SPEED_GAIN),
  torque_scale=(TORQUE_SCALE, TORQUE_SCALE),
  mass_scale=MASS_SCALE,
  leg_start_deg: float = LEG_START_DEG,
) -> dict[str, EventTermCfg]:
  """All reset events. The defaults are the training ranges; pass degenerate (v, v) ranges for a fixed robot."""
  if foot_friction is None:
    foot_friction = {FEET[0]: FOOT_FRICTION, FEET[1]: FOOT_FRICTION}
  leg = math.radians(leg_start_deg)
  return {
    "reset_root": EventTermCfg(
      func=reset_root_state_uniform, mode="reset", params={"pose_range": {}, "velocity_range": {}}
    ),
    "reset_legs": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={"position_range": (-leg, leg), "velocity_range": (0.0, 0.0), "asset_cfg": _LEGS},
    ),
    # Ranges keyed by geom name: each foot gets its own draw.
    "foot_friction": EventTermCfg(
      func=dr.geom_friction, mode="reset", params={"ranges": foot_friction, "operation": "abs", "axes": [0]}
    ),
    "motors": EventTermCfg(
      func=randomize_motors, mode="reset", params={"speed_gain": speed_gain, "torque_scale": torque_scale}
    ),
    "chassis_mass": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="reset",
      params={"alpha_range": _mass_alpha(mass_scale), "asset_cfg": SceneEntityCfg("robot", body_names=("chassis",))},
    ),
  }


# ---- environment -------------------------------------------------------------------------------------------


def walker_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "leg_angle": ObservationTermCfg(func=leg_angle_sin_cos),
    "leg_vel": ObservationTermCfg(
      func=leg_vel, scale=1.0 / MAX_SPEED, noise=UniformNoiseCfg(n_min=-0.3, n_max=0.3)  # noise in rad/s
    ),
    "last_action": ObservationTermCfg(func=last_action),
    "gyro_z": ObservationTermCfg(func=gyro_z, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
    "heading_error": ObservationTermCfg(func=heading_error, noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02)),
  }
  observations = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=not play),
    # The critic is only used in training, so it may also see the true velocity.
    "critic": ObservationGroupCfg({**actor_terms, "world_vel_xy": ObservationTermCfg(func=world_vel_xy)}),
  }

  actions: dict[str, ActionTermCfg] = {
    "legs": JointVelocityActionCfg(
      entity_name="robot", actuator_names=JOINTS, scale=MAX_CMD_FRAC * MAX_SPEED, use_default_offset=True,
      preserve_order=True,
    ),
  }

  rewards = {
    "forward_vel": RewardTermCfg(func=forward_vel, weight=1.0),
    "sideways_vel": RewardTermCfg(func=sideways_vel_sq, weight=-0.5),
    "heading_error": RewardTermCfg(func=heading_error_sq, weight=-1.0),
    "yaw_rate": RewardTermCfg(func=yaw_rate_sq, weight=-0.02),
    # -0.01 left the commands jumping between -1 and +1 every second (60-iteration CPU test); -0.1 makes
    # smoothness worth about as much as the sideways term.
    "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.1),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    "flipped": TerminationTermCfg(func=bad_orientation, params={"limit_angle": math.radians(TILT_LIMIT_DEG)}),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      num_envs=1,
      env_spacing=1.0,
    ),
    observations=observations,
    actions=actions,
    events=make_events(),
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="chassis",
      distance=0.45,
      elevation=-25.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=SIM_DT, integrator="implicitfast")),
    decimation=DECIMATION,
    episode_length_s=1e10 if play else EPISODE_S,
  )


def walker_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(64, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(64, 64), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="beam_walker",
    logger="tensorboard",    # the default, wandb, needs an account
    upload_model=False,
    clip_actions=1.0,        # action 1.0 = MAX_CMD_FRAC (60%) of MAX_SPEED
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=300,
  )


register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=walker_env_cfg(),
  play_env_cfg=walker_env_cfg(play=True),
  rl_cfg=walker_ppo_runner_cfg(),
)
