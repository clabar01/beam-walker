"""Train the beam walker to walk straight (mjlab PPO), then export the policy and make the writeup figures.

On the Tufts HPC, from the beam-walker repo root:
    python3 "$HOME/Documents/ME193-Robotics/Public stuff/mujoco-hpc/runIt.py" --gpu --time 01:00:00 --mem 16G \
        --cpus 4 train/train_walker.py --with models --with train

Tiny local test (CPU, Python 3.11-3.13 venv with mjlab 1.6.0):
    .venv-mjlab/bin/python train/train_walker.py --num-envs 8 --iterations 3 --video-seconds 2

Everything is written to --out: the Slurm job folder on the cluster (so runIt.py brings it home), or
./train_out/<time>/ locally.
    walker_final.pt          final training checkpoint (rsl_rl format; resume / reload with mjlab)
    walker_policy.pt         TorchScript policy for the laptop + real robot: actions = policy(obs[1, 10])
    walker_policy.onnx       the same policy as ONNX
    policy_io.json           observation order, scaling and action meaning, needed to run the policy for real
    reward_curves.png/.csv   mean reward and every reward term per iteration
    rollout_policy.mp4       10 s with the trained policy, on a deliberately lopsided robot
    rollout_baseline.mp4     the same robot with equal constant motor speeds
    rollout_summary.txt      distance / drift / heading for both rollouts
    logs/                    mjlab / rsl_rl run folder (tensorboard events, model_<n>.pt, params)

--checkpoint some/model_299.pt --iterations 0 skips training and redoes the export, plots and videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The task module is next to this file in the repo (train/), or in train/ inside the cluster job folder.
for _p in (HERE, HERE / "train"):
  if (_p / "walker_task.py").exists():
    sys.path.insert(0, str(_p))
    break

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import walker_task as wt  # noqa: E402  (registers the task)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.scripts.train import TrainConfig, run_train  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402


def parse_args():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--num-envs", type=int, default=2048)
  ap.add_argument("--iterations", type=int, default=300, help="PPO iterations (0 = skip training, needs --checkpoint)")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out", type=Path, default=None, help="output folder (default: Slurm job folder, else ./train_out/<time>)")
  ap.add_argument("--checkpoint", type=Path, default=None, help="use this checkpoint instead of training")
  ap.add_argument("--video-seconds", type=float, default=10.0)
  ap.add_argument("--no-video", action="store_true")
  ap.add_argument("--baseline-frac", type=float, default=0.5, help="baseline: both motors at this fraction of max speed")
  # The robot used for BOTH rollout videos: fixed, lopsided values inside the training ranges, so the baseline
  # curves and the policy has something to correct.
  ap.add_argument("--eval-friction", type=float, nargs=2, default=(0.6, 1.0), metavar=("LEFT", "RIGHT"))
  ap.add_argument("--eval-speed-gain", type=float, nargs=2, default=(1.0, 0.9), metavar=("LEFT", "RIGHT"))
  ap.add_argument("--eval-torque-scale", type=float, nargs=2, default=(1.0, 1.0), metavar=("LEFT", "RIGHT"))
  return ap.parse_args()


def say(msg):
  print(msg, flush=True)


# ---- training ------------------------------------------------------------------------------------------------


def train(args, out: Path) -> Path:
  cfg = TrainConfig.from_task(wt.TASK_ID)
  cfg.env.scene.num_envs = args.num_envs
  cfg.agent.max_iterations = args.iterations
  cfg.agent.seed = args.seed
  log_dir = out / "logs" / "rsl_rl" / cfg.agent.experiment_name / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  say(f"[train] {args.num_envs} envs x {args.iterations} iterations -> {log_dir}")
  t0 = time.time()
  run_train(wt.TASK_ID, cfg, log_dir)   # mjlab's own training loop (MjlabOnPolicyRunner, PPO)
  say(f"[train] done in {time.time() - t0:.0f} s")
  ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
  if not ckpts:
    raise SystemExit(f"no checkpoint written in {log_dir}")
  return ckpts[-1]


# ---- reward curves -------------------------------------------------------------------------------------------


def plot_rewards(log_dir: Path, out: Path):
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  acc.Reload()
  tags = acc.Tags().get("scalars", [])
  wanted = ["Train/mean_reward", "Train/mean_episode_length"] + sorted(t for t in tags if t.startswith("Episode_Reward/"))
  series = {t: [(e.step, e.value) for e in acc.Scalars(t)] for t in wanted if t in tags}
  if not series:
    say(f"[plot] no scalars found in {log_dir} (tags: {tags[:10]})")
    return

  steps = sorted({s for v in series.values() for s, _ in v})
  with open(out / "reward_curves.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["iteration", *series])
    lookup = {t: dict(v) for t, v in series.items()}
    for s in steps:
      w.writerow([s, *(lookup[t].get(s, "") for t in series)])

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  terms = [t for t in series if t.startswith("Episode_Reward/")]
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
  ax = axes[0]
  if "Train/mean_reward" in series:
    s, v = zip(*series["Train/mean_reward"])
    ax.plot(s, v, color="black", lw=2)
  ax.set_title("Mean episode return")
  ax.set_xlabel("PPO iteration")
  ax.grid(alpha=0.3)
  ax = axes[1]
  for t in terms:
    s, v = zip(*series[t])
    ax.plot(s, v, label=t.split("/", 1)[1])
  ax.axhline(0, color="gray", lw=0.8)
  ax.set_title("Reward terms (weighted, per second of episode)")
  ax.set_xlabel("PPO iteration")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)
  fig.suptitle("Beam walker: drive straight (PPO, mjlab)")
  fig.tight_layout()
  fig.savefig(out / "reward_curves.png", dpi=130)
  plt.close(fig)
  last = {t.split("/", 1)[-1]: round(series[t][-1][1], 4) for t in series}
  say(f"[plot] reward_curves.png; last values: {last}")


# ---- export --------------------------------------------------------------------------------------------------


def export_policy(ckpt: Path, out: Path, device: str):
  env_cfg = load_env_cfg(wt.TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(wt.TASK_ID)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=agent_cfg.clip_actions)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=device)
  runner.export_policy_to_jit(str(out), "walker_policy.pt")
  runner.export_policy_to_onnx(str(out), "walker_policy.onnx")

  # Check the TorchScript file gives the same actions as the policy inside the runner.
  obs = env.get_observations()
  policy = runner.get_inference_policy(device=device)
  jit = torch.jit.load(str(out / "walker_policy.pt"), map_location="cpu")
  with torch.no_grad():
    a_runner = policy(obs).cpu()
    a_jit = jit(obs["actor"].cpu())
  diff = (a_runner - a_jit).abs().max().item()
  say(f"[export] walker_policy.pt + walker_policy.onnx; TorchScript vs runner max diff {diff:.2e}")
  if diff > 1e-4:
    raise SystemExit("exported policy does not match the trained one")

  terms = env.unwrapped.observation_manager.active_terms["actor"]
  dims = env.unwrapped.observation_manager.group_obs_term_dim["actor"]
  layout, i = [], 0
  for name, d in zip(terms, dims):
    n = int(np.prod(d))
    layout.append({"name": name, "index": [i, i + n]})
    i += n
  io = {
    "task": wt.TASK_ID,
    "policy_rate_hz": 1.0 / (wt.SIM_DT * wt.DECIMATION),
    "obs_dim": i,
    "obs_layout": layout,
    "obs_meaning": {
      "leg_angle": "[sin L, sin R, cos L, cos R] of the leg angle in rad; 0 = leg straight forward, + = rolled forward",
      "leg_vel": "[L, R] leg speed in rad/s divided by max_speed_rad_s (+ = rolling forward)",
      "last_action": "the previous policy output [L, R] (after clipping to [-1, 1]); zeros at the start",
      "gyro_z": "yaw rate in rad/s, chassis frame, + = turning left (counterclockwise seen from above)",
      "heading_error": "yaw in rad since the start (dm.imu_reset_yaw_axis() at the start), + = turned left",
    },
    "action": "[L, R] in [-1, 1]; motor speed command = action * max_speed_rad_s (= action * 100 percent)",
    "max_speed_rad_s": wt.MAX_SPEED,
    "normalization": "inside the exported model (feed raw observations as above)",
  }
  (out / "policy_io.json").write_text(json.dumps(io, indent=2) + "\n")
  env.close()


# ---- rollouts ------------------------------------------------------------------------------------------------


def make_eval_env(args, device: str) -> ManagerBasedRlEnv:
  env_cfg = load_env_cfg(wt.TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  fl, fr = args.eval_friction
  gl, gr = args.eval_speed_gain
  tl, tr = args.eval_torque_scale
  env_cfg.events = wt.make_events(
    foot_friction={wt.FEET[0]: (fl, fl), wt.FEET[1]: (fr, fr)},
    speed_gain=((gl, gl), (gr, gr)),
    torque_scale=((tl, tl), (tr, tr)),
    mass_scale=(1.0, 1.0),
    leg_start_deg=0.0,
  )
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def rollout(env: ManagerBasedRlEnv, act_fn, seconds: float, video: Path | None):
  """Run one episode from reset; return per-second rows and write an mp4 (track camera, like sim_check.py)."""
  obs, _ = env.reset()
  robot = env.scene["robot"]
  origin = env.scene.env_origins[0, :2].cpu().numpy()
  steps = int(round(seconds / (wt.SIM_DT * wt.DECIMATION)))
  fps = int(round(1.0 / (wt.SIM_DT * wt.DECIMATION)))
  renderer = data = None
  frames = []
  if video is not None:
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
  rows, flipped = [], False
  last_action = torch.zeros(1, 2, device=env.device)
  with torch.no_grad():
    for k in range(steps):
      action = torch.clamp(act_fn(obs["actor"]), -1.0, 1.0).to(env.device)
      obs, _, terminated, _, _ = env.step(action)
      last_action = action
      if bool(terminated[0]):
        flipped = True
      if renderer is not None:
        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        mujoco.mj_forward(env.sim.mj_model, data)
        renderer.update_scene(data, camera="robot/track")
        frames.append(renderer.render())
      if (k + 1) % fps == 0 or k == steps - 1:
        x, y = (robot.data.root_link_pos_w[0, :2].cpu().numpy() - origin).tolist()
        yaw = math.degrees(wt._yaw(env)[0].item())
        rows.append({"t": (k + 1) / fps, "x_cm": 100 * x, "y_cm": 100 * y, "yaw_deg": yaw,
                     "cmd_L": last_action[0, 0].item(), "cmd_R": last_action[0, 1].item()})
  if renderer is not None:
    imageio.mimwrite(video, frames, fps=fps, quality=8, macro_block_size=1)
    renderer.close()
  return rows, flipped


def run_rollouts(args, out: Path, device: str):
  env = make_eval_env(args, device)
  jit = torch.jit.load(str(out / "walker_policy.pt"), map_location="cpu")
  base = args.baseline_frac
  runs = {
    "policy": lambda o: jit(o.cpu()),
    "baseline": lambda o: torch.full((o.shape[0], 2), base),
  }
  lines = [
    f"Eval robot (same for both): foot friction L/R {args.eval_friction}, speed gain L/R {args.eval_speed_gain}, "
    f"torque scale L/R {args.eval_torque_scale}, nominal mass, legs start at 0.",
    f"Baseline: both motors at {base:.0%} of {wt.MAX_SPEED} rad/s.",
    "",
  ]
  for name, fn in runs.items():
    video = None if args.no_video else out / f"rollout_{name}.mp4"
    rows, flipped = rollout(env, fn, args.video_seconds, video)
    lines.append(f"== {name}{'  (FLIPPED and reset during the run)' if flipped else ''}")
    lines.append("   t(s)   x(cm)   y(cm)  yaw(deg)  cmd L   cmd R")
    for r in rows:
      lines.append(f"  {r['t']:5.1f} {r['x_cm']:7.1f} {r['y_cm']:7.1f} {r['yaw_deg']:8.1f}  {r['cmd_L']:+.2f}   {r['cmd_R']:+.2f}")
    f = rows[-1]
    lines.append(f"   final: forward {f['x_cm']:.1f} cm, sideways {f['y_cm']:+.1f} cm, heading {f['yaw_deg']:+.1f} deg")
    lines.append("")
  env.close()
  text = "\n".join(lines)
  (out / "rollout_summary.txt").write_text(text + "\n")
  say(text)


# ---- main ----------------------------------------------------------------------------------------------------


def main():
  args = parse_args()
  if "SLURM_JOB_ID" in os.environ and args.out is None:
    out = Path(os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()))
  else:
    out = args.out or Path.cwd() / "train_out" / datetime.now().strftime("%Y%m%d-%H%M%S")
  out = out.resolve()
  out.mkdir(parents=True, exist_ok=True)

  # mjlab's run_train picks the device from CUDA_VISIBLE_DEVICES ("" = CPU).
  if torch.cuda.is_available():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = "cuda:0"
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"
  say(f"[setup] model {wt.MODEL_XML}\n[setup] output {out}\n[setup] device {device}, torch {torch.__version__}, "
      f"mujoco {mujoco.__version__}")

  if args.iterations > 0:
    ckpt = train(args, out)
  elif args.checkpoint is not None:
    ckpt = args.checkpoint.resolve()
  else:
    raise SystemExit("--iterations 0 needs --checkpoint")
  shutil.copy2(ckpt, out / "walker_final.pt")
  say(f"[ckpt] {ckpt.name} -> walker_final.pt")

  plot_rewards(ckpt.parent, out)
  export_policy(ckpt, out, device)
  run_rollouts(args, out, device)
  say(f"[done] outputs in {out}")


if __name__ == "__main__":
  main()
