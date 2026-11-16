"""Headless sanity check for models/walker.xml.

Drives both motors at the same speed for 5 s, prints forward distance, sideways
drift and final yaw, then renders the tracking camera to videos/sim_check.mp4.

Run from the repo root:
    .venv/bin/python scripts/sim_check.py
    .venv/bin/python scripts/sim_check.py --speed-frac 0.8 --duration 10

On macOS, mujoco.Renderer uses the native CGL backend, so no MUJOCO_GL setting
or display is needed.
"""

import argparse
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "walker.xml"
VIDEO_PATH = ROOT / "videos" / "sim_check.mp4"
VIDEO_WARN_BYTES = 50 * 1024 * 1024


def quat_to_yaw_deg(q):
    w, x, y, z = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def chassis_up_z(q):
    """World-Z component of the chassis +Z axis (1 = upright, <0 = upside down)."""
    w, x, y, z = q
    return 1 - 2 * (x * x + y * y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=5.0, help="seconds")
    ap.add_argument("--speed-frac", type=float, default=0.5, help="fraction of max motor speed")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--foot-friction", type=float, default=None,
                    help="override the feet's sliding friction (for tuning sweeps)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if args.foot_friction is not None:
        for name in ("foot_left", "foot_right"):
            model.geom(name).friction[0] = args.foot_friction
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    act_l = model.actuator("motor_left").id
    act_r = model.actuator("motor_right").id
    max_speed = model.actuator_ctrlrange[act_l, 1]
    cmd = args.speed_frac * max_speed
    mismatch = float(model.numeric("speed_mismatch_right").data[0])
    cmd_r = cmd * (1.0 + mismatch)

    start_pos = data.sensor("chassis_pos").data.copy()
    start_yaw = quat_to_yaw_deg(data.sensor("chassis_quat").data)

    renderer = None
    frames = []
    if not args.no_video:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    steps_per_frame = max(1, round(1.0 / (args.fps * model.opt.timestep)))

    n_steps = int(round(args.duration / model.opt.timestep))
    min_up = 1.0
    max_tilt_deg = 0.0
    for i in range(n_steps):
        data.ctrl[act_l] = cmd
        data.ctrl[act_r] = cmd_r
        mujoco.mj_step(model, data)

        up = chassis_up_z(data.sensor("chassis_quat").data)
        min_up = min(min_up, up)
        max_tilt_deg = max(max_tilt_deg, math.degrees(math.acos(np.clip(up, -1, 1))))

        if renderer is not None and i % steps_per_frame == 0:
            renderer.update_scene(data, camera="track")
            frames.append(renderer.render())

    pos = data.sensor("chassis_pos").data
    quat = data.sensor("chassis_quat").data
    dx, dy = pos[0] - start_pos[0], pos[1] - start_pos[1]
    yaw = quat_to_yaw_deg(quat) - start_yaw
    yaw = (yaw + 180) % 360 - 180

    qvel_l = data.sensor("hinge_left_vel").data[0]
    qvel_r = data.sensor("hinge_right_vel").data[0]
    revs_l = data.sensor("hinge_left_pos").data[0] / (2 * math.pi)
    revs_r = data.sensor("hinge_right_pos").data[0] / (2 * math.pi)

    print(f"model:           {MODEL_PATH.relative_to(ROOT)}")
    print(f"command:         {cmd:.2f} rad/s on both motors ({args.speed_frac:.0%} of {max_speed:.1f})")
    print(f"R/L mismatch:    {mismatch:+.1%} (right cmd {cmd_r:.2f})")
    print(f"foot friction:   {model.geom('foot_left').friction[0]:.3f}")
    print(f"duration:        {args.duration:.1f} s")
    print(f"forward (x):     {dx * 100:+.2f} cm")
    print(f"sideways (y):    {dy * 100:+.2f} cm")
    print(f"final yaw:       {yaw:+.1f} deg")
    print(f"final height z:  {pos[2] * 1000:.1f} mm")
    print(f"max tilt:        {max_tilt_deg:.1f} deg")
    print(f"leg speed L/R:   {qvel_l:.2f} / {qvel_r:.2f} rad/s (cmd {cmd:.2f})")
    print(f"leg revs L/R:    {revs_l:.2f} / {revs_r:.2f}")

    if min_up < 0:
        verdict = "FLIPPED (chassis went upside down)"
    elif abs(dx) < 0.01 and abs(yaw) > 30:
        verdict = "SPINNING IN PLACE"
    elif dx > 0.01:
        verdict = "MOVES FORWARD"
    elif dx < -0.01:
        verdict = "MOVES BACKWARD"
    else:
        verdict = "BARELY MOVES"
    print(f"verdict:         {verdict}")

    if renderer is not None:
        VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(VIDEO_PATH, frames, fps=args.fps, quality=8, macro_block_size=1)
        renderer.close()
        size = VIDEO_PATH.stat().st_size
        print(f"video:           {VIDEO_PATH.relative_to(ROOT)} ({len(frames)} frames, {size / 1e6:.1f} MB)")
        if size > VIDEO_WARN_BYTES:
            print("WARNING: video is over 50 MB; add it to .gitignore before committing.")


if __name__ == "__main__":
    main()
