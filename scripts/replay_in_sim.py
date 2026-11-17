"""Replay a real-robot run in the sim: same motor commands, same timing, rendered from behind like a phone video.

    .venv/bin/python scripts/replay_in_sim.py data/policy_run_20261116-194157.csv
    .venv/bin/python scripts/replay_in_sim.py data/policy_run_<time>.csv --follow-real-heading --azimuth 35 --fixed-camera

Reads a log from scripts/run_policy.py (policy or --baseline). Each logged command (cmd_pct_left/right, signed
percent, + = forward) is applied to models/walker.xml when the hub would have received it: the time it was sent
(t_s) plus half its measured round trip, and is held until the next one. After the last step the motors brake,
as run_policy.py does. The legs start at the logged starting angles.

Two kinds of replay:
  default                pure physics. On 2026-11-16 this did NOT match the real robot: the sim turns far more for
                         the same left/right speed difference (one run: sim -167 deg vs real -7 deg). No friction
                         setting matches both the real heading and the real distance (docs/sim_vs_real.md).
  --follow-real-heading  a yaw torque on the chassis makes the sim heading follow the logged IMU heading; the legs
                         and the floor still produce all the forward motion. Shows the real path with sim physics.
                         Label videos made this way as such: the heading is taken from the real robot.

--foot-friction / --body-friction override walker.xml for this replay only (walker.xml itself is frozen).

Video: frame 0 = motor start (the first command reaching the hub), so it lines up with the moment the legs start
moving in a phone video. The camera sits behind and above the robot and follows it smoothly; --azimuth turns the
view (0 = looking straight along the starting heading). --floor tiles (default) draws 10 cm high-contrast tiles
so the motion across the floor is easy to see; --floor checker / plain for walker.xml's checker or flat grey.
Portrait 540x960 by default, like a phone held upright.
"""

import argparse
import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "walker.xml"
BODY_GEOMS = ("chassis_box", "hub_left", "hub_right", "beam_left", "beam_right")


def load_log(path):
    rows = list(csv.DictReader(open(path)))
    if not rows or "cmd_pct_left" not in rows[0]:
        raise SystemExit(f"{path}: not a run_policy.py log (no cmd_pct_left column)")
    t_send = np.array([float(r["t_s"]) for r in rows])
    rt = np.array([float(r.get("round_trip_ms") or 0) / 1000 for r in rows])
    cmd = np.array([[float(r["cmd_pct_left"]), float(r["cmd_pct_right"])] for r in rows])
    heading = np.degrees(np.unwrap(np.radians([float(r["heading_deg"]) for r in rows])))
    return rows, t_send, rt, cmd, heading


def yaw_rad(data):
    w, x, y, z = data.sensor("chassis_quat").data
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


FLOORS = ("tiles", "checker", "plain")


def build_model(foot_friction, body_friction, size, floor="checker"):
    """walker.xml with replay-only overrides. floor: "checker" (walker.xml's own), "tiles" (high-contrast 10 cm
    tiles with white grout lines, easy to see the robot cross), "plain" (flat grey)."""
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    if floor == "tiles":
        tex, mat = spec.texture("grid"), spec.material("floor_mat")
        tex.rgb1, tex.rgb2 = [0.78, 0.76, 0.72], [0.30, 0.31, 0.34]   # light / dark tiles
        tex.mark, tex.markrgb = mujoco.mjtMark.mjMARK_EDGE, [1.0, 1.0, 1.0]
        tex.width = tex.height = 64                                    # 2x2 cells -> edge line ~1/32 of a tile
        mat.texrepeat = [5, 5]                                         # texture spans 20 cm -> 10 cm tiles
    elif floor == "plain":
        spec.geom("floor").material = ""
        spec.geom("floor").rgba = [0.42, 0.42, 0.42, 1.0]
    model = spec.compile()
    if foot_friction is not None:
        for g, mu in zip(("foot_left", "foot_right"), foot_friction):
            model.geom(g).friction[0] = mu
    if body_friction is not None:
        for g in BODY_GEOMS:
            model.geom(g).friction[0] = body_friction
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, size[0])
    model.vis.global_.offheight = max(model.vis.global_.offheight, size[1])
    return model


def aim_camera(cam, cam_pos, target):
    """Point a free camera standing at cam_pos at target (keeps the camera where it is, like a phone panning)."""
    v = np.asarray(target) - cam_pos
    d = float(np.linalg.norm(v))
    cam.lookat[:] = target
    cam.distance = d
    cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
    cam.elevation = math.degrees(math.asin(v[2] / d))


def replay(log, model, follow_heading=False, fps=30.0, tail=0.5, size=(540, 960), distance=1.0,
           elevation=-50.0, azimuth=0.0, render=True, fixed_camera=False):
    """Run the replay. Returns a summary dict and the rendered frames (list of HxWx3 arrays)."""
    rows, t_send, rt, cmd, real_heading = log
    t_apply = t_send + rt / 2
    period = float(np.median(np.diff(t_send))) if len(t_send) > 1 else 0.062
    t_stop = t_send[-1] + period + float(np.median(rt)) / 2     # run_policy's blocking stop after the last step
    t0, t_end = t_apply[0], t_stop + tail                        # motor start = video frame 0

    data = mujoco.MjData(model)
    act = [model.actuator(n).id for n in ("motor_left", "motor_right")]
    max_speed = model.actuator_ctrlrange[act[0], 1]
    hinges = [model.joint(n) for n in ("hinge_left", "hinge_right")]
    for j, key in zip(hinges, ("leg_angle_left_deg", "leg_angle_right_deg")):
        data.qpos[j.qposadr[0]] = math.radians(float(rows[0][key]))
    mujoco.mj_forward(model, data)
    start_xy = data.sensor("chassis_pos").data[:2].copy()
    start_yaw = yaw_rad(data)
    chassis = model.body("chassis").id

    # Heading follower: a PD yaw torque on the chassis toward the logged IMU heading (rad, unwrapped).
    # Gains are ~10 Hz for the chassis' tiny yaw inertia (~7e-5 kg m^2).
    kp, kd = 0.3, 0.006
    target = np.radians(real_heading)

    renderer = cam = None
    frames = []
    if render:
        renderer = mujoco.Renderer(model, height=size[1], width=size[0])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        cam.lookat[:] = data.xpos[chassis]
        # Where that camera stands; with fixed_camera it stays there and only turns to follow the robot.
        az, el = math.radians(azimuth), math.radians(elevation)
        view_dir = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        cam_pos = data.xpos[chassis] - distance * view_dir
    smooth = data.xpos[chassis].copy()
    tau = 0.3                                                    # s; camera follow smoothing

    i = -1
    dt = model.opt.timestep
    next_frame = t0
    t = 0.0
    sim_heading, prev_yaw, unwrapped = [], start_yaw, 0.0
    k = 0
    while t < t_end:
        while i + 1 < len(t_apply) and t >= t_apply[i + 1]:
            i += 1
        if i >= 0 and t < t_stop:
            data.ctrl[act] = cmd[i] / 100.0 * max_speed
        else:
            data.ctrl[act] = 0.0                                 # before the first command / after the stop: brake
        yaw = yaw_rad(data)
        unwrapped += (yaw - prev_yaw + math.pi) % (2 * math.pi) - math.pi
        prev_yaw = yaw
        if follow_heading:
            goal = float(np.interp(t, t_send, target))
            data.xfrc_applied[chassis, 5] = kp * (goal - unwrapped) - kd * data.qvel[5]
        mujoco.mj_step(model, data)
        t += dt
        while k < len(t_send) and t >= t_send[k]:
            sim_heading.append(math.degrees(unwrapped))
            k += 1
        smooth += (data.xpos[chassis] - smooth) * min(1.0, dt / tau)
        if renderer is not None and t >= next_frame:
            if fixed_camera:
                aim_camera(cam, cam_pos, smooth)
            else:
                cam.lookat[:] = smooth
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
            next_frame += 1.0 / fps
    if renderer is not None:
        renderer.close()

    sim_heading = np.array(sim_heading[:len(real_heading)])
    dxy = 100 * (data.sensor("chassis_pos").data[:2] - start_xy)
    return {
        "forward_cm": float(dxy[0]), "sideways_cm": float(dxy[1]),
        "final_heading_deg": float(sim_heading[-1]), "real_final_heading_deg": float(real_heading[-1]),
        "heading_rms_deg": float(np.sqrt(np.mean((sim_heading - real_heading[:len(sim_heading)]) ** 2))),
        "sim_revs": [math.degrees(data.qpos[j.qposadr[0]]) / 360 for j in hinges],
        "real_revs": [float(rows[-1][k]) / 360 for k in ("leg_angle_left_deg", "leg_angle_right_deg")],
        "duration_s": t_send[-1] + period,
    }, frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="data/policy_run_<time>.csv")
    ap.add_argument("--out", type=Path, default=None, help="mp4 path (default videos/replay_<log name>.mp4)")
    ap.add_argument("--follow-real-heading", action="store_true",
                    help="make the sim heading follow the logged IMU heading (yaw torque on the chassis)")
    ap.add_argument("--foot-friction", type=float, nargs=2, metavar=("LEFT", "RIGHT"), help="override walker.xml")
    ap.add_argument("--body-friction", type=float, help="override walker.xml (chassis, hubs, beams)")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--width", type=int, default=540)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--tail", type=float, default=0.5, help="seconds to keep rendering after the motors stop")
    ap.add_argument("--distance", type=float, default=1.0, help="camera distance from the robot (m)")
    ap.add_argument("--elevation", type=float, default=-50.0, help="camera pitch, degrees (negative = looking down)")
    ap.add_argument("--azimuth", type=float, default=0.0, help="camera azimuth, degrees (0 = along the start heading)")
    ap.add_argument("--floor", choices=FLOORS, default="tiles",
                    help="tiles: 10 cm high-contrast tiles (default); checker: walker.xml's; plain: flat grey")
    ap.add_argument("--fixed-camera", action="store_true",
                    help="camera stays where it starts and only turns to follow (like a hand-held phone)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    log = load_log(args.log)
    size = (args.width, args.height)
    model = build_model(args.foot_friction, args.body_friction, size, args.floor)
    s, frames = replay(log, model, args.follow_real_heading, args.fps, args.tail, size, args.distance,
                       args.elevation, args.azimuth, render=not args.no_video,
                       fixed_camera=args.fixed_camera)

    kind = "heading follows the real IMU" if args.follow_real_heading else "pure physics"
    print(f"log:            {args.log}  ({log[0][0].get('mode', '?')}, {len(log[0])} commands, {s['duration_s']:.1f} s)")
    print(f"replay:         {kind}; foot friction {args.foot_friction or 'walker.xml'}, "
          f"body friction {args.body_friction or 'walker.xml'}")
    print(f"sim:            forward {s['forward_cm']:+.1f} cm, sideways {s['sideways_cm']:+.1f} cm (+ = left), "
          f"final heading {s['final_heading_deg']:+.1f} deg")
    print(f"vs real IMU:    final heading {s['real_final_heading_deg']:+.1f} deg; RMS heading error over the run "
          f"{s['heading_rms_deg']:.1f} deg")
    print(f"leg revs L/R:   sim {s['sim_revs'][0]:.2f} / {s['sim_revs'][1]:.2f}   "
          f"real {s['real_revs'][0]:.2f} / {s['real_revs'][1]:.2f} (sim includes the run to the stop)")
    if frames:
        out = (args.out or ROOT / "videos" / f"replay_{args.log.stem}.mp4").expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
        print(f"video:          {out} ({len(frames)} frames, {len(frames) / args.fps:.1f} s from motor start)")


if __name__ == "__main__":
    main()
