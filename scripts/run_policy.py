"""Run a trained drive-straight policy on the real beam walker. **Moves the robot.**

    .venv/bin/python scripts/run_policy.py runs/<job>/walker_policy.onnx --home      # policy, 10 s
    .venv/bin/python scripts/run_policy.py --baseline --home                         # both motors at 50%, for comparison
    .venv/bin/python scripts/run_policy.py walker_policy.onnx --home --seconds 5

The policy file's policy_io.json (same folder, or --io) gives the observation order, the units and the 60%
command cap. The loop is the rate-limited one from imu_rate_check.py:
  read legs + IMU -> build the observation -> run the policy -> send both motor commands as ONE blocking batch
  (never a new command before the last is confirmed) -> wait for the next tick (policy rate, ~16 Hz).
A blocking stop is always sent at the end, on Ctrl-C, or on any error.

Before starting it zeroes the leg angles at the home calibration (0 = leg straight forward, like the sim),
resets the IMU yaw so the heading error starts at 0, and measures the gyro z bias while the robot is still.
Everything is logged to data/policy_run_<time>.csv (raw readings, observations, actions, commands, timing).

Conversions (docs/real_robot_io.md, measured 2026-11-16):
  leg angle  rad   = radians(forward_sign * motor.position)                 (degrees, zeroed at home)
  leg speed  rad/s = forward_sign * motor.speed / 100 * 17.42               (speed is percent of max)
  heading    rad   = wrap_to_pi(radians(imu.yaw / 10))                      (yaw is 0.1 deg; + = left)
  gyro z     rad/s = radians(imu.gyroscopeZ - bias)                         (deg/s; + = turning left)
  command    percent = round(100 * cap * |action|), cap = 0.6; direction = forward if action >= 0
"""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import legoeducation as le

import lego_common as lc
from imu_rate_check import blocking_stop, pct, wait_for_imu

HARD_CAP = 0.6           # never command more than 60%, whatever policy_io.json says
SIDES = (("left", le.MOTOR_LEFT, lc.LEFT_FORWARD), ("right", le.MOTOR_RIGHT, lc.RIGHT_FORWARD))


def wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def isnan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

class Policy:
    """Exported policy: walker_policy.onnx (onnxruntime) or walker_policy.pt (TorchScript, needs torch).
    Normalization is inside the model, so it takes the raw observation."""

    def __init__(self, path: Path):
        self.path = path
        if path.suffix == ".onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inp = self.sess.get_inputs()[0]
            self.input_name, self.obs_dim = inp.name, int(inp.shape[-1])
            self.kind = "onnx"
        elif path.suffix == ".pt":
            import torch
            self.torch = torch
            self.model = torch.jit.load(str(path), map_location="cpu").eval()
            self.obs_dim = None
            self.kind = "torchscript"
        else:
            raise SystemExit(f"unknown policy file type: {path} (want .onnx or .pt)")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)[None, :]
        if self.kind == "onnx":
            out = self.sess.run(None, {self.input_name: x})[0]
        else:
            with self.torch.no_grad():
                out = self.model(self.torch.from_numpy(x)).numpy()
        return np.asarray(out, dtype=np.float64).reshape(-1)[:2]


def load_io(policy_path, io_arg):
    path = Path(io_arg) if io_arg else (policy_path.parent / "policy_io.json" if policy_path else None)
    if path is None or not path.exists():
        raise SystemExit(f"policy_io.json not found ({path}); pass --io")
    io = json.loads(path.read_text())
    for key in ("obs_layout", "obs_dim", "max_speed_rad_s", "max_command_frac", "policy_rate_hz"):
        if key not in io:
            raise SystemExit(f"{path} has no '{key}'; re-export the policy with the current train_walker.py")
    return io, path


# ---------------------------------------------------------------------------
# Sensors -> observation, action -> motors
# ---------------------------------------------------------------------------

def read_sensors(dm, gyro_bias, max_speed):
    """Raw readings and the converted quantities the policy needs (left, right)."""
    s = {}
    for name, idx, fwd in SIDES:
        m = dm.motor[idx]
        sign = lc.forward_sign(fwd)
        s[f"raw_pos_{name}"], s[f"raw_speed_{name}"] = m.position, m.speed
        s[f"leg_angle_{name}"] = math.radians(sign * m.position)
        s[f"leg_vel_{name}"] = sign * m.speed / 100.0 * max_speed
    imu = dm.imu_device
    s["imu_obj"] = imu
    s["raw_yaw"], s["raw_gyro_z"] = imu.yaw, imu.gyroscopeZ
    s["heading"] = wrap_pi(math.radians(imu.yaw / 10.0))
    s["gyro_z"] = math.radians(imu.gyroscopeZ - gyro_bias)
    bad = [k for k in ("raw_pos_left", "raw_pos_right", "raw_speed_left", "raw_speed_right", "raw_yaw", "raw_gyro_z")
           if isnan(s[k])]
    if bad:
        raise RuntimeError(f"no valid reading for {bad} (notifications stopped?)")
    return s


def build_obs(s, last_action, io):
    ms = io["max_speed_rad_s"]
    pieces = {
        "leg_angle": [math.sin(s["leg_angle_left"]), math.sin(s["leg_angle_right"]),
                      math.cos(s["leg_angle_left"]), math.cos(s["leg_angle_right"])],
        "leg_vel": [s["leg_vel_left"] / ms, s["leg_vel_right"] / ms],
        "last_action": list(last_action),
        "gyro_z": [s["gyro_z"]],
        "heading_error": [s["heading"]],
    }
    obs = []
    for term in io["obs_layout"]:
        name = term["name"]
        if name not in pieces:
            raise SystemExit(f"policy_io.json asks for observation '{name}', which this script doesn't know")
        a, b = term["index"]
        if len(pieces[name]) != b - a:
            raise SystemExit(f"observation '{name}' should have {b - a} values, got {len(pieces[name])}")
        obs.extend(pieces[name])
    if len(obs) != io["obs_dim"]:
        raise SystemExit(f"built {len(obs)} observation values, policy_io.json says {io['obs_dim']}")
    return np.array(obs, dtype=np.float64)


def to_motor(action, cap):
    """Policy action (clipped to [-1, 1]) -> [(percent, direction)] for left, right. percent <= 100 * cap."""
    cmds = []
    for a, (_, _, fwd) in zip(action, SIDES):
        a = float(np.clip(a, -1.0, 1.0))
        percent = int(round(100 * cap * abs(a)))
        cmds.append((min(percent, int(round(100 * HARD_CAP))), fwd if a >= 0 else lc.opposite(fwd)))
    return cmds


def send(dm, cmds):
    """Both motors as ONE batch, BLOCKING: returns after the hub confirms."""
    with dm.batch(blocking=True):
        for (percent, direction), (_, idx, _) in zip(cmds, SIDES):
            dm.motor_run(direction=direction, motor=idx, speed=percent)


def measure_gyro_bias(dm, seconds=0.5):
    vals, last = [], None
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        imu = dm.imu_device
        if imu is not last and not isnan(imu.gyroscopeZ):
            last = imu
            vals.append(imu.gyroscopeZ)
        time.sleep(0.005)
    return statistics.mean(vals) if vals else 0.0, len(vals)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", type=Path, nargs="?", help="walker_policy.onnx (or .pt) from train_walker.py")
    ap.add_argument("--io", help="policy_io.json (default: next to the policy file)")
    ap.add_argument("--baseline", action="store_true", help="no policy: both motors at --baseline-pct")
    ap.add_argument("--baseline-pct", type=int, default=50)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--max-hz", type=float, default=None, help="loop cap (default: the policy rate, ~16 Hz)")
    ap.add_argument("--home", action="store_true", help="turn both legs to straight forward first (needs --calibrate-home "
                                                         "from real_walk_test.py)")
    ap.add_argument("--yes", action="store_true", help="skip the 'press Enter' safety prompt")
    args = ap.parse_args(argv)
    if not args.baseline and args.policy is None:
        ap.error("give a policy file, or use --baseline")
    if not 0 < args.baseline_pct <= 100 * HARD_CAP:
        ap.error(f"--baseline-pct must be 1..{100 * HARD_CAP:.0f}")
    return args


def main(argv=None, connect=None):
    args = parse_args(argv)
    connect = connect or lc.connect_double_motor

    # Baseline runs need no policy; if a policy/--io is given anyway, its observations are still logged.
    io, io_path = load_io(args.policy, args.io) if (args.policy or args.io) else (None, None)
    max_speed = io["max_speed_rad_s"] if io else 17.42
    cap = min(io["max_command_frac"], HARD_CAP) if io else HARD_CAP
    rate = args.max_hz or (io["policy_rate_hz"] if io else 16.0)
    policy = None
    if not args.baseline:
        policy = Policy(args.policy)
        if policy.obs_dim is not None and policy.obs_dim != io["obs_dim"]:
            raise SystemExit(f"{args.policy} takes {policy.obs_dim} observations, {io_path} says {io['obs_dim']}")
    mode = "baseline" if args.baseline else "policy"
    baseline_action = args.baseline_pct / (100 * cap)      # in policy units, for the logged last_action

    home = lc.load_home()
    if args.home and home is None:
        raise SystemExit("--home needs a calibration first: scripts/real_walk_test.py --calibrate-home")
    print(f"mode {mode}" + (f" ({policy.kind}: {args.policy})" if policy else f" ({args.baseline_pct}% both)")
          + f", cap {100 * cap:.0f}%, {rate:.1f} Hz, {args.seconds:g} s")

    dm = connect()
    rows, round_trips, tick_starts = [], [], []
    ended, gyro_bias, n_bias = "not started", 0.0, 0
    try:
        if not wait_for_imu(dm):
            raise RuntimeError("no IMU notifications from the Double Motor")
        if args.home:
            print("Homing legs to straight forward (slow, 20%)...")
            err = lc.home_legs(dm, home)
            print(f"Homed: left {err[0]:+.0f} deg, right {err[1]:+.0f} deg from straight forward.")
        what = (f"run the POLICY (motors up to {100 * cap:.0f}%)" if policy
                else f"walk at {args.baseline_pct}% on both motors (baseline)")
        lc.confirm(f"\nAbout to {what} for {args.seconds:g} s.\n"
                   "Put the robot on the floor at your start mark, pointing where 'straight' should be.", args.yes)

        dm.motor_set_end_state(le.MOTOR_END_STATE_BRAKE, motor=le.MOTOR_BOTH)
        if home is not None:
            off = lc.zero_relative_at_home(dm, home)
            print(f"Leg angles zeroed at straight forward (start offset raw L {off[0]:+.0f}, R {off[1]:+.0f}).")
        else:
            dm.motor_reset_relative_position(motor=le.MOTOR_BOTH, position=0)
            print("WARNING: no home calibration, so leg angle 0 = the starting pose, not straight forward "
                  "(the policy assumes straight forward). Run real_walk_test.py --calibrate-home and use --home.")
        dm.imu_reset_yaw_axis(0)                        # heading error starts at 0
        time.sleep(0.2)
        gyro_bias, n_bias = measure_gyro_bias(dm)
        print(f"Yaw reset; gyro z bias {gyro_bias:+.2f} deg/s ({n_bias} samples). Go.")

        period = 1.0 / rate
        last_action = np.zeros(2)
        last_imu = None
        t0 = time.monotonic()
        next_tick = t0
        ended = "time"
        step = 0
        while True:
            t_start = time.monotonic()
            if t_start - t0 >= args.seconds:
                break
            tick_starts.append(t_start)
            s = read_sensors(dm, gyro_bias, max_speed)
            fresh = s["imu_obj"] is not last_imu
            last_imu = s["imu_obj"]
            obs = build_obs(s, last_action, io) if io else None
            raw = policy(obs) if policy else np.full(2, baseline_action)
            action = np.clip(raw, -1.0, 1.0)
            cmds = to_motor(action, cap)
            t_cmd = time.monotonic()
            send(dm, cmds)
            rt = time.monotonic() - t_cmd
            round_trips.append(rt)

            row = {"t_s": round(t_start - t0, 4), "step": step, "mode": mode,
                   "raw_pos_left": s["raw_pos_left"], "raw_pos_right": s["raw_pos_right"],
                   "raw_speed_left": s["raw_speed_left"], "raw_speed_right": s["raw_speed_right"],
                   "raw_yaw": s["raw_yaw"], "raw_gyro_z": s["raw_gyro_z"], "imu_fresh": int(fresh),
                   "leg_angle_left_deg": round(math.degrees(s["leg_angle_left"]), 2),
                   "leg_angle_right_deg": round(math.degrees(s["leg_angle_right"]), 2),
                   "leg_vel_left": round(s["leg_vel_left"], 4), "leg_vel_right": round(s["leg_vel_right"], 4),
                   "heading_deg": round(math.degrees(s["heading"]), 2),
                   "gyro_z_dps": round(math.degrees(s["gyro_z"]), 3),
                   "action_left": round(float(action[0]), 4), "action_right": round(float(action[1]), 4),
                   "raw_action_left": round(float(raw[0]), 4), "raw_action_right": round(float(raw[1]), 4)}
            for (percent, direction), (name, _, fwd) in zip(cmds, SIDES):
                row[f"cmd_pct_{name}"] = percent if direction == fwd else -percent
            row["round_trip_ms"] = round(1000 * rt, 1)
            if obs is not None:
                row.update({f"obs_{i}": round(float(v), 5) for i, v in enumerate(obs)})
            rows.append(row)

            last_action = action
            step += 1
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        ended = "Ctrl-C"
        print("\nCtrl-C: stopping the motors.")
    except Exception as e:
        ended = f"error: {e!r}"
        print(f"\nERROR: {e!r}. Stopping the motors.")
    finally:
        stop_s = blocking_stop(dm)
        lc.safe_stop(dm)                              # stop again (blocking) and disconnect

    if not rows:
        print(f"Nothing ran (ended by {ended}).")
        return 1 if ended != "time" else 0
    path = lc.ROOT / "data" / f"policy_run_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    lc.write_csv(path, rows, fields)

    elapsed = rows[-1]["t_s"] + period
    rts = [1000 * x for x in round_trips]
    print(f"\n== {mode} run: {len(rows)} steps in {elapsed:.1f} s = {len(rows) / elapsed:.1f} Hz (target {rate:.1f}), "
          f"ended by {ended} ==")
    print(f"command round trip: median {statistics.median(rts):.0f} ms, 95th pct {pct(rts, 0.95):.0f} ms, "
          f"max {max(rts):.0f} ms; final stop {1000 * stop_s:.0f} ms")
    print(f"final heading: {rows[-1]['heading_deg']:+.1f} deg (+ = left);  "
          f"mean command L {statistics.mean(r['cmd_pct_left'] for r in rows):+.0f}%, "
          f"R {statistics.mean(r['cmd_pct_right'] for r in rows):+.0f}%")
    print(f"Saved {len(rows)} rows to {path.relative_to(lc.ROOT)}")
    print("\nNow measure: forward distance (cm), sideways drift (cm, + = left), final heading (deg, + = left).")
    return 0 if ended == "time" else 1


if __name__ == "__main__":
    raise SystemExit(main())
