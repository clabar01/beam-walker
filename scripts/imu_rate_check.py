"""Two hardware checks before running a policy on the real robot.

1. CONTROL LOOP RATE (**moves the robot**): for 10 s, read both leg angles and the IMU (yaw, gyro z) and send a
   speed command to both motors (20%, forward), the way a policy loop would. Rate-limited like a real policy:
   every command is BLOCKING (the next one is only sent after the hub confirms the last, so nothing queues up),
   the loop is capped at --max-hz (default 30, about the IMU rate), and a blocking stop is always sent at the end,
   on Ctrl-C, or on an error. Prints the achieved rate, command round-trip times, and how often FRESH sensor data
   arrived (notifications every NOTIFY_MS = 20 ms are requested; the IMU actually updates ~32 /s).
   (The first version sent non-blocking commands as fast as possible; they queued faster than Bluetooth could
   send them and the robot kept walking after the loop ended.)
2. GYRO UNITS (motors stopped): you turn the robot by hand 90 deg to the LEFT over ~2 s. Integrating gyro z
   and comparing it to the yaw change (and to the true 90 deg) tells whether gyro is deg/s, rad/s, or raw counts.

Each phase waits for Enter before anything happens. Logs go to data/imu_check_<time>.csv.

    .venv/bin/python scripts/imu_rate_check.py               # both checks
    .venv/bin/python scripts/imu_rate_check.py --skip-loop   # gyro units only (nothing moves)
    .venv/bin/python scripts/imu_rate_check.py --skip-gyro --max-hz 50   # loop only, try a higher cap
"""

import argparse
import math
import statistics
import time
from pathlib import Path

import legoeducation as le

from lego_common import (
    LEFT_FORWARD, RIGHT_FORWARD, connect_double_motor, confirm, safe_stop, wrap180, write_csv,
)

ROOT = Path(__file__).resolve().parent.parent
TRUE_TURN_DEG = 90.0

# Candidate gyro units: how many raw units one deg/s would read as.
GYRO_UNITS = {
    "deg/s": 1.0,
    "rad/s": math.pi / 180,
    "0.1 deg/s (decideg/s)": 10.0,
    "mdeg/s": 1000.0,
    "raw, +-2000 dps (16.4 LSB per deg/s)": 16.4,
    "raw, +-1000 dps (32.8 LSB per deg/s)": 32.8,
    "raw, +-500 dps (65.5 LSB per deg/s)": 65.5,
    "raw, +-250 dps (131 LSB per deg/s)": 131.0,
}


def isnan(v):
    return isinstance(v, float) and math.isnan(v)


def wait_for_imu(dm, timeout=3.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if not isnan(dm.imu_device.yaw):
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# 1. Control loop rate
# ---------------------------------------------------------------------------

def pct(values, q):
    s = sorted(values)
    return s[int(q * (len(s) - 1))]


def send_speeds(dm, speed):
    """Both motors forward at `speed` percent, as one batch. BLOCKING: returns only after the hub confirms,
    so a new command is never sent while the previous one is still in flight."""
    with dm.batch(blocking=True):
        dm.motor_run(direction=LEFT_FORWARD, motor=le.MOTOR_LEFT, speed=speed)
        dm.motor_run(direction=RIGHT_FORWARD, motor=le.MOTOR_RIGHT, speed=speed)


def blocking_stop(dm):
    """Stop both motors and wait for the confirmation. Retried once if interrupted. Returns seconds taken."""
    t = time.monotonic()
    for attempt in range(2):
        try:
            dm.motor_stop(motor=le.MOTOR_BOTH, blocking=True)
            break
        except KeyboardInterrupt:
            print("\n(stopping the motors, please wait)")
        except Exception as e:
            print(f"stop attempt {attempt + 1} failed: {e!r}")
    return time.monotonic() - t


def loop_rate_check(dm, seconds, speed, max_hz):
    """Policy-style loop: read sensors, send one confirmed command, wait for the next tick (at most max_hz).
    Returns True if it ended normally, False on Ctrl-C."""
    dm.motor_set_end_state(le.MOTOR_END_STATE_BRAKE, motor=le.MOTOR_BOTH)
    period = 1.0 / max_hz
    cmd_times, tick_starts, ages = [], [], []
    fresh_loops, overruns = 0, 0
    last_imu, last_imu_change = None, None
    ended = "time"
    t0 = time.monotonic()
    t_end = t0 + seconds
    next_tick = t0
    try:
        while True:
            t_start = time.monotonic()
            if t_start >= t_end:
                break
            tick_starts.append(t_start)
            # --- read, like building the policy observation ---
            ml, mr, imu = dm.motor[le.MOTOR_LEFT], dm.motor[le.MOTOR_RIGHT], dm.imu_device
            _obs = (ml.position, ml.speed, mr.position, mr.speed, imu.yaw, imu.gyroscopeZ)
            if imu is not last_imu:                   # the library swaps in a new object per notification
                last_imu, last_imu_change = imu, t_start
                fresh_loops += 1
            ages.append(t_start - last_imu_change)    # how old the IMU data used by this step is (lower bound)
            # --- send, like applying the policy action (waits for the hub's confirmation) ---
            t_cmd = time.monotonic()
            send_speeds(dm, speed)
            cmd_times.append(time.monotonic() - t_cmd)
            # --- wait for the next tick ---
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                overruns += 1
                next_tick = time.monotonic()          # don't try to catch up with a burst
    except KeyboardInterrupt:
        ended = "Ctrl-C"
    except Exception as e:
        ended = f"error: {e!r}"
    finally:
        t_stop = time.monotonic()
        stop_s = blocking_stop(dm)
    elapsed = t_stop - t0
    n = len(cmd_times)

    print(f"\n== Control loop (blocking commands, cap {max_hz:.0f} Hz, {speed}% both, {elapsed:.1f} s, ended by {ended}) ==")
    if n == 0:
        print("no loops ran")
        return ended == "time"
    cms = [1000 * x for x in cmd_times]
    gaps = [1000 * (b - a) for a, b in zip(tick_starts, tick_starts[1:])]
    print(f"loops:               {n} -> {n / elapsed:.1f} loops/s  (cap {max_hz:.0f}; {overruns} ticks overran the period)")
    print(f"command round trip:  avg {statistics.mean(cms):.1f} ms, median {statistics.median(cms):.1f} ms, "
          f"95th pct {pct(cms, 0.95):.1f} ms, max {max(cms):.1f} ms")
    if gaps:
        print(f"time between ticks:  avg {statistics.mean(gaps):.1f} ms, 95th pct {pct(gaps, 0.95):.1f} ms, max {max(gaps):.1f} ms")
    print(f"fresh IMU per loop:  {fresh_loops}/{n} loops ({fresh_loops / n:.0%}) saw a new IMU reading")
    a = [1000 * x for x in ages]
    print(f"IMU data age:        avg {statistics.mean(a):.1f} ms, max {max(a):.1f} ms (since the reading changed)")
    print(f"final stop took:     {1000 * stop_s:.0f} ms (should be about one round trip if nothing was queued)")
    max_rate = 1000.0 / statistics.median(cms)
    print(f"=> achieved {n / elapsed:.1f} Hz. Blocking commands alone would allow ~{max_rate:.0f} Hz (1 / median round "
          f"trip); the IMU gives ~32 new readings/s.")
    return ended == "time"


# ---------------------------------------------------------------------------
# 2. Gyro units
# ---------------------------------------------------------------------------

def record_imu(dm, seconds, phase, t0, rows):
    last = None
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        imu = dm.imu_device
        if imu is not last and not isnan(imu.yaw):
            last = imu
            rows.append({"t_s": round(time.monotonic() - t0, 4), "phase": phase, "yaw": imu.yaw,
                         "pitch": imu.pitch, "roll": imu.roll, "gyro_x": imu.gyroscopeX,
                         "gyro_y": imu.gyroscopeY, "gyro_z": imu.gyroscopeZ,
                         "accel_x": imu.accelerometerX, "accel_y": imu.accelerometerY, "accel_z": imu.accelerometerZ})
        time.sleep(0.002)


def integrate(rows, key, bias):
    total = 0.0
    for a, b in zip(rows, rows[1:]):
        total += 0.5 * ((a[key] - bias) + (b[key] - bias)) * (b["t_s"] - a["t_s"])
    return total


def gyro_units_check(dm, rows, settle_s, turn_s):
    dm.motor_stop(motor=le.MOTOR_BOTH)
    dm.imu_reset_yaw_axis(0)
    time.sleep(0.3)
    t0 = time.monotonic()
    print(f"\nHold the robot STILL for {settle_s:.0f} s (measuring gyro bias)...", flush=True)
    record_imu(dm, settle_s, "still", t0, rows)
    print(f">>> GO: turn the robot {TRUE_TURN_DEG:.0f} deg to the LEFT (counterclockwise seen from above) over ~2 s, "
          f"keep it flat on the floor, then hold it still. Recording {turn_s:.0f} s...", flush=True)
    record_imu(dm, turn_s, "turn", t0, rows)
    print("Done recording.")

    still = [r for r in rows if r["phase"] == "still"]
    turn = [r for r in rows if r["phase"] == "turn"]
    if len(still) < 5 or len(turn) < 10:
        print(f"Too few IMU samples (still {len(still)}, turn {len(turn)}). Is the IMU sending notifications?")
        return
    allr = still + turn
    dur = allr[-1]["t_s"] - allr[0]["t_s"]
    print(f"\n== Gyro units ({len(allr)} IMU samples in {dur:.1f} s = {len(allr) / dur:.0f} /s) ==")
    bias = {k: statistics.mean(r[k] for r in still) for k in ("gyro_x", "gyro_y", "gyro_z")}
    noise = {k: statistics.pstdev(r[k] for r in still) for k in bias}
    yaw0 = statistics.mean(r["yaw"] for r in still)
    yaw1 = statistics.mean(r["yaw"] for r in turn[-10:])          # last ~10 samples, robot held still again
    dyaw = yaw1 - yaw0
    print(f"yaw:      before {yaw0:.1f}, after {yaw1:.1f}  ->  change {dyaw:+.1f} (raw yaw units)")
    if abs(abs(wrap180(dyaw)) - TRUE_TURN_DEG) < 20:
        print("          -> yaw reads in DEGREES (about 90 for a 90 deg turn)")
    elif abs(abs(dyaw) - 10 * TRUE_TURN_DEG) < 200:
        print("          -> yaw reads in TENTHS of a degree")
    else:
        print("          -> yaw scale unclear; check that the turn really was ~90 deg")
    peak = {k: max(abs(r[k] - bias[k]) for r in turn) for k in bias}
    integ = {k: integrate(allr, k, bias[k]) for k in bias}
    for k in bias:
        print(f"{k}:   bias {bias[k]:+.2f}, still noise {noise[k]:.2f}, peak {peak[k]:.1f}, integral {integ[k]:+.1f}")
    axis = max(integ, key=lambda k: abs(integ[k]))
    if axis != "gyro_z":
        print(f"NOTE: the turn shows up mostly on {axis}, not gyro_z. The yaw axis may need imu_set_yaw_face(), "
              "and the policy's gyro_z observation must use this axis.")
    g = integ[axis]
    per_degps_true = abs(g) / TRUE_TURN_DEG
    print(f"\n{axis} integral / true 90 deg      = {per_degps_true:.4g} units per deg/s")
    if abs(dyaw) > 1:
        print(f"{axis} integral / measured yaw change = {abs(g / dyaw):.4g} units per (yaw unit)/s")
    best = sorted(GYRO_UNITS.items(), key=lambda kv: abs(math.log(per_degps_true / kv[1])))
    print("closest unit guesses (from the true 90 deg turn):")
    for name, scale in best[:3]:
        print(f"  {name:40s} expects {scale:8.4g} units per deg/s  (off by {per_degps_true / scale - 1:+.0%})")
    sign = "positive" if g * (dyaw if abs(dyaw) > 1 else 1) > 0 else "negative"
    print(f"sign: turning LEFT gave {axis} integral {g:+.1f} and yaw change {dyaw:+.1f} "
          f"(gyro and yaw agree: {'yes' if sign == 'positive' else 'NO'}).")
    print("Sim convention: + = turned left (counterclockwise from above) for both yaw and gyro z.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=10.0, help="loop test duration")
    ap.add_argument("--speed", type=int, default=20, help="loop test motor speed, percent")
    ap.add_argument("--max-hz", type=float, default=30.0, help="loop rate cap (default 30, about the IMU rate)")
    ap.add_argument("--skip-loop", action="store_true")
    ap.add_argument("--skip-gyro", action="store_true")
    ap.add_argument("--turn-seconds", type=float, default=5.0, help="recording time after GO")
    args = ap.parse_args()

    dm = connect_double_motor()
    rows = []
    try:
        if not wait_for_imu(dm):
            print("WARNING: no IMU notification yet (dm.imu_device.yaw is NaN). Continuing; results may be empty.")
        else:
            imu = dm.imu_device
            print(f"IMU now: yaw {imu.yaw}, pitch {imu.pitch}, roll {imu.roll}, "
                  f"gyro ({imu.gyroscopeX}, {imu.gyroscopeY}, {imu.gyroscopeZ})")

        if not args.skip_loop:
            confirm(f"\nLOOP TEST: the robot will WALK at {args.speed}% for {args.seconds:.0f} s "
                    "(about 1 m on a hard floor).\nGive it clear floor in front, or hold it up so the legs spin freely.")
            if not loop_rate_check(dm, args.seconds, args.speed, args.max_hz):
                return                                  # Ctrl-C or an error: motors are stopped, skip the rest

        if not args.skip_gyro:
            confirm("\nGYRO TEST: motors stay stopped. Put the robot flat on the floor and keep your hands on it.\n"
                    "After Enter: hold still 1 s, then at GO turn it 90 deg LEFT over ~2 s and hold still.")
            gyro_units_check(dm, rows, settle_s=1.0, turn_s=args.turn_seconds)
    finally:
        safe_stop(dm)

    if rows:
        path = ROOT / "data" / f"imu_check_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        write_csv(path, rows, fields=list(rows[0]))
        print(f"\nSaved {len(rows)} IMU samples to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
