"""Walk the real robot for exactly 5 s so you can tape-measure the distance.

Logs both legs' encoder angles to data/walk_<speed>pct_<time>.csv and prints
the average leg RPM under load against the commanded RPM.

Compare against:  .venv/bin/python scripts/sim_check.py --speed-frac 0.5

    # one time (motors unpowered; you turn the legs by hand):
    .venv/bin/python scripts/real_walk_test.py --calibrate-home

    .venv/bin/python scripts/real_walk_test.py 50 --home     # home legs, then walk
    .venv/bin/python scripts/real_walk_test.py 50            # walk from wherever the legs are
    .venv/bin/python scripts/real_walk_test.py 100 --reverse # if it walked backward

The 5 s is timed on the motor hub itself (motor_run_for_time), and both outputs
start in a single BLE packet, so Bluetooth lag doesn't change the run length.

--home turns both legs slowly (20%) to straight forward and horizontal (sim
qpos 0) BEFORE the Enter prompt, then holds them there. When a home calibration
exists, the logged leg_angle_deg is measured from that pose, so it matches the
sim's hinge angle. Without one, it's measured from wherever the legs started.
"""

import argparse
import sys
import time
from pathlib import Path

import legoeducation as le

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lego_common as lc  # noqa: E402

RUN_MS = 5000
SETTLE_S = 0.5          # skip the spin-up when fitting steady-state RPM


def summarize(rows, speed_pct, signs):
    """Print average and steady-state leg RPM under load vs the commanded RPM."""
    cmd_rpm = speed_pct / 100 * lc.MAX_RPM_MEASURED
    run_s = RUN_MS / 1000
    print(f"\nCommanded: {speed_pct}% = {cmd_rpm:.1f} RPM "
          f"(measured no-load max {lc.MAX_RPM_MEASURED} RPM)")
    print(f"{'leg':>5} | {'avg RPM (5 s)':>13} | {'steady RPM':>10} | {'% of cmd':>8} | {'revs':>5}")
    print("-" * 55)
    avgs = {}
    for side in ("left", "right"):
        pts = [r for r in rows if r["side"] == side]
        before = [r for r in pts if r["phase"] == "baseline"] or pts[:1]
        start_angle = before[-1]["leg_angle_deg"]
        end_angle = pts[-1]["leg_angle_deg"]          # hub brakes at 5 s; "after" samples show it stopped
        net = end_angle - start_angle
        avg = net / 360 / run_s * 60
        steady, _ = lc.fit_rpm(rows, "walk", side, SETTLE_S)
        steady *= signs[side]                          # fit uses raw counts; make forward positive
        avgs[side] = avg
        print(f"{side:>5} | {avg:>+13.1f} | {steady:>+10.1f} | {avg / cmd_rpm * 100:>7.0f}% | "
              f"{net / 360:>+5.2f}")
    if avgs["left"]:
        print(f"R/L average RPM difference: {(avgs['right'] - avgs['left']) / avgs['left'] * 100:+.1f}%")
    print("(+ = rolling forward. Below 100% of commanded means the legs slowed under load or stalled.)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speed", type=int, nargs="?", choices=[25, 50, 100], help="motor speed in percent")
    ap.add_argument("--home", action="store_true",
                    help="before the Enter prompt, turn both legs to straight-forward/horizontal")
    ap.add_argument("--calibrate-home", action="store_true",
                    help="record the straight-forward leg angles by hand (no walking), then exit")
    ap.add_argument("--reverse", action="store_true", help="swap forward/backward for both legs")
    ap.add_argument("--yes", action="store_true", help="skip the 'press Enter' safety prompt")
    args = ap.parse_args()
    if args.speed is None and not args.calibrate_home:
        ap.error("give a speed (25, 50 or 100), or use --calibrate-home")

    home = lc.load_home()
    if args.home and home is None and not args.calibrate_home:
        sys.exit(f"--home needs a calibration first: run with --calibrate-home "
                 f"(creates {lc.HOME_PATH.relative_to(lc.ROOT)}).")

    left_dir, right_dir = lc.LEFT_FORWARD, lc.RIGHT_FORWARD
    if args.reverse:
        left_dir, right_dir = lc.opposite(left_dir), lc.opposite(right_dir)
    signs = {"left": lc.forward_sign(left_dir), "right": lc.forward_sign(right_dir)}

    dm = lc.connect_double_motor()
    log = None
    walked = False
    try:
        if args.calibrate_home:
            lc.calibrate_home(dm)
            return

        if args.home:
            print("Homing legs to straight forward (slow, 20%)...")
            err = lc.home_legs(dm, home)
            print(f"Homed: left {err[0]:+.0f} deg, right {err[1]:+.0f} deg from straight forward.")

        start_hint = ("Legs are homed and holding. Put the robot on the floor at your start mark."
                      if args.home else
                      "Put the robot on the floor at your start mark with both legs pointing forward.")
        lc.confirm(f"\nAbout to walk at {args.speed}% for {RUN_MS / 1000:g} s.\n{start_hint}", args.yes)

        dm.motor_set_end_state(le.MOTOR_END_STATE_BRAKE, motor=le.MOTOR_BOTH)
        if home is not None:
            off = lc.zero_relative_at_home(dm, home)
            angle_ref = (f"from straight forward (start offset raw counts: "
                         f"L {off[0]:+.0f}, R {off[1]:+.0f})")
        else:
            dm.motor_reset_relative_position(motor=le.MOTOR_BOTH, position=0)
            angle_ref = "from the starting pose (no home calibration)"
        time.sleep(0.2)

        log = lc.Logger(dm, time.monotonic())
        log.phase, log.cmd = "baseline", 0
        log.start()
        time.sleep(0.3)

        log.phase, log.cmd = "walk", args.speed
        with dm.batch(blocking=False):
            dm.motor_run_for_time(RUN_MS, direction=left_dir, motor=le.MOTOR_LEFT, speed=args.speed)
            dm.motor_run_for_time(RUN_MS, direction=right_dir, motor=le.MOTOR_RIGHT, speed=args.speed)

        # The hub stops the motors itself at 5 s. Wait past that before the
        # host-side stop below, so the backstop can never cut the run short.
        time.sleep(RUN_MS / 1000)
        log.phase, log.cmd = "after", 0
        time.sleep(0.5)
        walked = True
        print(f"Done (the hub timed {RUN_MS / 1000:g} s).")
    except KeyboardInterrupt:
        print("\nCancelled - stopping motors.")
    finally:
        if log is not None:
            log.stop()
        lc.safe_stop(dm)

    if log is not None and log.rows:
        for r in log.rows:
            r["leg_angle_deg"] = signs[r["side"]] * r["position_deg"]
        path = lc.ROOT / "data" / f"walk_{args.speed}pct_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        lc.write_csv(path, log.rows, lc.FIELDS + ["leg_angle_deg"])
        print(f"\nSaved {len(log.rows)} samples to {path.relative_to(lc.ROOT)}")
        print(f"leg_angle_deg = degrees rolled forward, {angle_ref}")
        if walked:
            summarize(log.rows, args.speed, signs)

    if walked:
        print("\nNow measure:")
        print("  - forward distance (cm) from start mark to the same point on the robot")
        print("  - sideways drift (cm) and the final heading (deg; + = turned left)")
        print(f"Sim to compare: .venv/bin/python scripts/sim_check.py --speed-frac {args.speed / 100:g}")


if __name__ == "__main__":
    main()
