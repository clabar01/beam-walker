"""Walk the real robot for exactly 5 s so you can tape-measure the distance.

Compare against:  .venv/bin/python scripts/sim_check.py --speed-frac 0.5

    .venv/bin/python scripts/real_walk_test.py 50
    .venv/bin/python scripts/real_walk_test.py 100 --reverse   # if it walked backward

The 5 s is timed on the motor hub itself (motor_run_for_time), and both outputs
start in a single BLE packet, so Bluetooth lag doesn't change the run length.
"""

import argparse
import sys
import time
from pathlib import Path

import legoeducation as le

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lego_common as lc  # noqa: E402

RUN_MS = 5000


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("speed", type=int, choices=[25, 50, 100], help="motor speed in percent")
    ap.add_argument("--reverse", action="store_true", help="swap forward/backward for both legs")
    ap.add_argument("--yes", action="store_true", help="skip the 'press Enter' safety prompt")
    args = ap.parse_args()

    left_dir, right_dir = lc.LEFT_FORWARD, lc.RIGHT_FORWARD
    if args.reverse:
        left_dir, right_dir = lc.opposite(left_dir), lc.opposite(right_dir)

    dm = lc.connect_double_motor()
    try:
        lc.confirm(f"\nAbout to walk at {args.speed}% for {RUN_MS / 1000:g} s.\n"
                   "Put the robot on the floor at your start mark with both legs pointing forward.",
                   args.yes)
        dm.motor_set_end_state(le.MOTOR_END_STATE_BRAKE, motor=le.MOTOR_BOTH)

        with dm.batch(blocking=False):
            dm.motor_run_for_time(RUN_MS, direction=left_dir, motor=le.MOTOR_LEFT, speed=args.speed)
            dm.motor_run_for_time(RUN_MS, direction=right_dir, motor=le.MOTOR_RIGHT, speed=args.speed)

        # The hub stops the motors itself at 5 s. Wait past that before the
        # host-side stop below, so the backstop can never cut the run short.
        time.sleep(RUN_MS / 1000 + 0.5)
        print(f"Done (the hub timed {RUN_MS / 1000:g} s).")
    except KeyboardInterrupt:
        print("\nCancelled - stopping motors.")
    finally:
        lc.safe_stop(dm)

    print("\nNow measure:")
    print("  - forward distance (cm) from start mark to the same point on the robot")
    print("  - sideways drift (cm) and the final heading (deg; + = turned left)")
    print(f"Sim to compare: .venv/bin/python scripts/sim_check.py --speed-frac {args.speed / 100:g}")


if __name__ == "__main__":
    main()
