"""Measure the Double Motor's no-load speed on each output.

Runs both outputs at 50% for 5 s, then 100% for 5 s, logging each output's
angle against time. Prints RPM per output per speed (to spot left/right
mismatch) and saves the raw log to data/motor_rpm.csv.

LIFT THE ROBOT so the legs spin freely. This measures no-load speed, which is
what the sim's ctrlrange (PH_SPEED) should use.

    .venv/bin/python scripts/measure_motor.py
    .venv/bin/python scripts/measure_motor.py --duration 8 --settle 1.0
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import legoeducation as le
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lego_common as lc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "motor_rpm.csv"
def run_both(dm, pct):
    """Start both outputs together in the forward direction at pct %."""
    with dm.batch():
        dm.motor_run(direction=lc.LEFT_FORWARD, motor=le.MOTOR_LEFT, speed=pct)
        dm.motor_run(direction=lc.RIGHT_FORWARD, motor=le.MOTOR_RIGHT, speed=pct)


def mean_power(rows, phase, side, settle_s):
    """Mean |power| after spin-up. The hub regulates speed, so a weaker or
    higher-friction side shows up here (more effort for the same RPM)."""
    pts = [(r["t_s"], r["power"]) for r in rows if r["phase"] == phase and r["side"] == side]
    if not pts:
        return float("nan")
    t0 = pts[0][0]
    vals = [abs(p) for t, p in pts if t - t0 >= settle_s]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=5.0, help="seconds per speed (default 5)")
    ap.add_argument("--settle", type=float, default=0.75,
                    help="seconds of spin-up to skip in each phase before fitting RPM")
    ap.add_argument("--yes", action="store_true", help="skip the 'press Enter' safety prompt")
    args = ap.parse_args()

    phases = [("50pct", 50), ("100pct", 100)]

    dm = lc.connect_double_motor()
    log = None
    try:
        lc.confirm(f"\nAbout to spin BOTH outputs: 50% for {args.duration:g} s, then 100% for "
                   f"{args.duration:g} s.\nLift the robot so the legs spin freely.", args.yes)
        dm.motor_reset_relative_position(motor=le.MOTOR_BOTH, position=0)
        time.sleep(0.2)

        t0 = time.monotonic()
        log = lc.Logger(dm, t0)
        log.start()
        time.sleep(0.3)                                      # baseline at rest

        for name, pct in phases:
            log.phase, log.cmd = name, pct
            print(f"  {name}: running {pct}% for {args.duration:g} s...")
            run_both(dm, pct)
            time.sleep(args.duration)

        dm.motor_stop(motor=le.MOTOR_BOTH)
        log.phase, log.cmd = "stop", 0
        time.sleep(0.5)                                      # record the coast-down
    except KeyboardInterrupt:
        print("\nCancelled - stopping motors.")
    finally:
        if log is not None:
            log.stop()
        lc.safe_stop(dm)

    if log is None or not log.rows:
        print("No data recorded.")
        return

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=lc.FIELDS)
        w.writeheader()
        w.writerows(log.rows)
    n_left = sum(r["side"] == "left" for r in log.rows)
    n_right = len(log.rows) - n_left
    print(f"\nSaved {len(log.rows)} samples ({n_left} left / {n_right} right) to "
          f"{CSV_PATH.relative_to(ROOT)}")

    print(f"\n{'speed':>6} | {'left RPM':>9} | {'right RPM':>9} | {'R/L mismatch':>12} | "
          f"{'mean rad/s':>10} | {'L power':>7} | {'R power':>7} | {'R/L power':>9}")
    print("-" * 90)
    for name, pct in phases:
        rl, nl = lc.fit_rpm(log.rows, name, "left", args.settle)
        rr, nr = lc.fit_rpm(log.rows, name, "right", args.settle)
        # Signs depend on how each output counts; the outputs are mirrored, so compare magnitudes.
        al, ar = abs(rl), abs(rr)
        mismatch = (ar - al) / al * 100 if al > 0 else float("nan")
        mean_rads = (al + ar) / 2 * 2 * math.pi / 60
        pl, pr = mean_power(log.rows, name, "left", args.settle), mean_power(log.rows, name, "right", args.settle)
        pmis = (pr - pl) / pl * 100 if pl > 0 else float("nan")
        print(f"{pct:>5}% | {rl:>+9.1f} | {rr:>+9.1f} | {mismatch:>+11.1f}% | {mean_rads:>10.2f} | "
              f"{pl:>7.0f} | {pr:>7.0f} | {pmis:>+8.1f}%")
    print("\nThe hub regulates each output's speed, so RPM usually matches closely;")
    print("a real left/right difference shows up as a power (effort) mismatch instead.")
    print("RPM signs show which way each output's encoder counts; magnitudes are what matter.")
    print("Use the 100% mean rad/s for ctrlrange (PH_SPEED) in models/walker.xml.")


if __name__ == "__main__":
    main()
