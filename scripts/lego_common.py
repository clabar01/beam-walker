"""Shared LEGO Double Motor setup for the beam-walker hardware scripts.

Connection pattern copied from clabar01/Lego world_cup/robot.py (same card).
"""

import json
import math
import threading
import time
from pathlib import Path

import legoeducation as le
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HOME_PATH = ROOT / "data" / "leg_home.json"

# MEASURED no-load speed at 100% (scripts/measure_motor.py, 2026-11-15).
MAX_RPM_MEASURED = 166.3

CARD_SERIAL = "0999"
CARD_COLOR = le.LEGO_COLOR_ORANGE   # public constant (= 9); connect() matches against this

# Which way each output turns to move the robot FORWARD. The two outputs face
# opposite sides, so they spin in opposite directions to roll the same way.
# This matches the World Cup car (LEFT_FLIP = 1, RIGHT_FLIP = -1, run CCW).
# If real_walk_test.py walks backward, swap both; if it spins, swap one.
LEFT_FORWARD = le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
RIGHT_FORWARD = le.MOTOR_MOVE_DIRECTION_CLOCKWISE

# Sensor notification period. 15 ms is the library minimum; 20 ms is safer over BLE.
NOTIFY_MS = 20


def opposite(direction):
    if direction == le.MOTOR_MOVE_DIRECTION_CLOCKWISE:
        return le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
    return le.MOTOR_MOVE_DIRECTION_CLOCKWISE


def connect_double_motor(attempts=5):
    """Connect to the Double Motor, retrying the usual "not ready" / asleep cases."""
    dm = le.DoubleMotor()
    print(f"Connecting to Double Motor (card {CARD_SERIAL}, ORANGE)...")
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            dm.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL,
                       device_notification_delay=NOTIFY_MS)
        except Exception as e:
            if "not ready" in str(e).lower() and not last:
                time.sleep(1)
                continue
            raise
        if dm.connected:
            break
        if not last:
            print(f"  not found yet - press the hub button to wake it (try {attempt + 2}/{attempts})...")
            time.sleep(1)
    if not dm.connected:
        raise ConnectionError(
            f"Could not connect to the Double Motor (card {CARD_SERIAL}). Press the hub "
            "button, check the Connection Card, and close any other script using it.")
    print("Connected.")
    return dm


def safe_stop(dm):
    """Stop both outputs and disconnect, never raising."""
    try:
        dm.motor_stop(motor=le.MOTOR_BOTH)
    except Exception as e:
        print(f"stop failed: {e!r}")
    try:
        dm.disconnect()
    except Exception as e:
        print(f"disconnect failed: {e!r}")


def confirm(prompt, skip=False):
    if skip:
        return
    input(f"{prompt}\nPress Enter to start (Ctrl-C to cancel)... ")


# ---------------------------------------------------------------------------
# Encoder logging (shared by measure_motor.py and real_walk_test.py)
# ---------------------------------------------------------------------------

FIELDS = ["t_s", "phase", "cmd_pct", "side", "position_deg", "abs_position_deg",
          "speed_pct", "power"]


class Logger(threading.Thread):
    """Polls dm.motor[] and records each NEW notification with a host timestamp.

    The library replaces dm.motor[i] with a fresh object on every notification,
    so an identity change means a new reading arrived.
    """

    def __init__(self, dm, t0, poll_s=0.002):
        super().__init__(daemon=True)
        self.dm, self.t0, self.poll_s = dm, t0, poll_s
        self.rows = []
        self.phase, self.cmd = "idle", 0
        self._stop = threading.Event()

    def run(self):
        last = [None, None]
        while not self._stop.is_set():
            now = time.monotonic() - self.t0
            for i, side in ((le.MOTOR_LEFT, "left"), (le.MOTOR_RIGHT, "right")):
                m = self.dm.motor[i]
                if m is not last[i] and not (isinstance(m.position, float) and math.isnan(m.position)):
                    last[i] = m
                    self.rows.append({
                        "t_s": round(now, 4), "phase": self.phase, "cmd_pct": self.cmd,
                        "side": side, "position_deg": m.position,
                        "abs_position_deg": m.absolutePosition,
                        "speed_pct": m.speed, "power": m.power,
                    })
            time.sleep(self.poll_s)

    def stop(self):
        self._stop.set()
        self.join(timeout=1)


def fit_rpm(rows, phase, side, settle_s):
    """RPM from a straight-line fit of position vs time, skipping the spin-up."""
    pts = [(r["t_s"], r["position_deg"]) for r in rows if r["phase"] == phase and r["side"] == side]
    if len(pts) < 5:
        return float("nan"), 0
    t0 = pts[0][0]
    pts = [p for p in pts if p[0] - t0 >= settle_s]
    if len(pts) < 5:
        return float("nan"), 0
    t, pos = np.array(pts).T
    deg_per_s = np.polyfit(t, pos, 1)[0]
    return deg_per_s / 6.0, len(pts)          # 360 deg/rev, 60 s/min


def write_csv(path, rows, fields=None):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def forward_sign(direction):
    """Sign that turns this output's encoder count into 'degrees rolled forward'.

    From data/motor_rpm.csv: a COUNTERCLOCKWISE run counts down, CLOCKWISE counts up.
    Positive forward angle = the sim hinge's positive direction (leg tip goes down).
    """
    return -1 if direction == le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE else 1


# ---------------------------------------------------------------------------
# Leg homing: both legs straight forward and horizontal = sim qpos 0
# ---------------------------------------------------------------------------

def wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def read_abs(dm, timeout=1.0):
    """(left, right) absolute encoder angles 0-359, waiting for valid readings."""
    t_end = time.monotonic() + timeout
    while True:
        vals = [dm.motor[i].absolutePosition for i in (le.MOTOR_LEFT, le.MOTOR_RIGHT)]
        if not any(isinstance(v, float) and math.isnan(v) for v in vals):
            return float(vals[0]), float(vals[1])
        if time.monotonic() > t_end:
            raise RuntimeError("No encoder readings from the Double Motor yet.")
        time.sleep(0.02)


def load_home():
    """Saved home angles, or None if --calibrate-home hasn't been run."""
    if not HOME_PATH.exists():
        return None
    return json.loads(HOME_PATH.read_text())


def calibrate_home(dm):
    """Let the user point both legs forward BY HAND, then save their encoder angles.
    Doesn't drive the motors: both outputs are set to coast so they turn freely."""
    dm.motor_set_end_state(le.MOTOR_END_STATE_COAST, motor=le.MOTOR_BOTH)
    dm.motor_stop(motor=le.MOTOR_BOTH)
    input("\nHOME CALIBRATION (motors are not powered)\n"
          "Turn both legs BY HAND so each beam points straight FORWARD and is HORIZONTAL\n"
          "(use a square or the floor edge). Then press Enter... ")
    left, right = read_abs(dm)
    home = {"left_abs": left, "right_abs": right,
            "calibrated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "absolute encoder angles (0-359) with each leg straight forward, horizontal"}
    HOME_PATH.parent.mkdir(parents=True, exist_ok=True)
    HOME_PATH.write_text(json.dumps(home, indent=2) + "\n")
    print(f"Saved home: left {left:.0f} deg, right {right:.0f} deg -> {HOME_PATH.relative_to(ROOT)}")
    return home


def zero_relative_at_home(dm, home):
    """Set each output's relative position = its current offset from home,
    so position 0 means 'leg straight forward' for the whole run."""
    left, right = read_abs(dm)
    off_l, off_r = wrap180(left - home["left_abs"]), wrap180(right - home["right_abs"])
    dm.motor_reset_relative_position(motor=le.MOTOR_LEFT, position=int(round(off_l)))
    dm.motor_reset_relative_position(motor=le.MOTOR_RIGHT, position=int(round(off_r)))
    return off_l, off_r


def home_legs(dm, home, speed=20, timeout=4.0, tol_deg=3.0):
    """Turn both legs to the saved home angles (shortest way), slowly, and hold there.
    Returns the final (left, right) error in degrees."""
    dm.motor_set_end_state(le.MOTOR_END_STATE_HOLD, motor=le.MOTOR_BOTH)
    with dm.batch(blocking=False):
        dm.motor_run_to_absolute_position(int(round(home["left_abs"])), motor=le.MOTOR_LEFT,
                                          direction=le.MOTOR_MOVE_DIRECTION_SHORTEST, speed=speed)
        dm.motor_run_to_absolute_position(int(round(home["right_abs"])), motor=le.MOTOR_RIGHT,
                                          direction=le.MOTOR_MOVE_DIRECTION_SHORTEST, speed=speed)
    t_end = time.monotonic() + timeout
    time.sleep(0.3)
    while True:
        left, right = read_abs(dm)
        err = (wrap180(left - home["left_abs"]), wrap180(right - home["right_abs"]))
        settled = all(abs(e) <= tol_deg for e in err)
        if settled or time.monotonic() > t_end:
            break
        time.sleep(0.05)
    if not settled:
        dm.motor_stop(motor=le.MOTOR_BOTH)       # don't leave a stalled leg pushing
        print(f"WARNING: homing timed out (error L {err[0]:+.0f}, R {err[1]:+.0f} deg). "
              "A leg may be blocked by the floor; lift the robot and retry.")
    time.sleep(0.2)                              # let the hold settle
    return err
