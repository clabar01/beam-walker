"""Shared LEGO Double Motor setup for the beam-walker hardware scripts.

Connection pattern copied from clabar01/Lego world_cup/robot.py (same card).
"""

import time

import legoeducation as le

CARD_SERIAL = "0999"
CARD_COLOR = le.LEGO_COLOR_RED

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
    print(f"Connecting to Double Motor (card {CARD_SERIAL}, RED)...")
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
