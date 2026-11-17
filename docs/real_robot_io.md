# Real robot: sensor units and control rate

Measured 2026-11-16 on the hard floor with `scripts/imu_rate_check.py` (legoeducation 1.2.0,
Double Motor card 0999, notifications requested every 20 ms). The trained policy needs these to
turn legoeducation readings into its observations. The same table goes into `policy_io.json`
with every exported policy.

## IMU (`dm.imu_device`)

From `data/imu_check_20261116-184818.csv`: held still 1 s, then turned about 90° left by hand.

| Field | Raw unit | Convert for the policy | Sign | Evidence |
|---|---|---|---|---|
| `yaw` | 0.1° | `rad = wrap_to_pi(radians(raw / 10))`, after `dm.imu_reset_yaw_axis(0)` at the start | + = turned left (same as sim) | +856 for a ~85–90° left turn |
| `gyroscopeZ` | deg/s | `rad/s = radians(raw)` | + = turning left (same as sim) | integral 97.2 vs yaw change 85.6° (1.14) and vs 90° (1.08) |
| `gyroscopeX`, `gyroscopeY` | — | not used | — | barely moved during the turn, so z is the yaw axis |

- Gyro z reads deg/s to within about 10%. The two estimates (1.08 and 1.14 raw per deg/s) bracket the
  uncertainty in how far the robot was actually turned.
- The IMU updates about **32 times per second**: 192 samples in 5.97 s.

## Control loop rate

Policy-style loop: read both leg angles + yaw + gyro z, then send both motor speeds as one batch of `motor_run`
commands, **blocking** (wait for the hub's confirmation), capped at 30 Hz, for 10 s at 20%.

| | Value |
|---|---|
| Achieved rate | **16.0 Hz** (160 loops in 10 s; never reached the 30 Hz cap) |
| Command round trip | median 60 ms, avg 62.6, 95th pct 90, max 118 ms |
| Fresh IMU data | new reading on 100% of loops |
| Final stop | 60 ms (one round trip: nothing queued) |

- **Bluetooth limits the rate, not the IMU.** One confirmed command batch takes about 60 ms.
- **Don't send non-blocking commands as fast as possible.** The first version of the test did that. Commands
  queued faster than Bluetooth could send them, and the robot kept walking after the loop ended until Ctrl-C.
- **The rate might go up** if the loop sends fewer messages per step. Each `motor_run(speed=...)` sends a set-speed
  and a run message, so a step is 4 messages. `motor_set_speed` on already-running motors, or one
  `movement_move_tank(left, right)`, may be faster. Not tested.

## What the sim uses

| | Sim | Why |
|---|---|---|
| Policy rate | 16.1 Hz (`DECIMATION = 31` × 2 ms = 62 ms) | measured 16.0 Hz |
| Action delay | 0–1 control steps (0–62 ms), uniform, redrawn every step | round trips vary 60–118 ms |
| Yaw / gyro z | rad, rad/s, + = left | convert the raw values as above |
| Motor command cap | 60% (action 1.0 = 60% = 10.45 rad/s) | sim validated only up to ~50%; see `sim_vs_real.md` |

The first CPU checks in `docs/training_notes.md` were run at the old 50 Hz with no delay.
