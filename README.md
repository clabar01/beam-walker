# beam-walker

ME193 robotics project. A LEGO Education Double Motor (45651) has a LEGO Technic
beam on each output shaft in place of a wheel. Turning the beams makes the robot
shuffle (walk) along the floor. The plan is to train a drive-straight policy
with [MJLab](https://github.com/mujocolab/mjlab) on the Tufts HPC and then run
it on the real robot.

## Layout

| Path | What |
|---|---|
| `models/walker.xml` | MuJoCo MJCF model: chassis (free joint), two leg hinges, velocity actuators, sensors |
| `scripts/sim_check.py` | Headless sanity test: drives both motors at the same speed, prints x/y/yaw, writes `videos/sim_check.mp4` |
| `scripts/measure_motor.py` | **Moves the motors.** Measures each output's RPM at 50%/100% (legs off the ground), logs to `data/motor_rpm.csv` |
| `scripts/real_walk_test.py` | **Moves the robot.** Walks at 25/50/100% for exactly 5 s so you can tape-measure the distance and compare with the sim |
| `scripts/lego_common.py` | Double Motor connection (card 0999, ORANGE) and the per-side forward directions |
| `data/` | Measurements from the real robot |
| `videos/` | Rendered rollouts |

## Model at a glance

- Frame: origin at the center of the chassis bottom face, +X forward, +Y left, +Z up. Axles run along Y.
- Actuators: `motor_left` and `motor_right` are velocity actuators. ctrl is in rad/s, in the range ±15.7.
  A positive command rolls the legs forward.
- Sensors match the real hardware (built-in IMU and absolute encoders):
  `chassis_pos`, `chassis_quat`, `chassis_gyro`, `hinge_{left,right}_{pos,vel}`.
- **Placeholder values** (mass, max speed, torque, kv, friction) are listed at
  the top of `models/walker.xml`. Each one is tagged `PH_*` so you can search for it.

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/sim_check.py            # 5 s at 50% speed
.venv/bin/python scripts/sim_check.py --help     # options
```

To view the model interactively, run `.venv/bin/python -m mujoco.viewer --mjcf models/walker.xml`
(on macOS, use `mjpython` in place of `python` for the viewer).

## Real robot

Both scripts connect, then wait for Enter before anything moves (`--yes` skips the prompt).

```bash
.venv/bin/python scripts/measure_motor.py        # lift the robot first
.venv/bin/python scripts/real_walk_test.py 50    # on the floor; add --reverse if it walks backward
```

## TODO

- [ ] Weigh the robot and the legs, then update the masses
- [ ] Measure the no-load speed and the stall torque
- [ ] Measure the friction between the feet and the floor
- [ ] Set up the MJLab environment and the drive-straight reward
