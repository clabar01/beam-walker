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
| `scripts/real_walk_test.py` | **Moves the robot.** Walks at 25/50/100% for exactly 5 s and logs both legs' angles. Optional homing to straight forward (sim qpos 0) |
| `scripts/lego_common.py` | Double Motor connection (card 0999, ORANGE) and the per-side forward directions |
| `scripts/run_policy.py` | **Moves the robot.** Runs a trained policy (`walker_policy.onnx` + `policy_io.json`) at ~16 Hz, or `--baseline` 50%; `--home` first. Logs to `data/policy_run_<time>.csv` |
| `scripts/imu_rate_check.py` | **Moves the robot.** Measures the real control-loop rate (blocking commands) and the IMU gyro/yaw units. Results: `docs/real_robot_io.md` |
| `train/walker_task.py` | mjlab task `BeamWalker-DriveStraight`: observations, rewards, terminations, domain randomization |
| `train/train_walker.py` | Trains with PPO, then writes the checkpoint, TorchScript/ONNX policy, reward plot and rollout videos |
| `requirements-mjlab.txt` | Pinned training environment (Python 3.13), same versions as the cluster |
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
.venv/bin/python scripts/real_walk_test.py --calibrate-home   # once: point legs forward by hand
.venv/bin/python scripts/real_walk_test.py 50 --home  # home legs, Enter, walk 5 s; --reverse if it walks backward
```

`real_walk_test.py` logs both legs' angles to `data/walk_<speed>pct_<time>.csv`. At the end it prints
the average leg RPM under load against the commanded RPM. `--home` moves the legs
(slowly) **before** the Enter prompt.

## Training (mjlab, Tufts HPC)

The task in `train/walker_task.py` only gives the policy what the real robot can sense: leg angles
(sin/cos), leg speeds, its last command, and the Double Motor's IMU (gyro z and yaw since start, read with
`dm.imu_device` in legoeducation). Actions are the two motor speeds, capped at 60% of 17.42 rad/s
(the range where the sim matches the real robot; see `docs/sim_vs_real.md`). Each
episode randomizes foot friction, motor speed gain and torque limit separately for each side, plus chassis
mass and the starting leg angles, so the policy has left/right differences to correct.

Run on a GPU node from the repo root (`runIt.py` copies the outputs back to `mujoco-hpc/runs/<name>/`):

```bash
python3 "$HOME/Documents/ME193-Robotics/Public stuff/mujoco-hpc/runIt.py" --gpu --time 00:30:00 --mem 16G \
    --cpus 4 train/train_walker.py --with models --with train
```

Options go after the script name: `--num-envs 2048 --iterations 300` (defaults), `--no-video`.
Outputs: `walker_final.pt`, `walker_policy.pt` (TorchScript), `walker_policy.onnx`, `policy_io.json` (what
goes into the policy, in which order), `reward_curves.png`, `rollout_policy.mp4`, `rollout_baseline.mp4`, and
`rollout_summary.txt`. Both videos (30 fps, sampled from the physics; the policy acts at 16 Hz) use the
same deliberately lopsided robot (right foot friction 0.6 vs 1.0, left motor 10% slow), chosen so the
baseline drifts left like the real robot.

Quick local test before using the GPU (CPU, Python 3.13 venv from `requirements-mjlab.txt`):

```bash
.venv-mjlab/bin/python train/train_walker.py --num-envs 8 --iterations 3 --video-seconds 2
```

## TODO

- [ ] Weigh the robot and the legs, then update the masses
- [ ] Measure the no-load speed and the stall torque
- [ ] Measure the friction between the feet and the floor
- [ ] Set up the MJLab environment and the drive-straight reward
