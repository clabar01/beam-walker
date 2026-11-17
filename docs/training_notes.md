# Training notes: drive-straight policy (mjlab)

2026-11-15. CPU check runs on the laptop, not the GPU run. The GPU job has **not** been submitted yet.

## Result: no stall

The 256-env x 150-iteration CPU check (reward: capped forward velocity, action rate -0.1) walks the full 10 s
on the lopsided test robot. It did not stall, so I did **not** switch to the velocity-tracking reward
or change the action-rate penalty from -0.1. The earlier stall (60 iterations, 128 envs) was too little
training.

## 10 s rollout on the lopsided test robot

Test robot: left foot friction 0.6 vs right 1.0, right motor speed gain 0.9, nominal mass, legs start at 0.

| | Trained policy | Baseline (both motors 50%) |
|---|---|---|
| Walks the full 10 s | yes, no flip | yes |
| Forward | 84.4 cm | 81.6 cm |
| Final lateral offset | **-5.9 cm** | -41.3 cm |
| Final heading error | **-3.4 deg** | -36.7 deg |
| Heading during the run | within +-3.4 deg | -45 to -22 deg |

## Reward terms

Weighted, per second of episode (`forward_vel` maxes out at 1.0):

| Term | Iteration 0 | 75 | 149 |
|---|---|---|---|
| Mean return | -0.02 | 5.12 | 6.34 |
| forward_vel | 0.013 | 0.754 | 0.822 |
| sideways_vel | -0.007 | -0.120 | -0.092 |
| heading_error | -0.001 | -0.007 | -0.003 |
| yaw_rate | -0.002 | -0.042 | -0.046 |
| action_rate | -0.005 | -0.061 | -0.050 |

Every episode ran the full 500 steps (10 s); 0% flipped. Return was still rising at iteration 149.
Curves: [training_cpu_check_reward_curves.png](training_cpu_check_reward_curves.png).

## Open points

- The commands are smoother than at -0.01 but still change a lot (one leg swings about -0.8 to +0.5 from second to
  second). If that is a problem on the real robot, try action rate -0.03 or the velocity-tracking reward
  `exp(-(v - 0.07)^2 / 0.02^2)` on the GPU run.
- Check the IMU gyro units on the real robot (legoeducation does not document them); yaw is in degrees.
- Next: GPU run (2048 envs x 300 iterations), command in the README.

---

# 2026-11-16: CPU check at the real control rate (passed)

Setup changes since the run above: policy at **16.1 Hz** (measured real rate) with a random **0–1 step command delay**,
motor commands **capped at 60%**, fitted `walker.xml` (body friction 0.3, torque limit 0.032 N·m), and a test robot
that drifts **left** like the real one (right foot friction 0.6, left motor 10% slow). Same reward (capped forward
velocity, action rate -0.1). 256 envs x 150 iterations, CPU.

## 10 s rollout on the left-drifting test robot

| | Trained policy | Baseline (both motors 50%) |
|---|---|---|
| Walks the full 10 s | yes, no stall, no flip | yes |
| Forward | 138.8 cm | 145.2 cm |
| Final lateral offset | **+7.2 cm** | +51.4 cm |
| Final heading error | **-14.8 deg** | +18.9 deg |
| Heading during the run | -14.8 to +15.4 deg | -4 to +60 deg |

The policy corrects mostly by slowing the left motor (left 23-60%, right 8-60%). In the last 3 s it overcorrects:
the left motor drops to ~25% and the heading drifts right to -15 deg. Watch for this in the GPU run.

## Reward terms

Weighted, per second of episode:

| Term | Iteration 0 | 75 | 149 |
|---|---|---|---|
| Mean return | -0.18 | 8.09 | 8.48 |
| forward_vel | 0.027 | 0.978 | 0.991 |
| sideways_vel | -0.009 | -0.073 | -0.060 |
| heading_error | -0.011 | -0.028 | -0.022 |
| yaw_rate | -0.005 | -0.052 | -0.049 |
| action_rate | -0.017 | -0.011 | -0.007 |

Episodes all ran the full 10 s (162 steps); 0% flipped.

## Open point

`forward_vel` is saturated at ~0.99: the policy already walks ~0.14 m/s, above the 0.08 m/s cap (`V_CAP`),
so speed earns nothing extra and the reward is driven by the steering terms. That was expected after the
model fit (the real robot does 0.107 m/s at 25%). **Done 2026-11-16: `V_CAP` raised to 0.12 m/s** (no CPU
recheck; judged from the GPU run). The sideways penalty keeps its old 0.08 m/s scale (`V_SIDE_SCALE`) so it
doesn't get weaker.
