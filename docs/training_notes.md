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
