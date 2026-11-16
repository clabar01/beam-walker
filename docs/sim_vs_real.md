# Sim vs real: 5 s straight walk

Both motors get the same speed command for 5 s. The real robot is measured with
a tape measure. The sim numbers come from `scripts/sim_check.py --speed-frac <f>`.

**Status (2026-11-15): no valid real walk data yet. Nothing is tuned.** The
first walk runs were on a soft comforter and don't count. Hard-floor runs are
next. Foot friction is back to the 0.8 placeholder.

## Measured parameters (2026-11-15)

The motor was measured with the legs off the ground, so the walking surface
doesn't affect these numbers.

| Parameter | Value | Source |
|---|---|---|
| Total mass | 97 g | scale |
| No-load speed at 100% | 166.3 RPM = 17.42 rad/s (both outputs) | `scripts/measure_motor.py`, `data/motor_rpm.csv` |
| No-load speed at 50% | 82.9 RPM = 8.68 rad/s (both outputs) | same; speed is linear in % |
| L/R speed mismatch | 0.0% at 50% and 100% | same |
| L/R power mismatch | right +2.4% (50%), +4.8% (100%) | same; **not modeled yet** |

Still placeholders: how the 97 g splits between chassis and legs, motor torque
limit (±0.3 N·m), kv, armature, damping, and all friction values (0.8).

## Results

Sim: measured mass and speed, foot friction 0.8 (untuned).

| Speed | Real forward (hard floor) | Real drift (hard floor) | Sim forward | Sim drift / yaw |
|---|---|---|---|---|
| 25% | *pending* | *pending* | 26.1 cm | 0.0 cm / 0.0° |
| 50% | *pending* | *pending* | 31.6 cm | 0.0 cm / 0.0° |
| 100% | *pending* | *pending* | 57.7 cm | 0.0 cm / 0.0° |

### ⚠️ Comforter runs (INVALID, kept for the record only)

Run on a soft comforter, so they don't represent the floor contact being
modeled. **Don't tune against these.**

| Speed | Forward | Drift |
|---|---|---|
| 25% | 12 in = 30.5 cm | 1 in = 2.5 cm |
| 50% | 7 in = 17.8 cm | ~3 (written "~3 c") |
| 100% | 8 in = 20.3 cm | 8 (written "8incm"; units unclear) |

Foot friction was briefly tuned to 0.2 to match the 50% comforter run. That
change has been **reverted**.

## Sim reference: how foot friction affects distance

Sim forward distance in cm, 5 s, all other parameters as above. This is sim-only
data and doesn't depend on the comforter runs. It shows roughly where foot
friction will need to land once hard-floor numbers are in.

| Foot μ | 25% | 50% | 100% |
|---|---|---|---|
| 0.05 | 13.0 | 17.9 | 21.0 |
| 0.10 | 13.5 | 17.8 | 21.0 |
| 0.15 | 13.9 | 18.2 | 21.9 |
| 0.20 | 14.3 | 18.1 | 23.7 |
| 0.30 | 15.5 | 19.2 | 27.4 |
| 0.40 | 16.5 | 20.4 | 32.2 |
| 0.50 | 18.0 | 22.1 | 38.9 |
| 0.60 | 20.2 | 24.1 | 44.8 |
| **0.80 (current)** | **26.1** | **31.6** | **57.7** |
| 1.00 | 37.6 | 48.3 | 69.4 |
| 1.20 | 41.0 | 63.0 | 79.1 |
| 1.50 | 41.7 | 70.4 | 89.2 |

## Hard-floor protocol (next session)

- Use a hard, flat floor. Write down the surface (tile, wood, concrete).
- Start with both legs pointing forward. Measure from the same point on the robot each time.
- Record forward distance, sideways drift, and final heading, with units on every number.
- Do at least 3 runs per speed so run-to-run spread is visible.
- Only after that: tune foot friction against the 50% runs, then check 25% and 100%.
