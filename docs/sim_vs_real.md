# Sim vs real: 5 s straight walk

Both motors get the same speed command for 5 s. The real robot is measured with
a tape measure. The sim numbers come from `scripts/sim_check.py --speed-frac <f>`.

**Status (2026-11-16): hard-floor data in; partly fitted.** Chassis/hub/beam
friction 0.3 and torque limit 0.032 N·m bring 25% and 50% within 20% of the real
robot, and the sim legs turn at the measured speed at all three settings. **100%
is still 2× too far**, so the ~30% target is not met. No setting tried fixes
that without making the legs turn slower than the encoders measured (see
"What was tried").

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

Still placeholders: how the 97 g splits between chassis and legs, kv, armature,
damping. Fitted on 2026-11-16 (below): chassis/hub/beam friction 0.3, torque
limit 0.032 N·m. Foot friction stays 0.8.

## Hard-floor runs (2026-11-16)

`scripts/real_walk_test.py <pct> --home`, 5 s, legs homed to straight forward,
3 runs per speed. Raw encoder logs: `data/walk_<pct>pct_20261116-*.csv`.
The surface type was not recorded.

| Speed | Forward (cm) | Mean | Drift (cm, all to the LEFT) | Avg leg RPM (5 s) | Steady RPM | Revs in 5 s | cm per rev |
|---|---|---|---|---|---|---|---|
| 25% | 58, 52, 50 | 53.3 | 4, 3, 1 | 41.5 (100% of cmd) | 41.3–41.8 | 3.46 | 15.4 |
| 50% | 67, 48, 67 | 60.7 | 16, 36, 1 | 81.3 (98%) | 83.0–83.7 | 6.78 | 9.0 |
| 100% | 52, 48, 44 | 48.0 | 5, 3, 1 | 158.4 (95%) | 165.8–167.3 | 13.20 | 3.6 |

- Past 50% the real robot gains nothing: 100% goes slightly *less* far than 50%.
- The hub holds the commanded speed under load: steady RPM is 100% of the command
  at every speed. The 95% average at 100% comes from spin-up at the start, not
  from the motor slowing under load.
- Distance per leg revolution collapses with speed (15.4 → 9.0 → 3.6 cm/rev). The
  extra leg turns at high speed are lost to slipping or hopping, not the motor.
- Drift is always to the left even though L/R speed matches within 0.5%. So the
  asymmetry is in the contact or the build, not the motors. The 50% runs vary a lot
  (1–36 cm). Final heading was not recorded.

## Results: sim before / after fitting

Sim = `scripts/sim_check.py --speed-frac <f> --duration 5` (symmetric model, so
sim drift and yaw are 0.0 throughout).

| Speed | Real mean | Sim before (body μ 0.8, τ 0.3) | Sim after (body μ 0.3, τ 0.032) | Sim revs after (real) |
|---|---|---|---|---|
| 25% | 53.3 cm | 26.1 cm (−51%) | 44.9 cm (**−16%**) | 3.34 (3.46) |
| 50% | 60.7 cm | 31.6 cm (−48%) | 72.7 cm (**+20%**) | 6.70 (6.78) |
| 100% | 48.0 cm | 57.7 cm (+20%) | 95.9 cm (**+100%**) | 13.19 (13.20) |

Before, the sim fell 51% short at 25%. After, 25% and 50% are within 20% (the
50% value is inside the real run-to-run spread of 48–67 cm). 100% overshoots by 2×.

### What each step did

1. **Chassis/hub/beam friction 0.8 → 0.3** (foot stays 0.8): 25% −51% → −15%, 50%
   −48% → +19%, but 100% +20% → +100%. Lower body friction lets the chassis slide,
   which helps at every speed, and most at 100%.
2. **Torque limit from the under-load RPM:** 0.032 N·m makes the sim's 5 s average
   at 100% 157–159 RPM, matching the real 158.4. It barely changes distance: from
   0.3 down to 0.03 N·m, 100% stays at 92–99 cm. The 97 g robot needs very little
   torque, so a torque limit cannot produce the real plateau. Below ~0.02 N·m the
   legs slow far below the measured speed.

### What was tried (all 5 s, all three speeds)

Sweep scripts were scratch files, not committed.

| Knob | Range | Best worst-case error | Problem |
|---|---|---|---|
| Body friction (τ 0.3) | 0.1–0.8 | 49% | 100% always overshoots once 25% is close |
| Torque limit (body 0.3) | 0.015–0.3 N·m | 63% (at 0.015) | 0.015 only "works" by slowing the legs to 93 RPM at 100% |
| Body × foot friction (0.6–1.0) × contact solref | 75 combos | 35% | needs foot 0.6; 25% falls 35% short |
| Rotor armature × leg mass split | arm up to 2e-2, beam 7–17 g | 25–30% | **only by slowing the legs 35–75% below the measured RPM**. Rejected |
| Same, with sim leg RPM within 5% of real | 120 combos | **56%** | 100% stays ~75 cm or more |

With the legs held to the measured speed, no combination of friction, torque,
armature or leg mass gets 100% within 30%. The sim is missing a loss that grows
with leg speed.

### Ideas for the 100% gap (not done)

- Film a 100% run in slow motion: do the feet slip, does the chassis hop, does it bounce?
  That tells which mechanism to model.
- Candidates: softer, more dissipative foot contact (solref/solimp per foot), compliance
  in the beam-to-hub joint, or speed-dependent friction. All are new model features, not
  parameter changes.

### Validated range: 0–60% (decision 2026-11-16)

This tuning is accepted and `walker.xml` is frozen. The sim is only trusted where it was checked against
the real robot:

- **25% and 50% are validated:** within 20%, with leg revolutions within 3% of the encoders.
- **100% is outside the validated range.** The real robot goes no farther at 100% than at 50% (48 vs
  60.7 cm in 5 s), losing the extra leg turns to slip or hops, while the sim keeps speeding up (95.9 cm, 2×
  real). A policy trained there would learn to walk fast in a way the real robot can't.
- **So the policy's motor commands are capped at 60% of max speed,** in training and in `policy_io.json`
  for the real robot (action 1.0 = 60% = 10.45 rad/s). 60% is a small step beyond the 50% check;
  nothing above it is used.

### What this meant for training (written before the 60% cap)

- The capped reward keeps the policy near 0.08 m/s. The real robot already goes about
  0.107 m/s at 25% and 0.12 m/s at 50%, and the fitted sim about 0.09 / 0.145 m/s. So the cap
  now sits below 25% throttle. Revisit `V_CAP` before the GPU run.
- The sim matches best at 25–50%. Keeping the policy there keeps it in the part of
  the model that agrees with the real robot.
- The torque-scale randomization (±20% of 0.032 N·m) now spans 0.026–0.038 N·m. Leg
  speed stays within about 4% of the measured value across that range.

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

**Old model (body friction 0.8, torque limit 0.3 N·m), before the 2026-11-16 fit.**
Sim forward distance in cm, 5 s. This is sim-only
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

## Hard-floor protocol

Followed on 2026-11-16, except that the surface and the final heading were not
recorded. Record both next time.

- Use a hard, flat floor. Write down the surface (tile, wood, concrete).
- Start with both legs pointing forward. Measure from the same point on the robot each time.
- Record forward distance, sideways drift, and final heading, with units on every number.
- Do at least 3 runs per speed so run-to-run spread is visible.
- Only after that: tune foot friction against the 50% runs, then check 25% and 100%.
