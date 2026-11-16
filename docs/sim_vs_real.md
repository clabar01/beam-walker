# Sim vs real: 5 s straight walk

Both motors get the same speed command for 5 s. The real robot was measured
with a tape measure. The sim numbers come from `scripts/sim_check.py --speed-frac <f>`.

## Measured parameters (2026-11-15)

| Parameter | Value | Source |
|---|---|---|
| Total mass | 97 g | scale |
| No-load speed at 100% | 166.3 RPM = 17.42 rad/s (both outputs) | `scripts/measure_motor.py`, `data/motor_rpm.csv` |
| No-load speed at 50% | 82.9 RPM = 8.68 rad/s (both outputs) | same; speed is linear in % |
| L/R speed mismatch | 0.0% at 50% and 100% | same |
| L/R power mismatch | right +2.4% (50%), +4.8% (100%) | same; **not modeled yet** |

Still placeholders: how the 97 g splits between chassis and legs, motor torque
limit (±0.3 N·m), kv, armature, damping, and friction on non-foot geoms (0.8).

## Results

Error = (sim − real) / real. Foot friction is the only value tuned so far.

| Speed | Real forward | Real drift | Sim forward, foot μ = 0.8 (before) | Sim forward, **foot μ = 0.2 (tuned)** | Sim drift / yaw |
|---|---|---|---|---|---|
| 25% | 12 in = 30.5 cm | 1 in = 2.5 cm | 26.1 cm (−14%) | **14.4 cm (−53%)** | 0.0 cm / 0.0° |
| 50% | 7 in = 17.8 cm | ~3 cm (assumed cm) | 31.6 cm (+78%) | **18.1 cm (+2%)** ✅ | 0.0 cm / 0.0° |
| 100% | 8 in = 20.3 cm | 8 (units unclear: in or cm) | 57.7 cm (+184%) | **23.7 cm (+17%)** | 0.0 cm / 0.0° |

The 50% target was within ±20% (14.2–21.3 cm). It's met at **+2%**.

## Foot friction sweep (sim forward distance, cm)

| Foot μ | 25% | 50% | 100% |
|---|---|---|---|
| 0.05 | 13.0 | 17.9 | 21.0 |
| 0.10 | 13.5 | 17.8 | 21.0 |
| 0.15 | 13.9 | 18.2 | 21.9 |
| **0.20** | **14.3** | **18.1** | **23.7** |
| 0.30 | 15.5 | 19.2 | 27.4 |
| 0.40 | 16.5 | 20.4 | 32.2 |
| 0.50 | 18.0 | 22.1 | 38.9 |
| 0.60 | 20.2 | 24.1 | 44.8 |
| 0.80 | 26.1 | 31.6 | 57.7 |
| 1.00 | 37.6 | 48.3 | 69.4 |
| 1.20 | 41.0 | 63.0 | 79.1 |
| 1.50 | 41.7 | 70.4 | 89.2 |
| *real* | *30.5* | *17.8* | *20.3* |

Every value from 0.05 to 0.40 puts 50% inside ±20%. I chose 0.20 because it's
the highest value that also keeps 100% within 20%, so it's the closest to
typical plastic-on-floor friction.

## Open issues

1. **25% doesn't match at any friction that fits 50%.** On the real robot, 25%
   went further than 50% or 100% (30.5 cm vs 17.8 and 20.3). The sim travels
   further per second as speed goes up. So something that depends on speed is
   missing from the model: probably the stance phase at low speed (legs push
   instead of slapping the floor), or how much torque the motor has. Friction
   alone can't fix this. Next things to fit: torque limit (`PH_TORQUE`) and
   `kv`. Also repeat the 25% run to rule out a one-off.
2. **Drift isn't modeled.** The sim is symmetric and the speed mismatch is 0, so
   sim drift is 0. Real drift goes up with speed (2.5 cm → ~3 cm → 8 in/cm).
   That fits the right side needing more effort as speed rises. Candidates:
   different torque limits per side, or friction per foot.
3. **Real-run units:** the 50% drift was written as "~3 c" and the 100% drift as
   "8incm". I assumed cm for 50%. 100% is left unresolved.
4. One real run per speed. Repeats would show run-to-run spread.
