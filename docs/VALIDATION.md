# Validation record

Fill this in from the real run's `logs/slurm-<jobid>.out`. A script's pass/fail line alone is not the explanation the rubric asks for.

Run: job ID `<FILL>`, node `<FILL>`, GPU `<FILL>`, wall time `<FILL>` s, peak GPU memory `<FILL>` MiB (from `logs/gpu_usage_<jobid>.csv`).

## Smoke tests

| Check | Result |
|---|---|
| Pre-flight `scripts/smoke_test.py` (imports, version guards, artifacts, shape) | |
| CUDA visible on GPU node | |
| Output HDF5 found, has `rmse` and `acc` | |
| Output arrays finite and non-empty | |

Detected array layout printed by validate.py: `<FILL>`

## Substantive checks (Z500, channel 14)

| Check | Expected | Measured | Pass/fail |
|---|---|---|---|
| ACC within [-1, 1] | by definition | | |
| RMSE >= 0 | by definition | | |
| ACC at lead 0 | about 1 (>= 0.99), confirms index 0 is the initial condition | | |
| ACC at 24 h | >= 0.6 | | |
| ACC decays, RMSE grows with lead time | ACC(24 h) > ACC(last), RMSE(last) > RMSE(24 h) | | |

Skill summary (copy from the log): ACC / RMSE at 24, 48, 72, 120, 240 h: `<FILL>`

**Why 0.6:** an anomaly correlation of 0.6 is the long-standing operational convention for calling a synoptic-scale forecast useful rather than noise. It is a verification standard, not a number from the FourCastNet paper. These checks catch the common failure modes (wrong stats files, wrong channel, normalization bug, broken checkpoint load) but they cannot prove digit-level agreement with the paper.

## Comparison with the paper

Put `outputs/skill_vs_leadtime.png` next to Figures 3 (RMSE) and 4 (ACC) of Pathak et al. and write 3 to 5 sentences: does ACC fall and RMSE rise with lead time at a broadly similar rate, where do they differ, and is the difference explained by a single initialization versus the paper's average over many, or by GPU numerics.

## Verdict

State plainly whether the pipeline produced a valid, working forecast, and what would increase trust: more initializations, a longer evaluation set, a second run on a different GPU to measure run-to-run numerical drift.
