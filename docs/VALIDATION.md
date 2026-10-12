# Validation record

Fill this in after an actual run. `scripts/validate.py` prints most of these numbers; this file is where you record them alongside your interpretation, which the rubric specifically asks for ("sufficient explanation of why the output is valid," not just a pasted number).

## Smoke test

| Check | Result |
|---|---|
| Output HDF5 file found | pass / fail |
| Expected datasets present (`rmse`, `acc`) | pass / fail |
| No NaN/Inf in output arrays | pass / fail |

## Substantive validation

| Check | Expected | Measured | Pass/fail |
|---|---|---|---|
| ACC within [-1, 1] | by definition | | |
| RMSE non-negative | by definition | | |
| Day-1 Z500 ACC vs. skill threshold | ACC >= 0.6 (conventional "useful forecast" threshold in synoptic-scale NWP verification, see note below) | | |

**Why 0.6:** an anomaly correlation coefficient of 0.6 is the long-standing operational threshold meteorologists use to call a synoptic-scale forecast "useful" rather than noise; it predates FourCastNet and isn't a paper-specific number, it's a standard verification convention. If the measured Day-1 Z500 ACC clears it, that's independent evidence the forecast is doing something real, not just an artifact of the pipeline running without crashing.

## Qualitative comparison against the paper

Open `<run_dir>/skill_vs_leadtime.png` next to Figure 3 (RMSE) and Figure 4 (ACC) of Pathak et al. 2022 (https://arxiv.org/abs/2202.11214) and describe what you see:

- Does ACC decrease with lead time in both, at a roughly similar rate?
- Does RMSE increase with lead time in both?
- Where do they diverge, and is that divergence explainable (different GPU/cuFFT numerics, a single initialization vs. the paper's full out-of-sample average, etc.) per the Risks section of the Model Selection Report?

(Write 3-5 sentences here after the run. Don't claim digit-for-digit agreement; the Model Selection Report already scoped this to shape/ordering agreement, not exact reproduction.)

## Overall verdict

State plainly: did this run produce a valid, working FourCastNet inference pipeline? What would someone need to change to trust these numbers more (more initializations, the full out-of-sample average instead of one date, etc.)?
