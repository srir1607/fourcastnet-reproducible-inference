# Troubleshooting

Known risks, carried over from the Model Selection Report's Risks section, plus space to log what actually happened. Update this file with real entries once you run the job; a generic risk list without an actual-incident log is exactly what the rubric's "Significant Gaps" tier for Execution Evidence describes.

## Known risks (anticipated before running)

**Globus/CDS account friction.** Both require account setup with variable turnaround. If `download_artifacts.sh` hangs or fails authentication, check `globus login` was completed interactively first; it cannot be scripted.

**Normalization/variable-selection mismatch.** The released `stats_v0` files must be used as-is, not recomputed. If `scripts/validate.py`'s ACC bounds check fails (values outside [-1, 1] or implausibly low), check first that `configure_paths.py` pointed at the stats that shipped with this exact checkpoint release, not a mismatched set.

**Numerical drift from GPU/cuFFT version differences.** Don't expect digit-for-digit agreement with the paper. Compare the shape of the measured skill curve (`skill_vs_leadtime.png`), not exact values.

**PACE-ICE job caps.** At the time of the Model Selection Report, the known cap was 50 queued/running GPU jobs and 32 concurrent GPU-hours per user. This run's estimated usage (well under an hour) is far inside both; a larger multi-seed sweep later would need to be sequenced.

**Old base image vs. new GPU generation.** See `environment/VERSION_NOTES.md`: the official Dockerfile's PyTorch/CUDA pairing predates Hopper (H200) support, which is why `environment.yml` pins newer versions instead of reproducing the Dockerfile verbatim.

## Actual incidents (fill in during/after the real run)

| Date | What happened | How it was diagnosed | Fix / workaround |
|---|---|---|---|
| | | | |

## Cleanup

- Remove the `$OUTPUT_DIR` scratch directory from `slurm/run_inference.sbatch` once results are copied into `outputs/` and no longer needed on scratch (scratch storage on most Slurm clusters is not backed up and may be purged automatically).
- Downloaded artifacts in `$ARTIFACT_ROOT` can be deleted and re-fetched later via `scripts/download_artifacts.sh`; they aren't committed to this repo (see README's note on large files).
- `conda env remove -n fourcastnet-inference` if you want to fully tear down the environment.
