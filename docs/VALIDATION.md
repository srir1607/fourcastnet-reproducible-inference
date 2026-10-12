# Validation record

Run recorded here: Slurm job 6205007 on node `atl1-1-02-011-31-0` (Tesla V100-PCIE-16GB), account `coc`. Wall time 95 s for the job, 77 s for inference. Peak GPU memory 11,292 MiB (sampled twice per second, `logs/gpu_usage_6205007.csv`). Full output: `logs/slurm-6205007.out` and `.err`. The CSV and `validate.py --csv-out` option were added after this job ran (they only add outputs; the checks are unchanged), so the CSV for this job was regenerated from its saved output.

## Smoke tests

| Check | Result |
|---|---|
| Pre-flight `scripts/smoke_test.py` (imports, numpy<2, torch<2.6, artifact files, checkpoint size, data shape) | pass (`logs/smoke_test.txt`) |
| CUDA visible on a GPU node | pass, Tesla V100-PCIE-16GB (`logs/cuda_check.txt`) |
| Artifact checksums match `docs/CHECKSUMS.sha256` | see `logs/download_manifest.txt` |
| Output HDF5 found with `rmse` and `acc` | pass (also `acc_coarse`, `ground_truth`, `predicted` and others) |
| Output arrays finite and non-empty | pass; both have shape (1, 40, 20) |

## Substantive checks (Z500, channel 14)

| Check | Expected | Measured | Result |
|---|---|---|---|
| ACC within [-1, 1], all 20 channels | by definition | min 0.0922, max 1.0000 | pass |
| RMSE >= 0 | by definition | min 0.0000 | pass |
| ACC at lead 0 | about 1 (initial condition) | 1.0000 | pass |
| ACC at 24 h | >= 0.6 | 0.9952 | pass |
| ACC decays and RMSE grows with lead time | monotone trend | ACC 0.9952 to 0.2874, RMSE 79.61 to 1069.85 (24 h to 234 h) | pass |

Z500 skill by lead time: 24 h ACC 0.9952, RMSE 79.61; 48 h ACC 0.9813, RMSE 156.09; 72 h ACC 0.9437, RMSE 271.82; 120 h ACC 0.8252, RMSE 478.35. From the inference log, ACC falls from 0.613 at 168 h to 0.574 at 174 h, so it crosses 0.6 at about 7 days. The final value at 234 h (step 39) is 0.2874.

The 24 h check at 0.6 is a floor that a working model clears easily here, so the lead time at which ACC crosses 0.6 is the more informative number. The 0.6 threshold itself is the long-standing operational convention for a useful synoptic-scale forecast. It is a verification standard, not a number from the FourCastNet paper. RMSE units are those of the Z500 field as scored by `inference.py` and were not independently verified.

## Comparison with the paper
The closest figure in Pathak et al. (2022) is Figure 7(a), which plots Z500 ACC and RMSE (m^2/s^2) against lead time for the unperturbed FourCastNet control, averaged over many forecasts across 2018 (the paper lists about 32 to 36 initial conditions). Paper values below are read off the figure by eye and are approximate. This run uses a single initial condition and covers 234 h, almost the same horizon as the figure.

The two agree early. Both start at an ACC of about 1 and stay near 0.99 through 24 to 48 h (0.995 at 24 h and 0.981 at 48 h here), both curves decline smoothly, and RMSE is on the same scale as the paper's (hundreds of m^2/s^2 at 4 to 8 days), so the scoring appears to be on the same units.

They disagree later. This run falls faster after about 72 h. At 96 h ACC is 0.878 here versus roughly 0.93 in the paper's control, at 144 h 0.739 versus roughly 0.83, at 192 h 0.443 versus roughly 0.71, and at 234 h 0.287 versus roughly 0.61. This run crosses 0.6 at about 170 h, while the paper's control stays above 0.6 across the whole plotted range. RMSE shows the same pattern, 1070 at 234 h here versus roughly 880 in the paper.

This run therefore does not reproduce the paper's long-lead skill. A single initial condition can differ a lot from an average over many, and ACC at 5 to 10 days is the most variable part of the curve, so a below-average case is a plausible explanation. This workflow could not test that, because the bundled evaluation file holds only one 10-day window. A difference between this checkpoint and the one behind the paper's figure, or in how ACC is computed, is not ruled out. Agreement through about 2 days is supported. The shortfall beyond about 4 days is an open discrepancy.
## Repeatability

A second run (job 6205661, same node and GPU, 52 s of walltime) was compared to job 6205007 with `scripts/compare_runs.py outputs/z500_skill_6205007.csv outputs/z500_skill_6205661.csv`. Over all 40 steps the maximum absolute difference was 0.000e+00 for both ACC and RMSE, so the pipeline is deterministic on this hardware. Both jobs used the same node and GPU, so this does not measure drift between different GPU types or cuFFT versions.

## Verdict
The pipeline ran end to end from checksummed artifacts to scored output twice with identical results, and the output is physically sensible: a correct initial state, smoothly decaying skill, and agreement with the paper's Figure 7(a) through about 2 days. Beyond about 4 days skill is clearly below the paper's average curve, and this run cannot say whether that is single-initialization variance or a real discrepancy. It supports the claim that the workflow works and produces a sensible forecast, not that the paper's long-lead skill is reproduced. To resolve it, evaluate many independent initial conditions (this needs the full 2018 file, not the demo bundle's 40-step window), confirm the calendar dates of these 40 steps, and repeat on a different GPU type.

