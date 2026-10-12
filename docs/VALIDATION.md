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

The paper (Pathak et al., 2022) shows Z500 ACC and RMSE only in figures (Figure 6(d) and Figure 13(d)), not in text, so the comparison is by eye against `outputs/skill_vs_leadtime.png`. The paper averages 36 initial conditions across 2018, while this run uses one, so the curve here will be noisier and should not be expected to match the paper's average value for value. Qualitatively, the measured curve has the expected shape: ACC near 1 at the start, a smooth decline, crossing 0.6 at about 7 days, and RMSE growing steadily with lead time.

COMPARISON TO FILL IN: look at the two figures side by side and add two or three sentences on where the curves agree or differ, and whether the difference is explainable by one initial condition versus 36.

## Repeatability

REPEAT RUN TO FILL IN: job ID of the second run, and the output of `scripts/compare_runs.py` between the two CSVs.

## Verdict

The pipeline ran end to end from checksummed artifacts to scored output, and the result is physically sensible: a correctly initialized state, forecast skill that decays smoothly, and a useful-skill horizon of about 7 days for Z500. This supports the claim that the workflow is working, not that the paper's headline numbers are reproduced. To trust the numbers more, run many initial conditions and average, confirm the calendar dates of the 40 steps, and repeat on a different GPU type to measure numerical drift.
