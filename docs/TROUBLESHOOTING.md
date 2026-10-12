# Troubleshooting

## Incidents actually hit while building this workflow (2026-10-12)

| What happened | Diagnosis | Fix |
|---|---|---|
| `module load miniforge` printed a conda `ParserError` pointing at `~/.condarc` line 2 | Config file had been appended to with malformed YAML | Backed it up, rewrote it with `>` (not `>>`) as a single valid `envs_dirs:` block |
| `wget` of `ccai_demo.tar` died at about 95% with `Exit 137` | Login node killed a long, heavy process (SIGKILL) | Resumed with `wget -c`; for repeat failures run the download in an `srun` session |
| Home directory at 97% of its 30 GB quota during env build | `du`/quota output; pip and conda caches in home | `pip cache purge` and `conda clean --all -y` freed about 7 GB; env and artifacts live on scratch |
| `tar -tv ccai_demo.tar` errored | The archive is gzip-compressed | Use `tar -tzf` / `tar -xzf` |
| My first plan assumed Globus transfers and a different bundle layout | Listing the NERSC portal and the tar showed the real layout (`ccai_demo/...`) and that plain HTTPS works | Rewrote `download_artifacts.sh` and all paths; added checksums |
| Default config would run off the end of the data | Measured `2018.h5` has 40 steps while `AFNO.yaml` defaults to 41 | `configure_paths.py` sets `prediction_length` from the data shape |

Job-script pitfalls found by review before the first submission (fixed in `slurm/run_inference.sbatch`): `$0` inside a Slurm job is a spool copy, so paths come from `$SLURM_SUBMIT_DIR`; relative `#SBATCH --output` is relative to the submit directory, so submit from the repo root; `set -u` breaks `conda activate`.

Findings from the first real run (job 6205007, 1 min 35 s, completed):

- Peak GPU memory was 11,292 MiB, not the 2 to 4 GB estimated in the Model Selection Report. The likely cause is that the run keeps the full predicted and ground-truth fields in memory for saving (the output file is 6.6 GB), which the estimate did not account for; this was not profiled. It fits on a 16 GB V100 but may not fit on a smaller card. The output file is 6.6 GB.
- The first version of the job script copied the 6.6 GB prediction file into `outputs/`. It now stays on scratch and a small CSV of Z500 skill is committed instead.
- `git pull` on ICE refused to merge because the ICE copy of the job script had a local account edit. Fixed with `git checkout -- slurm/run_inference.sbatch` (the repo version already contains `--account=coc`).

## Known risks

- **Stats/checkpoint mismatch:** use `stats_v0` exactly as shipped. If ACC is out of [-1, 1] or very low at 24 h, check this first.
- **Version drift:** `inference.py` needs numpy<2 and torch<2.6; `smoke_test.py` checks both.
- **Numerical differences:** different GPUs and cuFFT versions will not reproduce the paper to the digit. Compare curve shape.
- **Job limits:** PACE-ICE caps roughly 50 GPU jobs and 32 GPU-hours per user; this run uses well under an hour.
- **Old Docker base image:** the official image predates Hopper GPUs, which is why the environment pins newer wheels (`environment/VERSION_NOTES.md`).
- **GPU memory:** about 11.3 GiB peak with `--vis`; request a GPU with at least 16 GB.

## Cleanup

- Per-run output on scratch: `rm -rf ~/scratch/fcn-run-<jobid>` after the plot and logs are copied into this repo.
- Artifacts: `rm -rf ~/scratch/fcn-artifacts` (re-fetch with `scripts/download_artifacts.sh`). They are not committed.
- Environment: `conda env remove -p ~/scratch/.conda/envs/fourcastnet-inference`.
- Caches: `pip cache purge; conda clean --all -y` to protect the home quota.
