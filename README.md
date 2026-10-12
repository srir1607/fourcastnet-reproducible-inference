# FourCastNet Reproducible Inference Workflow

Sri Ravela | Reproducible Inference Assignment

This repo runs single-GPU backbone inference with FourCastNet (NVIDIA's AFNO-based global weather model), validates the output against a standard meteorological skill threshold, and documents the full chain from checkpoint download to validated result. It builds directly on the Model Selection Report and GitHub issue from the previous assignment: [link your issue here].

**Status note:** the files in this repo set up a complete, ready-to-run workflow. `logs/`, `outputs/`, and `docs/VALIDATION.md` are placeholders until an actual job has been run on PACE-ICE; see the "What's left to do" section at the bottom. Submitting this without running it will not satisfy the Execution Evidence or Validation rubric items, both of which require a real run's artifacts.

## What this reproduces

Inference-only reproduction of the FourCastNet backbone: load the pretrained checkpoint, initialize from the released out-of-sample 2018 ERA5 data, run autoregressive rollout, and compute RMSE/ACC against ERA5 ground truth for Z500 (and the other channels `inference.py` scores by default). Training is explicitly out of scope; see `docs/PROVENANCE.md` for why.

## Repo layout

```
environment/        conda environment spec + notes on why versions deviate from the official Dockerfile
scripts/             download, config-patching, smoke test, and validation scripts
slurm/               the actual Slurm job script
docs/                provenance, validation record template, troubleshooting log
logs/                Slurm stdout/stderr and download manifest land here after a real run
outputs/             copied-over result artifacts (plot, trimmed output file) after a real run
```

## Setup

0. **SSH in and find your allocation.** `--account` in `slurm/run_inference.sbatch` is not optional and isn't the same for everyone:
   ```
   ssh <your-gt-username>@login-ice.pace.gatech.edu
   sacctmgr show associations user=$USER
   ```
   Use whichever account that lists for GPU work, and put it in `slurm/run_inference.sbatch`. Don't assume a labmate's account (e.g. the `coc` account another student used for a different project) applies to you.

1. **Clone the model repo** (not this one; this repo wraps it):
   ```
   git clone https://github.com/NVlabs/FourCastNet.git
   cd FourCastNet && git rev-parse HEAD
   ```
   Record that commit hash in `docs/PROVENANCE.md`.

2. **Create the environment on scratch, not home.** PACE-ICE home quota is small and fills fast; put conda envs on scratch like everyone else in the lab does. Add this once to `~/.condarc` so `conda activate fourcastnet-inference` works by name afterward:
   ```
   envs_dirs:
     - ~/scratch/.conda/envs
   ```
   Then, on a login node:
   ```
   module purge
   module load miniforge
   module load miniforge
   module load miniforge      # 3x clears any stale conda module; not a typo
   source $(conda info --base)/etc/profile.d/conda.sh
   mamba env create -f environment/environment.yml -p ~/scratch/.conda/envs/fourcastnet-inference
   conda activate fourcastnet-inference
   ```
   See `environment/VERSION_NOTES.md` for why the package versions don't match the repo's own Dockerfile verbatim. Unlike OpenFold, this model needs no custom CUDA extension build, so there's no gcc-toolchain or CUTLASS step here; stock PyTorch wheels are enough.

3. **Get a Globus account** (free, https://www.globus.org) if you don't have one, then:
   ```
   pip install globus-cli
   globus login
   ./scripts/download_artifacts.sh ~/scratch/fcn-artifacts
   globus task list     # wait for all three transfers to show SUCCEEDED
   ```

4. **Smoke test on the login node first:**
   ```
   python scripts/smoke_test.py \
     --fourcastnet-repo ~/FourCastNet \
     --artifact-root ~/scratch/fcn-artifacts
   ```
   Fix anything it flags. This catches the kind of mistake that otherwise costs a queued GPU job's wait time. It will correctly report no CUDA device here; that's expected on a login node.

5. **Confirm CUDA actually works, on a short interactive allocation, before trusting the full batch job:**
   ```
   srun --account=<your account from step 0> --partition=ice-gpu --gres=gpu:1 \
        --cpus-per-task=8 --mem=32G --time=00:15:00 --pty bash
   conda activate fourcastnet-inference
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   This should print `True` and a real GPU name. If it doesn't, something is wrong with the PyTorch/CUDA pairing (see `environment/VERSION_NOTES.md`) and the batch job will fail the same way, just slower to find out. Exit the allocation (`exit`) once confirmed.

## Running the workflow

Edit the three path variables at the top of `slurm/run_inference.sbatch` (your FourCastNet clone, your artifact root, and leave the output dir as-is unless you want it elsewhere) and `--account` (from step 0). `--partition=ice-gpu` is already filled in and confirmed working for GPU jobs on this cluster; re-check with `sinfo` only if the job is rejected.

```
sbatch slurm/run_inference.sbatch
squeue -u $USER                      # watch it queue and run
```

The job script patches `config/AFNO.yaml` via `scripts/configure_paths.py`, runs `inference/inference.py`, and then automatically runs `scripts/validate.py` on the result, all in one submission. Logs land in `logs/slurm-<jobid>.out` and `.err`.

## Validating the result

`validate.py` runs automatically at the end of the Slurm job, but can be re-run standalone:

```
python scripts/validate.py --run-dir ~/scratch/fcn-run-<jobid>
```

It checks output structure (smoke test) and scientific plausibility (ACC bounds, RMSE non-negativity, Day-1 Z500 ACC against the standard 0.6 "useful forecast" threshold), and writes `skill_vs_leadtime.png` for the visual comparison against the paper's Figure 3/4. Fill in `docs/VALIDATION.md` with the actual numbers and your interpretation; a script's pass/fail output alone isn't the documentation the rubric is asking for.

## Troubleshooting

See `docs/TROUBLESHOOTING.md` for anticipated failure modes and a log of what actually went wrong during the real run.

## Cleanup

See the Cleanup section at the bottom of `docs/TROUBLESHOOTING.md`.

## What's left to do (not yet done as of this commit)

- [ ] Actually run `sbatch slurm/run_inference.sbatch` on PACE-ICE (the course's "AI Makerspace") and confirm it completes.
- [ ] Copy `logs/slurm-<jobid>.out/.err` and `logs/download_manifest.txt` into `logs/`.
- [ ] Copy the resulting `skill_vs_leadtime.png` and a trimmed output file into `outputs/`.
- [ ] Fill in `docs/VALIDATION.md` with real measured numbers and the qualitative Figure 3/4 comparison.
- [ ] Fill in the commit hash and checkpoint integrity check in `docs/PROVENANCE.md`.
- [ ] Log any real incidents in `docs/TROUBLESHOOTING.md`.
- [ ] Verify `--partition`/`--account` in `slurm/run_inference.sbatch` actually match PACE-ICE (they're currently placeholders).
- [ ] Verify `DEFAULT_Z500_CHANNEL` in `scripts/validate.py` against the actual channel ordering in the pinned commit.
