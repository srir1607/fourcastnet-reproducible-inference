# FourCastNet Reproducible Inference Workflow

Sri Ravela | Reproducible Inference Assignment

Single-GPU inference with FourCastNet (NVIDIA's AFNO-based global weather model) on PACE-ICE, with validation of the output and a documented chain from artifact download to result. It builds on the Model Selection Report and GitHub issue from the previous assignment: https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/24.

## Scope

- **What is reproduced:** one autoregressive rollout of the pretrained backbone, initialized at index 0 of the released out-of-sample 2018 ERA5 file, run for 40 six-hour states (10 days), scored against ERA5 for 20 channels. Z500 (channel 14) is the headline variable.
- **What is not:** this is a single-initialization sanity reproduction, not the paper's multi-initialization average, so numbers will be noisier than the paper's Figure 7(a). Compare curve shape, not digits. Training, the precipitation model, and the HuggingFace 26-channel release are out of scope (`docs/PROVENANCE.md`).
- **Compute needed:** one GPU with at least 16 GB (measured peak 11.3 GiB with `--vis`), 64 GB host RAM requested (21.7 GB used), about 95 s wall time on a V100.
- **Known limits of the evaluation file:** it holds 40 time steps, so the repo's default 41-step rollout would run off the end; `scripts/configure_paths.py` sets `prediction_length` from the file. The calendar dates of those 40 steps were not verified.
- **Status:** see the checklist at the bottom. Items are only ticked once real logs are committed.

## Repo layout

```
environment/   conda spec + notes on why versions differ from the official Dockerfile
scripts/       download, config patching, smoke test, validation
slurm/         the job script
docs/          provenance, checksums, validation record, troubleshooting + cleanup
logs/          download manifest, Slurm stdout/stderr, config diff, GPU usage samples
outputs/       skill plot from the real run
```

## Reproduction steps (PACE-ICE)

Connect to the GT VPN, then `ssh <gt-username>@login-ice.pace.gatech.edu`.

0. **Find your account.** `sacctmgr show associations user=$USER format=account,qos,partition` and put the account in `slurm/run_inference.sbatch` (`--account=`). This run used `coc` with partition `ice-gpu`.

1. **Clone both repos** (home is small, so put this repo on scratch):
   ```
   git clone https://github.com/NVlabs/FourCastNet.git ~/FourCastNet
   git -C ~/FourCastNet rev-parse HEAD            # record in docs/PROVENANCE.md
   git clone https://github.com/srir1607/fourcastnet-reproducible-inference.git ~/scratch/fcn-repro
   ```

2. **Create the environment on scratch.** Home quota is about 30 GB and fills quickly; `~/.condarc` must contain valid YAML:
   ```
   envs_dirs:
     - ~/scratch/.conda/envs
   ```
   ```
   module purge; module load miniforge; module load miniforge; module load miniforge
   source "$(conda info --base)/etc/profile.d/conda.sh"
   mamba env create -f environment/environment.yml -p ~/scratch/.conda/envs/fourcastnet-inference
   conda activate fourcastnet-inference
   python -c "import torch; print(torch.__version__)"      # 2.2.2+cu121
   ```
   Pins and rationale: `environment/environment.yml`, `environment/VERSION_NOTES.md`. In particular `numpy<2` and `torch<2.6` are required by `inference.py`.

3. **Download and verify artifacts** (about 4 GB, plain HTTPS, no Globus):
   ```
   ./scripts/download_artifacts.sh ~/scratch/fcn-artifacts
   ```
   This resumes with `wget -c`, extracts, and checks every file against `docs/CHECKSUMS.sha256`, writing `logs/download_manifest.txt`. If the login node kills the transfer (exit 137), run the same script inside `srun --account=<acct> --partition=ice-gpu --gres=gpu:1 --time=01:00:00 --pty bash`.

4. **Smoke test** (login node; no CUDA is expected there):
   ```
   python scripts/smoke_test.py --fourcastnet-repo ~/FourCastNet --artifact-root ~/scratch/fcn-artifacts
   ```

5. **CUDA check on a GPU node:**
   ```
   srun --account=<acct> --partition=ice-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --pty bash
   conda activate fourcastnet-inference
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   exit
   ```

6. **Submit from the repo root** (paths in the script depend on this):
   ```
   cd ~/scratch/fcn-repro
   sbatch slurm/run_inference.sbatch
   squeue -u $USER
   ```
   The job patches `config/AFNO.yaml` (diff saved to `logs/config_patch.diff`), samples GPU memory to `logs/gpu_usage_<jobid>.csv`, runs `inference/inference.py --config=afno_backbone --run_num=0 --vis`, then runs `scripts/validate.py`. Logs: `logs/slurm-<jobid>.out/.err`. Evidence: `outputs/skill_vs_leadtime.png`, `outputs/z500_skill_<jobid>.csv`, `logs/inference_out_<jobid>.log`. The 6.6 GB prediction file stays on scratch. The job exits non-zero if validation fails.

## Validation

`scripts/validate.py` (also runnable alone: `python scripts/validate.py --run-dir ~/scratch/fcn-run-<jobid>`):

- Smoke: output HDF5 exists, has `rmse`/`acc`, finite and non-empty.
- Substantive: ACC within [-1, 1], RMSE >= 0; ACC at lead 0 near 1 (confirms the first state is the initial condition); Z500 ACC at 24 h >= 0.6 (conventional useful-forecast threshold); ACC falls and RMSE rises with lead time. It also reports the lead time at which Z500 ACC first drops below 0.6.
- Repeatability: `python scripts/compare_runs.py outputs/z500_skill_A.csv outputs/z500_skill_B.csv` compares two runs.

Measured numbers and interpretation go in `docs/VALIDATION.md`.

## Troubleshooting and cleanup

`docs/TROUBLESHOOTING.md` has the incidents hit while building this (with fixes) and the cleanup steps.

## Status checklist

- [x] Environment built and verified (torch 2.2.2+cu121)
- [x] Artifacts downloaded and checksummed (`docs/CHECKSUMS.sha256`)
- [x] Evaluation file shape measured: (40, 21, 721, 1440) float32
- [x] Smoke test passed (`logs/smoke_test.txt`)
- [x] CUDA check on a GPU node (`logs/cuda_check.txt`: Tesla V100-PCIE-16GB)
- [x] `sbatch` run completed: job 6205007, exit 0, validation passed (`logs/slurm-6205007.out`)
- [x] `docs/VALIDATION.md` filled with measured numbers
- [x] FourCastNet commit hash recorded in `docs/PROVENANCE.md`
- [x] Repeat run and `scripts/compare_runs.py` result recorded in `docs/VALIDATION.md`
