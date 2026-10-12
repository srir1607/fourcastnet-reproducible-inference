# FourCastNet Reproducible Inference Workflow

Sri Ravela | Reproducible Inference Assignment

This repo runs single-GPU inference with FourCastNet, NVIDIA's AFNO-based global weather model, on PACE-ICE. It scores the output against ERA5 ground truth and documents every step from artifact download to result. It follows from my Model Selection Report and my GitHub issue: https://github.com/cyber-shuttle/cybershuttle-ai4science-foundry/issues/24

## What I reproduced

I ran one autoregressive rollout of the pretrained backbone. It starts at the first time step of the released out-of-sample 2018 ERA5 file and runs 40 six-hour steps, which covers 10 days. The model scores 20 channels, and Z500 (channel 14) is the headline variable. My issue also proposed T850, but `inference.py` only scores Z500, so T850 is left out.

For Z500, ACC is 0.995 at 24 hours and 0.981 at 48 hours, and it crosses 0.6 at about 170 hours. That agrees with Figure 7(a) of the paper through about 2 days. Beyond about 4 days my curve falls below the paper's average, and I couldn't settle why. One initialization against the paper's average over many could explain it, but the demo bundle holds a single 10-day window, so I had no way to test that. Two runs on the same GPU gave identical numbers. The full record is in `docs/VALIDATION.md`.

With a single initialization, my numbers are noisier than an average over many, so I treat this as a sanity reproduction of the pipeline. Training, the precipitation model, and the HuggingFace 26-channel release are out of scope (see `docs/PROVENANCE.md`).

## What it takes to run

The job needs one GPU with at least 16 GB. Peak GPU memory was 11.3 GiB with `--vis`, host RAM used was 21.7 GB (I requested 64 GB), and the job took about 95 seconds on a V100. The evaluation file holds 40 time steps, so the repo's default 41-step rollout would run off the end. `scripts/configure_paths.py` sets `prediction_length` from the file to avoid that. I didn't verify the calendar dates of those 40 steps.

## Repo layout

```
environment/   conda spec, and notes on why versions differ from the official Dockerfile
scripts/       download, config patching, smoke test, validation, run comparison
slurm/         the job script
docs/          provenance, checksums, validation record, troubleshooting and cleanup
logs/          download manifest, Slurm stdout and stderr, config diff, GPU usage samples
outputs/       skill plot and Z500 skill CSVs from the real runs
```

## Reproduction steps (PACE-ICE)

Connect to the GT VPN, then `ssh <gt-username>@login-ice.pace.gatech.edu`.

0. **Find your account.** Run `sacctmgr show associations user=$USER format=account,qos,partition` and put the account in `slurm/run_inference.sbatch` (`--account=`). I used `coc` with partition `ice-gpu`.

1. **Clone both repos.** Home space is small, so this repo goes on scratch:
   ```
   git clone https://github.com/NVlabs/FourCastNet.git ~/FourCastNet
   git -C ~/FourCastNet rev-parse HEAD            # record in docs/PROVENANCE.md
   git clone https://github.com/srir1607/fourcastnet-reproducible-inference.git ~/scratch/fcn-repro
   ```

2. **Create the environment on scratch.** The home quota is about 30 GB and fills quickly. `~/.condarc` has to contain valid YAML:
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
   The pins and my reasons for them are in `environment/environment.yml` and `environment/VERSION_NOTES.md`. In particular, `inference.py` needs `numpy<2` and `torch<2.6`.

3. **Download and verify the artifacts.** This is about 4 GB over plain HTTPS, and it needs no Globus account:
   ```
   ./scripts/download_artifacts.sh ~/scratch/fcn-artifacts
   ```
   The script resumes with `wget -c`, extracts the archive, and checks every file against `docs/CHECKSUMS.sha256`, writing `logs/download_manifest.txt`. If the login node kills the transfer (exit code 137), run the same script inside `srun --account=<acct> --partition=ice-gpu --gres=gpu:1 --time=01:00:00 --pty bash`.

4. **Run the smoke test** on the login node. It reports no CUDA device there, which is expected:
   ```
   python scripts/smoke_test.py --fourcastnet-repo ~/FourCastNet --artifact-root ~/scratch/fcn-artifacts
   ```

5. **Check CUDA on a GPU node:**
   ```
   srun --account=<acct> --partition=ice-gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=00:15:00 --pty bash
   conda activate fourcastnet-inference
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   exit
   ```

6. **Submit from the repo root.** The paths in the script depend on that:
   ```
   cd ~/scratch/fcn-repro
   sbatch slurm/run_inference.sbatch
   squeue -u $USER
   ```
   The job patches `config/AFNO.yaml` and saves the diff to `logs/config_patch.diff`. It samples GPU memory into `logs/gpu_usage_<jobid>.csv`, runs `inference/inference.py --config=afno_backbone --run_num=0 --vis`, and then runs `scripts/validate.py`. The logs are `logs/slurm-<jobid>.out` and `.err`. The evidence files are `outputs/skill_vs_leadtime.png`, `outputs/z500_skill_<jobid>.csv` and `logs/inference_out_<jobid>.log`. The 6.6 GB prediction file stays on scratch, and the job exits with a non-zero code if validation fails.

## Validation

`scripts/validate.py` runs at the end of the job and can also run alone with `python scripts/validate.py --run-dir ~/scratch/fcn-run-<jobid>`. The smoke checks confirm that the output HDF5 exists, holds `rmse` and `acc`, and contains finite values. The substantive checks cover these points:

- ACC stays within [-1, 1] and RMSE stays at or above 0.
- ACC at lead 0 is near 1, which confirms the first state is the initial condition.
- Z500 ACC at 24 hours reaches at least 0.6, the usual threshold for a useful forecast.
- ACC falls and RMSE rises with lead time.

The script also reports the lead time where Z500 ACC first drops below 0.6. To compare two runs, use `python scripts/compare_runs.py outputs/z500_skill_<jobidA>.csv outputs/z500_skill_<jobidB>.csv`. My measured numbers and what I make of them are in `docs/VALIDATION.md`.

## Troubleshooting and cleanup

`docs/TROUBLESHOOTING.md` lists the problems I hit while building this, with the fix for each, and the cleanup steps.

## Status checklist

- [x] Environment built and verified (torch 2.2.2+cu121)
- [x] Artifacts downloaded and checksummed (`docs/CHECKSUMS.sha256`)
- [x] Evaluation file shape measured: (40, 21, 721, 1440) float32
- [x] Smoke test passed (`logs/smoke_test.txt`)
- [x] CUDA check on a GPU node (`logs/cuda_check.txt`: Tesla V100-PCIE-16GB)
- [x] `sbatch` run completed: job 6205007, exit 0, validation passed (`logs/slurm-6205007.out`)
- [x] `docs/VALIDATION.md` filled with measured numbers
- [x] FourCastNet commit hash recorded in `docs/PROVENANCE.md`
- [x] Repeat run (job 6205661) compared with `scripts/compare_runs.py`, recorded in `docs/VALIDATION.md`
