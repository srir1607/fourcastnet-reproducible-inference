# Provenance

Every artifact this workflow depends on, with source and version.

## Model

- FourCastNet, NVIDIA's AFNO-based global weather model. Pathak et al., 2022, https://arxiv.org/abs/2202.11214
- 20 prognostic channels on a 720x1440 grid; patch size 8, 8 AFNO blocks, embedding width 56.

## Code

- Repository: https://github.com/NVlabs/FourCastNet (BSD 3-Clause)
- **Commit used:** `93360c1720a9f97aabf970689f21c9fad8737788` (also printed at the top of every Slurm log)
- Scripts used: `inference/inference.py`, `config/AFNO.yaml` (section `afno_backbone`, patched by `scripts/configure_paths.py`; diff in `logs/config_patch.diff`)
- Behaviors of `inference.py` that shaped this workflow: scores field `z500` only; uses `--vis` to run a single initial condition; imports `numpy.core.numeric.False_` (needs numpy<2); calls `torch.load` without `weights_only` (needs torch<2.6).

## Data bundle

- **Source:** https://portal.nersc.gov/project/m4134/ccai_demo.tar (gzip-compressed tar, 3,998,497,644 bytes), published by the FourCastNet authors on NERSC.
- **SHA-256 of the tar and of each file used:** `docs/CHECKSUMS.sha256` (measured after download; re-verified by `scripts/download_artifacts.sh`).
- **Contents used (paths under `ccai_demo/`):**
  - `model_weights/FCN_weights_v0/backbone.ckpt` (896,430,597 bytes). `precip.ckpt` is in the bundle but unused.
  - `additional/stats_v0/{global_means,global_stds,time_means}.npy`. These must come from the same release as the checkpoint; do not recompute them.
  - `data/FCN_ERA5_data_v0/out_of_sample/2018.h5`: dataset `fields`, shape (40, 21, 721, 1440), float32 (40 six-hour steps, about 3.5 GB). The model reads the first 20 channels and the first 720 latitude rows.
- **Underlying data:** ERA5 reanalysis (ECMWF/Copernicus). This file is the authors' preprocessed out-of-sample subset; no raw ERA5 was downloaded. 2018 is outside the 1979-2015 training years, so the evaluation is out-of-sample. The calendar dates of the 40 steps were not verified.

## Software environment

- PACE-ICE, `module load miniforge`, conda env `fourcastnet-inference` on scratch. Spec: `environment/environment.yml`; rationale: `environment/VERSION_NOTES.md`.
- Verified: Python 3.10, torch 2.2.2+cu121, numpy 1.26.4.
- Hardware used for the recorded run: Tesla V100-PCIE-16GB (16,384 MiB), driver 580.178.04, node `atl1-1-02-011-31-0`, account `coc`, partition `ice-gpu`, QOS `coc-ice`.

## Out of scope

Training; `precip.ckpt`; the HuggingFace `nvidia/fourcastnet1` (26 channels) and hosted APIs, which are different releases and not interchangeable with these files.
