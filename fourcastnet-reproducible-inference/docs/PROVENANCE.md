# Provenance

This document traces every artifact this workflow depends on: the model, its weights, the code that runs it, and the data it runs against. Everything here is pinned to a specific source and version so the run can be reproduced exactly, not approximately.

## Model and architecture

- **Model:** FourCastNet, NVIDIA's Adaptive Fourier Neural Operator (AFNO) based global weather forecasting model.
- **Paper:** Pathak et al., "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators" (2022). https://arxiv.org/abs/2202.11214
- **Architecture summary:** Vision-Transformer-style patch embedding (patch size 8) over a 720x1440 global grid (20 channels), with AFNO blocks (8 blocks, embedding width 56, 32 retained Fourier modes) replacing self-attention for near-linear-complexity spatial mixing. A separate head handles precipitation.

## Source repository

- **Repository:** https://github.com/NVlabs/FourCastNet
- **Commit pinned for this workflow:** record the exact commit hash here after cloning, e.g. `git rev-parse HEAD` run right after clone. Do not leave this as "latest" once you've actually cloned it.
- **License:** BSD 3-Clause (permissive, no academic-only or non-commercial restriction). See `LICENSE` in the repo.
- **Scripts used:** `inference/inference.py` (backbone inference), `config/AFNO.yaml` (run configuration).

## Checkpoint / weights

- **Source:** `FCN_weights_v0/` at https://portal.nersc.gov/project/m4134/FCN_weights_v0/ (served via NERSC Globus)
- **Files used:** `backbone.ckpt` (this workflow is inference-only against the backbone; `precip.ckpt` is out of scope here).
- **Integrity check:** record the file size and, if available, a checksum for `backbone.ckpt` after download, in `logs/download_manifest.txt` (the download script writes this automatically).
- **Important constraint:** the normalization statistics below must match this exact checkpoint release. Do not mix `stats_v0` with a different weights release.

## Normalization statistics

- **Source:** `stats_v0/` folder, same NERSC Globus location as the weights.
- **Files used:** `global_means.npy`, `global_stds.npy`, `time_means.npy`, `land_sea_mask.npy`, `latitude.npy`, `longitude.npy`.

## Evaluation / input data

- **Dataset:** ERA5 reanalysis, published by ECMWF/Copernicus.
- **Primary access:** Copernicus Climate Data Store, https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels (requires free CDS account registration).
- **Mirror used for this workflow:** the `out_of_sample/2018.h5` file at the same NERSC Globus endpoint as the weights, which is the out-of-sample 2018 ERA5 subset the FourCastNet authors packaged for inference testing. This is what the inference script expects by default.
- **Scope:** this workflow runs inference against that pre-packaged out-of-sample file. It does not build a custom ERA5 slice from raw Copernicus netCDF, which the source repo supports separately via `copernicus/get_data_*.py` but which is out of scope for this assignment.

## What is explicitly out of scope

- Training reproduction (the original training run used a ~5TB ERA5 corpus on NERSC's Perlmutter HPC cluster; not attempted here).
- The `precip.ckpt` precipitation head.
- The HuggingFace `nvidia/fourcastnet1` checkpoint (26 channels, Apache 2.0) and the build.nvidia.com hosted inference API. Both are separate releases from the one used here (different channel count, different license) and are not interchangeable with the artifacts above.

## Compute environment

- **Platform:** PACE-ICE (the course calls this "AI Makerspace"), GT's Slurm-scheduled GPU cluster, login at login-ice.pace.gatech.edu.
- **Base software environment:** adapted from the official FourCastNet Docker image (`nvcr.io/nvidia/pytorch:21.11-py3`); see `environment/environment.yml` for the pinned dependency list used in this workflow and why some package versions were updated from the original Docker image.
