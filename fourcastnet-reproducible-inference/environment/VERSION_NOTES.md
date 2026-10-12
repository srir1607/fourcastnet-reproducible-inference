# Why these versions, not the original Docker image's

The official FourCastNet repo ships a Dockerfile built on `nvcr.io/nvidia/pytorch:21.11-py3`, which bundles an old PyTorch/CUDA pairing. That image predates Hopper-generation GPUs (H200), so it does not include GPU kernels built for `sm_90`. Running it as-is on an H200 node would either fail outright or silently fall back to an unsupported compute mode.

`environment.yml` instead pins modern versions (PyTorch 2.2.2 with CUDA 12.1 wheels) that do support H200, while keeping the same package set the original image installs (h5py, timm, einops, ruamel.yaml, tqdm, wandb) since `inference/inference.py` imports all of them directly.

**mpi4py and benchy are deliberately dropped.** Both are training-only dependencies (`benchy` is a profiling tool invoked from `submit_batch.sh`, `mpi4py` is used for distributed data loading across multiple nodes). This workflow runs single-GPU inference and doesn't import either.

**What to verify on first run:** `torchvision.utils.save_image` is imported but only used for debug visualization, not metric computation, so a version mismatch there is unlikely to be fatal. If `inference.py` throws an import or API-mismatch error against these pinned versions, check `inference/inference.py`'s actual import list in whatever commit you pinned in `docs/PROVENANCE.md` and record the fix in `docs/TROUBLESHOOTING.md`, since the public repo has evolved since these scripts were last touched.

**wandb:** the script imports `wandb` but logging to a remote project isn't needed for this workflow. Run with `WANDB_MODE=disabled` (already set in `slurm/run_inference.sbatch`) so it initializes without requiring an account or network access.
