#!/bin/bash
#SBATCH --job-name=fcn-inference
#SBATCH --partition=ice-gpu                          # confirmed real PACE-ICE partition name; re-check with `sinfo` if it rejects
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT          # find yours: sacctmgr show associations user=$USER
#SBATCH --gres=gpu:1                                 # any GPU type is enough for this model; add a type e.g. gpu:h200:1 only if the partition requires it (see `sinfo -o "%P %G"`)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00                              # measured runtime should be well under this; see README
#SBATCH --output=../logs/slurm-%j.out
#SBATCH --error=../logs/slurm-%j.err

# ---------------------------------------------------------------------------
# Single-GPU backbone inference for FourCastNet, inference-only reproduction.
# Resource requests here are sized from the Model Selection Report's analysis
# of the released config (patch_size 8, embed width 56, 8 AFNO blocks):
# ~2-4 GB GPU memory, under an hour of GPU time for a full evaluation sweep.
# This script runs one initialization; loop over --run_num for a sweep.
# ---------------------------------------------------------------------------

set -euo pipefail

# --- EDIT THESE THREE PATHS FOR YOUR ENVIRONMENT ---
FOURCASTNET_REPO="$HOME/FourCastNet"                 # where you cloned NVlabs/FourCastNet
ARTIFACT_ROOT="$HOME/scratch/fcn-artifacts"           # output of download_artifacts.sh
OUTPUT_DIR="$HOME/scratch/fcn-run-$SLURM_JOB_ID"
# ----------------------------------------------------

module purge
module load miniforge
module load miniforge
module load miniforge      # loading 3x clears any other conda module first; a known PACE-ICE quirk, not a typo
source $(conda info --base)/etc/profile.d/conda.sh

conda activate fourcastnet-inference

export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

echo "=== Job $SLURM_JOB_ID starting on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== GPU visible to job: ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

cd "$FOURCASTNET_REPO"

python "$(dirname "$0")/../scripts/configure_paths.py" \
  --fourcastnet-repo "$FOURCASTNET_REPO" \
  --artifact-root "$ARTIFACT_ROOT" \
  --output-dir "$OUTPUT_DIR"

echo "=== Running inference ==="
python inference/inference.py \
  --config=afno_backbone \
  --run_num=0 \
  --weights "$ARTIFACT_ROOT/weights/backbone.ckpt" \
  --override_dir "$OUTPUT_DIR"

echo "=== Inference finished at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Peak GPU memory used: ==="
nvidia-smi --query-gpu=memory.used --format=csv

echo "=== Output files written: ==="
ls -la "$OUTPUT_DIR"

echo "=== Running validation ==="
python "$(dirname "$0")/../scripts/validate.py" --run-dir "$OUTPUT_DIR"
