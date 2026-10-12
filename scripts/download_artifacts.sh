#!/usr/bin/env bash
# Downloads the FourCastNet backbone checkpoint, normalization stats, and the
# out-of-sample 2018 ERA5 evaluation file from the NERSC Globus endpoint.
#
# Requires: a Globus account (free, https://www.globus.org) and the Globus CLI
# (`pip install globus-cli`, then `globus login` once interactively before
# running this script for the first time — that step needs a browser and
# can't be scripted).
#
# Usage: ./download_artifacts.sh /path/to/artifacts/root

set -euo pipefail

ARTIFACT_ROOT="${1:?Usage: ./download_artifacts.sh /path/to/artifacts/root}"
mkdir -p "$ARTIFACT_ROOT"/{weights,stats,out_of_sample}

# NERSC Globus endpoint hosting FCN_weights_v0 (weights + stats + out-of-sample data).
# Confirm this endpoint ID against https://portal.nersc.gov/project/m4134/FCN_weights_v0/
# before running -- Globus endpoint IDs occasionally change when NERSC migrates hosting.
NERSC_ENDPOINT_ID="9d6d99eb-6d04-11e5-ba46-22000b92c6ec"  # NERSC DTN, verify at portal link above
LOCAL_ENDPOINT_ID="$(globus endpoint local-id 2>/dev/null || true)"

if [ -z "$LOCAL_ENDPOINT_ID" ]; then
  echo "No local Globus Connect Personal endpoint found."
  echo "On a Slurm cluster, use the cluster's Globus Collection (ask PACE-ICE docs/admins"
  echo "for its collection name) instead of a personal endpoint, and substitute it below."
  exit 1
fi

echo "Transferring weights..."
globus transfer "$NERSC_ENDPOINT_ID:/FCN_weights_v0/backbone.ckpt" \
  "$LOCAL_ENDPOINT_ID:$ARTIFACT_ROOT/weights/backbone.ckpt" --label "fcn-backbone-ckpt"

echo "Transferring normalization stats..."
for f in global_means.npy global_stds.npy time_means.npy land_sea_mask.npy latitude.npy longitude.npy; do
  globus transfer "$NERSC_ENDPOINT_ID:/stats_v0/$f" "$LOCAL_ENDPOINT_ID:$ARTIFACT_ROOT/stats/$f" --label "fcn-stats-$f"
done

echo "Transferring out-of-sample 2018 evaluation data..."
globus transfer "$NERSC_ENDPOINT_ID:/out_of_sample/2018.h5" \
  "$LOCAL_ENDPOINT_ID:$ARTIFACT_ROOT/out_of_sample/2018.h5" --label "fcn-2018-eval"

echo "Transfers queued. Check status with: globus task list"
echo "Once complete, record file sizes for the integrity check:"
mkdir -p ../logs
{
  echo "# Download manifest -- generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  find "$ARTIFACT_ROOT" -type f -exec ls -la {} \;
} > ../logs/download_manifest.txt
echo "Manifest written to logs/download_manifest.txt (sizes only queued until transfers finish --"
echo "re-run the find/ls block once 'globus task list' shows all tasks as SUCCEEDED)."
