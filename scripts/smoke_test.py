#!/usr/bin/env python3
"""
Smoke test: run this on a login node (not inside the Slurm job) before
submitting the real job, to catch environment and path problems cheaply
instead of burning a queued GPU allocation on a typo.

Checks:
  1. All packages inference.py imports are importable.
  2. CUDA is visible to PyTorch (only meaningful if run on a GPU node/allocation;
     on a login node this check is expected to report no CUDA device and that's fine).
  3. The three required artifact paths exist and are non-empty.
  4. config/AFNO.yaml parses and has the afno_backbone section.

Exit code 0 means "safe to submit the real job." Non-zero means fix the
printed issue first.

Usage:
    python smoke_test.py --fourcastnet-repo /path/to/FourCastNet --artifact-root /path/to/artifacts
"""
import argparse
import importlib
import os
import sys

REQUIRED_MODULES = [
    "numpy", "h5py", "torch", "torchvision", "timm", "einops",
    "ruamel.yaml", "tqdm", "wandb", "matplotlib",
]


def check_imports():
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError as e:
            missing.append((mod, str(e)))
    if missing:
        print("FAIL: missing or broken imports:")
        for mod, err in missing:
            print(f"  - {mod}: {err}")
        return False
    print(f"OK: all {len(REQUIRED_MODULES)} required modules import cleanly.")
    return True


def check_cuda():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"OK: CUDA available, device 0 = {name}")
    else:
        print("INFO: no CUDA device visible here. Expected on a login node; "
              "this must say OK (not this message) when re-run inside the Slurm job.")
    return True  # not a hard failure on a login node


def check_artifacts(artifact_root):
    required = [
        os.path.join(artifact_root, "weights", "backbone.ckpt"),
        os.path.join(artifact_root, "stats", "global_means.npy"),
        os.path.join(artifact_root, "stats", "global_stds.npy"),
        os.path.join(artifact_root, "stats", "time_means.npy"),
        os.path.join(artifact_root, "out_of_sample", "2018.h5"),
    ]
    ok = True
    for p in required:
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            print(f"FAIL: missing or empty artifact: {p}")
            ok = False
    if ok:
        print(f"OK: all {len(required)} required artifact files present and non-empty.")
    return ok


def check_config(fourcastnet_repo):
    config_path = os.path.join(fourcastnet_repo, "config", "AFNO.yaml")
    if not os.path.isfile(config_path):
        print(f"FAIL: config not found at {config_path}")
        return False
    from ruamel.yaml import YAML
    yaml = YAML()
    with open(config_path) as f:
        cfg = yaml.load(f)
    if "afno_backbone" not in cfg:
        print("FAIL: config/AFNO.yaml has no afno_backbone section.")
        return False
    print("OK: config/AFNO.yaml parses and has afno_backbone section.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fourcastnet-repo", required=True)
    ap.add_argument("--artifact-root", required=True)
    args = ap.parse_args()

    results = [
        check_imports(),
        check_cuda(),
        check_artifacts(args.artifact_root),
        check_config(args.fourcastnet_repo),
    ]

    if all(results):
        print("\nSMOKE TEST PASSED. Safe to submit slurm/run_inference.sbatch.")
        sys.exit(0)
    else:
        print("\nSMOKE TEST FAILED. Fix the issues above before submitting a GPU job.")
        sys.exit(1)


if __name__ == "__main__":
    main()
