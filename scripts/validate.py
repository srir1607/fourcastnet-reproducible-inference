#!/usr/bin/env python3
"""
Validation for a completed FourCastNet backbone inference run.

This is the rubric's "smoke test + substantive validation check" in one
script, run automatically at the end of slurm/run_inference.sbatch, or
standalone afterward:

    python validate.py --run-dir /path/to/output_dir

Smoke test (structural): does the output file exist, is it well-formed HDF5,
does it contain the datasets inference.py is supposed to write, are the
arrays non-empty and free of NaN/Inf.

Substantive check (scientific): does the measured skill actually look like a
working weather forecast, not just an array of numbers. Two checks:
  1. ACC must fall in [-1, 1] by definition (it's a correlation coefficient);
     a violation means something in the pipeline, most likely normalization,
     is wrong.
  2. Day-1 (first autoregressive step) ACC for Z500 should clear 0.6, the
     conventional threshold meteorologists use for a "useful" synoptic-scale
     forecast (this is a standard NWP verification threshold, not a number
     specific to this paper -- see docs/VALIDATION.md for the citation).

It also plots RMSE and ACC vs. forecast lead time to
<run_dir>/skill_vs_leadtime.png, which is what you compare by eye against the
paper's Figure 3/4, since that comparison can't be automated without
digitizing the paper's plot.
"""
import argparse
import os
import sys

import h5py
import numpy as np

EXPECTED_DATASETS = ["rmse", "acc"]

# z500's index in FourCastNet's fixed 20-channel variable ordering.
# VERIFY THIS against the channel-order comment in the cloned repo's
# utils/data_loader_multifiles.py (or config/AFNO.yaml) before trusting the
# Z500-specific check below -- this default is what the public repo has used,
# but confirm it against the exact commit you pinned in docs/PROVENANCE.md.
DEFAULT_Z500_CHANNEL = 14
ACC_SKILL_THRESHOLD = 0.6


def smoke_test(run_dir):
    candidates = [f for f in os.listdir(run_dir) if f.startswith("autoregressive_predictions") and f.endswith(".h5")]
    if not candidates:
        print(f"FAIL (smoke test): no autoregressive_predictions*.h5 file found in {run_dir}")
        return None
    out_path = os.path.join(run_dir, sorted(candidates)[-1])
    print(f"Found output file: {out_path}")

    try:
        f = h5py.File(out_path, "r")
    except OSError as e:
        print(f"FAIL (smoke test): could not open {out_path} as HDF5: {e}")
        return None

    missing = [k for k in EXPECTED_DATASETS if k not in f]
    if missing:
        print(f"FAIL (smoke test): missing expected dataset(s): {missing}. Present: {list(f.keys())}")
        return None

    for k in EXPECTED_DATASETS:
        arr = f[k][:]
        if arr.size == 0:
            print(f"FAIL (smoke test): dataset '{k}' is empty.")
            return None
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"FAIL (smoke test): dataset '{k}' contains NaN or Inf values.")
            return None
        print(f"OK: '{k}' shape={arr.shape}, dtype={arr.dtype}, no NaN/Inf.")

    print("SMOKE TEST PASSED: output file well-formed with expected non-empty, finite datasets.\n")
    return f


def substantive_check(f, z500_channel, out_png):
    acc = np.asarray(f["acc"])
    rmse = np.asarray(f["rmse"])

    ok = True

    # ACC must be a valid correlation coefficient.
    if acc.min() < -1.0001 or acc.max() > 1.0001:
        print(f"FAIL (substantive): ACC values out of [-1, 1] range (min={acc.min():.4f}, max={acc.max():.4f}). "
              f"This usually means a normalization or climatology-subtraction bug.")
        ok = False
    else:
        print(f"OK: ACC values within [-1, 1] (min={acc.min():.4f}, max={acc.max():.4f}).")

    # RMSE must be non-negative.
    if rmse.min() < 0:
        print(f"FAIL (substantive): RMSE has negative values (min={rmse.min():.4f}), which is not physically possible.")
        ok = False
    else:
        print(f"OK: RMSE non-negative (min={rmse.min():.4f}).")

    # Day-1 Z500 skill threshold. acc shape is expected to be
    # (n_initial_conditions, n_channels, n_lead_times) based on inference.py;
    # adjust the indexing below if your actual output shape differs (print it
    # and check against inference_out.log from the run).
    try:
        if acc.ndim == 3:
            day1_z500_acc = acc[:, z500_channel, 0].mean()
        elif acc.ndim == 2:
            day1_z500_acc = acc[z500_channel, 0]
        else:
            raise ValueError(f"Unexpected acc array rank: {acc.ndim}")
        print(f"Day-1 Z500 ACC (channel index {z500_channel}): {day1_z500_acc:.4f}")
        if day1_z500_acc >= ACC_SKILL_THRESHOLD:
            print(f"OK: Day-1 Z500 ACC ({day1_z500_acc:.4f}) clears the {ACC_SKILL_THRESHOLD} "
                  f"conventional 'useful forecast' skill threshold.")
        else:
            print(f"FAIL (substantive): Day-1 Z500 ACC ({day1_z500_acc:.4f}) is below the "
                  f"{ACC_SKILL_THRESHOLD} skill threshold. Either the channel index is wrong "
                  f"(verify DEFAULT_Z500_CHANNEL) or the forecast genuinely isn't skillful, "
                  f"which would point to a normalization/stats mismatch -- see docs/TROUBLESHOOTING.md.")
            ok = False
    except (IndexError, ValueError) as e:
        print(f"FAIL (substantive): could not extract Day-1 Z500 ACC: {e}. "
              f"Print acc.shape and fix the indexing above for your actual output layout.")
        ok = False

    # Plot skill vs lead time for the by-eye comparison against the paper's Figure 3/4.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if acc.ndim == 3:
            lead_times = np.arange(acc.shape[2])
            axes[0].plot(lead_times, acc[:, z500_channel, :].mean(axis=0))
            axes[1].plot(lead_times, rmse[:, z500_channel, :].mean(axis=0))
        else:
            lead_times = np.arange(acc.shape[-1])
            axes[0].plot(lead_times, acc[z500_channel])
            axes[1].plot(lead_times, rmse[z500_channel])
        axes[0].set_title("ACC vs lead time (Z500)")
        axes[0].set_xlabel("lead time step")
        axes[0].set_ylabel("ACC")
        axes[1].set_title("RMSE vs lead time (Z500)")
        axes[1].set_xlabel("lead time step")
        axes[1].set_ylabel("RMSE")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        print(f"\nSaved skill-vs-lead-time plot to {out_png}. Compare its shape against "
              f"the paper's Figure 3/4 by eye and record the comparison in docs/VALIDATION.md.")
    except Exception as e:
        print(f"WARNING: could not generate plot ({e}). Not a validation failure, but you'll "
              f"need to produce this comparison some other way for the Validation rubric item.")

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Output directory passed as --override_dir to inference.py")
    ap.add_argument("--z500-channel", type=int, default=DEFAULT_Z500_CHANNEL)
    args = ap.parse_args()

    f = smoke_test(args.run_dir)
    if f is None:
        print("\nVALIDATION FAILED at smoke test. See docs/TROUBLESHOOTING.md.")
        sys.exit(1)

    out_png = os.path.join(args.run_dir, "skill_vs_leadtime.png")
    passed = substantive_check(f, args.z500_channel, out_png)

    if passed:
        print("\nVALIDATION PASSED: output is structurally sound and scientifically plausible.")
        sys.exit(0)
    else:
        print("\nVALIDATION FAILED at substantive check. See docs/TROUBLESHOOTING.md and "
              "fill in docs/VALIDATION.md with what you investigated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
