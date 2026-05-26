"""Plot ATE RMSE vs runtime for incremental BA add_size sweep.

Usage:
    python evals/plot_incremental_ba_sweep.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_BASE = REPO_ROOT / "evals" / "results"

# Fixed reference points from the prior eval run
REFERENCE = {
    "baseline": {"ate_rmse": 0.05576780438423157, "time_s": 91.34},
    "ba (all-at-once)": {"ate_rmse": 0.058328576385974884, "time_s": 102.79},
}

# Previously measured incremental points
KNOWN = {
    5: {"ate_rmse": 0.04653976857662201, "time_s": 180.94},
    10: {"ate_rmse": 0.05328147113323212, "time_s": 149.49},
}

# Regex to extract "Time: Xs" and ATE from eval log or metrics.json
_TIME_RE = re.compile(r"Time:\s+([\d.]+)s")
_ATE_RE = re.compile(r'"ate_rmse":\s+([\d.]+)')


def load_sweep_results() -> dict[int, dict]:
    """Load results from sweep output dirs and known prior runs."""
    results = dict(KNOWN)
    # Search both legacy chess_sweep_N dirs and new incremental_ba_sweep/N dirs
    candidates = list(RESULTS_BASE.glob("chess_sweep_*")) + list((RESULTS_BASE / "incremental_ba_sweep").glob("*"))
    for result_dir in sorted(candidates):
        m = re.search(r"(\d+)$", result_dir.name)
        if not m:
            continue
        add_size = int(m.group(1))

        metrics_path = result_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        key = f"incremental_ba-{add_size}"
        if key not in metrics:
            continue

        entry = metrics[key]
        ate = entry["ate"]["rmse"]
        time_s = entry.get("time_s")

        results[add_size] = {"ate_rmse": ate, "time_s": time_s}

    return results


def main() -> None:
    sweep = load_sweep_results()

    # Sort by add_size
    add_sizes = sorted(sweep)
    ates = [sweep[k]["ate_rmse"] * 100 for k in add_sizes]  # convert to cm
    times = [sweep[k]["time_s"] for k in add_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: ATE vs add_size ---
    ax1.plot(add_sizes, ates, "o-", color="tab:blue", linewidth=2, markersize=7, label="incremental BA")
    for ref_name, ref in REFERENCE.items():
        ax1.axhline(ref["ate_rmse"] * 100, linestyle="--",
                    color="tab:red" if "baseline" in ref_name else "tab:orange",
                    label=ref_name, linewidth=1.5)
    ax1.set_xlabel("add_size (frames per step)")
    ax1.set_ylabel("ATE RMSE (cm)")
    ax1.set_title("Accuracy vs add_size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Right: ATE vs runtime (Pareto plot) ---
    valid = [(sweep[k]["time_s"], sweep[k]["ate_rmse"] * 100, k) for k in add_sizes if sweep[k]["time_s"] is not None]
    if valid:
        vt, va, vk = zip(*valid)
        ax2.scatter(vt, va, color="tab:blue", s=80, zorder=5)
        ax2.plot(vt, va, "-", color="tab:blue", alpha=0.4)
        for t, a, k in zip(vt, va, vk):
            ax2.annotate(f"add_size={k}", (t, a), textcoords="offset points",
                         xytext=(5, 5), fontsize=8)

    for ref_name, ref in REFERENCE.items():
        color = "tab:red" if "baseline" in ref_name else "tab:orange"
        ax2.scatter(ref["time_s"], ref["ate_rmse"] * 100,
                    color=color, s=100, marker="^", zorder=5, label=ref_name)

    ax2.set_xlabel("Runtime (s)")
    ax2.set_ylabel("ATE RMSE (cm)")
    ax2.set_title("Accuracy vs Runtime (lower-left = better)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Incremental BA — 7-Scenes chess seq-01, 50 frames", fontweight="bold")
    plt.tight_layout()

    out = REPO_ROOT / "evals" / "incremental_ba_sweep.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
