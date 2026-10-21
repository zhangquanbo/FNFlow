"""Generate scientific plots only from completed diagnostic results."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.research_io import ROOT, output_path


def main():
    target = output_path(ROOT / "figures")
    target.mkdir(parents=True, exist_ok=True)
    full = ROOT / "outputs/D1_full/summary.json"
    if full.exists():
        state = json.loads(full.read_text())
        if state["status"] == "completed":
            fig, ax = plt.subplots(figsize=(9, 4.8))
            names = ["p03", "p05_original", "p07"]
            labels = ["dropout 0.3", "dropout 0.5 (original)", "dropout 0.7"]
            for offset, mode, label, color in [(-.25, "no_cfg", "Conditional", "#3579aa"),
                                                (0, "fixed_cfg", "Fixed CFG 1.5", "#d38d33"),
                                                (.25, "geom_cfg", "Geom-CFG", "#3b9469")]:
                values = [next(row["legacy"]["mean"] for row in state["rows"]
                               if row["model"] == name and row["mode"] == mode) for name in names]
                bars = ax.bar(np.arange(3) + offset, values, width=.24, label=label, color=color)
                ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
            ax.set(xticks=np.arange(3), xticklabels=labels, ylabel="Mean angular error (degrees)",
                   title="D1: same-checkpoint guidance comparison (1,998 images)", ylim=(0, 35))
            ax.legend(frameon=False)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
            fig.text(.5, .015, "Historical face-mask and clamp protocol; seed 42. Lower is better.",
                     ha="center", fontsize=8)
            fig.tight_layout(rect=(0, .04, 1, 1))
            fig.savefig(target / "D1_full.png", dpi=180)
            plt.close(fig)
    pilot = json.loads((ROOT / "outputs/P0_D1_D5_pilot/summary.json").read_text())
    if pilot["status"] != "completed":
        return
    rows = pilot["rows"]
    ref = "p05_best"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    steps = [1, 2, 4, 8]
    errors = [next(row["float"]["mean"] for row in rows if row["model"] == ref
                   and row["mode"] == "geom_cfg" and row["steps"] == step and row["seed"] == 42)
              for step in steps]
    axes[0].plot(steps, errors, marker="o", color="#3579aa")
    axes[0].set(xticks=steps, xlabel="Euler steps", ylabel="MAE (degrees)",
                title="D2: steps, 32 identities")
    seeds = list(range(42, 50))
    errors = [next(row["float"]["mean"] for row in rows if row["model"] == ref
                   and row["mode"] == "geom_cfg" and row["steps"] == 1 and row["seed"] == seed)
              for seed in seeds]
    axes[1].plot(seeds, errors, marker="o", color="#3b9469", label="Single sample")
    ensemble = next(row["float"]["mean"] for row in rows if row["task"] == "D4")
    axes[1].axhline(ensemble, color="#d38d33", linestyle="--", label="8-sample direction mean")
    axes[1].set(xticks=seeds, xlabel="Inference seed", ylabel="MAE (degrees)",
                title="D3/D4: seeds, 32 identities")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle("Exploratory subset; floating unit normals; same historical mask", fontsize=10)
    fig.tight_layout()
    fig.savefig(target / "D2_D4_pilot.png", dpi=180)
    plt.close(fig)
    print("Diagnostic plots written to", target, flush=True)


if __name__ == "__main__":
    main()
