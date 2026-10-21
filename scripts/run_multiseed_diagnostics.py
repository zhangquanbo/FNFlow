"""Full, paired seed/step follow-up using the exact existing evaluator."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import traceback
import numpy as np
from scripts.run_diagnostics import Experiment
from src.research_io import ROOT, sha256, write_json, compare_best


def summarize(experiment):
    rows = experiment.state["rows"]
    groups = {}
    for mode, steps in [("geom_cfg", 1)] + [("no_cfg", k) for k in experiment.config["steps"]]:
        group = [r for r in rows if r["mode"] == mode and r["steps"] == steps]
        if len(group) != len(experiment.config["inference_seeds"]):
            continue
        protocols = {}
        for protocol in ("legacy", "float"):
            values = np.array([r[protocol]["mean"] for r in group])
            protocols[protocol] = {"mean_mae_deg": float(values.mean()),
                                   "std_seed_ddof1_deg": float(values.std(ddof=1)),
                                   "std_seed_ddof0_deg": float(values.std(ddof=0)),
                                   "seed_maes_deg": {str(r["seed"]): r[protocol]["mean"] for r in group}}
        groups[f"{mode}_k{steps}"] = {"num_seeds": len(group), "protocols": protocols,
                                     "mean_ms_per_image": float(np.mean([r["mean_ms"] for r in group]))}
    paired = {}
    reference = {r["seed"]: r for r in rows if r["mode"] == "no_cfg" and r["steps"] == 1}
    for steps in experiment.config["steps"]:
        if steps == 1:
            continue
        group = {r["seed"]: r for r in rows if r["mode"] == "no_cfg" and r["steps"] == steps}
        if len(group) != 8 or set(group) != set(reference):
            continue
        differences = np.array([group[s]["float"]["mean"] - reference[s]["float"]["mean"]
                                for s in sorted(group)])
        paired[str(steps)] = {"mean_delta_mae_deg_vs_one_step": float(differences.mean()),
                             "std_seed_delta_ddof1_deg": float(differences.std(ddof=1)),
                             "seed_deltas_deg": dict(zip(map(str, sorted(group)), differences.tolist())),
                             "positive_means": "higher error than one step"}
    experiment.state["multiseed"] = {
        "groups": groups, "paired_no_cfg_step_differences": paired,
        "std_definition": "sample SD over 8 inference-seed dataset MAEs (ddof=1); also retain population SD for prior-pilot comparison",
        "seed_policy": "historical per-image local generator reset: a seed gives the same initial field for every image; identical fields paired across steps/modes",
        "scope": "fixed checkpoint, historical development domain; not training-seed variance or general-domain evidence",
        "matrix_completed": len(rows) == 40}
    write_json(experiment.directory / "multiseed_summary.json", experiment.state["multiseed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diagnostics_full_multiseed.json")
    parser.add_argument("--run-id")
    parser.add_argument("--subjects", type=int, default=0)
    args = parser.parse_args()
    args.stage = "d1"  # Permit full records without changing the completed D0-D5 runner.
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    target = ROOT / "outputs" / (args.run_id or config["run_id"]) / "summary.json"
    extension = {"path": str(Path(__file__).relative_to(ROOT)), "sha256": sha256(__file__)}
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("extension_code") != extension:
            raise RuntimeError("Follow-up code changed; use a new run-id.")
    experiment = None
    try:
        experiment = Experiment(args)
        experiment.state["extension_code"] = extension
        experiment.state["diagnostic_design"] = "paired inference seed x no-CFG integration steps; plus historical one-step Geom-CFG multi-seed"
        experiment.save()
        experiment.run_parity()
        reference = config["reference"]
        # Establish historical one-step robustness before the more expensive step sweep.
        for seed in config["inference_seeds"]:
            experiment.evaluate(reference, "geom_cfg", steps=1, seed=seed, task="D3-full")
            summarize(experiment)
            experiment.save()
        for steps in config["steps"]:
            for seed in config["inference_seeds"]:
                experiment.evaluate(reference, "no_cfg", steps=steps, seed=seed, task="D2-D3-full")
                summarize(experiment)
                experiment.save()
        experiment.state["status"] = "completed"
        experiment.state["current_task"] = "completed"
    except Exception:
        if experiment is not None:
            experiment.state["status"] = "failed"
            experiment.state["error"] = traceback.format_exc()
        raise
    finally:
        if experiment is not None:
            baseline = json.loads((ROOT / "provenance/best_before.json").read_text())
            experiment.state["best_audit"] = compare_best(baseline)
            experiment.save()
            if not experiment.state["best_audit"]["verified_unchanged"]:
                raise RuntimeError("Protected reference changed; inspect the audit.")


if __name__ == "__main__":
    main()
