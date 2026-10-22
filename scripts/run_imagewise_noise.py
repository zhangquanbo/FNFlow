"""One-step image-specific noise, paired guidance and within-image sensitivity."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import json
import time
import traceback
import numpy as np
import torch
from scripts.run_diagnostics import Experiment, aggregate, hwc, write_csv
from src.flow_inference import infer
from src.diagnostic_data import prediction_metrics
from src.noise_protocol import PROTOCOL, image_noise_seed, mae_seed_statistics, directional_sensitivity
from src.research_io import ROOT, BEST, sha256, write_json, compare_best


def public_summary(experiment, buckets, directions):
    seeds = experiment.config["inference_seeds"]
    groups, sensitivity, reference = {}, {}, {}
    experiment.state["rows"] = []
    for mode in experiment.config["modes"]:
        protocols = {}
        for protocol in ("legacy", "float"):
            matrix = np.array([[item[protocol]["mean"] for item in buckets[mode][seed]]
                               for seed in seeds]).T
            stat = mae_seed_statistics(matrix)
            protocols[protocol] = {"mean_mae_deg": stat["mean_mae_deg"],
                                   "std_seed_ddof1_deg": stat["dataset_mae_sd_ddof1_deg"],
                                   "std_seed_ddof0_deg": float(matrix.mean(axis=0).std(ddof=0)),
                                   "seed_maes_deg": dict(zip(map(str, seeds), stat["dataset_maes_deg"]))}
            if protocol == "float":
                sensitivity[mode] = stat
        values = directions[mode]
        undefined = sum(item["undefined_mean_direction_pixels"] for item in values)
        sensitivity[mode].update({
            "mean_pixel_directional_variance_1_minus_R": float(np.mean([
                item["mean_pixel_directional_variance_1_minus_R"] for item in values])),
            "mean_pixel_angular_dispersion_deg": None if undefined else float(np.mean([
                item["mean_pixel_angular_dispersion_deg"] for item in values])),
            "undefined_mean_direction_pixels": undefined,
            "aggregation": "macro mean over images; pixel direction statistics use each image's historical mask"})
        group_times = []
        for seed in seeds:
            images = buckets[mode][seed]
            times = [item["ms"] for item in images]
            group_times.extend(times)
            experiment.state["rows"].append({
                "task": "D3-imagewise", "model": experiment.config["reference"], "mode": mode,
                "steps": 1, "seed": seed, "nfe": 1 if mode == "no_cfg" else 2,
                "legacy": aggregate(images, "legacy"), "float": aggregate(images, "float"),
                "mean_ms": float(np.mean(times)), "p95_ms": float(np.percentile(times, 95)),
                "mask_fallbacks": experiment.state["mask_fallbacks"]})
            write_csv(experiment.directory / "per_image" / f"{mode}_k1_base{seed}.csv", [
                {"id": item["id"], "float_mae": item["float"]["mean"],
                 "legacy_mae": item["legacy"]["mean"], "ms": item["ms"]} for item in images])
        groups[f"{mode}_k1"] = {"num_seeds": len(seeds), "protocols": protocols,
                                 "mean_ms_per_image": float(np.mean(group_times))}
        old = ROOT / "outputs" / experiment.config["shared_reference_run"] / "per_image"
        columns = []
        for seed in seeds:
            if experiment.subjects != 0:
                break  # A declared smoke subset cannot be paired with full reference CSVs.
            path = old / f"{experiment.config['reference']}_{mode}_k1_seed{seed}.csv"
            if not path.exists():
                break
            with path.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if [row["id"] for row in rows] != [item["id"] for item in buckets[mode][seed]]:
                raise RuntimeError("Shared-noise reference image order differs")
            columns.append([float(row["float_mae"]) for row in rows])
        reference[mode] = mae_seed_statistics(np.array(columns).T) if len(columns) == len(seeds) else {
            "available": False, "reason": "not all paired historical CSVs available"}
    no_cfg = np.array(list(groups["no_cfg_k1"]["protocols"]["float"]["seed_maes_deg"].values()))
    geom = np.array(list(groups["geom_cfg_k1"]["protocols"]["float"]["seed_maes_deg"].values()))
    delta = geom - no_cfg
    experiment.state["multiseed"] = {
        "groups": groups, "noise_sensitivity": sensitivity,
        "shared_noise_reference": {"run_id": experiment.config["shared_reference_run"], "statistics": reference},
        "paired_cfg_delta_mae_deg": {"mean_geom_minus_no_cfg": float(delta.mean()),
                                     "std_replicate_ddof1": float(delta.std(ddof=1)),
                                     "replicates_geom_better": int((delta < 0).sum())},
        "seed_policy": "SHA256([protocol, base_seed, namespace/dataset-relative RGB path]) first 63 bits; modes paired",
        "std_definition": "dataset MAE SD is Monte Carlo variation across imagewise-noise replicates; image MAE SD and pixel dispersion measure within-image sensitivity; not training SD or generalization CI",
        "scope": "fixed checkpoint, historical development set, one-step only",
        "matrix_completed": True}
    write_json(experiment.directory / "multiseed_summary.json", experiment.state["multiseed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diagnostics_imagewise_noise.json")
    parser.add_argument("--run-id")
    parser.add_argument("--subjects", type=int, default=0)
    args = parser.parse_args()
    args.stage = "d1"
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if config["noise_protocol"] != PROTOCOL or config["steps"] != [1]:
        raise ValueError("Unexpected noise protocol or step count")
    if config["modes"] != ["no_cfg", "geom_cfg"] or len(set(config["inference_seeds"])) < 2:
        raise ValueError("Need both paired guidance modes and multiple distinct replicates")
    extension = {"code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in [
        Path(__file__), ROOT / "src/noise_protocol.py"]},
        "checkpoint_sha256": sha256(BEST / config["models"][config["reference"]]["checkpoint"])}
    target = ROOT / "outputs" / (args.run_id or config["run_id"]) / "summary.json"
    if target.exists() and json.loads(target.read_text()).get("extension_code") != extension:
        raise RuntimeError("Code or checkpoint changed; use a new run-id")
    experiment = None
    try:
        experiment = Experiment(args)
        experiment.state["extension_code"] = extension
        experiment.state["diagnostic_design"] = "one-step imagewise independent noise x paired modes; dataset and image sensitivity separate"
        experiment.save()
        experiment.run_parity()
        seeds, modes = config["inference_seeds"], config["modes"]
        buckets = {mode: {seed: [] for seed in seeds} for mode in modes}
        directions = {mode: [] for mode in modes}
        dataset = (BEST / "datasets/FacescapeRender1").resolve()
        keys = [config["dataset_namespace"] + "/" + Path(r["pair"]["rgb_path"]).resolve().relative_to(dataset).as_posix()
                for r in experiment.records]
        all_seeds = [image_noise_seed(seed, key) for key in keys for seed in seeds]
        if len(set(keys)) != len(keys) or len(set(all_seeds)) != len(all_seeds):
            raise RuntimeError("Duplicate image key or derived seed collision")
        experiment.state["noise_protocol"] = {"name": PROTOCOL, "num_unique_image_keys": len(keys),
                                               "num_unique_derived_seeds": len(all_seeds),
                                               "uses_gt_or_row_index": False,
                                               "key": "namespace/dataset-relative RGB path"}
        model = experiment.get_model(config["reference"])
        for index, (record, key) in enumerate(zip(experiment.records, keys)):
            path = experiment.directory / "image_records" / f"{record['id']}.json"
            hashes = {kind: sha256(record["pair"][kind + "_path"]) for kind in ["rgb", "normal"]}
            derived = {str(seed): image_noise_seed(seed, key) for seed in seeds}
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                if result["image_key"] != key or result["input_sha256"] != hashes or result["derived_seeds"] != derived:
                    raise RuntimeError("Saved image inputs or noise mapping changed")
            else:
                result = {"id": record["id"], "image_key": key, "input_sha256": hashes,
                          "derived_seeds": derived, "samples": {}, "directional": {}}
                mask = torch.as_tensor(record["mask"], device=experiment.device)[None]
                for mode in modes:
                    samples, items = [], []
                    for seed in seeds:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        prediction = infer(model, record["condition"], steps=1, mode=mode,
                                           seed=derived[str(seed)], scale=config["cfg_scale"], stats_mask=mask)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - start) * 1000
                        if prediction["invalid"][0, 0][mask[0]].any():
                            raise RuntimeError("Zero-length prediction in evaluation mask")
                        normal = hwc(prediction["normal"])
                        metrics = prediction_metrics(hwc(prediction["raw"]), normal, record)
                        items.append({"id": record["id"], "base_seed": seed, "ms": elapsed, **metrics})
                        samples.append(normal)
                    result["samples"][mode] = items
                    result["directional"][mode] = directional_sensitivity(np.stack(samples), record["mask"])
                write_json(path, result)
            for mode in modes:
                items = result["samples"][mode]
                if [item["base_seed"] for item in items] != seeds:
                    raise RuntimeError("Saved image has incomplete or reordered replicates")
                for item in items:
                    buckets[mode][item["base_seed"]].append(item)
                directions[mode].append(result["directional"][mode])
            if (index + 1) % 100 == 0 or index + 1 == len(keys):
                experiment.state["current_task"] = f"imagewise noise: {index + 1}/{len(keys)} images x {len(seeds)} replicates x 2 modes"
                experiment.state["completed_images"] = index + 1
                experiment.save()
                print(experiment.state["current_task"], flush=True)
        public_summary(experiment, buckets, directions)
        experiment.state["status"] = "completed"
        experiment.state["current_task"] = "completed"
    except BaseException:
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
                raise RuntimeError("Protected reference changed")


if __name__ == "__main__":
    main()
