"""Historical-domain D0-D5 diagnostics. All writes are confined to Research."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import hashlib
import json
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.models.model_unet import FaceNormalUNet_B
from src.flow_inference import infer
from src.diagnostic_data import (select_pairs, cache_pairs, prediction_metrics,
                                  angle_map, metrics_from_angles, legacy)
from src.research_io import ROOT, BEST, now, output_path, sha256, write_json, compare_best
from scripts.render_record import render


def aggregate(images, protocol):
    keys = images[0][protocol]
    result = {key: float(np.mean([item[protocol][key] for item in images])) for key in keys}
    result["std_image_mean"] = float(np.std([item[protocol]["mean"] for item in images]))
    result["num_images"] = len(images)
    result["aggregation"] = "macro mean over per-image metrics; std is across image MAEs"
    return result


def hwc(tensor):
    return tensor[0].detach().cpu().permute(1, 2, 0).numpy()


def save_normal(path, normal):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, normal.astype(np.float32))


def write_csv(path, records):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def load_model(path):
    model = FaceNormalUNet_B()
    state = torch.load(path, map_location="cpu")
    state = state.get("model_state_dict", state)
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.cuda().eval()


class Experiment:
    def __init__(self, args):
        self.config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
        self.run_id = args.run_id or self.config["run_id"]
        self.directory = output_path(ROOT / "outputs" / self.run_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stage = args.stage
        self.subjects = args.subjects if args.subjects is not None else self.config["subset_subjects"]
        if self.stage != "d1" and self.subjects == 0:
            raise ValueError("Full-set run is restricted to D1; run other diagnostics on declared subsets.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required for this experimental run.")
        torch.set_num_threads(4)
        self.device = torch.device("cuda")
        self.pairs, self.full_pair_count = select_pairs(self.subjects, self.config["selection_seed"])
        signature_data = {"config": self.config, "stage": self.stage, "subjects": self.subjects,
                          "pairs": self.pairs,
                          "code": {str(path.relative_to(ROOT)): sha256(path)
                                   for path in [Path(__file__), ROOT / "src/flow_inference.py",
                                                ROOT / "src/diagnostic_data.py",
                                                ROOT / "src/models/model_unet.py",
                                                ROOT / "scripts/legacy/eval_facescape_modes.py"]}}
        self.signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
        summary = self.directory / "summary.json"
        if summary.exists():
            self.state = json.loads(summary.read_text(encoding="utf-8"))
            if self.state["signature"] != self.signature:
                raise RuntimeError("Existing run has different data/code/config; use a new run_id.")
        else:
            self.state = {"run_id": self.run_id, "signature": self.signature,
                          "stage": self.stage, "created_at": now(), "rows": [],
                          "num_images": len(self.pairs),
                          "num_subjects": len({pair["subject"] for pair in self.pairs}),
                          "scope": "full historical FaceScape set" if self.subjects == 0 else
                                   "exploratory subset: one fixed image per sampled identity",
                          "checkpoints": {}, "config": self.config, "code_sha256": signature_data["code"],
                          "mask_policy": "historical dlib hull intersect closed/eroded GT; explicit fallback",
                          "metric_protocols": {
                              "legacy": "raw channel clamp, encode, then normalize; no PNG roundtrip",
                              "float": "FP32 normal normalized before export; same GT mask"}}
        self.state["status"] = "running"
        self.state.pop("error", None)
        self.state["current_task"] = "preparing data and masks"
        write_json(self.directory / "subset.json",
                   {"selection_seed": self.config["selection_seed"], "all_available": self.full_pair_count,
                    "selection": self.state["scope"], "pairs": self.pairs})
        self.save()
        self.records = cache_pairs(self.pairs, self.config["image_size"], self.device)
        self.state["mask_fallbacks"] = sum(record["mask_fallback"] for record in self.records)
        self.model = None
        self.loaded_name = None

    def save(self):
        self.state["updated_at"] = now()
        write_json(self.directory / "summary.json", self.state)
        render()

    def get_model(self, name):
        if self.loaded_name == name:
            return self.model
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        self.state["current_task"] = f"loading {name}"
        self.save()
        checkpoint = (BEST / self.config["models"][name]["checkpoint"]).resolve()
        self.state["checkpoints"][name] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
        self.model = load_model(checkpoint)
        self.loaded_name = name
        # Warm up outside the recorded per-image measurements.
        infer(self.model, self.records[0]["condition"], mode="no_cfg",
              seed=self.config["inference_seeds"][0], amp=True)
        torch.cuda.synchronize()
        return self.model

    def run_parity(self):
        if self.state.get("parity"):
            return
        name = self.config["reference"]
        model = self.get_model(name)
        record = self.records[0]
        legacy.DEVICE = "cuda"
        worst = 0.0
        angle_difference = 0.0
        for mode, old_mode in [("no_cfg", "no_cfg"), ("geom_cfg", "no_tta")]:
            result = infer(model, record["condition"], seed=42, mode=mode,
                           scale=self.config["cfg_scale"])
            reference = legacy.infer_flow(model, record["rgb"], num_steps=1,
                                          size=self.config["image_size"],
                                          inference_mode=old_mode, legacy_steps=False)
            encoded = hwc((result["raw"].clamp(-1, 1) + 1) / 2)
            difference = float(np.max(np.abs(encoded - reference)))
            worst = max(worst, difference)
            old_metrics = legacy.compute_angular_error(reference, record["gt_encoded"],
                                                       legacy.create_face_mask(record["rgb"],
                                                                               self.config["image_size"]))
            new_metrics = prediction_metrics(hwc(result["raw"]), hwc(result["normal"]), record)
            angle_difference = max(angle_difference,
                                   abs(old_metrics["mean"] - new_metrics["legacy"]["mean"]))
        self.state["parity"] = {"max_abs_encoded_difference": worst,
                               "angular_mean_difference_deg": angle_difference,
                               "modes": ["no_cfg", "geom_cfg"],
                               "tolerance_encoded": 1e-6, "tolerance_angle_deg": 1e-5}
        if worst > 1e-6 or angle_difference > 1e-5:
            raise RuntimeError(f"Historical parity failed: {self.state['parity']}")
        self.save()
        print("D0 parity passed", self.state["parity"], flush=True)

    def row(self, name, mode, steps, seed):
        return next((row for row in self.state["rows"]
                     if row["model"] == name and row["mode"] == mode and
                     row["steps"] == steps and row["seed"] == seed), None)

    def evaluate(self, name, mode, steps=1, seed=42, task="D1", save_samples=False):
        existing = self.row(name, mode, steps, seed)
        if existing:
            return existing
        model = self.get_model(name)
        tag = f"{name}_{mode}_k{steps}_seed{seed}"
        self.state["current_task"] = f"{task}: {tag}"
        self.save()
        per_image = []
        for index, record in enumerate(self.records):
            mask = torch.as_tensor(record["mask"], device=self.device)[None]
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = infer(model, record["condition"], steps=steps, mode=mode, seed=seed,
                           scale=self.config["cfg_scale"], stats_mask=mask)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if result["invalid"][0, 0][mask[0]].any():
                raise RuntimeError("Zero-length prediction on valid evaluation pixels")
            raw, normal = hwc(result["raw"]), hwc(result["normal"])
            metrics = prediction_metrics(raw, normal, record)
            item = {"id": record["id"], "subject": record["pair"]["subject"],
                    "legacy": metrics["legacy"], "float": metrics["float"],
                    "mask_fallback": record["mask_fallback"], "ms": elapsed,
                    "guidance": result["trace"]}
            per_image.append(item)
            if save_samples:
                save_normal(self.directory / "samples" / f"seed_{seed}" / f"{record['id']}.npy", normal)
                if seed == 42 and index < 4:
                    image = np.rint((np.clip(normal, -1, 1) + 1) * 127.5).astype(np.uint8)
                    Image.fromarray(image).save(output_path(self.directory / "samples" /
                                                            f"seed_{seed}" / f"{record['id']}.png"))
            if (index + 1) % 100 == 0 or index + 1 == len(self.records):
                self.state["current_task"] = f"{task}: {tag} {index+1}/{len(self.records)}"
                self.save()
                print(self.state["current_task"], flush=True)
        write_json(self.directory / "per_image" / f"{tag}.json", per_image)
        flat = [{"id": item["id"], "subject": item["subject"], "mask_fallback": item["mask_fallback"],
                 "legacy_mae": item["legacy"]["mean"], "float_mae": item["float"]["mean"],
                 "float_median": item["float"]["median"], "float_rmse": item["float"]["rmse"],
                 "ms": item["ms"]} for item in per_image]
        write_csv(self.directory / "per_image" / f"{tag}.csv", flat)
        row = {"task": task, "model": name, "mode": mode, "steps": steps, "seed": seed,
               "legacy": aggregate(per_image, "legacy"), "float": aggregate(per_image, "float"),
               "mean_ms": float(np.mean([item["ms"] for item in per_image])),
               "p95_ms": float(np.percentile([item["ms"] for item in per_image], 95)),
               "nfe": steps * (1 if mode == "no_cfg" else 2),
               "mask_fallbacks": sum(item["mask_fallback"] for item in per_image)}
        self.state["rows"].append(row)
        self.save()
        print(tag, f"legacy={row['legacy']['mean']:.6f} float={row['float']['mean']:.6f}", flush=True)
        return row

    def d1(self):
        for name in ["p0", "p03", "p05_original", "p07", "p05_best"]:
            modes = ["no_cfg"] if name == "p0" else ["no_cfg", "fixed_cfg", "geom_cfg"]
            if name == "p05_best":
                modes = ["geom_cfg"]
            for mode in modes:
                self.evaluate(name, mode, task="D1",
                              save_samples=(self.stage == "pilot" and name == "p05_best"))

    def noise_ensemble(self):
        if self.state.get("noise"):
            return
        dispersions, errors = [], []
        per_image = []
        for record in self.records:
            samples = np.stack([np.load(self.directory / "samples" / f"seed_{seed}" /
                                        f"{record['id']}.npy")
                                for seed in self.config["inference_seeds"]])
            average = samples.mean(axis=0)
            lengths = np.linalg.norm(average, axis=-1, keepdims=True)
            if np.any(lengths[record["mask"]] < 1e-8):
                raise RuntimeError("Degenerate directional sample mean")
            average = average / np.maximum(lengths, 1e-8)
            # Angular dispersion around the directional sample mean, not a calibrated posterior.
            dots = np.clip((samples.astype(np.float64) * average).sum(axis=-1), -1, 1)
            dispersion = np.degrees(np.arccos(dots)).mean(axis=0)
            error = angle_map(average, record)
            dispersions.append(dispersion[record["mask"]])
            errors.append(error[record["mask"]])
            metrics = prediction_metrics(average, average, record)
            per_image.append({"id": record["id"], "legacy": metrics["legacy"], "float": metrics["float"]})
            save_normal(self.directory / "ensemble" / f"{record['id']}.npy", average)
        x, y = np.concatenate(dispersions), np.concatenate(errors)
        correlation = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-12 and y.std() > 1e-12 else None
        self.state["noise"] = {"mean_angular_dispersion_deg": float(x.mean()),
                               "pearson_dispersion_error": correlation, "num_pixels": len(x),
                               "num_seeds": len(self.config["inference_seeds"]),
                               "error_target": "angular error of directional ensemble mean",
                               "scope": "pixel-weighted exploratory correlation"}
        self.state["rows"].append(
            {"task": "D4", "model": self.config["reference"], "mode": "directional_ensemble_geom_cfg",
             "steps": 1, "seed": ",".join(map(str, self.config["inference_seeds"])),
             "legacy": aggregate(per_image, "legacy"), "float": aggregate(per_image, "float"),
             "mean_ms": 0, "nfe": len(self.config["inference_seeds"]) * 2,
             "timing_note": "not separately timed; eight model samples plus averaging"})
        write_json(self.directory / "ensemble_metrics.json", per_image)
        self.save()

    def ood(self):
        if self.state.get("ood"):
            return
        model = self.get_model(self.config["reference"])
        preprocess = legacy.get_preprocess(self.config["image_size"])
        files = sorted(path for path in (BEST / "data/test_images_1/imgs").iterdir()
                       if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
        rows = []
        for path in files:
            rgb = Image.open(path).convert("RGB")
            result = infer(model, preprocess(rgb).unsqueeze(0).cuda(), mode="geom_cfg", seed=42,
                           scale=self.config["cfg_scale"])
            normal = hwc(result["normal"])
            save_normal(self.directory / "ood" / f"{path.stem}.npy", normal)
            visual = Image.fromarray(np.rint((np.clip(normal, -1, 1) + 1) * 127.5).astype(np.uint8))
            comparison = Image.new("RGB", (self.config["image_size"] * 2, self.config["image_size"]))
            comparison.paste(rgb.resize((self.config["image_size"], self.config["image_size"])), (0, 0))
            comparison.paste(visual, (self.config["image_size"], 0))
            comparison.save(output_path(self.directory / "ood" / f"{path.stem}_compare.png"))
            rows.append({"image": str(path), "mean_guidance": result["trace"][0]["mean"],
                         "zero_fraction": result["trace"][0]["zero_fraction"],
                         "lt_0_1_fraction": result["trace"][0]["lt_0_1_fraction"]})
        if not rows:
            raise RuntimeError("No OOD diagnostic images")
        write_csv(self.directory / "ood" / "guidance.csv", rows)
        self.state["ood"] = {"num_images": len(rows),
                             "mean_guidance": float(np.mean([row["mean_guidance"] for row in rows])),
                             "mean_zero_fraction": float(np.mean([row["zero_fraction"] for row in rows])),
                             "scope": "whole image, no GT, qualitative only"}
        self.save()

    def execute(self):
        self.run_parity()
        self.d1()
        if self.stage == "pilot":
            reference = self.config["reference"]
            for steps in self.config["steps"]:
                self.evaluate(reference, "geom_cfg", steps=steps, task="D2",
                              save_samples=(steps == 1))
            for seed in self.config["inference_seeds"]:
                self.evaluate(reference, "geom_cfg", seed=seed, task="D3", save_samples=True)
            self.noise_ensemble()
            self.ood()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diagnostics_pilot.json")
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=["pilot", "d1"], default="pilot")
    parser.add_argument("--subjects", type=int, help="0 selects all historical pairs for D1")
    args = parser.parse_args()
    experiment = None
    try:
        experiment = Experiment(args)
        experiment.execute()
        experiment.state["status"] = "completed"
        experiment.state["current_task"] = "completed"
    except Exception:
        if experiment is not None:
            experiment.state["status"] = "failed"
            experiment.state["error"] = traceback.format_exc()
            experiment.state["current_task"] = "failed; completed rows retained"
        raise
    finally:
        if experiment is not None:
            baseline = json.loads((ROOT / "provenance/best_before.json").read_text())
            experiment.state["best_audit"] = compare_best(baseline)
            experiment.save()
            if not experiment.state["best_audit"]["verified_unchanged"]:
                print("Best audit detected differences; inspect summary before conclusions.", flush=True)


if __name__ == "__main__":
    main()
