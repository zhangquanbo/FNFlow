"""Build a reviewed publication copy without changing protected source data."""
import sys
from pathlib import Path, PurePosixPath
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import base64
import hashlib
import importlib.metadata
import json
import posixpath
import re
from src.research_io import ROOT, BEST, output_path, now, sha256, write_json

EXPORT = output_path(ROOT / "publish/FNFlow")
files = {}


def scrub(text):
    replacements = {
        # Braceless placeholders also remain literal inside exported f-strings.
        str(ROOT): "$FNFLOW_ROOT",
        str(BEST): "$FACEFLUX_BEST_ROOT",
        str(ROOT.parent) + "/": "$WORKSPACE_ROOT/",
        str(Path.home()) + "/": "$HOME/",
        "$FNFLOW_ROOT": "$FNFLOW_ROOT",
        "X:\\FaceFlux-Research": "$FNFLOW_ROOT",
        "$FACEFLUX_BEST_ROOT": "$FACEFLUX_BEST_ROOT",
        "X:\\FaceFlux-Best": "$FACEFLUX_BEST_ROOT",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def markdown(text, destination):
    def link(match):
        target = match.group(1).strip("<>")
        anchor = ""
        line = re.search(r":(\d+)$", target)
        if line:
            anchor = "#L" + line.group(1)
            target = target[:line.start()]
        for prefix, replacement in [("$FNFLOW_ROOT/", ""),
                                     ("$FACEFLUX_BEST_ROOT/", "baseline/")]:
            if target.startswith(prefix):
                relative = replacement + target[len(prefix):]
                if relative.startswith("outputs/"):
                    relative = "results/" + relative[len("outputs/"):]
                relative = {
                    "provenance/environment.json": "reproducibility/tested_environment.json",
                    "provenance/data_inventory.json": "docs/数据来源初查.md",
                    "provenance/best_after.json": "reproducibility/protected_reference_check.json",
                    "provenance/historical_reproduction.json": "reproducibility/historical_reproduction.json",
                    "baseline/logs/repro_p05_full_exact.log": "reproducibility/baseline_training_header.txt",
                    "baseline/outputs/clean_experiment_finals_20260801_212431/summary_sorted.md":
                        "baseline/results/historical_summary.md",
                }.get(relative, relative)
                relative = posixpath.relpath(relative, str(PurePosixPath(destination).parent))
                return "(" + relative.replace(" ", "%20") + anchor + ")"
        return match.group(0)
    return scrub(re.sub(r"\((<?X:/FaceFlux-(?:Research|Best)[^)]+>?)\)", link, text))


def add(source, destination, category):
    source = Path(source)
    if source.suffix.lower() == ".png":
        data = source.read_bytes()
    else:
        text = source.read_text(encoding="utf-8-sig")
        text = markdown(text, destination) if source.suffix.lower() == ".md" else scrub(text)
        data = text.encode("utf-8")
    files[destination] = {"data": data, "category": category, "source_sha256": sha256(source)}


def text_file(destination, text, category="release_metadata"):
    files[destination] = {"data": text.encode("utf-8"), "category": category}


def main():
    files.clear()
    EXPORT.mkdir(parents=True, exist_ok=True)
    for directory in ["src", "scripts", "tests", "configs"]:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}:
                add(path, path.relative_to(ROOT).as_posix(), "research_code")
    # User policy: future public updates contain code and aggregate results only.
    # Existing published documents are neither re-exported nor rewritten here.
    add(BEST / "train_unet.py", "baseline/train_unet.py", "historical_baseline")
    for directory in ["src", "scripts", "configs", "commands", "experiments/checkpoint_zoo"]:
        for path in sorted((BEST / directory).rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                    ".py", ".sh", ".yaml", ".yml", ".json", ".tsv", ".txt", ".sha256"}:
                if "__pycache__" not in path.parts and "retrained" not in path.parts:
                    add(path, "baseline/" + path.relative_to(BEST).as_posix(), "historical_baseline")
    add(BEST / "outputs/clean_experiment_finals_20260801_212431/eval_results.json",
        "baseline/results/historical_eval_results.json", "historical_metrics")
    add(BEST / "outputs/clean_experiment_finals_20260801_212431/summary_sorted.md",
        "baseline/results/historical_summary.md", "historical_metrics")
    header = (BEST / "logs/repro_p05_full_exact.log").read_text(encoding="utf-8").splitlines()[:8]
    text_file("reproducibility/baseline_training_header.txt", scrub("\n".join(header)) + "\n")
    for run in ["D1_full", "P0_D1_D5_pilot", "D2_D3_full_multiseed", "D3_full_imagewise_noise_v1",
                "D4_D6_face_supplements_v1", "P0_general_data_v1", "P0_prepared_hypersim_v1",
                "P0_prepared_hypersim_full_v1", "P0_object_benchmark_v1", "P0_arb_protocol_v1",
                "P0_nyu_scenes_v1", "P0_nyu_validation_v1", "P1_nyu_validation_v1", "P1_nyu_checked_v1",
                "P1_gltf_normal_textures_v1", "P1_nyu_rgb_diagnostic_v1", "P1_arb_prepared_pilot_v1",
                "P2_parameterization_pilot_v1", "P2_parameterization_v1", "P1_arb_boundary_audit_pilot_v1",
                "P1_arb_boundary_audit_formal_v1", "P1_arb_camera_cube_v1", "P1_arb_camera_blenderproc_v1",
                "P2_budget_preregistration_v2"]:
        source = ROOT / "outputs" / run / "summary.json"
        if not source.exists():
            continue
        state = json.loads(source.read_text(encoding="utf-8"))
        if state.get("status") != "completed":
            continue
        add(source, f"results/{run}/summary.json", "diagnostic_summary")
        aggregate_file = source.with_name("multiseed_summary.json")
        if aggregate_file.exists():
            add(aggregate_file, f"results/{run}/multiseed_summary.json", "diagnostic_summary")
        extra = source.with_name("aggregate_summary.json")
        if extra.exists():
            add(extra, f"results/{run}/aggregate_summary.json", "diagnostic_summary")
    prepared = ROOT / "data/hypersim_trainval_v1/summary.json"
    if prepared.exists() and json.loads(prepared.read_text()).get("status") == "completed":
        add(prepared, "results/P0_prepared_hypersim_full_v1/preparation_summary.json", "diagnostic_summary")
    add(ROOT / "outputs/protocol_tests.txt", "results/protocol_tests.txt", "test_result")
    for filename in ["D1_full.png", "D2_D4_pilot.png", "D2_D3_full_multiseed.png", "P2_convergence_center_v2.png"]:
        path = ROOT / "figures" / filename
        if path.exists():
            add(path, path.relative_to(ROOT).as_posix(), "aggregate_plot")
    add(ROOT / "provenance/historical_reproduction.json",
        "reproducibility/historical_reproduction.json", "reproducibility")
    audit = json.loads((ROOT / "provenance/best_after.json").read_text())
    text_file("reproducibility/protected_reference_check.json",
              scrub(json.dumps(audit, ensure_ascii=False, indent=2)) + "\n")
    env = json.loads((ROOT / "provenance/environment.json").read_text())
    for key in ["python", "project_root", "read_only_reference"]:
        env.pop(key, None)
    env["gpu"] = "NVIDIA GeForce RTX 4090, 24 GB"
    for package in ["matplotlib", "tqdm", "h5py", "scipy"]:
        env["packages"][package] = importlib.metadata.version(package)
    text_file("reproducibility/tested_environment.json",
              json.dumps(env, ensure_ascii=False, indent=2) + "\n")
    text_file("requirements.txt", "\n".join(
        f"{name}=={version}" for name, version in env["packages"].items()
        if name not in {"torch", "torchvision"} and version) + "\n")
    text_file("requirements-torch.txt", "torch==2.1.2\ntorchvision==0.16.2\n")
    text_file(".gitignore", """__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
.env
.env.*
*.pem
*.key
.ssh/
credentials*
data/
datasets/
checkpoints/
runtime/
logs/
outputs/
provenance/
publish/
docs/
**/per_image/
**/subset.json
**/ensemble_metrics.json
**/image_records/
*.pth
*.pt
*.ckpt
*.safetensors
*.npy
*.npz
*.dat
*.tar
*.tar.gz
*.tar.xz
*.zip
*.png
!figures/*.png
""")
    text_file("AGENTS.md", """# Repository instructions

- Treat FACEFLUX_BEST_ROOT as read-only, including code, weights and datasets.
- Write new experiments under this repository; never modify external reference files.
- Keep results/ as published snapshots. New runs write outputs/.
- Distinguish full historical results, pilot subsets, clamp metrics and float-normal metrics.
- FaceScape is a historical diagnostic domain, not a new generalization claim.
- Do not commit data, portraits, predicted image arrays, weights, credentials or machine inventories.
- Future public updates include code/configuration, operational instructions and aggregate results only.
- Keep research plans, interpretations, sample manifests and per-image records local.
- Do not change repository visibility or rewrite published history without user instruction.
- Record failures explicitly; never silently change masks or omit failed images.
""")
    imagewise_section = ""
    imagewise_path = ROOT / "outputs/D3_full_imagewise_noise_v1/summary.json"
    if imagewise_path.exists():
        imagewise = json.loads(imagewise_path.read_text(encoding="utf-8"))
        if imagewise.get("status") == "completed":
            lines = ["## Image-specific noise follow-up", "",
                     "Same fixed checkpoint and full 1,998-image set, one Euler step, eight inference replicates. "
                     "A versioned SHA256 of the base seed and dataset-relative RGB key derives a different initial field for each image. "
                     "The fields are paired across guidance modes; the historical shared-field results above are retained.", "",
                     "| Inference | MAE mean ± dataset-replicate SD, degrees | Mean within-image MAE SD, degrees | Mean pixel angular dispersion, degrees |",
                     "|---|---:|---:|---:|"]
            for mode in ["no_cfg", "geom_cfg"]:
                score = imagewise["multiseed"]["groups"][mode + "_k1"]["protocols"]["float"]
                sensitivity = imagewise["multiseed"]["noise_sensitivity"][mode]
                angle = sensitivity["mean_pixel_angular_dispersion_deg"]
                angle_text = "undefined (see JSON counts)" if angle is None else f"{angle:.4f}"
                lines.append(f"| {mode} | {score['mean_mae_deg']:.4f} ± {score['std_seed_ddof1_deg']:.4f} | "
                             f"{sensitivity['mean_image_mae_sd_ddof1_deg']:.4f} | {angle_text} |")
            lines.extend(["", "The dataset-replicate SD measures Monte Carlo variation of the dataset mean. "
                          "Within-image MAE SD and pixel directional dispersion measure noise sensitivity; they are distinct statistics. "
                          "Displayed SDs use ddof=1. None is training-seed SD or a generalization confidence interval. "
                          "Pixel dispersion is the mean angle to the directional sample mean within each evaluation mask, then macro-averaged over images.", "",
                          "[Aggregate results](results/D3_full_imagewise_noise_v1/multiseed_summary.json) include the shared-field reference "
                          "and the mean directional variance 1-R. Sample keys, seed mappings and per-image records are not exported.", ""])
            imagewise_section = "\n".join(lines)
    supplement_path = ROOT / "outputs/D4_D6_face_supplements_v1/aggregate_summary.json"
    if supplement_path.exists():
        supplement = json.loads(supplement_path.read_text())
        if supplement.get("status") == "completed":
            lines = ["## Directional averaging and identity resampling", "",
                     "Same full historical set and image-specific noise. Existing checkpoints are kept separate; neither was retrained.", "",
                     "| Checkpoint | Conditional MAE° | Geom-CFG MAE° | Geom-CFG ensemble of 8 MAE° | Single-sample Geom-CFG identity interval° |",
                     "|---|---:|---:|---:|---|"]
            for name, result in supplement["supplements"].items():
                groups = result["single_sample"]["groups"]
                conditional = groups["no_cfg_k1"]["protocols"]["float"]["mean_mae_deg"]
                geom = groups["geom_cfg_k1"]["protocols"]["float"]["mean_mae_deg"]
                ensemble = result["ensemble8"]["geom_cfg"]["float"]["mean"]
                ci = supplement["identity_bootstrap"]["metrics"][name + "/geom_cfg/single"]
                lines.append(f"| {name} | {conditional:.4f} | {geom:.4f} | {ensemble:.4f} | [{ci['lower_95']:.4f}, {ci['upper_95']:.4f}] |")
            lines.extend(["", "Intervals are 95% percentile cluster-bootstrap intervals: 100 identities resampled 10,000 times, "
                          "conditional on the fixed weights, inference draws and development-domain identity population. "
                          "Whole clusters are sampled and the macro-image ratio is recomputed. They do not include training-seed variation or establish generalization to new domains. "
                          "Eight-sample directional averaging costs eight samples and changes the estimator; its gain is not an isolated irreducible noise cost.", "",
                          "[Aggregate results and paired intervals](results/D4_D6_face_supplements_v1/aggregate_summary.json).", ""])
            imagewise_section += "\n" + "\n".join(lines)
    data_audit = ROOT / "outputs/P0_general_data_v1/summary.json"
    full_check = ROOT / "outputs/P0_prepared_hypersim_full_v1/summary.json"
    if data_audit.exists() and full_check.exists():
        imagewise_section += "\n## Data preparation status\n\n" + (
            "The existing Hypersim conversion contains 7,325 samples; 3,146 images have at least one zero normal inside the old valid mask. "
            "A new native-geometry FP32 cache contains 5,736 train and 700 validation images from 30/5 disjoint scenes. "
            "All prepared samples pass the included loader checks. Exact camera projection matrices are retained; an ordinary K has a separate validity flag. "
            "RGB comes from lossy final-preview images, not the original HDR color.\n\n"
            "The first comparison does not use ray inputs, DINO or CFG. Official NYUv2 scene metadata and train/test indices are checked: "
            "795 images from 249 rooms are reserved for real-indoor validation, with no room overlap with the 654-image test partition. "
            "All 795 packaged RGB/normal/K validation pairs have been prepared and checked; non-identical official MAT RGB does not block these pairs. "
            "Arb retains its BSDF normal-pass label tag and missing calibration; Hypersim retains normal_cam geometry labels. "
            "The 16-archive pilot acquired 64 objects, of which 62 and 184 views have valid labels; eight empty-label views are explicitly excluded. "
            "Masks use finite near-unit EXR normals, with alpha only auxiliary. The pilot inherits the complete formal asset split. "
            "All eight 100-update pilot recipes completed with finite gradients, shared configuration and matching data signatures. "
            "These are numerical preflight results, not method rankings or convergence evidence. The 2,000-asset formal acquisition is separate. "
            "The 300 DSINE ScanNet benchmark frames and 42 OLATverse validation objects remain reserved. "
            "Data, sample manifests and prepared caches are not distributed. "
            "[Inventory](results/P0_general_data_v1/summary.json), "
            "[prepared-cache checks](results/P0_prepared_hypersim_full_v1/summary.json), "
            "[object evaluation inventory](results/P0_object_benchmark_v1/summary.json), "
            "[Arb sample checks](results/P0_arb_protocol_v1/summary.json), "
            "[NYUv2 mapping checks](results/P0_nyu_validation_v1/summary.json), "
            "[packaged validation checks](results/P1_nyu_checked_v1/summary.json), "
            "[eight-recipe preflight](results/P2_parameterization_pilot_v1/summary.json).\n")
    text_file("README.md", """# FNFlow

Pixel-space conditional Flow Matching for face normals, with isolated research code for general object and indoor-scene normal estimation.

This repository contains the historical UNet baseline, experiment code and aggregate numerical results. **Controlled no-ray R/V/Xv/Xn preflight is complete; no completed multi-seed formal generalization result is claimed.** Future public updates are limited to code/configuration, operational instructions and aggregate results.

## Contents

- [Historical baseline](baseline/)
- [Full results](results/D1_full/summary.json)
- [Pilot results](results/P0_D1_D5_pilot/summary.json)
- [Full eight-seed results](results/D2_D3_full_multiseed/multiseed_summary.json)
- [Publication scope](PUBLICATION.md)

## First diagnostic results

UNet-B has 43,081,539 parameters. Full FaceScape diagnostics use 1,998 images and the historical face-mask protocol. The original best result is reproduced at **17.899513816 degrees**. These are historical development-domain results, not object/scene generalization results.

| Training checkpoint | Conditional | Fixed CFG 1.5 | Geom-CFG |
|---|---:|---:|---:|
| dropout 0.3 | 19.7527 | 21.6212 | 19.8666 |
| dropout 0.5, original | 17.9906 | 19.3185 | 17.9513 |
| dropout 0.7 | 25.6053 | 29.2225 | 27.3178 |

Mean angular error in degrees; lower is better. A checkpoint trained without condition dropout is evaluated only conditionally.

![Full CFG diagnostics](figures/D1_full.png)

Step-count, eight-seed and averaging diagnostics use a fixed **32-identity pilot**, not all 1,998 images. They show remaining noise dependence and do not establish calibrated uncertainty.

![Pilot diagnostics](figures/D2_D4_pilot.png)

## Full eight-seed follow-up

The follow-up uses the fixed p05_best checkpoint, all 1,998 historical images and inference seeds 42–49. Each score is first averaged over images, then over seeds. SD is the sample SD of the eight dataset MAEs (ddof=1), not training-seed variation or a generalization confidence interval. Identical initial fields are paired across step counts; the historical evaluator resets a seed per image.

| Inference | Euler steps | Float-normal MAE mean ± seed SD, degrees |
|---|---:|---:|
| Geom-CFG | 1 | 19.2632 ± 1.0382 |
| Conditional | 1 | 19.5688 ± 1.1633 |
| Conditional | 2 | 19.8400 ± 1.4971 |
| Conditional | 4 | 20.2608 ± 1.9732 |
| Conditional | 8 | 20.5925 ± 2.3104 |

All 40 configurations completed. Per-seed aggregate scores and paired step differences are in the linked JSON. No sample identifiers or per-image results are added by this follow-up release.

__IMAGEWISE_NOISE_RESULTS__

## Setup

Tested on Linux, Python 3.10.19, PyTorch 2.1.2 + CUDA 12.1, torchvision 0.16.2 and RTX 4090 (24 GB). Observed versions are in [tested_environment.json](reproducibility/tested_environment.json).

Install an appropriate PyTorch build for your hardware using requirements-torch.txt, then the recorded dependencies in requirements.txt. dlib may require compiler/CMake support.

Datasets, weights and the landmark predictor are **not distributed**. Obtain them separately with the relevant permissions. Set a read-only reference directory:

~~~bash
export FACEFLUX_BEST_ROOT=/path/to/FaceFlux-Best
export PYTHONDONTWRITEBYTECODE=1
~~~

Expected checkpoint and data paths are declared in [diagnostics_pilot.json](configs/diagnostics_pilot.json).

## Run

From the repository root:

~~~bash
python -m pytest -q tests -o cache_dir=outputs/.pytest_cache
python scripts/preflight.py
python scripts/run_diagnostics.py
python scripts/run_diagnostics.py --stage d1 --subjects 0 --run-id D1_full
python scripts/run_multiseed_diagnostics.py
python scripts/run_imagewise_noise.py
python scripts/verify_best.py
python scripts/plot_diagnostics.py
python scripts/summarize_diagnostics.py
python scripts/render_record.py
~~~

Published snapshots are in results/; new runs write outputs/. The original preflight audit is intentionally not overwritten. Reusing a run ID requires identical code/configuration/sample selection.

Report scripts regenerate workspace-oriented documents. This publication export normalizes machine paths and document links.

The v1 no-ray pilot uses configs/parameterization_first_round.json and remains a historical numerical preflight. The current entry point scripts/run_parameterization_pipeline.py forwards to scripts/run_convergence_pipeline.py with configs/parameterization_first_round_v2_review1.json (the original v2 numerical-smoke configuration is preserved). It builds new native-center-crop Hypersim/NYUv2 caches, checks all samples, runs numerical smoke checks, then waits for the separately acquired 2,000-object data before the matched Xv clipping diagnostic and R/V/Xn validation curves. These use one training seed and do not establish a ranking. Every optimizer update records clipping and image-weighted time-bin losses. Every curve point also evaluates a fixed training-only probe (64 images per source domain). configs/convergence_decision_v2.json preregisters the 0.3-degree extension/tolerance rule and per-domain rank stability; the timestamp/hash receipt was recorded before any long-run curve existed. Arb boundary MAE is accompanied by foreground exclusion/coverage counts, with relaxed candidate edges kept distinct from trusted labels. The formal grid is locked; acquisition completion cannot launch it, and a --steps override cannot bypass the budget gate. Review the per-domain curves and common early-window clipping before creating a separate reviewed formal configuration. scripts/summarize_convergence.py writes local aggregate progress and plots once curve points exist. scripts/summarize_parameterization.py keeps training-seed and inference-seed variation separate for the v1 grid. Data/weights/manifests are not distributed. Camera checks require Blender 3.3 and the pinned BlenderProc 2.5 defaults. Completed checks are exported only as explicit numerical summary snapshots; no inferred K is inserted into frozen no-ray manifests. src/dino_conditioning.py and configs/dino_conditioning_scaffold.json provide code-only frozen-DINO conditioning; contract tests use a mock encoder. No DINO training or real pretrained-weight compatibility result is claimed.

## Scope

baseline/ preserves the historical face pipeline and checkpoint-zoo configurations. Its command snapshots require local path adaptation; they are not the new general training entry point.

The source and output protocols are covered by the included tests. Full-set and pilot measurements are distinct and should not be pooled.

No dataset, pretrained-weight license or new software-license grant is implied by repository visibility. See [NOTICE.md](NOTICE.md).
""".replace("__IMAGEWISE_NOISE_RESULTS__", imagewise_section))
    text_file("NOTICE.md", """# Attribution and usage

This is the user's FNFlow research snapshot. Historical model comments cite ADM and Rectified Flow. Research documents link to primary papers; original attributions are retained.

Datasets, external weights, landmark predictors and images are not included and remain subject to their own licenses/access requirements.

No new software license was selected for this upload. Public visibility alone does not establish a license grant.

Publication normalizes machine-specific paths, command examples and document links. Source-side and exported hashes are distinguished in PUBLICATION_MANIFEST.json.
""")
    text_file("PUBLICATION.md", """# Publication scope

Future updates include Python code, tests/configuration, operational instructions, historical baseline source/configuration, aggregate plots, completed numerical summaries and reproducibility checks.

Excluded from future exports: research plans/interpretations, sample manifests, per-image CSV/JSON, RGB/normal/depth data, portrait visualizations, predicted normal arrays, checkpoints, landmark binaries, raw machine inventories, private environment details and caches.

The user chose to keep this repository public on 2026-10-22. Previously published files and history remain; this policy changes future updates and does not retract the original snapshot. Old tracked documents/sample-level files are not re-exported or updated.

Local absolute paths are replaced with placeholders/relative links. Original source remains under the isolated workspace. FaceFlux-Best was not modified.

results/ stores published snapshots; new experiments write outputs/ and should be exported/reviewed before publication. PUBLICATION_MANIFEST.json records the files approved for the current export, excluding itself; it is not an inventory of unchanged legacy files. The upload bundle uses that exact allowlist, never a recursive scan of an existing export directory.
""")
    text_file("baseline/README.md", """# Historical baseline

Read-only copies of the canonical face model, training/evaluation/inference code, commands and checkpoint-zoo configurations. Data and weights are external.

Historical reproduction metrics are in results/. Machine-specific command paths are normalized and require adaptation. Some historical commands clean intermediate checkpoints; these snapshots are preserved as evidence and are not automatically executed.

Use the top-level experiment scripts for new runs.
""")
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
        "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
        "url_password": r"https?://[^/\s:@]+:[^/\s@]+@",
    }
    violations = []
    for name, record in files.items():
        if name.startswith("docs/") or "/per_image/" in name or "/image_records/" in name or name.endswith(("/subset.json", "/ensemble_metrics.json")):
            violations.append({"file": name, "rule": "private_research_or_sample_records"})
        if name.lower().endswith((".pth", ".pt", ".npy", ".npz", ".dat", ".pem", ".key")):
            violations.append({"file": name, "rule": "excluded_binary"})
        if name.startswith(("data/", "datasets/", "checkpoints/")):
            violations.append({"file": name, "rule": "excluded_directory"})
        if name.startswith("results/") and name.endswith(".json"):
            def inspect(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in {"subject", "subject_id", "rgb_path", "normal_path", "pairs", "per_image", "filename", "sample_id", "image_key", "derived_seeds", "input_sha256"}:
                            violations.append({"file": name, "rule": "sample_identifier_key"})
                        inspect(child)
                elif isinstance(value, list):
                    for child in value:
                        inspect(child)
            inspect(json.loads(record["data"]))
        if not name.endswith(".png"):
            content = record["data"].decode("utf-8")
            for rule, pattern in patterns.items():
                if re.search(pattern, content):
                    violations.append({"file": name, "rule": rule})
    if violations:
        write_json(ROOT / "publish/scan_failures.json", violations)
        raise RuntimeError("Publication scan failed; inspect filenames/rules only")
    manifest = []
    for name, record in sorted(files.items()):
        target = output_path(EXPORT / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(record["data"])
        manifest.append({"path": name, "category": record["category"], "bytes": len(record["data"]),
                         "exported_sha256": hashlib.sha256(record["data"]).hexdigest(),
                         **({"source_sha256": record["source_sha256"]} if "source_sha256" in record else {})})
    (EXPORT / "PUBLICATION_MANIFEST.json").write_text(json.dumps(
        {"created_at": now(), "scan_passed": True,
         "scope": "current code-and-aggregate export; unchanged legacy files are not re-exported",
         "files": manifest},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = []
    # Never scoop up stale research documents, sample manifests or unreviewed files.
    for name in sorted([*files, "PUBLICATION_MANIFEST.json"]):
        path = EXPORT / name
        raw = path.read_bytes()
        encoding = "base64" if path.suffix == ".png" else "utf-8"
        bundle.append({"path": name, "encoding": encoding,
                       "content": base64.b64encode(raw).decode() if encoding == "base64" else raw.decode("utf-8"),
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    transport = json.dumps(bundle, ensure_ascii=True, separators=(",", ":"))
    (ROOT / "publish/upload_bundle.json").write_text(transport, encoding="ascii")
    info = {"export_root": str(EXPORT), "files": len(bundle),
            "total_bytes": sum(item["bytes"] for item in bundle),
            "largest_file_bytes": max(item["bytes"] for item in bundle),
            "bundle_characters": len(transport),
            "bundle_sha256": hashlib.sha256(transport.encode("ascii")).hexdigest(),
            "scan_passed": True}
    write_json(ROOT / "publish/export_summary.json", info)
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
