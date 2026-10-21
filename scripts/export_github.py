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
        str(ROOT): "$" + "{FNFLOW_ROOT}",
        str(BEST): "$" + "{FACEFLUX_BEST_ROOT}",
        str(ROOT.parent) + "/": "$" + "{WORKSPACE_ROOT}/",
        str(Path.home()) + "/": "$" + "{HOME}/",
        "${FNFLOW_ROOT}": "$" + "{FNFLOW_ROOT}",
        "X:\\FaceFlux-Research": "$" + "{FNFLOW_ROOT}",
        "${FACEFLUX_BEST_ROOT}": "$" + "{FACEFLUX_BEST_ROOT}",
        "X:\\FaceFlux-Best": "$" + "{FACEFLUX_BEST_ROOT}",
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
        for prefix, replacement in [("${FNFLOW_ROOT}/", ""),
                                     ("${FACEFLUX_BEST_ROOT}/", "baseline/")]:
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
    EXPORT.mkdir(parents=True, exist_ok=True)
    for directory in ["src", "scripts", "tests", "configs"]:
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".yaml", ".yml"}:
                add(path, path.relative_to(ROOT).as_posix(), "research_code")
    for path in sorted((ROOT / "docs").glob("*.md")):
        add(path, path.relative_to(ROOT).as_posix(), "research_document")
    add(BEST / "docs/general_normal_research_2026-10-21.md",
        "docs/原始代码审计与文献.md", "research_document")
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
    for run in ["D1_full", "P0_D1_D5_pilot"]:
        for filename in ["summary.json", "subset.json"]:
            add(ROOT / "outputs" / run / filename, f"results/{run}/{filename}", "diagnostic_summary")
        for path in sorted((ROOT / "outputs" / run).rglob("*.csv")):
            add(path, "results/" + path.relative_to(ROOT / "outputs").as_posix(), "per_image_metrics")
        if (ROOT / "outputs" / run / "ensemble_metrics.json").exists():
            add(ROOT / "outputs" / run / "ensemble_metrics.json",
                f"results/{run}/ensemble_metrics.json", "ensemble_metrics")
    add(ROOT / "outputs/protocol_tests.txt", "results/protocol_tests.txt", "test_result")
    for path in sorted((ROOT / "figures").glob("*.png")):
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
    for package in ["matplotlib", "tqdm"]:
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
logs/
outputs/
provenance/
publish/
*.pth
*.pt
*.ckpt
*.safetensors
*.npy
*.npz
*.dat
*.tar
*.tar.gz
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
- Preserve experiment ordering and decision criteria; do not add calendar schedules.
- Record failures explicitly; never silently change masks or omit failed images.
""")
    text_file("README.md", """# FNFlow

Pixel-space conditional Flow Matching for face normals, with isolated research code for general object and indoor-scene normal estimation.

This repository contains the historical UNet baseline, research code, revised plan and numerical results from the first diagnostic experiments. **No new general-domain model has been trained yet.**

## Contents

- [Research plan](docs/研究方案.md)
- [Code and experiment record](docs/代码与实验记录.md)
- [First diagnostic findings](docs/首轮诊断结论.md)
- [Dataset source audit](docs/数据来源初查.md)
- [Historical baseline](baseline/)
- [Full results](results/D1_full/summary.json)
- [Pilot results](results/P0_D1_D5_pilot/summary.json)
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
python scripts/verify_best.py
python scripts/plot_diagnostics.py
python scripts/summarize_diagnostics.py
python scripts/render_record.py
~~~

Published snapshots are in results/; new runs write outputs/. The original preflight audit is intentionally not overwritten. Reusing a run ID requires identical code/configuration/sample selection.

Report scripts regenerate workspace-oriented documents. This publication export normalizes machine paths and document links.

## Scope

baseline/ preserves the historical face pipeline and checkpoint-zoo configurations. Its command snapshots require local path adaptation; they are not the new general training entry point.

There are 13 passing protocol tests. The research direction remains a hypothesis: reliable geometric detail under mixed-quality supervision, tested against matched regression/FM controls.

No dataset, pretrained-weight license or new software-license grant is implied by repository visibility. See [NOTICE.md](NOTICE.md).
""")
    text_file("NOTICE.md", """# Attribution and usage

This is the user's FNFlow research snapshot. Historical model comments cite ADM and Rectified Flow. Research documents link to primary papers; original attributions are retained.

Datasets, external weights, landmark predictors and images are not included and remain subject to their own licenses/access requirements.

No new software license was selected for this upload. Public visibility alone does not establish a license grant.

Publication normalizes machine-specific paths, command examples and document links. Source-side and exported hashes are distinguished in PUBLICATION_MANIFEST.json.
""")
    text_file("PUBLICATION.md", """# Publication scope

Included: research Python code, tests/configuration, historical baseline source/configuration, research documents, aggregate plots, all completed per-image CSVs, selected summaries/sample manifests and reproducibility checks.

Excluded: RGB/normal/depth data, portrait visualizations, predicted normal arrays, checkpoints, landmark binaries, raw machine inventories, private environment details, caches and repetitive per-image JSON (equivalent per-image CSVs and aggregate JSONs are included).

Local absolute paths are replaced with placeholders/relative links. Original source remains under the isolated workspace. FaceFlux-Best was not modified.

results/ stores published snapshots; new experiments write outputs/ and should be exported/reviewed before publication. PUBLICATION_MANIFEST.json records exported sizes and hashes, excluding itself.
""")
    text_file("baseline/README.md", """# Historical baseline

Read-only copies of the canonical face model, training/evaluation/inference code, commands and checkpoint-zoo configurations. Data and weights are external.

Historical reproduction metrics are in results/. Machine-specific command paths are normalized and require adaptation. Some historical commands clean intermediate checkpoints; these snapshots are preserved as evidence and are not automatically executed.

Use the top-level research scripts and revised plan for new experiments.
""")
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
        "github_token": r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
        "aws_key": r"\bAKIA[0-9A-Z]{16}\b",
        "url_password": r"https?://[^/\s:@]+:[^/\s@]+@",
    }
    violations = []
    for name, record in files.items():
        if name.lower().endswith((".pth", ".pt", ".npy", ".npz", ".dat", ".pem", ".key")):
            violations.append({"file": name, "rule": "excluded_binary"})
        if name.startswith(("data/", "datasets/", "checkpoints/")):
            violations.append({"file": name, "rule": "excluded_directory"})
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
        {"created_at": now(), "scan_passed": True, "files": manifest},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    bundle = []
    for path in sorted(EXPORT.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            raw = path.read_bytes()
            encoding = "base64" if path.suffix == ".png" else "utf-8"
            bundle.append({"path": path.relative_to(EXPORT).as_posix(), "encoding": encoding,
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
