"""Capture the read-only baseline before running experiments."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.metadata
import platform
import subprocess
import torch
from src.research_io import ROOT, BEST, now, sha256, write_json, best_snapshot


def main():
    target = ROOT / "provenance/best_before.json"
    if target.exists():
        raise RuntimeError("Baseline snapshot already exists; do not overwrite the original audit.")
    write_json(target, best_snapshot())
    packages = {}
    for name in ["torch", "torchvision", "numpy", "Pillow", "opencv-python",
                 "dlib", "accelerate", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                          "--format=csv,noheader"], text=True, capture_output=True)
    write_json(ROOT / "provenance/environment.json",
               {"captured_at": now(), "python": sys.executable,
                "python_version": sys.version, "platform": platform.platform(),
                "packages": packages, "cuda_available": torch.cuda.is_available(),
                "torch_cuda": torch.version.cuda, "gpu": gpu.stdout.strip(),
                "project_root": str(ROOT), "read_only_reference": str(BEST)})
    source_pairs = {
        "src/models/model_unet.py": "src/models/model_unet.py",
        "scripts/legacy/eval_facescape_modes.py": "scripts/eval/eval_facescape_modes.py",
        "provenance/train_unet_original.py.txt": "train_unet.py",
        "docs/研究方案.md": "docs/normal_research_execution_plan_2026-10-21.md",
    }
    copies = [{"local": local, "source": str(BEST / source),
               "sha256": sha256(ROOT / local),
               "matches_source": sha256(ROOT / local) == sha256(BEST / source)}
              for local, source in source_pairs.items() if (ROOT / local).is_file()]
    # Published documents/snapshots can have normalized paths; require exact model bytes.
    model_copy = next(item for item in copies if item["local"] == "src/models/model_unet.py")
    if not model_copy["matches_source"]:
        raise RuntimeError("Copied model differs from the protected reference.")
    write_json(ROOT / "provenance/source_copies.json", copies)
    data = {"captured_at": now(), "sources": []}
    for relative in ["datasets/ffhq_rgb_crop256", "datasets/ffhq_rgb_crop256N",
                     "datasets/FacescapeRender1/RImage",
                     "datasets/FacescapeRender1/RNormal", "data/test_images_1/imgs"]:
        path = BEST / relative
        data["sources"].append({"path": str(path), "resolved": str(path.resolve()),
                                "exists": path.exists(),
                                "png_files": sum(1 for _ in path.rglob("*.png")),
                                "usage": "read-only; face labels have unconfirmed provenance"})
    general = ROOT.parent / "Dataset/General"
    data["general_sources_present"] = sorted(p.name for p in general.iterdir()) if general.exists() else []
    write_json(ROOT / "provenance/data_inventory.json", data)
    print("Preflight captured. Best snapshot:", len(__import__("json").loads(target.read_text())["entries"]),
          "entries; CUDA:", torch.cuda.is_available(), flush=True)


if __name__ == "__main__":
    main()
