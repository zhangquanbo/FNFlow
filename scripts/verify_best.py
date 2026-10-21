"""Verify protected source entries and the contents of every used checkpoint."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from src.research_io import ROOT, now, sha256, write_json, compare_best


def main():
    baseline = json.loads((ROOT / "provenance/best_before.json").read_text())
    result = compare_best(baseline)
    weights = {}
    for path in (ROOT / "outputs").glob("*/summary.json"):
        run = json.loads(path.read_text())
        for checkpoint in run.get("checkpoints", {}).values():
            source = checkpoint["path"]
            weights[source] = {"expected_sha256": checkpoint["sha256"],
                               "current_sha256": sha256(source)}
    for item in weights.values():
        item["matches"] = item["expected_sha256"] == item["current_sha256"]
    result["checkpoint_contents"] = weights
    result["verified_unchanged"] &= all(item["matches"] for item in weights.values())
    write_json(ROOT / "provenance/best_after.json", result)
    print(json.dumps({"checked_at": now(), "best_unchanged": result["verified_unchanged"],
                      "entries": result["entry_count"], "used_checkpoints": len(weights)}))
    if not result["verified_unchanged"]:
        raise SystemExit("Protected reference differs; inspect provenance/best_after.json")


if __name__ == "__main__":
    main()
