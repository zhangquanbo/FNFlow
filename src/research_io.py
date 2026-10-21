"""Workspace-bound output helpers and explicit provenance."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
BEST = Path(os.environ.get("FACEFLUX_BEST_ROOT", ROOT.parent / "FaceFlux-Best")).expanduser().resolve()


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def output_path(path):
    target = Path(path).resolve()
    if target.is_relative_to(BEST):
        raise ValueError(f"Protected reference cannot be an output target: {target}")
    if not target.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Output must remain inside {ROOT}: {target}")
    return target


def write_json(path, value):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, target)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def best_snapshot():
    """Do not follow directory symlinks; capture all Best entries and code hashes."""
    entries = {}
    code_extensions = {".py", ".md", ".yaml", ".yml", ".json", ".sh", ".tsv"}
    for directory, subdirs, files in os.walk(BEST, followlinks=False):
        for name in sorted(subdirs + files):
            path = Path(directory) / name
            stat = path.lstat()
            item = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "kind": "symlink" if path.is_symlink() else
                            "directory" if path.is_dir() else "file"}
            if path.is_symlink():
                item["target"] = os.readlink(path)
            elif path.is_file() and path.suffix.lower() in code_extensions:
                item["sha256"] = sha256(path)
            entries[path.relative_to(BEST).as_posix()] = item
    return {"captured_at": now(), "root": str(BEST), "entries": entries}


def compare_best(snapshot):
    current = best_snapshot()
    old = snapshot["entries"]
    new = current["entries"]
    return {"checked_at": now(), "entry_count": len(new),
            "added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "changed": sorted(key for key in set(old) & set(new) if old[key] != new[key]),
            "verified_unchanged": old == new,
            "scope": "all non-followed entries: size/mtime/type; code/config/document content hashes"}
