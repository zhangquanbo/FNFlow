"""Serialize research training and rendering jobs on the shared single GPU."""
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def gpu_job_lock():
    import fcntl
    from src.research_io import ROOT
    path=ROOT/'runtime/gpu_job.lock';path.parent.mkdir(exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)
