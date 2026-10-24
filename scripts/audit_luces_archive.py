"""Read-only gzip/tar integrity check and member index; never extracts images."""
import gzip, hashlib, json, sys, tarfile, time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from src.research_io import write_json, sha256

archive = ROOT / 'data/real_object_raw/luces_opendatalab/OpenDataLab___LUCES/raw/LUCES.tar.gz'
out = ROOT / 'data/real_object_selected_v1/luces'
out.mkdir(parents=True, exist_ok=True)
protocol = out / 'intake_before_extraction.json'
if not protocol.exists():
    write_json(protocol, {
        'source': 'https://opendatalab.com/OpenDataLab/LUCES/tree/main/raw',
        'archive': str(archive), 'archive_size_bytes': archive.stat().st_size,
        'selection': 'All original 14 objects; lowest numerical lighting ID, lexical tie break; official GT and masks. No image or error inspection for selection.',
        'parent_amendment_sha256': sha256(ROOT / 'docs/object_calibration_intake_amendment_20261023.md'),
        'authorization': 'User requested server download commands and reported download complete. Whole archive acquisition authorized by that request.',
        'created_unix': time.time(), 'scores_computed': False,
    })
    protocol.with_suffix('.json.sha256').write_text(sha256(protocol) + '\n')

class HashingReader:
    def __init__(self, f):
        self.f = f; self.digest = hashlib.sha256(); self.count = 0; self.last = time.monotonic()
    def read(self, size=-1):
        b = self.f.read(size); self.digest.update(b); self.count += len(b)
        if time.monotonic() - self.last > 20:
            write_json(out / 'audit_progress.json', {'compressed_bytes_read': self.count, 'archive_bytes': archive.stat().st_size, 'updated_unix': time.time()})
            print(f'compressed bytes {self.count}/{archive.stat().st_size}', flush=True)
            self.last = time.monotonic()
        return b

try:
    before = archive.stat(); entries = []; seen = set()
    with archive.open('rb') as raw:
        stream = HashingReader(raw)
        with gzip.GzipFile(fileobj=stream) as gz:
            with tarfile.open(fileobj=gz, mode='r|') as tar:
                for m in tar:
                    p = PurePosixPath(m.name)
                    if p.is_absolute() or '..' in p.parts: raise ValueError(f'Unsafe path: {m.name}')
                    if m.name in seen: raise ValueError(f'Duplicate member: {m.name}')
                    seen.add(m.name)
                    entries.append({'name': m.name, 'size': m.size, 'type': m.type.decode('ascii', errors='replace')})
            # Reading through EOF verifies gzip CRC/length, including trailing blocks.
            while gz.read(8 * 1024 * 1024): pass
        if stream.count != before.st_size: raise ValueError('Compressed stream was not fully read')
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns): raise ValueError('Archive changed during audit')
    write_json(out / 'archive_index_private.json', {'archive': str(archive), 'entries': entries})
    write_json(out / 'archive_integrity.json', {'status': 'passed', 'gzip_crc_and_length_checked': True, 'tar_headers_checked': True, 'archive_size_bytes': before.st_size, 'archive_sha256': stream.digest.hexdigest(), 'member_count': len(entries), 'index_sha256': sha256(out / 'archive_index_private.json'), 'completed_unix': time.time()})
    print(json.dumps({'status': 'passed', 'members': len(entries), 'sha256': stream.digest.hexdigest()}), flush=True)
except Exception as e:
    write_json(out / 'audit_failure.json', {'error': repr(e), 'time': time.time()})
    raise
