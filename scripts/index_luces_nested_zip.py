"""Materialize the nested ZIP as a temporary intake cache, then index headers."""
import hashlib,json,os,shutil,sys,tarfile,time,zipfile
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.research_io import write_json,sha256
out=ROOT/'data/real_object_selected_v1/luces'
audit=json.loads((out/'archive_integrity.json').read_text());assert audit['status']=='passed'
archive=Path(json.loads((out/'archive_index_private.json').read_text())['archive'])
assert archive.stat().st_size==audit['archive_size_bytes']
cache=ROOT/'runtime/downloads/luces_inner_audited.zip'
receipt=out/'nested_zip_receipt.json'
if not (cache.is_file() and receipt.is_file() and cache.stat().st_size==json.loads(receipt.read_text())['size_bytes'] and sha256(cache)==json.loads(receipt.read_text())['sha256']):
    part=cache.with_suffix('.zip.part');expected=12123247159
    assert shutil.disk_usage(cache.parent).free>expected+3*1024**3,'Insufficient temporary disk space'
    digest=hashlib.sha256();count=0;last=time.monotonic()
    with tarfile.open(archive,'r|gz') as tar:
        for m in tar:
            if m.name!='LUCES/Luces_code.zip':continue
            assert m.isfile() and m.size==expected
            with tar.extractfile(m) as src,part.open('wb') as dest:
                for b in iter(lambda:src.read(8*1024*1024),b''):
                    dest.write(b);digest.update(b);count+=len(b)
                    if time.monotonic()-last>20:print(f'inner ZIP {count}/{expected}',flush=True);last=time.monotonic()
            break
        else:raise ValueError('Expected nested ZIP absent')
    assert count==expected
    os.replace(part,cache)
    write_json(receipt,{'path':str(cache),'size_bytes':count,'sha256':digest.hexdigest(),'parent_archive_sha256':audit['archive_sha256'],'temporary_generated_cache':True})
with zipfile.ZipFile(cache) as z:
    infos=z.infolist();names=[i.filename for i in infos];assert len(names)==len(set(names))
    for name in names:
        p=PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in name,name
    entries=[{'name':i.filename,'size':i.file_size,'compressed_size':i.compress_size,'crc32':f'{i.CRC:08x}'} for i in infos]
write_json(out/'nested_zip_index_private.json',{'entries':entries,'receipt_sha256':sha256(receipt)})
print(json.dumps({'members':len(entries),'first_100':entries[:100]}),flush=True)
