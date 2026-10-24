"""Index Google Drive ZIP parts inside the mirror archive using bounded temp files."""
import hashlib,json,shutil,sys,tempfile,zipfile
from collections import Counter
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.research_io import write_json,sha256
out=ROOT/'data/real_object_selected_v1/luces';cache=ROOT/'runtime/downloads/luces_inner_audited.zip'
receipt=json.loads((out/'nested_zip_receipt.json').read_text());assert sha256(cache)==receipt['sha256']
entries=[];parts=[]
with zipfile.ZipFile(cache) as outer:
    for member in sorted(i.filename for i in outer.infolist() if i.filename.endswith('.zip')):
        info=outer.getinfo(member);assert shutil.disk_usage(cache.parent).free>info.file_size+1024**3
        digest=hashlib.sha256()
        with tempfile.TemporaryFile(dir=cache.parent) as f:
            with outer.open(member) as src:
                for b in iter(lambda:src.read(8*1024*1024),b''):f.write(b);digest.update(b)
            assert f.tell()==info.file_size;f.seek(0)
            with zipfile.ZipFile(f) as inner:
                infos=inner.infolist();assert len({i.filename for i in infos})==len(infos)
                for i in infos:
                    p=PurePosixPath(i.filename)
                    assert not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename,i.filename
                    entries.append({'part':member,'name':i.filename,'size':i.file_size,'compressed_size':i.compress_size,'crc32':f'{i.CRC:08x}'})
        parts.append({'member':member,'size_bytes':info.file_size,'sha256':digest.hexdigest(),'outer_member_crc_checked':True,'members':len(infos)})
        print(member,len(infos),'members',flush=True)
write_json(out/'parts_index_private.json',{'entries':entries,'parts':parts,'parent_sha256':receipt['sha256']})
print(json.dumps({'members':len(entries),'prefixes':dict(Counter('/'.join(e['name'].split('/')[:3]) for e in entries)), 'metadata':[e for e in entries if not e['name'].endswith('/') and not PurePosixPath(e['name']).suffix.lower() in ['.png','.jpg','.jpeg','.tif','.tiff','.npz','.mat','.obj','.ply','.stl']][:80]}),flush=True)
