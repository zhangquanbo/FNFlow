"""Freeze lowest-ID processed RGB choices before extracting selected ZIP members."""
import hashlib,json,re,shutil,sys,tempfile,time,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.research_io import write_json,sha256
out=ROOT/'data/real_object_selected_v1/luces';index=out/'parts_index_private.json'
doc=json.loads(index.read_text());entries=doc['entries'];prefix='Luces_25082022/data/'
named={e['name']:e for e in entries};assert len(named)==len(entries)
objects=sorted({e['name'][len(prefix):].split('/')[0] for e in entries if e['name'].startswith(prefix) and '/' in e['name'][len(prefix):]})
assert len(objects)==14,objects
selection=[];wanted=set()
for obj in objects:
    base=prefix+obj+'/'
    rgb=[e for e in entries if e['name'].startswith(base) and re.fullmatch(r'\d+\.png',e['name'][len(base):])]
    assert {int(PurePosixPath(e['name']).stem) for e in rgb}==set(range(1,53)),obj
    choice=min(rgb,key=lambda e:(int(PurePosixPath(e['name']).stem),e['name']))
    row={'object':obj,'lighting_id':int(PurePosixPath(choice['name']).stem),'rgb_member':choice['name'],'normal_member':base+'normals.png','mask_member':base+'mask.png','calibration_member':base+'led_params.txt'}
    assert all(row[k] in named for k in ['rgb_member','normal_member','mask_member','calibration_member'])
    selection.append(row);wanted.update(row[k] for k in ['rgb_member','normal_member','mask_member','calibration_member'])
wanted.update(e['name'] for e in entries if e['name'].startswith('Luces_25082022/') and ((e['name'].count('/')==1 and PurePosixPath(e['name']).suffix.lower()=='.txt') or (e['name'].startswith('Luces_25082022/code/') and PurePosixPath(e['name']).suffix.lower()=='.py') or e['name']=='Luces_25082022/data/camera.yml'))
target=out/'selection_before_extract_private.json'
frozen={'index_sha256':sha256(index),'rule':'lowest numeric lighting ID, lexical tie break; processed demosaiced rectified RGB, not Bayer RAW','selection':selection,'members':[named[n] for n in sorted(wanted)],'created_unix':time.time(),'no_pixels_or_model_errors_inspected_for_selection':True}
if target.exists():
    old=json.loads(target.read_text());assert old['selection']==selection and old['members']==frozen['members']
else:write_json(target,frozen);target.with_suffix('.json.sha256').write_text(sha256(target)+'\n')
cache=ROOT/'runtime/downloads/luces_inner_audited.zip';assert sha256(cache)==doc['parent_sha256']
copied=[];parts={p['member']:p for p in doc['parts']}
with zipfile.ZipFile(cache) as outer:
    for part in sorted({named[n]['part'] for n in wanted}):
        digest=hashlib.sha256()
        with tempfile.TemporaryFile(dir=cache.parent) as f:
            with outer.open(part) as src:
                for b in iter(lambda:src.read(8*1024*1024),b''):f.write(b);digest.update(b)
            assert digest.hexdigest()==parts[part]['sha256'];f.seek(0)
            with zipfile.ZipFile(f) as z:
                for name in sorted(n for n in wanted if named[n]['part']==part):
                    p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts
                    dest=out/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True)
                    # read() verifies each selected original ZIP member CRC.
                    data=z.read(name);assert len(data)==named[name]['size']
                    if dest.exists():assert hashlib.sha256(data).hexdigest()==sha256(dest)
                    else:dest.write_bytes(data)
                    copied.append({'member':name,'path':str(dest),'bytes':len(data),'sha256':sha256(dest),'zip_crc_checked':True})
        print(part,'selected members copied',len(copied),flush=True)
assert {c['member'] for c in copied}==wanted
write_json(out/'selected_intake_receipt_private.json',{'selection_sha256':sha256(target),'archive_integrity_sha256':sha256(out/'archive_integrity.json'),'members':copied,'total_selected_bytes':sum(c['bytes'] for c in copied),'completed_unix':time.time()})
# Remove only our explicitly named, hash-verified duplicate intake cache.
assert cache.resolve()==(ROOT/'runtime/downloads/luces_inner_audited.zip').resolve()
cache.unlink()
write_json(out/'temporary_cache_cleanup.json',{'removed_generated_duplicate':str(cache),'sha256':doc['parent_sha256'],'original_download_preserved':True})
print(json.dumps({'objects':len(objects),'selected_files':len(copied),'selected_bytes':sum(c['bytes'] for c in copied),'temporary_duplicate_removed':True}),flush=True)
