"""Download only preselected official DiLiGenT ZIP members, with byte caps."""
import sys,json,re,zipfile,urllib.request,time,zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.http_range import HTTPRangeReader
from src.research_io import sha256,write_json
URL='https://drive.usercontent.google.com/download?id=1EgC3x8daOWL4uQmc6c4nXVe4mdAMJVfg&export=download&confirm=t'
out=ROOT/'data/real_object_selected_v1/diligent';out.mkdir(parents=True,exist_ok=True)
def main():
 with urllib.request.urlopen(urllib.request.Request(URL,method='HEAD'),timeout=25) as r:
  length=int(r.headers['Content-Length']);head={'status':r.status,'content_length':length,'accept_ranges':r.headers.get('Accept-Ranges'),'content_type':r.headers.get('Content-Type')}
 write_json(out/'head_receipt.json',head);print(head,flush=True)
 reader=HTTPRangeReader(URL,length,budget=128*1024*1024,max_request=16*1024*1024,timeout=40)
 with zipfile.ZipFile(reader) as z:
  names=z.namelist();write_json(out/'archive_index_private.json',{'names':names,'length':length});groups={}
  for name in names:
   m=re.search(r'(?:^|/)pmsData/([^/]+)/(\d+)\.png$',name,re.I)
   if m:groups.setdefault(m[1],[]).append((int(m[2]),name))
  assert len(groups)==10,sorted(groups)
  selected=[]
  for obj,images in sorted(groups.items()):
   number,rgb=min(images);prefix=rgb.rsplit('/',1)[0]+'/'
   required=[rgb]
   for base in ['mask.png','Normal_gt.mat','light_intensities.txt','light_directions.txt','filenames.txt']:
    matches=[n for n in names if n.lower()==(prefix+base).lower()]
    if not matches and base in ['mask.png','Normal_gt.mat']:raise RuntimeError('Missing '+prefix+base)
    required+=matches
   selected.append({'object':obj,'lighting_id':number,'rgb_member':rgb,'members':required})
  write_json(out/'selection_before_extract_private.json',{'rule':'lowest numerical lighting ID per official pmsData object','selection':selected,'amendment_sha256':sha256(ROOT/'docs/object_calibration_intake_amendment_20261023.md')})
  extracted=[]
  for choice in selected:
   folder=out/choice['object'];folder.mkdir(exist_ok=True)
   for name in choice['members']:
    p=folder/Path(name).name
    info=z.getinfo(name);raw=p.read_bytes() if p.exists() else None
    if raw is None or len(raw)!=info.file_size or zlib.crc32(raw)!=info.CRC:
     for attempt in range(4):
      try:raw=z.read(name);break
      except (OSError,TimeoutError):
       if attempt==3:raise
       time.sleep(2)
     p.write_bytes(raw)
    extracted.append({'member':name,'path_relative':p.relative_to(ROOT).as_posix(),'bytes':len(raw),'sha256':sha256(p)})
   print('extracted',choice['object'],flush=True)
 write_json(out/'receipt_private.json',{'source_url':URL,'archive_bytes':length,'range_bytes_downloaded':reader.downloaded,'range_requests':reader.requests,'objects':len(groups),'files':extracted,'script_sha256':sha256(__file__)})
 print('COMPLETE',reader.downloaded,flush=True)
if __name__=='__main__':main()
