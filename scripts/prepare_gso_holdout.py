"""Versioned official-collection subset, conservative source/count duplicate screen."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,urllib.request,urllib.parse,hashlib,zipfile,time,re,gzip
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from src.research_io import ROOT,write_json,sha256,now
from scripts.prepare_arb_training import fixed_formal_partition
from src.http_range import HTTPRangeReader

BASE='https://fuel.gazebosim.org/1.0/GoogleResearch/'
OUT=ROOT/'outputs/P1_gso_intake_v1';RAW=ROOT/'data/gso_raw_v1'


def fetch_json(url):
    error=None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=40) as r:return json.load(r)
        except OSError as e:
            error=e;time.sleep(min(8,2**attempt))
    raise error


def normalize_name(name):return re.sub('[^a-z0-9]','',name.lower())


def download_ranges(url,folder,version):
    request=urllib.request.Request(url,headers={'Range':'bytes=0-0','Accept-Encoding':'identity'})
    with urllib.request.urlopen(request,timeout=45) as response:
        content=response.headers.get('Content-Range','');match=re.fullmatch(r'bytes 0-0/(\d+)',content)
        if response.status!=206 or not match or len(response.read(2))!=1:raise RuntimeError('Exact GSO length probe refused')
    length=int(match[1])
    if length>150*1024**2:raise RuntimeError('GSO archive cap exceeded')
    parts=folder/f'zip_parts_v{version}';parts.mkdir(exist_ok=True);block=1024*1024
    def part(start):
        count=min(block,length-start);path=parts/f'{start:012d}.part';checksum=path.with_suffix('.sha256')
        if path.exists() and checksum.exists() and path.stat().st_size==count and sha256(path)==checksum.read_text():return path
        last=None
        for attempt in range(3):
            try:
                reader=HTTPRangeReader(url,length,budget=count+16,max_request=block,timeout=45)
                reader.seek(start);data=reader.read(count);path.write_bytes(data);checksum.write_text(sha256(path));return path
            except OSError as e:last=e;time.sleep(2**attempt)
        raise last
    with ThreadPoolExecutor(max_workers=4) as pool:pieces=list(pool.map(part,range(0,length,block)))
    with (folder/'source.partial').open('wb') as f:
        for piece in pieces:f.write(piece.read_bytes())
    if (folder/'source.partial').stat().st_size!=length:raise RuntimeError('GSO joined archive length mismatch')
    (folder/'source.partial').replace(folder/'source.zip')


def download(row):
    name=row['name'];folder=RAW/name;folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt_private.json'
    if receipt.exists():
        result=json.loads(receipt.read_text())
        if sha256(folder/'source.zip')!=result['archive_sha256']:raise RuntimeError('GSO cached archive changed')
        return result
    detail=fetch_json(BASE+'models/'+urllib.parse.quote(name,safe=''))
    version=int(detail['version']);url=BASE+f'models/{urllib.parse.quote(name,safe="")}/{version}/{urllib.parse.quote(name,safe="")}.zip'
    archive=folder/'source.zip';last=None
    for attempt in range(3):
        try:
            if not archive.exists():download_ranges(url,folder,version)
            with zipfile.ZipFile(archive) as z:
                if z.testzip():raise RuntimeError('GSO ZIP CRC failure')
                for info in z.infolist():
                    if not (folder/'extracted'/info.filename).resolve().is_relative_to((folder/'extracted').resolve()):raise RuntimeError('Unsafe GSO member')
                    if info.file_size>300*1024**2:raise RuntimeError('Oversized GSO member')
                z.extractall(folder/'extracted')
            last=None;break
        except Exception as e:
            last=e;write_json(folder/'source_failure_private.json',{'source_url':url,'attempt':attempt+1,'error':repr(e),'created_at':now()})
            time.sleep(attempt+1)
    if last:raise last
    objs=list((folder/'extracted').rglob('*.obj'))
    # Official model SDF specifies the visual mesh; prefer model.obj over collision meshes.
    candidates=[p for p in objs if p.name=='model.obj']
    if len(candidates)!=1:raise RuntimeError('GSO visual model.obj missing or ambiguous')
    mesh=candidates[0];vertices=[];faces=0
    with mesh.open(errors='strict') as f:
        for line in f:
            if line.startswith('v '):vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith('f '):faces+=len(line.split())-3
    if not vertices or not faces:raise RuntimeError('Empty GSO geometry')
    position=np.asarray(vertices,np.float64);extent=np.ptp(position,axis=0)
    centered=(position-(position.max(0)+position.min(0))/2)/max(extent)
    # Stable exact vertex-set fingerprint within this GSO subset, not a global near-duplicate guarantee.
    quant=np.unique(np.round(centered,5),axis=0)
    fingerprint=hashlib.sha256(quant.astype('<f8').tobytes()).hexdigest()
    result={'name':name,'source_url':url,'version':version,'license':detail['license_name'],
        'license_url':detail['license_url'],'archive_sha256':sha256(archive),'archive_bytes':archive.stat().st_size,
        'mesh':str(mesh),'mesh_sha256':sha256(mesh),'vertices':len(vertices),'triangles':faces,
        'geometry_fingerprint':fingerprint,'source_detail':detail}
    write_json(receipt,result);return result


def main():
    OUT.mkdir(parents=True,exist_ok=True);RAW.mkdir(parents=True,exist_ok=True)
    index_file=OUT/'collection_index_private.json'
    if index_file.exists():rows=json.loads(index_file.read_text())['records']
    else:
        rows=[];page=1;expected=None;snapshot=str(time.time_ns())
        while True:
            url=BASE+'collections/Scanned%20Objects%20by%20Google%20Research/models?per_page=100&page='+str(page)+'&research_snapshot='+snapshot
            with urllib.request.urlopen(url,timeout=40) as response:
                values=json.load(response)
                if expected is None:expected=int(response.headers['X-Total-Count'])
                if int(response.headers['X-Total-Count'])!=expected:raise RuntimeError('Collection membership changed during snapshot')
            if not values:raise RuntimeError('Collection ended before declared count')
            rows.extend(values);page+=1
            if len(rows)==expected:break
            if len(rows)>expected:raise RuntimeError('GSO count drift during pagination')
            if page>30:raise RuntimeError('Unexpected GSO pagination')
        unique={};duplicate_memberships=[]
        for r in rows:
            key=(r['owner'],r['name'])
            if key in unique:
                previous=unique[key]
                if any(previous.get(k)!=r.get(k) for k in ['license_id','filesize','modify_date']):
                    raise RuntimeError('Conflicting metadata for repeated collection membership')
                duplicate_memberships.append({'owner':r['owner'],'name':r['name']})
            else:unique[key]=r
        if len(unique)<1000:raise RuntimeError('Unexpectedly incomplete official GSO collection')
        rows=list(unique.values())
        write_json(index_file,{'records':rows,'source':'official named collection','created_at':now(),
            'raw_memberships':expected,'duplicate_memberships':duplicate_memberships})
    eligible=[r for r in rows if r['owner']=='GoogleResearch' and r.get('license_id')==2 and not r.get('private')]
    eligible.sort(key=lambda r:hashlib.sha256(('gso-holdout-v1/'+r['name']).encode()).hexdigest())
    selected=eligible[:200]
    write_json(OUT/'selection_private.json',{'records':selected,'target':200,'selection':'first 200 official collection names by fixed SHA256 before model evaluation'})
    received=[];failures=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(download,r):r['name'] for r in selected}
        for job in as_completed(jobs):
            try:received.append(job.result())
            except Exception as e:
                failures.append({'name':jobs[job],'error':repr(e)})
                print('GSO source failed',jobs[job],repr(e),flush=True)
                write_json(OUT/'failures_private.json',failures)
            write_json(OUT/'status.json',{'status':'downloading','completed':len(received),'failed':len(failures),'target':len(selected),'updated_at':now()})
            print('GSO downloaded',len(received),'failed',len(failures),flush=True)
    plan=fixed_formal_partition();obj=[]
    for folder in (ROOT/'outputs/P1_arb_acquisition_v1').glob('000-*'):
        path=folder/'upstream_metadata.json.gz'
        if path.exists():
            with gzip.open(path,'rt') as f:metadata=json.load(f)
            obj.extend(v for uid,v in metadata.items() if uid in plan)
    if len(obj)!=2000:raise RuntimeError('GSO duplicate screen must cover complete planned 2000 Objaverse assets')
    names={normalize_name(r.get('name','')) for r in obj};counts={(r.get('vertexCount'),r.get('faceCount')) for r in obj}
    seen={};accepted=[];excluded=[]
    for r in sorted(received,key=lambda x:hashlib.sha256(('gso-holdout-v1/'+x['name']).encode()).hexdigest()):
        reason=None
        if normalize_name(r['name']) in names:reason='normalized_source_name_matches_planned_objaverse'
        elif (r['vertices'],r['triangles']) in counts:reason='potential_duplicate_vertex_and_triangle_counts; conservative exclusion'
        elif r['geometry_fingerprint'] in seen:reason='exact_centered_vertex_set_duplicate_within_gso'
        if reason:excluded.append({'name':r['name'],'reason':reason});continue
        seen[r['geometry_fingerprint']]=r['name'];accepted.append(r)
    # Keep product-name families on one side; no validation/test split by view.
    families={}
    for r in accepted:
        family='_'.join(r['name'].lower().split('_')[:2]);families.setdefault(family,[]).append(r)
    keys=sorted(families,key=lambda k:hashlib.sha256(('gso-family-split-v1/'+k).encode()).hexdigest())
    records=[]
    for key in keys:
        parity=int(hashlib.sha256(('gso-family-split-v1/'+key).encode()).hexdigest(),16)%2
        for r in families[key]:records.append({**r,'family':key,'split':'val' if parity==0 else 'test','training_allowed':False})
    write_json(OUT/'manifest_private.json',{'records':records,'exclusions':excluded,'failures':failures})
    summary={'status':'completed' if not failures else 'completed_with_download_failures','created_at':now(),
        'official_collection_assets':len(rows),'selected_assets':len(selected),'downloaded_assets':len(received),
        'accepted_assets':len(records),'excluded_assets':len(excluded),'failed_assets':len(failures),
        'assets_by_split':{s:sum(r['split']==s for r in records) for s in ['val','test']},
        'training_allowed':False,'planned_objaverse_assets_screened':2000,'group_split':'product-name first-two-token family; all views stay with asset',
        'duplicate_checks':['normalized source names against planned 2000','conservative exact vertex/triangle count matches against planned 2000','exact centered rounded vertex-set hashes within GSO'],
        'near_duplicate_geometry_across_sources_exhaustively_verified':False,
        'scope':'candidate object holdouts under declared source/count checks; cannot guarantee absence of re-meshed/re-exported cross-source duplicates',
        'upstream_archive_checksum_published':False,'integrity':'versioned official HTTPS ZIP; ZIP CRC and computed SHA256 retained, not an upstream published checksum claim'}
    write_json(OUT/'summary.json',summary);write_json(OUT/'status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write_json(OUT/'status.json',{'status':'failed','error':repr(error),'updated_at':now()})
        raise
