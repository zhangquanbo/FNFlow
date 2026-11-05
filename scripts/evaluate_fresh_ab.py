"""Resumable, fixed-checkpoint fresh A/B evaluation; GSO-test is never opened."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,os,time,gc,fcntl
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.research_io import write_json,sha256,now
from src.noise_protocol import image_noise_seed
from scripts.train_object_dino_retrain import make_model
OUT=ROOT/'outputs/P3_object_dino_evaluation_fresh_20261027'
PANELS=['primary','original','arb_style','diligent','luces']
def read(p):return json.loads(Path(p).read_text())
def signature():return sha256(OUT/'preregistration.json')
def assert_registration():
    reg=read(OUT/'preregistration.json')
    assert reg['test_evaluation'] is False
    for name,h in reg['source_sha256'].items():assert sha256(ROOT/'scripts'/name)==h,('Source changed after registration',name)
    for name,h in reg['shared_source_sha256'].items():assert sha256(ROOT/name)==h,('Shared source changed',name)
    return reg
def prepare(panel):
    assert_registration();p=OUT/'manifests_private'/(panel+'.json')
    if p.exists():
        man=read(p);assert man['registration_sha256']==signature();return man
    if panel=='primary':source=ROOT/'data/gso_multielevation_fresh_20261026/val/private_manifest.json'
    elif panel in ['original','arb_style']:
        status=read(OUT/f'lighting_{panel}_status.json')
        assert status['status']=='completed' and status['objects']==93 and not status.get('smoke')
        source=Path(status['manifest']);assert sha256(source)==status['manifest_sha256']
    else:source=ROOT/f'data/evaluation_fresh_20261026/{panel}/manifest_private.json'
    rows=read(source)['records'];records=[]
    expected={'primary':1860,'original':372,'arb_style':372,'diligent':10,'luces':14}
    assert len(rows)==expected[panel]
    for row in sorted(rows,key=lambda r:r['image_key']):
        if panel in ['diligent','luces']:
            r=dict(row);r.update(white=row['ours_rgb'],rgba=row['nlr_rgba'],elevation_bin=None,style=panel)
            for field in ['ours_rgb','nlr_rgba']:assert sha256(r[field])==r[field+'_sha256']
        else:
            assert row['split']=='val' and '/test/' not in row['path']
            key=row['image_key'];identifier=hashlib.sha256(key.encode()).hexdigest()
            d=OUT/'inputs_private'/identifier;d.mkdir(parents=True,exist_ok=True)
            assert sha256(row['path'])==row['npz_sha256']
            with np.load(row['path']) as z:
                valid=z['valid'].astype(bool);rgb=z['rgb'];assert valid.any()
                Image.fromarray(np.where(valid[...,None],rgb,255).astype(np.uint8)).save(d/'white.png')
                Image.fromarray(np.dstack([rgb,valid.astype(np.uint8)*255])).save(d/'rgba.png')
            r={'id':identifier,'object':row['scene'],'image_key':key,'style':panel,'domain':'gso',
               'elevation_bin':row['elevation_bin'],'elevation_actual':row['elevation_actual'],
               'gt_path':row['path'],'gt_sha256':row['npz_sha256'],'white':str(d/'white.png'),'rgba':str(d/'rgba.png')}
            r['white_sha256']=sha256(r['white']);r['rgba_sha256']=sha256(r['rgba'])
        records.append(r)
    man={'records':records,'source_manifest_sha256':sha256(source),'registration_sha256':signature(),'panel':panel}
    write_json(p,man);return man
def metrics(pred,gt,valid,extra=None):
    if pred.shape[:2]!=valid.shape:
        pred=F.interpolate(torch.from_numpy(pred.copy()).permute(2,0,1)[None],size=valid.shape,mode='bilinear',align_corners=False)[0].permute(1,2,0).numpy()
    assert valid.any() and pred.shape==gt.shape
    p=pred[valid].astype(np.float64);g=gt[valid].astype(np.float64)
    for v in [p,g]:assert np.isfinite(v).all() and (np.linalg.norm(v,axis=1)>1e-8).all(),'Invalid valid-pixel normal; image cannot be omitted'
    p/=np.linalg.norm(p,axis=1)[:,None];g/=np.linalg.norm(g,axis=1)[:,None]
    angle=np.degrees(np.arccos(np.clip((p*g).sum(1),-1,1)))
    up=(np.argmax(np.abs(g),axis=1)==1)&(g[:,1]>0)
    out={'mean_deg':float(angle.mean()),'median_deg':float(np.median(angle)),'rms_deg':float(np.sqrt(np.mean(angle**2))),
         'valid_pixels':len(g),'up_pixels':int(up.sum()),'up_mae_deg':float(angle[up].mean()) if up.any() else None,
         'up_gt_z_sum':float(g[up,2].sum()),'up_pred_z_sum':float(p[up,2].sum()),
         'up_z_signed_error_sum':float((p[up,2]-g[up,2]).sum()),'up_z_absolute_error_sum':float(np.abs(p[up,2]-g[up,2]).sum())}
    for threshold in [11.25,22.5,30.]:out[f'fraction_below_{threshold:g}']=float(np.mean(angle<threshold))
    for field,array in (extra or {}).items():
        if field.startswith('boundary_'):
            band=array[valid].astype(bool)
            out[field+'_pixels']=int(band.sum());out[field+'_mae_deg']=float(angle[band].mean()) if band.any() else None
            out[field+'_outside_mae_deg']=float(angle[~band].mean()) if (~band).any() else None
        elif field.startswith('normal_'):
            alt=array[valid].astype(np.float64);norm=np.linalg.norm(alt,axis=1);assert (norm>1e-8).all()
            out[field+'_mae_deg']=float(np.degrees(np.arccos(np.clip((p*(alt/norm[:,None])).sum(1),-1,1))).mean())
    return out
def rowpath(panel,model,seed,r):return OUT/'per_image_private'/panel/model/str(seed)/(r['id']+'.json')
def cached(panel,model,seed,r,manifest_hash):
    p=rowpath(panel,model,seed,r)
    if not p.exists():return False
    old=read(p);assert old['registration_sha256']==signature() and old['manifest_sha256']==manifest_hash
    return True
def save(panel,model,seed,r,pred,manifest_hash,extra=None):
    assert '/test/' not in r['gt_path'] and sha256(r['gt_path'])==r['gt_sha256']
    with np.load(r['gt_path']) as z:
        fields={f:z[f] for f in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if f in z.files}
        result=metrics(pred,z['normal'],z['valid'].astype(bool),fields)
    write_json(rowpath(panel,model,seed,r),{'model':model,'seed':seed,'id':r['id'],'object':r['object'],'style':panel,
       'elevation_bin':r['elevation_bin'],'image_key':r['image_key'],'metrics':result,
       'registration_sha256':signature(),'manifest_sha256':manifest_hash,**(extra or {})})
def ours(panel):
    reg=assert_registration();man=prepare(panel);mh=sha256(OUT/'manifests_private'/(panel+'.json'))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    with torch.inference_mode():
        for name,method in [('A','Xn'),('B','R')]:
            info=reg['checkpoints'][name];assert sha256(info['path'])==info['sha256']
            ck=torch.load(info['path'],map_location='cpu',weights_only=False);assert ck['step']==50000
            model=make_model().eval();state=model.state_dict()
            expected={k for k in state if not k.startswith('features.')}
            assert set(ck['trainable_model'])==expected,'Checkpoint trainable keys mismatch'
            state.update(ck['trainable_model']);model.load_state_dict(state,strict=True);del state,ck
            assert not model.features.encoder.training and all(not p.requires_grad for p in model.features.encoder.parameters())
            started=time.perf_counter();seeds=list(range(42,50)) if name=='A' else [42]
            for i,r in enumerate(man['records']):
                if all(cached(panel,name,s,r,mh) for s in seeds):continue
                rgb=np.asarray(Image.open(r['white']).convert('RGB').resize((256,256),Image.Resampling.BILINEAR))
                rgb=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
                for seed in seeds:
                    if cached(panel,name,seed,r,mh):continue
                    noise=torch.zeros_like(rgb) if method=='R' else torch.randn(rgb.shape,device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,r['image_key'])))
                    with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device='cuda'),rgb)
                    pred=raw.float()[0].permute(1,2,0).cpu().numpy()
                    save(panel,name,seed,r,pred,mh)
                    if seed==42 and i<3:
                        path=OUT/'previews_private'/panel/name/(r['id']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(path,prediction=pred)
                if i%10==0 or i+1==len(man['records']):
                    status={'phase':'ours','panel':panel,'model':name,'images':i+1,'total_images':len(man['records']),
                       'elapsed_seconds':time.perf_counter()-started,'updated_at':now(),'GPU':torch.cuda.get_device_name()}
                    write_json(OUT/'ours_status.json',status);print(json.dumps(status),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    write_json(OUT/f'ours_{panel}_completed.json',{'status':'completed','models':['A','B'],'images':len(man['records']),'updated_at':now(),'manifest_sha256':mh})
def aggregate():
    reg=assert_registration();summary={'updated_at':now(),'test_images_scored':0,'panels':{},'complete':True,'C_training_started':False}
    for panel in PANELS:
        p=OUT/'manifests_private'/(panel+'.json')
        if not p.exists():summary['complete']=False;continue
        man=read(p);mh=sha256(p);models={}
        for model,seeds in [('A',list(range(42,50))),('B',[42]),('NeuralLightRig',[511])]:
            if not all(cached(panel,model,s,r,mh) for r in man['records'] for s in seeds):summary['complete']=False;continue
            images=[]
            for r in man['records']:
                vals=[read(rowpath(panel,model,s,r))['metrics'] for s in seeds]
                merged={k:float(np.mean([v[k] for v in vals])) if vals[0][k] is not None else None for k in vals[0]}
                images.append({'object':r['object'],'pitch':r['elevation_bin'],**merged})
            bins=[0,15,30,45,60] if panel=='primary' else [30] if panel in ['original','arb_style'] else [None]
            pitches={}
            for angle in bins:
                rows=[r for r in images if r['pitch']==angle];objects=sorted({r['object'] for r in rows});vals={}
                for metric in rows[0]:
                    if metric in ['object','pitch']:continue
                    valid=[r for r in rows if r[metric] is not None]
                    vals[metric]=float(np.mean([np.mean([r[metric] for r in valid if r['object']==obj]) for obj in objects if any(r['object']==obj for r in valid)])) if valid else None
                n=sum(r['up_pixels'] for r in rows)
                vals['up_z_signed_error_pixel_pooled']=sum(r['up_z_signed_error_sum'] for r in rows)/n if n else None
                vals['up_z_absolute_error_pixel_pooled']=sum(r['up_z_absolute_error_sum'] for r in rows)/n if n else None
                pitches[str(angle)]={'objects':len(objects),'images':len(rows),**vals}
            models[model]={'by_pitch':pitches,'equal_pitch_object_macro_mae_deg':float(np.mean([v['mean_deg'] for v in pitches.values()]))}
        summary['panels'][panel]=models
    primary=summary['panels'].get('primary',{})
    if all(k in primary for k in ['A','B','NeuralLightRig']):
        gaps={'five_angles':primary['A']['equal_pitch_object_macro_mae_deg']-primary['NeuralLightRig']['equal_pitch_object_macro_mae_deg'],
              'fresh_30':primary['A']['by_pitch']['30']['mean_deg']-primary['NeuralLightRig']['by_pitch']['30']['mean_deg']}
        route='success_review_multi_seed_Gbuffer' if all(v<=3 for v in gaps.values()) else 'capacity_unfreeze_review' if all(v>5 for v in gaps.values()) else 'inconclusive_mixed_or_gray'
        summary['decision']={'A_minus_NLR_deg':gaps,'route_for_review':route,'C_requires_AB_review':True}
    write_json(OUT/'summary.json',summary);print(json.dumps(summary),flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','ours','aggregate']);parser.add_argument('--panel',choices=PANELS,default='primary');a=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'ours.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.stage=='prepare':prepare(a.panel)
        elif a.stage=='ours':ours(a.panel)
        else:aggregate()
if __name__=='__main__':
    try:main()
    except Exception as e:
        write_json(OUT/'ours_failure.json',{'error':repr(e),'updated_at':now()});raise
