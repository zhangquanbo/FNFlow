"""Independent aggregation from private records plus sampled pixel-level checks."""
import argparse,csv,hashlib,json,math,statistics,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from src.research_io import sha256,write_json
OUT=ROOT/'outputs/object_comparison_report_20261024';OUT.mkdir(exist_ok=True)
FIELDS=['mean','median','lt_11_25','lt_22_5','lt_30','boundary_mean','outside_mean']
MODELS={'R':[42],'V':list(range(42,50)),'Xn':list(range(42,50)),'NeuralLightRig':[511],'NeuralLightRig_variability':[511,512,513]}
DATA={'main':('object_comparison_20261023','object_comparison_20261023'),'diligent':('object_comparison_diligent_20261023','object_comparison_diligent_20261023'),'luces':('object_comparison_luces_20261023','object_comparison_luces_20261023')}
def pct(x,q):
    x=sorted(x);pos=(len(x)-1)*q;i=int(pos);return x[i]+(x[min(i+1,len(x)-1)]-x[i])*(pos-i)
def resize(a,h,w):
    if a.shape[:2]==(h,w):return a.astype(np.float64)
    sh,sw=a.shape[:2];y=np.clip((np.arange(h)+.5)*sh/h-.5,0,sh-1);x=np.clip((np.arange(w)+.5)*sw/w-.5,0,sw-1)
    y0=np.floor(y).astype(int);x0=np.floor(x).astype(int);y1=np.minimum(y0+1,sh-1);x1=np.minimum(x0+1,sw-1)
    wy=(y-y0)[:,None,None];wx=(x-x0)[None,:,None]
    return ((a[y0[:,None],x0[None,:]]*(1-wx)+a[y0[:,None],x1[None,:]]*wx)*(1-wy)+(a[y1[:,None],x0[None,:]]*(1-wx)+a[y1[:,None],x1[None,:]]*wx)*wy)
def pixel_check(r,row,model,matrix):
    path=Path(row['prediction_path']);assert sha256(path)==row['prediction_sha256']
    with np.load(r['gt_path']) as z:gt=z['normal'].astype(np.float64);valid=z['valid'].astype(bool)
    if model=='NeuralLightRig':p=(np.asarray(Image.open(path).convert('RGB')).astype(np.float64)/127.5-1)@matrix.T
    else:
        with np.load(path) as z:p=z['normal'].astype(np.float64)
    p=resize(p,*valid.shape)[valid];g=gt[valid]
    assert np.isfinite(p).all() and (np.linalg.norm(p,axis=-1)>1e-8).all()
    p/=np.linalg.norm(p,axis=-1)[:,None];g/=np.linalg.norm(g,axis=-1)[:,None]
    angles=np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',p,g),-1,1)))
    values={'mean':float(angles.mean()),'median':float(np.median(angles)),**{k:float(np.count_nonzero(angles<t)/len(angles)*100) for k,t in [('lt_11_25',11.25),('lt_22_5',22.5),('lt_30',30)]}}
    errors={k:abs(v-row['metrics']['main'][k]) for k,v in values.items()}
    for k,error in errors.items():assert error<(0.001 if k in ['mean','median'] else 0.01),(r['image_key'],model,k,error)
    return {'domain':r['domain'],'id':r['id'],'model':model,'absolute_differences':errors,'valid_pixels':int(valid.sum())}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--datasets',nargs='+',choices=DATA,default=list(DATA));args=parser.parse_args()
    aggregate=[];checks=[];variability=[];paired={};receipts=[];nrecords=0
    matrix=np.asarray(json.loads((ROOT/'outputs/lightrig_calibration_v1/coordinate_lock.json').read_text())['nlr_to_common'])
    for dataset in args.datasets:
        data_name,output_name=DATA[dataset];mp=ROOT/'data'/data_name/'comparison_manifest_private.json';folder=ROOT/'outputs'/output_name
        digest=sha256(mp);assert mp.with_suffix('.json.sha256').read_text().strip()==digest
        man=json.loads(mp.read_text());summary=json.loads((folder/'summary.json').read_text());assert summary['manifest_sha256']==digest
        assert all(r['complete'] for r in summary['rows']),dataset
        expected_summary={(r['model'],r['domain'],r['gt'],r['aggregation']):r for r in summary['rows']}
        loaded={};supports={};object_means=defaultdict(dict)
        for model,seeds in MODELS.items():
            records=[r for r in man['records'] if model!='NeuralLightRig_variability' or r['id'] in man['nlr_variability_ids']]
            seed_values=defaultdict(lambda:defaultdict(list));image_values=defaultdict(list);objects=defaultdict(lambda:defaultdict(list))
            for seed in seeds:
                paths=folder/'per_image_private'/model/str(seed)
                assert {p.stem for p in paths.glob('*.json')}=={r['id'] for r in records},(dataset,model,seed,'coverage')
            for r in records:
                rows=[]
                for seed in seeds:
                    row=json.loads((folder/'per_image_private'/model/str(seed)/(r['id']+'.json')).read_text());nrecords+=1
                    assert row['manifest_sha256']==digest and row['id']==r['id'] and row['seed']==seed and row['model']==model
                    assert row['domain']==r['domain'] and row['object']==r['object'] and row['coordinate_lock_sha256']==man['coordinate_lock_sha256']
                    loaded[(model,seed,r['id'])]=row;rows.append(row)
                    for gt,m in row['metrics'].items():
                        assert m['valid_pixels']>0 and m['boundary_pixels']+m['outside_pixels']==m['valid_pixels']
                        key=(r['id'],gt);support=(m['valid_pixels'],m['boundary_pixels'],m['outside_pixels'])
                        if key in supports:assert supports[key]==support,'Model-dependent mask'
                        supports[key]=support
                        assert 0<=m['lt_11_25']<=m['lt_22_5']<=m['lt_30']<=100
                        for f in FIELDS:
                            value=m[f]
                            if value is None:assert f in ['boundary_mean','outside_mean'] and m['boundary_pixels' if f=='boundary_mean' else 'outside_pixels']==0
                            else:assert math.isfinite(value) and 0<=value<=180,(f,value)
                    seed_values[r['domain']][seed].append(row['metrics']['main']['mean'])
                for gt in rows[0]['metrics']:
                    vals={f:statistics.fmean(row['metrics'][gt][f] for row in rows) if rows[0]['metrics'][gt][f] is not None else None for f in FIELDS}
                    image_values[(r['domain'],gt)].append(vals);objects[(r['domain'],gt)][r['object']].append(vals)
                if model!='NeuralLightRig_variability':object_means[(r['domain'],r['object'])].setdefault(model,[]).append(statistics.fmean(row['metrics']['main']['mean'] for row in rows))
                if len(seeds)>1:
                    by_seed=[row['metrics']['main']['mean'] for row in rows]
                    variability.append({'dataset':dataset,'domain':r['domain'],'model':model,'id':r['id'],'image_mae_sd':statistics.stdev(by_seed),'image_mae_range':max(by_seed)-min(by_seed),'seeds':len(seeds)})
            for (domain,gt),values in image_values.items():
                for aggregation in ['image_macro','object_macro']:
                    vals=values if aggregation=='image_macro' else [{f:statistics.fmean(v[f] for v in group if v[f] is not None) if any(v[f] is not None for v in group) else None for f in FIELDS} for group in objects[(domain,gt)].values()]
                    actual={f:statistics.fmean(v[f] for v in vals if v[f] is not None) if any(v[f] is not None for v in vals) else None for f in FIELDS}
                    ref=expected_summary[(model,domain,gt,aggregation)]
                    for f,value in actual.items():assert value is None and ref[f] is None or value is not None and abs(value-ref[f])<1e-9,(dataset,model,domain,gt,aggregation,f)
                    assert ref['images_done']==len(values)==ref['images_expected'] and ref['objects_done']==len(objects[(domain,gt)])
                    aggregate.append({'dataset':dataset,'model':model,'domain':domain,'gt':gt,'aggregation':aggregation,'images':len(values),'objects':len(objects[(domain,gt)]),**actual})
            for domain,values in seed_values.items():
                means=[statistics.fmean(values[s]) for s in seeds];ref=next(v for v in summary['seed_variability'] if v['model']==model and v['domain']==domain)
                assert all(abs(a-b)<1e-9 for a,b in zip(means,ref['image_macro_mae_by_seed']))
                if len(seeds)>1:assert abs(statistics.stdev(means)-ref['sd_ddof1'])<1e-9
        for key,vals in object_means.items():paired[key]={model:statistics.fmean(v) for model,v in vals.items()}
        for domain in sorted({r['domain'] for r in man['records']}):
            chosen=[r for r in man['records'] if r['domain']==domain and r['id'] in man['case_ids']][:2]
            for r in chosen:
                for model,seed in [('R',42),('V',42),('Xn',42),('NeuralLightRig',511)]:checks.append(pixel_check(r,loaded[(model,seed,r['id'])],model,matrix))
        case=json.loads((folder/'failure_cases_private/receipt_private.json').read_text());assert case['complete'] and case['ready']==case['expected']==len(man['case_ids'])
        for panel in case['panels']:assert sha256(panel['path'])==panel['sha256']
        receipts.append({'dataset':dataset,'manifest_sha256':digest,'summary_sha256':sha256(folder/'summary.json'),'case_panels':case['ready']})
        print(dataset,'independent aggregates and fixed-case pixel checks PASSED',flush=True)
    paired_rows=[{'domain':d,'object':o,**vals} for (d,o),vals in sorted(paired.items())]
    wins=[]
    for d in sorted({r['domain'] for r in paired_rows}):
        rows=[r for r in paired_rows if r['domain']==d]
        for model in ['R','V','Xn']:
            delta=[r[model]-r['NeuralLightRig'] for r in rows]
            wins.append({'domain':d,'model':model,'objects':len(rows),'better_than_nlr':sum(v<0 for v in delta),'worse_than_nlr':sum(v>0 for v in delta),'mean_delta_degrees':statistics.fmean(delta),'median_delta_degrees':statistics.median(delta)})
    var_summary=[]
    for d,m in sorted({(v['domain'],v['model']) for v in variability}):
        rows=[r for r in variability if r['domain']==d and r['model']==m];sds=[r['image_mae_sd'] for r in rows]
        var_summary.append({'domain':d,'model':m,'images':len(rows),'seeds':rows[0]['seeds'],'mean_image_mae_sd':statistics.fmean(sds),'p90_image_mae_sd':pct(sds,.9),'max_image_mae_sd':max(sds),'mean_image_mae_range':statistics.fmean(r['image_mae_range'] for r in rows)})
    same_subset=[]
    for d in sorted({v['domain'] for v in variability}):
        ids={v['id'] for v in variability if v['domain']==d and v['model']=='NeuralLightRig_variability'}
        for m in ['NeuralLightRig_variability','V','Xn']:
            selected=[v for v in variability if v['domain']==d and v['model']==m and v['id'] in ids]
            assert len(selected)==len(ids)
            sds=[v['image_mae_sd'] for v in selected]
            same_subset.append({'domain':d,'model':m,'images':len(ids),'seeds':selected[0]['seeds'],'mean_image_mae_sd':statistics.fmean(sds),'p90_image_mae_sd':pct(sds,.9),'max_image_mae_sd':max(sds)})
    for name,rows in [('independent_aggregates.csv',aggregate),('paired_object_mae_private.csv',paired_rows),('paired_object_wins.csv',wins),('seed_sensitivity.csv',var_summary),('seed_sensitivity_same_subset.csv',same_subset)]:
        with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(OUT/'validation.json',{'status':'passed','datasets':args.datasets,'private_records_checked':nrecords,'aggregate_rows_recomputed':len(aggregate),'pixel_checks':checks,'pixel_check_scope':'Two pre-registered fixed cases per domain, four models, mean/median/strict thresholds independently recomputed with float64 NumPy bilinear interpolation and unit vectors. These are sampled pixel checks, not all-image pixel recomputation. Boundary aggregations recomputed from all original rows; shared region counts verified across every model/seed.','receipts':receipts,'paired_object_wins':wins,'seed_sensitivity':var_summary,'seed_sensitivity_same_subset':same_subset,'training':False})
    print(json.dumps({'status':'passed','datasets':args.datasets,'records':nrecords,'aggregate_rows':len(aggregate),'pixel_checks':len(checks)}),flush=True)
if __name__=='__main__':main()
