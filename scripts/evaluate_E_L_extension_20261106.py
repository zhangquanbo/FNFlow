"""Resumable registered external comparisons on real controls and sealed GSO-val inputs."""
from pathlib import Path
from collections import defaultdict
import argparse
import fcntl
import gc
import hashlib
import json
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.research_io import write_json, sha256, now
from src.noise_protocol import image_noise_seed
from scripts.evaluate_fresh_ab import metrics

OUT = ROOT / 'outputs/MoGe_E_round_20261106/E_evaluation'
DATA = ROOT / 'data/external_round_20261106'
REG = ROOT / 'provenance/E_extension_evaluation_registration_20261106.json'

def read(path):
    return json.loads(Path(path).read_text())

def aggregate(folder,panels):
    result={}
    for panel in panels:
        objects=defaultdict(lambda:defaultdict(list))
        for path in (folder/'metrics'/panel).glob('*.json'):
            row=read(path)
            objects[str(row['elevation_bin'])][row['object']].append(row['metrics']['mean_deg'])
        pitches={pitch:sum(sum(v)/len(v) for v in rows.values())/len(rows) for pitch,rows in objects.items()}
        result[panel]={'by_pitch_MAE_deg':pitches,'equal_pitch_object_macro_MAE_deg':sum(pitches.values())/len(pitches) if pitches else None}
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--scope',choices=['real','gso'],required=True)
    args=parser.parse_args()
    registration=read(REG)
    assert sha256(__file__)==registration['worker_sha256']
    assert all(sha256(ROOT/rel)==h for rel,h in registration['source_sha256'].items())
    spec=registration['models'][args.model]
    folder=OUT/args.model/args.scope
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock,torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        matrix=np.asarray(spec['matrix_to_common'],dtype=np.float32)
        from src.moge_bridge_20261106 import load_moge,normal_only,tensor_rgb,common256
        cfg=read(ROOT/spec['config_path']);assert sha256(ROOT/spec['config_path'])==spec['config_sha256']
        checkpoint=ROOT/spec['checkpoint_path'];training_summary=read(checkpoint.parent/'summary.json');assert training_summary['status']=='completed' and training_summary['last_update']==50000 and not training_summary['smoke'];assert sha256(checkpoint)==training_summary['final_checkpoint_sha256'];weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if args.model=='E3_L':
            model,_=load_moge('L');model.normal_head.load_state_dict(weights['head'],strict=True);model.eval()
            predictor=lambda image,record,seed:normal_only(model,tensor_rgb(image),True).cpu().numpy()
        else:
            if args.model=='E2_L_direct':from src.e2l_direct_models_20261106 import make_model
            else:from src.e2l_models_20261106 import make_model
            model=make_model(cfg).eval();state=model.state_dict();assert set(weights['trainable_model'])=={k for k in state if not k.startswith('features.')};state.update(weights['trainable_model']);model.load_state_dict(state,strict=True);del state
            coarse_model=load_moge('L')[0];last_condition={'key':None,'value':None}
            def predictor(image,record,seed):
                if last_condition['key']!=record['image_key']:
                    rgb=tensor_rgb(image,256)*2-1;coarse=common256(normal_only(coarse_model,tensor_rgb(image),True),'MoGe-L').permute(2,0,1);coarse=torch.nn.functional.normalize(coarse.half().float(),dim=0);last_condition.update(key=record['image_key'],value=torch.cat([rgb,coarse],dim=0)[None])
                noise=torch.randn((1,3,256,256),device='cuda',generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,record['image_key'])))
                with torch.autocast('cuda',dtype=torch.bfloat16):normal=model(noise,torch.zeros(1,device='cuda'),last_condition['value'])
                return normal.float()[0].permute(1,2,0).cpu().numpy()
        del weights
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panels=['single_diligent','single_luces','rose10_luces','rose10_diligent'] if args.scope=='real' else ['gso_primary']
        signature=sha256(REG)
        completed=0
        gt_path=None
        for panel in panels:
            manifest=DATA/(panel+'.json')
            while not manifest.exists():
                status=read(ROOT/'outputs/external_round_20261106/data_status.json')
                if status.get('phase')=='failed':raise RuntimeError('Data preparation failed: '+status.get('error',''))
                write_json(folder/'status.json',{'phase':'waiting_for_'+panel,'completed_images':completed,'updated_at':now()})
                time.sleep(10)
            assert sha256(manifest)==registration['manifest_sha256'][panel]
            records=read(manifest)['records']
            assert not any('/test/' in x['gt_path'] for x in records)
            for index,record in enumerate(sorted(records,key=lambda x:(x['object'],x.get('lighting_id',0),x['image_key']))):
                output=folder/'metrics'/panel/(record['id']+'.json')
                if output.exists():
                    old=read(output)
                    assert old['registration_sha256']==signature
                    completed+=1
                    continue
                assert sha256(record['white'])==record.get('white_sha256', record.get('ours_rgb_sha256'))
                if gt_path!=record['gt_path']:
                    assert sha256(record['gt_path'])==record['gt_sha256']
                    with np.load(record['gt_path']) as data:
                        gt=data['normal'].copy()
                        valid=data['valid'].astype(bool)
                        extra={k:data[k].copy() for k in ['normal_face','normal_crease45_diagnostic','boundary_primary_angular','boundary_crease45_angular_plus_silhouette'] if k in data.files}
                    gt_path=record['gt_path']
                image=Image.open(record['white']).convert('RGB')
                values=[]
                began=time.monotonic()
                for seed in spec['inference_seeds']:
                    derived=image_noise_seed(seed,record['image_key'])
                    random.seed(derived)
                    np.random.seed(derived%(2**32))
                    torch.manual_seed(derived)
                    prediction=np.asarray(predictor(image,record,seed),dtype=np.float32)@matrix.T
                    assert prediction.ndim==3 and prediction.shape[-1]==3 and np.isfinite(prediction).all()
                    values.append(metrics(prediction,gt,valid,extra))
                reduced={key:sum(x[key] for x in values)/len(values) if all(x[key] is not None for x in values) else None for key in values[0]}
                write_json(output,{'model':args.model,'object':record['object'],'image_key':record['image_key'],
                    'elevation_bin':record.get('elevation_bin'),'lighting_id':record.get('lighting_id'),
                    'registration_sha256':signature,'metrics':reduced,'elapsed_seconds':time.monotonic()-began})
                completed+=1
                write_json(folder/'status.json',{'phase':'scoring','panel':panel,'panel_images':index+1,
                    'panel_total':len(records),'completed_images':completed,'updated_at':now(),'GPU':torch.cuda.get_device_name()})
                if (index+1)%10==0:print(args.model,panel,index+1,'/',len(records),flush=True)
        summary={'status':'completed','model':args.model,'scope':args.scope,'registration_sha256':signature,
                 'panels':aggregate(folder,panels),'completed_images':completed,'test_GSO_images_scored':0,'updated_at':now()}
        write_json(folder/'summary.json',summary)
        write_json(folder/'status.json',summary)
        print('MODEL_SCOPE_COMPLETED',json.dumps(summary),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('EXTERNAL_FAILED',type(error).__name__,str(error),flush=True)
        raise
