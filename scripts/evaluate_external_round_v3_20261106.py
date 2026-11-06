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

OUT = ROOT / 'outputs/external_round_20261106'
DATA = ROOT / 'data/external_round_20261106'
REG = ROOT / 'provenance/external_comparison_v3_20261106.json'

def read(path):
    return json.loads(Path(path).read_text())

def load_d2():
    os.environ['FACEFLUX_STAGE_CONFIG'] = str(ROOT / 'configs/D2_actual_20261030.json')
    from scripts.train_D_stage_20261030 import make_model
    model = make_model().eval()
    checkpoint = ROOT / 'outputs/D2_train_20261030/D2_Xn_s1247/last_local.pth'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = model.state_dict()
    weights.update(state['trainable_model'])
    model.load_state_dict(weights, strict=True)
    def predict(image, record, seed):
        rgb = torch.from_numpy(np.asarray(image.resize((256,256), Image.Resampling.BILINEAR)).transpose(2,0,1).copy()).float()[None].cuda()/127.5-1
        noise = torch.randn(rgb.shape, device='cuda', generator=torch.Generator(device='cuda').manual_seed(image_noise_seed(seed,record['image_key'])))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            value = model(noise, torch.zeros(1,device='cuda'), rgb)
        return value.float()[0].permute(1,2,0).cpu().numpy()
    return predict

def load_nlr(scratch):
    from src.lightrig_adapter import load_models
    module, first, second = load_models()
    def predict(image, record, seed):
        temporary = scratch / 'inference_concat.png'
        module.inference_impl(first,second,record['rgba'],str(temporary),input_res=512,
                              guidance_scale=2.,guidance_rescale=.7,num_inference_steps=75)
        with Image.open(temporary) as result:
            assert result.size == (2560,512)
            value = np.asarray(result.crop((2048,0,2560,512)).convert('RGB')).astype(np.float32)/127.5-1
        temporary.unlink()
        return value
    return predict


def calibrate(name, predictor, matrix, folder):
    from scripts.calibrate_external_coordinates_20261030 import fixture
    import cv2
    rows=[];probes={};gates=[]
    for title,kind,angles in [('sphere','sphere',(0,0,0)),('sphere_opposite_light','sphere',(0,0,0)),('cubeC','cube',(27,47,-13)),('cubeD','cube',(-27,-47,13))]:
        image,gt,valid=fixture(kind,angles)
        if title=='sphere_opposite_light':
            physical=gt*np.array([-1.,1.,1.],np.float32)
            light=np.array([.4,.6,1.],np.float32);light/=np.linalg.norm(light)
            diffuse=np.maximum((physical*light).sum(-1),0)
            linear=np.array([.62,.56,.45],np.float32)*(.25+.65*diffuse[...,None])
            image=Image.fromarray(np.rint(np.where(valid[...,None],np.clip(linear,0,1)**(1/2.2),1.)*255).astype(np.uint8))
        torch.manual_seed(42)
        raw=np.asarray(predictor(image),dtype=np.float32)
        assert raw.ndim==3 and raw.shape[-1]==3 and np.isfinite(raw).all()
        value=cv2.resize(raw,(512,512),interpolation=cv2.INTER_LINEAR)@matrix.T
        value/=np.maximum(np.linalg.norm(value,axis=-1,keepdims=True),1e-8)
        mae=float(np.degrees(np.arccos(np.clip((value*gt).sum(-1)[valid],-1,1))).mean())
        rows.append({'fixture':title,'MAE_deg':mae,'role':'axis_acceptance' if kind=='sphere' else 'geometry_diagnostic'})
        if kind=='sphere':
            signs={}
            for axis in [0,1]:
                for sign in [-1,1]:
                    region=valid&(sign*gt[...,axis]>.5);assert region.sum()>100
                    mean=float(value[...,axis][region].mean());signs[str(axis)+str(sign)]={'mean':mean,'expected_sign':sign,'passed':mean*sign>.05}
            front=valid&(gt[...,2]>.8);assert front.sum()>100
            z=float(value[...,2][front].mean());passed=all(x['passed'] for x in signs.values()) and z>.2 and mae<40
            probes[title]={'cardinal_checks':signs,'front_Z_mean':z,'passed':passed};gates.append(passed)
    passed=all(gates)
    doc={'status':'passed' if passed else 'coordinate_review_required','model':name,'matrix':matrix.tolist(),'fixtures':rows,'sphere_probes':probes,'benchmark_inputs_opened':False,'updated_at':now(),'acceptance':'Two analytically rendered spheres under opposite horizontal light: each MAE<40deg, all4 X/Y cardinal means agree by>0.05, frontZmean>0.2. Cubes remain recorded geometry diagnostics; their geometry error is not a coordinate gate. No alternative axes were searched.'}
    old=folder/'coordinate_acceptance.json';archive=folder/'coordinate_acceptance_v1.json'
    if old.exists() and not archive.exists():archive.write_bytes(old.read_bytes())
    write_json(folder/'coordinate_acceptance_v3.json',doc);write_json(old,doc)
    assert passed,'Independent coordinate acceptance failed'

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
    assert sha256(ROOT/'src/external_normal_adapter_20261106.py')==registration['adapter_sha256']
    spec=registration['models'][args.model]
    folder=OUT/args.model/args.scope
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock,torch.inference_mode():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        matrix=np.asarray(spec['matrix_to_common'],dtype=np.float32)
        if args.model=='D2':
            predictor=load_d2()
        elif args.model=='NeuralLightRig':
            predictor=load_nlr(folder)
        else:
            from src.external_normal_adapter_20261106 import load
            native=load(args.model,folder/'runtime')
            if spec.get('independent_calibration_required'):
                calibrate(args.model,native,matrix,folder)
            predictor=lambda image,record,seed:native(image,record)
        panels=['single_diligent','single_luces','rose10_luces','rose10_diligent'] if args.scope=='real' else ['gso_primary']
        signature=sha256(REG)
        completed=0
        gt_path=None
        for panel in panels:
            manifest=DATA/(panel+'.json')
            while not manifest.exists():
                status=read(OUT/'data_status.json')
                if status.get('phase')=='failed':raise RuntimeError('Data preparation failed: '+status.get('error',''))
                write_json(folder/'status.json',{'phase':'waiting_for_'+panel,'completed_images':completed,'updated_at':now()})
                time.sleep(10)
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
