"""Read-only data audit; manifests and evidence are confined to Research."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from collections import Counter,defaultdict
import csv
import io
import json
import urllib.request
import numpy as np
from PIL import Image
import cv2
import h5py
from src.research_io import ROOT,write_json,sha256,output_path,now


def cache_official(name,url,directory):
    path=directory/name
    if not path.exists():
        data=urllib.request.urlopen(url,timeout=40).read()
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    return path


def hypersim(dataset,tools,out):
    source=dataset/'General/Hypersim/asf_v2_full'
    raw=dataset/'General/Hypersim/raw'
    rows=[json.loads(line) for line in (source/'conversion_manifest.jsonl').read_text().splitlines()]
    official_path=tools/'Hypersim/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv'
    official={(r['scene_name'],r['camera_name'],int(r['frame_id'])):r for r in csv.DictReader(official_path.open())}
    issues=Counter(); scenes=defaultdict(set); counts=Counter(); dtypes=Counter(); valid_fractions=[]
    keys=set(); details=[]; matches=[]; scenes_with_poses=set(); source_fields=Counter()
    for index,row in enumerate(rows):
        split=row['split']; scene=row['scene']; camera=row['camera']; frame=row['frame_id']
        key=(scene,camera,frame); counts[split]+=1; scenes[split].add(scene)
        problem=[]
        if key in keys: problem.append('duplicate_key')
        keys.add(key)
        ref=official.get(key)
        if ref is None: problem.append('missing_official_key')
        elif ref['included_in_public_release']!='True': problem.append('excluded_from_public_release')
        elif ref['split_partition_name']!=split: problem.append('official_split_mismatch')
        stem=f'{scene}__{camera}__frame_{frame:04d}'
        paths={'rgb':source/split/'rgb'/(stem+'.jpg'),
               'normal':source/split/'normal'/(stem+'.npy'),
               'valid':source/split/'valid'/(stem+'.png')}
        geometry=raw/scene/'images'/f'scene_{camera}_geometry_hdf5'
        native=geometry/f'frame.{frame:04d}.normal_cam.hdf5'
        for field in ['normal_cam','normal_bump_cam','depth_meters','position']:
            source_fields[field]+=int((geometry/f'frame.{frame:04d}.{field}.hdf5').exists())
        if (raw/scene/'_detail'/camera/'camera_keyframe_orientations.hdf5').exists():
            scenes_with_poses.add(scene)
        for kind,path in paths.items():
            if not path.exists(): problem.append('missing_'+kind)
        if not problem:
            try:
                normal=np.load(paths['normal']).astype(np.float32)
                mask=np.asarray(Image.open(paths['valid']))>0
                with Image.open(paths['rgb']) as image: rgb_size=image.size
                if normal.shape != (*mask.shape,3) or rgb_size!=(mask.shape[1],mask.shape[0]):
                    problem.append('unaligned_shapes')
                elif not mask.any(): problem.append('empty_valid_mask')
                else:
                    finite=np.isfinite(normal).all(axis=-1)
                    length=np.linalg.norm(np.nan_to_num(normal),axis=-1)
                    if (~finite&mask).any(): problem.append('nonfinite_valid_normal')
                    if (mask&(length<1e-8)).any(): problem.append('zero_valid_normal')
                    if (mask&(np.abs(length-1)>.01)).any(): problem.append('nonunit_valid_normal')
                    valid_fractions.append(float(mask.mean()))
                    dtypes[str(np.load(paths['normal'],mmap_mode='r').dtype)]+=1
                    if scene not in {r['scene'] for r in matches} and native.exists():
                        with h5py.File(native,'r') as handle: original=handle['dataset'][:].astype(np.float32)
                        resized=cv2.resize(np.nan_to_num(original),(normal.shape[1],normal.shape[0]),interpolation=cv2.INTER_LINEAR)
                        resized/=np.maximum(np.linalg.norm(resized,axis=-1,keepdims=True),1e-8)
                        matches.append({'scene':scene,'mse_native_gl_linear':float(np.mean((resized[mask]-normal[mask])**2))})
            except Exception as error: problem.append('decode_error:'+type(error).__name__)
        issues.update(problem)
        details.append({**row,'source_paths':{k:str(v) for k,v in paths.items()},
                        'raw_geometry_normal':str(native),'problems':problem,
                        'official_split':ref.get('split_partition_name') if ref else None})
        if (index+1)%1000==0: print('Hypersim audited',index+1,flush=True)
    overlap={f'{a}__{b}':len(scenes[a]&scenes[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    projection=tools/'Hypersim/contrib/mikeroberts3000/metadata_camera_parameters.csv'
    summary={'num_rows':len(rows),'num_unique_keys':len(keys),'images_by_split':dict(counts),
        'scenes_by_split':{k:len(v) for k,v in scenes.items()},'scene_overlap_counts':overlap,
        'issues':dict(issues),'normal_dtype_counts':dict(dtypes),
        'valid_fraction_min_median': [min(valid_fractions),float(np.median(valid_fractions))] if valid_fractions else [],
        'native_fields_present_counts':dict(source_fields),'scenes_with_local_camera_poses':len(scenes_with_poses),
        'projection_metadata_available':projection.exists(),
        'native_comparison_scenes':len(matches),'native_gl_linear_mse_median':float(np.median([r['mse_native_gl_linear'] for r in matches])) if matches else None,
        'label_semantics':'normal_cam geometry, OpenGL camera basis, ignored bump mapping; comparison is provenance diagnosis, not prediction evaluation',
        'official_metadata_sha256':sha256(official_path),'conversion_manifest_sha256':sha256(source/'conversion_manifest.jsonl'),
        'readiness':'requires explicit canonical coordinate conversion and edge-safe resizing; source RGB/depth provenance remains declared'}
    write_json(out/'hypersim_details.json',{'records':details,'raw_comparisons':matches})
    return summary


def real_indoor(dataset,dsine,out):
    root=dataset/'General/dsine_eval'
    result={}; details={}
    for split in ['train','test']:
        folder=root/'nyuv2'/split
        images=sorted(folder.glob('*_img.png'))
        missing=sum(not p.with_name(p.name.replace('_img.png','_normal.png')).exists() or
                    not p.with_name(p.name.replace('_img.png','_intrins.npy')).exists() for p in images)
        result['nyuv2_'+split]={'images':len(images),'missing_normal_or_K':missing,
            'group_mapping_available_in_package':False,
            'use':'train scenes must be mapped before holdout' if split=='train' else 'reserved benchmark test'}
    folder=root/'scannet'
    images=sorted(folder.glob('scene*/*_img.png'))
    scenes=sorted({p.parent.name for p in images})
    official={}
    for split in ['train','val']:
        url=f'https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_{split}.txt'
        try:
            p=cache_official('scannetv2_'+split+'.txt',url,out/'official_metadata')
            official[split]=set(p.read_text().split())
        except Exception as error:
            result['official_scannet_'+split+'_error']=type(error).__name__
    benchmark=dsine/'data/datasets/scannet/split/test.txt'
    reserved=set(benchmark.read_text().split())
    records=[]
    for image in images:
        relative=image.relative_to(folder).as_posix()
        records.append({'rgb_path':str(image),'scene':image.parent.name,'base_scene':image.parent.name.split('_')[0],
                        'official_split':next((k for k,v in official.items() if image.parent.name in v),'unknown'),
                        'reserved_dsine_test':relative in reserved})
    result['scannet']={'images':len(images),'scans':len(scenes),'base_scenes':len({s.split('_')[0] for s in scenes}),
        'scan_official_overlap_counts':{k:len(set(scenes)&v) for k,v in official.items()},
        'images_in_dsine_benchmark_test':sum(r['reserved_dsine_test'] for r in records),
        'train_candidate_images_excluding_reserved_benchmark':sum(r['official_split']=='train' and not r['reserved_dsine_test'] for r in records),
        'label_source':'FrameNet depth-derived normals; DSINE benchmark split preserved'}
    write_json(out/'real_indoor_details.json',{'scannet_records':records})
    return result


def objects(dataset):
    root=dataset/'General/OLATverse/val'
    objects=sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []
    return {'olatverse_validation_objects':len(objects),
            'objects_with_benchmark_normals':sum((p/'normal_benchmark').exists() for p in objects),
            'objects_with_reconstructed_meshes':sum((p/'model').exists() for p in objects),
            'training_archives_present':len(list((dataset/'General/OLATverse/downloads').glob('*Tr*.tar*'))),
            'readiness':'held-out object evaluation candidate only; no training use of curated validation objects',
            'normal_source':'photometric pseudo-GT and reconstructed meshes, not automatically direct-render geometric normals'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-root',type=Path,default=ROOT.parent/'Dataset')
    parser.add_argument('--run-id',default='P0_general_data_v1')
    args=parser.parse_args()
    out=output_path(ROOT/'outputs'/args.run_id); out.mkdir(parents=True,exist_ok=True)
    summary={'created_at':now(),'run_id':args.run_id,'status':'running','code_sha256':sha256(__file__)}
    write_json(out/'summary.json',summary)
    summary['hypersim']=hypersim(args.dataset_root,args.dataset_root/'Tools',out)
    summary['real_indoor']=real_indoor(args.dataset_root,ROOT.parent/'ExternalModels/DSINE/repo',out)
    summary['objects']=objects(args.dataset_root)
    summary['status']='completed'; summary['main_training_ready']=False
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
