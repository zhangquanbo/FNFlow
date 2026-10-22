"""Report foreground removed by near-unit masking and its boundary concentration."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import argparse,json
import cv2
import numpy as np
from PIL import Image
from src.boundary_audit import audit_mask,add_counts,fractions
from src.research_io import ROOT,write_json,sha256,now
from scripts.prepare_arb_training import fixed_formal_partition


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['pilot','formal'],required=True)
    args=parser.parse_args();source=ROOT/f'outputs/P1_arb_acquisition_v1/{args.phase}_manifest_private.json'
    assets=json.loads(source.read_text())['records'];plan=fixed_formal_partition()
    total={};by_split={'train':{},'val':{}};details=[];n=0
    for asset in assets:
        folder=ROOT/asset['folder'];split=plan[asset['asset']]
        for path in sorted(folder.glob('normal_*.exr')):
            view=path.stem.split('_')[-1];rgbs=list(folder.glob(f'color_*_{view}.png'))
            if len(rgbs)!=1:raise RuntimeError('Selected RGB must be unique')
            native=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
            if native is None:raise RuntimeError('EXR decode failed')
            rgba=np.asarray(Image.open(rgbs[0]).convert('RGBA'))
            if native.shape[:2]!=rgba.shape[:2]:raise RuntimeError('RGB normal mismatch')
            h,w=native.shape[:2];xs=((np.arange(256)+.5)*w/256).astype(int);ys=((np.arange(256)+.5)*h/256).astype(int)
            decoded=native[ys[:,None],xs[None,:]]*2-1
            alpha=rgba[ys[:,None],xs[None,:],3].astype(np.float32)/255
            result=audit_mask(decoded,alpha);add_counts(total,result);add_counts(by_split[split],result)
            details.append({'asset':asset['asset'],'view':view,'split':split,'counts':result});n+=1
        if n%150==0:print('boundary audit views',n,flush=True)
    summary={'status':'completed','created_at':now(),'phase':args.phase,'acquired_assets':len(assets),'views':n,
             'resolution':[256,256],'unit_tolerance':.05,'angle_threshold_deg':20,'band_dilation_radius':2,
             'includes_empty_label_views_and_assets':True,
             'foreground':'selected RGB alpha thresholds 0.01 / 0.5 / 0.99, auxiliary coverage proxy only; never changes training or error masks',
             'band_definitions':{'strict_gt_before_valid_intersection':'same valid-neighbor angular edges as MAE, dilate first; do not intersect final mask for exclusion audit',
                'relaxed_direction_plus_silhouette':'finite nonzero raw decoded directions, including nonunit normals, plus alpha silhouette; candidate band, not trusted GT on excluded pixels'},
             'denominator':'finite foreground pixels failing abs(norm-1)<=0.05; nonfinite and zero counts also reported; fractions pooled by pixels',
             'all':fractions(total),'by_split':{s:fractions(v) for s,v in by_split.items()},
             'caution':'No direction GT is restored for rejected pixels. Candidate boundaries cannot establish true geometric errors there.'}
    out=ROOT/f'outputs/P1_arb_boundary_audit_{args.phase}_v1'
    write_json(out/'summary.json',summary)
    write_json(out/'per_view_private.json',{'source_manifest_sha256':sha256(source),'records':details})
    print(json.dumps({'status':'completed','views':n,'summary':str(out/'summary.json')}),flush=True)


if __name__=='__main__':main()
