"""Index the existing curated object validation package without training on it."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from collections import Counter
from PIL import Image
import numpy as np
from src.research_io import ROOT,write_json,sha256,now

def main():
    source=ROOT.parent/'Dataset/General/OLATverse/val'
    records=[];issues=Counter();objects=set();views=set()
    for obj in sorted(source.iterdir()):
        if not obj.is_dir():continue
        objects.add(obj.name)
        benchmark=obj/'normal_benchmark'
        for normal in sorted((benchmark/'gt_normal').glob('*.png')):
            mask=benchmark/'mask'/normal.name
            if not mask.exists():issues['missing_mask']+=1;continue
            with Image.open(mask) as im:valid=np.asarray(im)>0;size=im.size
            with Image.open(normal) as im:
                if im.size!=size:issues['normal_mask_shape_mismatch']+=1
            if not valid.any():issues['empty_mask']+=1
            views.add((obj.name,normal.stem))
            for illumination in ['input','input1','input2','input3']:
                rgb=benchmark/illumination/normal.name
                if not rgb.exists():issues['missing_rgb']+=1;continue
                with Image.open(rgb) as im:
                    if im.size!=size:issues['rgb_mask_shape_mismatch']+=1
                records.append({'asset':obj.name,'camera':normal.stem,'appearance':illumination,
                    'rgb_path':str(rgb),'normal_path':str(normal),'mask_path':str(mask),
                    'split':'reserved_object_evaluation','K_valid':False,
                    'camera_warning':'all_cam calibration is for original captures; benchmark crop/resize transform not yet validated'})
    target=ROOT/'outputs/P0_object_benchmark_v1'
    write_json(target/'private_manifest.json',{'records':records})
    summary={'created_at':now(),'status':'completed','objects':len(objects),'geometry_views':len(views),
             'appearance_images':len(records),'appearance_variants_per_view':4,'issues':dict(issues),
             'training_images':0,'input_protocol':'single RGB; extra illumination not an inference input',
             'label_source':'photometric/reconstruction-derived benchmark normals; coordinate and crop calibration still require verification',
             'metric_protocol_ready':False,'code_sha256':sha256(__file__)}
    write_json(target/'summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main()
