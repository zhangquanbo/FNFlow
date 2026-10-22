"""Freeze a deterministic training-only subset before reading learning curves."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,hashlib
from src.research_io import ROOT,write_json,sha256,now
from src.general_dataset import PreparedNormalDataset


def select_probe(records,domain,count):
    rows=[r for r in records if r['split']=='train']
    key=lambda r:hashlib.sha256(('train-probe-v1/'+domain+'/'+r.get('image_key',Path(r['path']).name)).encode()).hexdigest()
    if len(rows)<count:raise ValueError('Insufficient training images for fixed probe')
    return sorted(rows,key=key)[:count]


def prepare(cfg):
    target=ROOT/cfg['output_root']/'training_probe_private.json'
    source={'hypersim':ROOT/cfg['hypersim_root'],'arb':ROOT/cfg['object_formal_root']}
    domains={d:select_probe(PreparedNormalDataset(path,'train').records,d,cfg['train_probe_images_per_domain']) for d,path in source.items()}
    record={'domains':domains,'selection':'SHA256 train-probe-v1/domain/stable image key; training split only',
            'source_manifest_sha256':{d:sha256(p/'private_manifest.json') for d,p in source.items()}}
    if target.exists():
        if json.loads(target.read_text())!=record:raise RuntimeError('Existing fixed training probe changed')
    else:write_json(target,record)
    write_json(target.with_name('training_probe_summary.json'),{'status':'completed','created_at':now(),
        'images_by_domain':{d:len(r) for d,r in domains.items()},'selection':record['selection'],
        'training_only':True,'augmentation':False,'subset_sha256':sha256(target)})
    return domains


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/parameterization_first_round_v2_review1.json');a=p.parse_args()
    prepare(json.loads((ROOT/a.config).read_text()))
