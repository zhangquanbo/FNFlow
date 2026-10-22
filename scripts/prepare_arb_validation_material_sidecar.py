"""Freeze metadata groups for all 200 planned validation assets without editing data."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from src.research_io import ROOT,write_json,sha256,now
from src.material_protocol import load_material_audit
from scripts.prepare_arb_training import fixed_formal_partition


def main():
    folder=ROOT/'outputs/P1_arb_materials_review4'
    if (folder/'validation_ready.json').exists():load_material_audit(ROOT);return True
    config=json.loads((ROOT/'configs/arb_material_audit_review4.json').read_text())
    plan=fixed_formal_partition();selected=sorted(uid for uid,split in plan.items() if split=='val')
    if len(selected)!=200:raise RuntimeError('Wrong fixed validation plan')
    rows=[]
    for uid in selected:
        path=folder/'private_assets'/f'{uid}.json';source=path.with_suffix('.gltf')
        if not path.exists():return False
        row=json.loads(path.read_text())
        if row['split']!='val' or row['source_revision']!=config['source_revision'] or row['document_sha256']!=sha256(source):
            raise RuntimeError('Validation metadata cache mismatch')
        rows.append(row)
    index=folder/'validation_material_index_private.json'
    write_json(index,{'assets':{r['asset']:r for r in rows},'source_revision':config['source_revision']})
    write_json(folder/'validation_ready.json',{'status':'completed','classified_assets':200,'failed_assets':0,
        'material_index_sha256':sha256(index),'created_at':now(),'training_filter_enabled':False})
    return True


if __name__=='__main__':print(json.dumps({'validation_metadata_ready':main()}))
