"""Aggregate the fixed validation metadata census, independent of model outputs."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.material_protocol import load_material_audit
from src.research_io import ROOT,write_json,now
import json


def main():
    assets,digest=load_material_audit(ROOT);rows=list(assets.values())
    result={'status':'completed','created_at':now(),'scope':'all 200 planned validation assets only; not the 2000-asset population',
        'classified_assets':len(rows),'flagged_assets':sum(r['flagged'] for r in rows),
        'flagged_fraction':sum(r['flagged'] for r in rows)/len(rows),'blend_assets':sum(r['blend'] for r in rows),
        'positive_transmission_assets':sum(r['transmission'] for r in rows),
        'referenced_material_flagged_assets':sum(r['referenced_material_flagged'] for r in rows),
        'unknown_assets':0,'material_sidecar_sha256':digest,'models_used':False,'training_data_changed':False,
        'flag':'any declared alphaMode=BLEND OR positive KHR_materials_transmission factor'}
    write_json(ROOT/'outputs/P1_arb_validation_materials_review4/summary.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
