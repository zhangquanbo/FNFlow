"""Register current public source for a fresh, independently signed local reproduction."""
from pathlib import Path
import argparse,json,hashlib,datetime
R=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--trainer',default='scripts/train_ICME_Rplus_20261110.py');a=p.parse_args();cfg=json.loads(a.config.read_text());dest=R/cfg['registration_path'];assert dest.resolve().is_relative_to((R/'outputs').resolve());assert not dest.exists(),'Registration exists; never overwrite a run signature'
 for key in ['training_manifest','validation_manifest','prior_weights']:
  target=R/cfg[key];assert target.is_file(),str(target);expected=cfg.get(key+'_sha256');assert expected is None or digest(target)==expected,('External prerequisite SHA mismatch',key)
 training=json.loads((R/cfg['training_manifest']).read_text())['records'];validation=json.loads((R/cfg['validation_manifest']).read_text())['records'];train=[x for x in training if x['split']=='train'];val=[x for x in validation if x['split']=='val'];assert len(train)==cfg['training_images'] and len({x['scene'] for x in train})==cfg['training_assets'];assert not {x['scene'] for x in train}&{x['scene'] for x in val}
 source=[x for x in (R/'src').rglob('*.py') if '__pycache__' not in x.parts]+[R/a.trainer,R/'scripts/train_parameterization.py',R/'scripts/evaluate_fresh_ab.py'];payload={'config_sha256':digest(a.config),'trainer_sha256':digest(R/a.trainer),'source_sha256':{str(x.relative_to(R)):digest(x) for x in source},'fixed_final_update':cfg['total_updates'],'registered_before_training':True,'scope':'fresh local reproduction; never replace a reported checkpoint signature','test_images_scored':0,'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(payload,indent=2)+'\n');print(dest)
if __name__=='__main__':main()
