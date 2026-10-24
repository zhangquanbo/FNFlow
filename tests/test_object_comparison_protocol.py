"""Analytic checks for normalization, masks, thresholds, and unequal-object macros."""
import sys,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from src.research_io import sha256
def main():
 from scripts import evaluate_object_comparison as ev
 with tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='analytic_metric_check_') as tmp:
  folder=Path(tmp);gt=np.zeros((2,2,3),np.float32);gt[...,2]=1;valid=np.ones((2,2),bool);p=folder/'gt.npz';np.savez(p,normal=gt,valid=valid)
  r={'domain':'arb','gt_path':str(p),'gt_sha256':sha256(p)};angles=np.deg2rad(np.array([[0,10],[20,40]]));pred=np.stack([np.sin(angles),np.zeros_like(angles),np.cos(angles)],axis=-1)*7
  m,_=ev.metrics(pred,r);a=m['main'];assert abs(a['mean']-17.5)<1e-4 and abs(a['median']-15)<1e-4
  assert a['lt_11_25']==50 and a['lt_22_5']==75 and a['lt_30']==75 and a['boundary_mean'] is None
  bad=pred.copy();bad[0,0]=0
  try:ev.metrics(bad,r)
  except ValueError:pass
  else:raise AssertionError('Zero GT-valid predictions were silently filtered')
  ev.OUT=folder/'metrics';ev.OUT.mkdir();ev.MAN={'coordinate_lock_sha256':ev.MAN['coordinate_lock_sha256'],'records':[{'id':str(i),'image_key':str(i),'domain':'arb','object':'A' if i<3 else 'B'} for i in range(4)],'nlr_variability_ids':[],'arb_overlap_note':{}}
  for row in ev.MAN['records']:
   value=10. if row['object']=='A' else 40.;vals={k:value for k in ['mean','median','rmse','lt_11_25','lt_22_5','lt_30','boundary_mean','outside_mean']};vals.update(valid_pixels=4,boundary_pixels=4,outside_pixels=0)
   ev.save_row('R',42,row,{'main':vals})
  ev.summarize();rows=json.loads((ev.OUT/'summary.json').read_text())['rows'];by={r['aggregation']:r for r in rows}
  assert by['image_macro']['mean']==17.5 and by['object_macro']['mean']==25
 print('PASS: analytic angles/thresholds, unit normalization, invalid predictions fail, unequal-object macros',flush=True)
if __name__=='__main__':main()
