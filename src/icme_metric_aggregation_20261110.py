from pathlib import Path
import sys,json,collections
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.cluster_bootstrap import cluster_percentile_intervals
FIELDS=['mean_deg','median_deg','fraction_below_11.25','fraction_below_22.5','fraction_below_30','boundary_primary_angular_mae_deg','boundary_primary_angular_outside_mae_deg']
def aggregate(rows):
 result={}
 for key in FIELDS:
  if key.startswith('boundary_') and not any(key in r['metrics'] for r in rows):continue
  groups=collections.defaultdict(lambda:collections.defaultdict(list));empty=0
  for row in rows:
   assert key in row['metrics'],key;v=row['metrics'][key]
   if v is None:empty+=1;continue
   assert np.isfinite(v);groups[row['object']][row.get('elevation_bin')].append(v)
  values={obj:float(np.mean([np.mean(v) for v in pitch.values()])) for obj,pitch in groups.items()};ids=sorted(values);result[key]={'mean':float(np.mean(list(values.values()))) if values else None,'eligible_assets':len(values),'undefined_empty_region_images':empty,'bootstrap':cluster_percentile_intervals([values[x] for x in ids],ids,draws=10000,seed=20261110) if len(values)>1 else None}
 pitches=sorted({r['elevation_bin'] for r in rows if r.get('elevation_bin') is not None});return {'assets':len({r['object'] for r in rows}),'images':len(rows),'metrics':result,'by_pitch':{str(p):aggregate([r for r in rows if r.get('elevation_bin')==p]) if len(pitches)>1 else None for p in pitches}}
