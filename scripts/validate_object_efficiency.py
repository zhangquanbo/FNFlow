import json,csv,statistics
from pathlib import Path
R=Path(__file__).resolve().parents[1];o=R/'outputs/object_efficiency_20261023';j=json.loads((o/'results.json').read_text());assert set(j['isolated_methods'])=={'R','V','Xn'};rows=[]
for model,r in j['models'].items():
 assert len(r['times_ms'])==30
 median=statistics.median(r['times_ms']);p90=statistics.quantiles(r['times_ms'],n=10,method='inclusive')[8]
 assert abs(median-r['median_ms'])<1e-8 and abs(p90-r['p90_ms'])<1e-8
 rows.append({'model':model,'median_ms':median,'p90_ms':p90,'peak_allocated_MiB':r['peak_allocated_bytes']/1024**2,'peak_reserved_MiB':r['peak_reserved_bytes']/1024**2,'parameters':r.get('parameters'),'stage1_parameters':r.get('stage1_parameters_including_frozen_components'),'stage2_parameters':r.get('stage2_parameters'),'diffusion_steps':r.get('diffusion_steps'),'UNet_calls':r.get('underlying_stage1_unet_calls_per_image',r.get('unet_forward_calls')),'precision':r['precision']})
with (o/'efficiency_table.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(o/'validation.json').write_text(json.dumps({'status':'ready_with_protocol_caveats','independent_recalculation':'Python statistics median and inclusive decile90 match all30 raw times','warmups':5,'timed':30,'same_GPU':'RTX4090','batch':1,'resolution':512,'caveats':['NLR officialfloat32; ours bf16 autocast.','End-to-end includes official NLR concatenated PNG plus normal PNG extraction; ours normal PNG only.','Our512 efficiency input differs from native256 accuracy input, reported explicitly.','Original own measurements with retained NLR vision encoder rejected; fresh processes replace them.'],'rows':rows},indent=2));print(json.dumps(rows,indent=2))