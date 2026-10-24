import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
jobs=[('sd2-community/stable-diffusion-2-1','bb2154823665391b4fb29b0b9cf82a198964ee05',['model_index.json','feature_extractor/preprocessor_config.json','scheduler/scheduler_config.json','text_encoder/config.json','text_encoder/model.safetensors','tokenizer/merges.txt','tokenizer/special_tokens_map.json','tokenizer/tokenizer_config.json','tokenizer/vocab.json','unet/config.json','unet/diffusion_pytorch_model.safetensors','vae/config.json','vae/diffusion_pytorch_model.safetensors']),('sd2-community/stable-diffusion-2-1-unclip','ac4aaff4a67d57bdab2e9faf277c33d8e49f4df4',['image_encoder/config.json','image_encoder/model.safetensors']),('sudo-ai/zero123plus-v1.2','2da07e89919e1a130c9b5add1584c70c7aa065fd',['scheduler/scheduler_config.json','feature_extractor_vae/preprocessor_config.json'])]
rows=[];missing=[]
for repo,rev,names in jobs:
 root=R/'external/base_models'/repo.replace('/','__')
 for name in names:
  p=root/name
  if not p.is_file():missing.append(str(p));continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  row={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
  meta=root/'.cache/huggingface/download'/(name+'.metadata')
  if meta.exists():
   lines=meta.read_text().splitlines();row.update(metadata_revision=lines[0],metadata_etag=lines[1]);assert lines[0]==rev,(p,lines[0],rev)
   if len(lines[1])==64:assert h.hexdigest()==lines[1],p
   elif len(lines[1])==40:assert hashlib.sha1(f'blob {p.stat().st_size}\0'.encode()+p.read_bytes()).hexdigest()==lines[1],p
  rows.append(row)
result={'all_required_present':not missing,'missing':missing,'verified_existing_files':rows,'model_scores_computed':False}
(R/'provenance/lightrig_base_intake_partial.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'all_required_present':not missing,'verified_existing_files':len(rows),'missing':missing},indent=2))
