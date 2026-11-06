
from pathlib import Path
import hashlib,json,sys,tarfile,time,shutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.research_io import write_json,sha256
from scripts.prepare_real_protocols_20261106 import save_rgb,publish,OUT,AB,STATUS
EXPECTED='5e57f70ea99001016c6c262f792e54db4639c2f49b93073a26b9134384ac77ce'
def main():
 archive=OUT/'diligent_lights_verified_20261106.tar.partial'
 while not archive.exists() or archive.stat().st_size!=88678400:
  write_json(STATUS,{'phase':'waiting_for_verified_mirror_upload','updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')});time.sleep(5)
 assert sha256(archive)==EXPECTED
 target=OUT/'verified_diligent_source_20261106';target.mkdir(exist_ok=True)
 with tarfile.open(archive,'r') as tar:
  for info in tar.getmembers():
   if not info.isfile():continue
   dest=target/info.name
   assert dest.resolve().is_relative_to(target.resolve())
   dest.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(info) as src,dest.open('wb') as dst:shutil.copyfileobj(src,dst)
 receipt=json.loads((target/'mirror_receipt_private.json').read_text())
 assert receipt['status']=='verified' and receipt['existing_source_files_verified']==60 and receipt['ten_light_files']==100
 for item in receipt['files']:assert sha256(target/item['member'])==item['sha256']
 old=json.loads((ROOT/'data/real_object_selected_v1/diligent/receipt_private.json').read_text())
 for item in old['files']:assert sha256(target/item['member'])==item['sha256']
 records=[]
 for row in json.loads((AB/'diligent.json').read_text())['records']:
  for light in range(21,31):
   name=f"DiLiGenT/pmsData/{row['object']}/{light:03d}.png"
   records.append(save_rgb(row,(target/name).read_bytes(),light,name))
 publish('diligent',records)
 write_json(ROOT/'provenance/diligent_mirror_amendment_20261106.json',{'status':'verified','reason':'Original Google route timed out. Alternate source from research authors download script.','source_url':receipt['source_url'],'source_evidence':receipt['source_evidence'],'archive_bytes':891519958,'uploaded_tar_sha256':EXPECTED,'existing_original_source_files_byte_identical':60,'new_lights_per_object':list(range(21,31)),'new_images':100,'mask_and_normal_GT_unchanged':True,'test_GSO_opened':False})
 assert (OUT/'rose10_luces.json').exists()
 write_json(STATUS,{'phase':'completed','single_images':24,'ten_light_images':240,'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
 print('REAL_DATA_COMPLETED_24_CONTROLS_240_TEN_LIGHT_IMAGES',flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  write_json(STATUS,{'phase':'failed','error':repr(error),'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')});raise
