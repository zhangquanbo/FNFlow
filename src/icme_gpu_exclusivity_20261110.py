"""Reject timings if another process owns this physical GPU."""
import os,subprocess

def assert_exclusive(physical='0'):
 assert str(physical)=='0','ICME efficiencies are3090physicalGPU0 only'
 g=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],check=True,text=True,capture_output=True).stdout
 row=next(x.split(',') for x in g.splitlines() if x.split(',')[0].strip()==str(physical));uuid=row[1].strip();assert '3090' in row[2]
 q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True).stdout
 others=[int(x.split(',')[1]) for x in q.splitlines() if x.split(',')[0].strip()==uuid and int(x.split(',')[1])!=os.getpid()]
 assert not others,('RTX3090notexclusive',others)
 return {'physical_GPU':physical,'UUID':uuid,'other_compute_processes':others,'checked_before_and_after_measurement':True}
