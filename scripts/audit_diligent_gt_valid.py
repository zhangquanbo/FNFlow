import sys,json
from pathlib import Path
import numpy as np,cv2
from scipy.io import loadmat
R=Path(__file__).resolve().parents[1];rows=[]
for folder in sorted((R/'data/real_object_selected_v1/diligent').iterdir()):
 if not folder.is_dir():continue
 mask=cv2.imread(str(folder/'mask.png'),0)>0;n=loadmat(folder/'Normal_gt.mat')['Normal_gt'];finite=np.isfinite(n).all(-1);nonzero=np.linalg.norm(np.nan_to_num(n),axis=-1)>1e-8
 rows.append({'object':folder.name,'official_mask':int(mask.sum()),'zero_gt_inside':int((mask&~nonzero).sum()),'nonfinite_gt_inside':int((mask&~finite).sum()),'gt_valid':int((mask&finite&nonzero).sum())})
p=R/'outputs/diligent_gt_valid_audit.json';p.write_text(json.dumps(rows,indent=2));print(json.dumps(rows))