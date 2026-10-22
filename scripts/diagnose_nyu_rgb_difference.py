"""Describe color/affine differences without altering packaged RGB, labels or K."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,numpy as np,cv2
from PIL import Image
from src.research_io import ROOT,write_json,now

def analyze(official,packaged):
    reference=official.astype(np.float64)/255;actual=packaged.astype(np.float64)/255
    channels=[]
    for c in range(3):
        x=reference[...,c].reshape(-1);y=actual[...,c].reshape(-1)
        gain,bias=np.linalg.lstsq(np.column_stack([x,np.ones_like(x)]),y,rcond=None)[0]
        fitted=gain*x+bias
        channels.append({'gain':float(gain),'bias_0_to_1':float(bias),
            'pearson_before':float(np.corrcoef(x,y)[0,1]),'pearson_after_linear_fit':float(np.corrcoef(fitted,y)[0,1]),
            'rmse_after_fit_0_to_255':float(np.sqrt(np.mean((fitted-y)**2))*255)})
    a=cv2.cvtColor(reference.astype(np.float32),cv2.COLOR_RGB2GRAY)
    b=cv2.cvtColor(actual.astype(np.float32),cv2.COLOR_RGB2GRAY)
    shift,response=cv2.phaseCorrelate(a,b)
    warp=np.array([[1,0,shift[0]],[0,1,shift[1]]],np.float32)
    try:
        ecc,warp=cv2.findTransformECC(a,b,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,100,1e-6),None,5)
        spatial={'status':'converged','ecc_correlation':float(ecc),'official_to_packaged_affine':warp.tolist(),
                 'linear_scale_singular_values':np.linalg.svd(warp[:,:2],compute_uv=False).tolist()}
    except cv2.error as error:spatial={'status':'not_converged','error':str(error).splitlines()[0]}
    return {'channel_linear_fits':channels,'phase_translation_pixels':list(shift),'phase_response':float(response),
        'affine_diagnostic':spatial,'raw_mean_abs_0_to_255':float(np.abs(reference-actual).mean()*255)}

def main():
    folder=ROOT/'outputs/P0_data_intake_local_v1'
    indices=[r['official_index_1based'] for r in json.loads((folder/'nyu_rgb_checks.json').read_text())['records']]
    records=[]
    for index in indices:
        reference=np.load(folder/f'nyu_official_rgb_{index:04d}.npy')
        path=ROOT.parent/f'Dataset/General/dsine_eval/nyuv2/train/{index-1:06d}_img.png'
        packaged=np.asarray(Image.open(path).convert('RGB'))
        records.append({'official_index_1based':index,**analyze(reference,packaged)})
    out=ROOT/'outputs/P1_nyu_rgb_diagnostic_v1'
    write_json(out/'private_details.json',{'records':records})
    summary={'status':'completed','created_at':now(),'non_test_images':len(records),
        'mean_raw_absolute_pixel_difference_0_to_255':float(np.mean([r['raw_mean_abs_0_to_255'] for r in records])),
        'per_channel_pearson_range':[min(c['pearson_before'] for r in records for c in r['channel_linear_fits']),max(c['pearson_before'] for r in records for c in r['channel_linear_fits'])],
        'max_affine_pearson_change':max(abs(c['pearson_before']-c['pearson_after_linear_fit']) for r in records for c in r['channel_linear_fits']),
        'affine_converged':sum(r['affine_diagnostic']['status']=='converged' for r in records),
        'inference':'diagnostic fits do not identify the full preprocessing chain; RGB/normal/K are used from the same DSINE package without correction',
        'blocks_no_ray_training':False,'packaged_K_overwritten':False,
        'statistical_note':'Per-channel Pearson correlation is invariant to a positive affine intensity transform; residual reduction, not a correlation rise, tests a gain/bias explanation.'}
    write_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
