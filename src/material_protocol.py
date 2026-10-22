"""Metadata-only asset flags and paired aggregate validation sensitivity."""
import json,hashlib
import numpy as np


def material_flags(document):
    materials=document.get('materials',[]);blend=[];transmission=[]
    for index,m in enumerate(materials):
        mode=m.get('alphaMode','OPAQUE')
        if mode not in {'OPAQUE','MASK','BLEND'}:raise ValueError('Unknown glTF alphaMode')
        factor=m.get('extensions',{}).get('KHR_materials_transmission',{}).get('transmissionFactor',0)
        if not isinstance(factor,(int,float)) or not np.isfinite(factor) or not 0<=factor<=1:raise ValueError('Invalid transmissionFactor')
        if mode=='BLEND':blend.append(index)
        if factor>0:transmission.append(index)
    used={p['material'] for m in document.get('meshes',[]) for p in m.get('primitives',[]) if 'material' in p}
    if any(not isinstance(i,int) or i<0 or i>=len(materials) for i in used):raise ValueError('Unknown referenced material')
    return {'flagged':bool(blend or transmission),'blend':bool(blend),'transmission':bool(transmission),
        'material_count':len(materials),'referenced_material_flagged':bool(used&(set(blend)|set(transmission))),
        'blend_slots':len(blend),'positive_transmission_slots':len(transmission)}


def load_material_audit(root):
    directory=root/'outputs/P1_arb_materials_review4';path=directory/'validation_material_index_private.json'
    summary=json.loads((directory/'validation_ready.json').read_text())
    if summary['status']!='completed' or summary['classified_assets']!=200 or summary['failed_assets']:
        raise RuntimeError('All 200 validation assets need metadata; unknown assets cannot be called clean')
    raw=path.read_bytes();index=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=summary['material_index_sha256']:raise RuntimeError('Material sidecar changed')
    if len(index['assets'])!=200:raise RuntimeError('Incomplete validation material sidecar')
    return index['assets'],hashlib.sha256(raw).hexdigest()


def validation_material_sensitivity(records,errors,boundary_errors,dispersion,flags):
    """Reuse existing paired per-image predictions; never run a model or change input data."""
    from src.noise_protocol import mae_seed_statistics
    assets={r['scene'] for r in records}
    if any(uid not in flags or not isinstance(flags[uid]['flagged'],bool) for uid in assets):
        raise RuntimeError('Unknown material classification in validation')
    indices=[i for i,r in enumerate(records) if not flags[r['scene']]['flagged']]
    remaining={records[i]['scene'] for i in indices};excluded=assets-remaining
    base={'status':'completed' if indices else 'empty_subset','full_usable_assets':len(assets),
        'excluded_flagged_assets':len(excluded),'remaining_assets':len(remaining),'full_images':len(records),'remaining_images':len(indices),
        'flag_definition':'any declared alphaMode=BLEND or positive KHR_materials_transmission factor',
        'same_predictions_noise_and_masks':True,'unknown_assets':0}
    if not indices:return base
    data=np.asarray(errors)[indices];stats=mae_seed_statistics(data) if data.shape[1]>1 else {'num_images':len(data),'mean_mae_deg':float(data.mean())}
    bands=[boundary_errors[i] for i in indices if boundary_errors[i] is not None]
    perasset=[np.asarray([errors[i] for i in indices if records[i]['scene']==uid]).mean() for uid in sorted(remaining)]
    stats.update(asset_macro_mean_mae_deg=float(np.mean(perasset)),boundary_mae_deg=float(np.mean(bands)) if bands else None,
        boundary_images=len(bands),mean_pixel_angular_dispersion_deg=float(np.mean([dispersion[i] for i in indices])))
    return {**base,'metrics':stats}
