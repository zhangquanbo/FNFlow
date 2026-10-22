"""Finalize descriptive quality flags after visual and source-material inspection."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
from src.research_io import ROOT,write_json,now


def main():
    out=ROOT/'outputs/P1_arb_outlier_visual_review3'
    details=json.loads((out/'details_private.json').read_text())['records']
    materials={r['asset']:r for r in json.loads((out/'material_details_private.json').read_text())['records']}
    for row in details:
        m=materials[row['asset']]['materials']
        mixed=any(r['alphaMode']=='BLEND' or r['extensions'].get('KHR_materials_transmission',{}).get('transmissionFactor',0)>0 for r in m)
        row.update(visual_review_completed=True,
            source_blend_or_transmission=mixed,source_double_sided_declared=any(r['doubleSided'] for r in m),
            zero_scale_normal_texture_slots=sum(r['normalTexture'] is not None and r['normalTexture'].get('scale',1)==0 for r in m),
            quality_flag='material_mixture_confounded_nonunit_pass' if mixed else 'geometry_or_coverage_unresolved',
            disposition='retain existing strict finite near-unit pixel exclusion; do not use rejected pixels as q evidence or restore them for a q experiment before label semantics are resolved',
            cause_proven=False)
    write_json(out/'reviewed_details_private.json',{'records':details,'created_at':now()})
    source=json.loads((ROOT/'outputs/P1_normal_length_texture_v1/details_private.json').read_text())['records']
    flagged={r['asset'] for r in details if r['group']=='present'}
    rest={'present':0,'not_declared_in_gltf':0}
    for r in source:
        if r['asset'] not in flagged:rest[r['group']]+=r['counts']['0.99']['relaxed_direction_plus_silhouette']['unit_rejected_outside_band']
    result={'status':'completed','created_at':now(),'visually_reviewed_assets':len(details),
        'source_blend_or_transmission_assets':sum(r['source_blend_or_transmission'] for r in details),
        'double_sided_declarations':sum(r['source_double_sided_declared'] for r in details),
        'zero_scale_normal_texture_slots':sum(r['zero_scale_normal_texture_slots'] for r in details),
        'dominant_three_present_assets_excluded_from_this_q_sensitivity_summary':3,
        'remaining_outside_pixel_counts':rest,
        'remaining_present_share':rest['present']/sum(rest.values()),
        'training_mask_or_asset_split_changed':False,
        'quality_exclusion':'existing finite and abs(norm(raw)-1)<=.05 pixel mask; material-confounded rejected pixels must not be restored as length-q labels without resolving label semantics',
        'q_claim_supported':False,'boundary_position_only_control_required_for_later_q_tests':True,
        'limitations':'visual/source-material audit does not prove exact rendering cause, flipped topology, or an unnormalized normal map; doubleSided is a material declaration, not proof of duplicate opposing faces'}
    write_json(out/'summary.json',result);print(json.dumps(result))


if __name__=='__main__':main()
