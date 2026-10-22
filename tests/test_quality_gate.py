import json,hashlib
import pytest
from src.quality_gate import require_precurve_quality_review


def test_missing_review_and_postreview_manifest_change_block_long_training(tmp_path):
    with pytest.raises(FileNotFoundError):require_precurve_quality_review(tmp_path)
    fields={'rule_sha256':'configs/precurve_data_quality_review3.json',
        'wsd_rule_sha256':'configs/convergence_decision_review2.json',
        'hypersim_manifest_sha256':'data/hypersim_trainval_center_v2/private_manifest.json',
        'direction_summary_sha256':'outputs/P1_hypersim_direction_review3/summary.json',
        'preregistration_sha256':'provenance/data_quality_review3_preregistration.json'}
    receipt={'status':'approved'}
    for key,relative in fields.items():
        path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        receipt[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    approval=tmp_path/'provenance/precurve_data_quality_approval_review3.json';approval.write_text(json.dumps(receipt))
    assert require_precurve_quality_review(tmp_path)==hashlib.sha256(approval.read_bytes()).hexdigest()
    (tmp_path/fields['hypersim_manifest_sha256']).write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='signature changed'):require_precurve_quality_review(tmp_path)
