"""Bind the completed precurve data review without changing WSD budget rules."""
import json,hashlib


def require_precurve_quality_review(root):
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    path=root/'provenance/precurve_data_quality_approval_review3.json'
    receipt=json.loads(path.read_text())
    if receipt.get('status')!='approved':raise RuntimeError('Precurve data quality review unresolved')
    checks={
        'rule_sha256':'configs/precurve_data_quality_review3.json',
        'wsd_rule_sha256':'configs/convergence_decision_review2.json',
        'hypersim_manifest_sha256':'data/hypersim_trainval_center_v2/private_manifest.json',
        'direction_summary_sha256':'outputs/P1_hypersim_direction_review3/summary.json',
        'preregistration_sha256':'provenance/data_quality_review3_preregistration.json'}
    if any(receipt[key]!=digest(root/relative) for key,relative in checks.items()):
        raise RuntimeError('Data-quality review signature changed')
    return digest(path)
