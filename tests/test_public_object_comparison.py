"""Publication must resist accidental propagation of sample-level additions."""
import pytest
from src.public_object_comparison import public_summary, select_fields, EFFICIENCY_FIELDS

def test_new_private_summary_fields_are_not_published():
    source={'rows':[{'model':'R','domain':'gso','complete':True,'mean':10.0,'image_key':'PRIVATE_ASSET','prediction_path':'/private/output.png'}],
            'seed_variability':[{'model':'R','domain':'gso','complete':True,'seeds':[42],'image_macro_mae_by_seed':[10.0],'derived_seeds':{'PRIVATE_ASSET':123}}],
            'private_records':[{'object':'PRIVATE_ASSET'}]}
    public=public_summary(source)
    assert 'PRIVATE_ASSET' not in str(public)
    assert 'prediction_path' not in str(public)
    assert public['rows'][0]['mean']==10.0

def test_partial_run_cannot_be_released_as_complete():
    with pytest.raises(AssertionError):public_summary({'rows':[{'complete':False}],'seed_variability':[]})

def test_benchmark_repetition_paths_are_not_published():
    public=select_fields({'median_ms':100,'times_ms':[99,100,101],'source_sha256':'PRIVATE_SOURCE','end_to_end':'private local path'},EFFICIENCY_FIELDS)
    assert public=={'median_ms':100}
