import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
from src.general_dataset import PreparedNormalDataset
from src.research_io import ROOT


@pytest.fixture
def sample_root():
    parent=ROOT/'outputs/.test_data';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        root=Path(directory)
        normal=np.array([[[0.,0.,1.],[np.nan,0,0]]],dtype=np.float32)
        np.savez(root/'sample.npz',rgb=np.zeros((1,2,3),dtype=np.uint8),normal=normal,
                 valid=np.array([[True,False]]),K=np.zeros((3,3)),K_valid=np.array(False),projection_3x3=np.eye(3))
        record={'scene':'synthetic-room','split':'train','path':str(root/'sample.npz')}
        (root/'private_manifest.json').write_text(json.dumps({'records':[record]}))
        yield root


def test_loader_masks_invalid_labels_and_does_not_fabricate_intrinsics(sample_root):
    sample=PreparedNormalDataset(sample_root,'train')[0]
    assert sample['condition'].shape==(3,1,2)
    assert sample['condition'].eq(-1).all()
    assert sample['normal'][:,:,1].eq(0).all()
    assert not sample['K_valid'] and not sample['has_depth']


def test_scene_leakage_is_rejected(sample_root):
    path=sample_root/'private_manifest.json';manifest=json.loads(path.read_text())
    manifest['records'].append({**manifest['records'][0],'split':'val'})
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='Scene crosses'):
        PreparedNormalDataset(sample_root,'train')


def test_valid_zero_normal_is_rejected(sample_root):
    path=sample_root/'sample.npz'
    with np.load(path) as handle:values={key:handle[key] for key in handle.files}
    values['normal'][0,0]=0
    np.savez(path,**values)
    with pytest.raises(ValueError,match='valid-pixel'):
        PreparedNormalDataset(sample_root,'train')[0]


def test_result_renderer_accepts_separate_data_audit_schema(sample_root,monkeypatch):
    from scripts import render_record
    folder=sample_root/'outputs/data_audit';folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps({'status':'completed','num_rows':4}))
    monkeypatch.setattr(render_record,'ROOT',sample_root)
    render_record.render()
    assert (sample_root/'docs/代码与实验记录.md').is_file()
