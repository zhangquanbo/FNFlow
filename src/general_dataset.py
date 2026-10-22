"""Loader for locally prepared, explicitly versioned normal-data protocol pilots."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class PreparedNormalDataset(Dataset):
    def __init__(self,root,split):
        self.root=Path(root)
        manifest=json.loads((self.root/'private_manifest.json').read_text())
        records=manifest['records']
        groups={}
        for item in records:
            previous=groups.setdefault(item['scene'],item['split'])
            if previous!=item['split']:
                raise ValueError('Scene crosses dataset partitions')
        self.records=[item for item in records if item['split']==split]
        if not self.records:raise ValueError('Empty requested partition')

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index]
        with np.load(record['path']) as handle:
            rgb=handle['rgb'];normal=handle['normal'].astype(np.float32)
            valid=handle['valid'].astype(bool)
            k=handle['K'].astype(np.float32);k_valid=bool(handle['K_valid'])
            projection=handle['projection_3x3'].astype(np.float32)
            projection_valid=bool(handle['projection_valid']) if 'projection_valid' in handle else True
        if rgb.shape!=normal.shape or valid.shape!=normal.shape[:2] or not valid.any():
            raise ValueError('Unaligned or empty sample')
        if not np.isfinite(normal[valid]).all() or not np.allclose(np.linalg.norm(normal[valid],axis=-1),1,atol=2e-4):
            raise ValueError('Invalid valid-pixel normals')
        if projection_valid and (not np.isfinite(projection).all() or abs(np.linalg.det(projection))<1e-8):
            raise ValueError('Invalid exact camera projection')
        if k_valid and (not np.isfinite(k).all() or not np.allclose(k[2],[0,0,1])):
            raise ValueError('Invalid pinhole intrinsics')
        normal=np.where(valid[...,None],normal,0)
        return {'condition':torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()/127.5-1,
                'normal':torch.from_numpy(np.ascontiguousarray(normal.transpose(2,0,1))),
                'valid':torch.from_numpy(valid),'K':torch.from_numpy(k),'K_valid':k_valid,
                'projection_3x3':torch.from_numpy(projection),'projection_valid':projection_valid,'has_depth':False,
                'scene':record['scene'],'label_source':record.get('label_source','rendered_geometry'),
                'input_protocol':'RGB only; camera metadata is not appended to condition'}
