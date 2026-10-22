import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import argparse
from src.general_dataset import PreparedNormalDataset
from src.research_io import ROOT,write_json,now

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepared-root',default='data/hypersim_protocol_v1')
    parser.add_argument('--run-id',default='P0_prepared_hypersim_v1')
    parser.add_argument('--splits',nargs='+',default=['train','val'])
    args=parser.parse_args()
    summary={'created_at':now(),'status':'completed','partitions':{}}
    for split in args.splits:
        data=PreparedNormalDataset(ROOT/args.prepared_root,split)
        counts=0;norm_error=0.;pinhole=0
        for sample in data:
            normal=sample['normal'].numpy().transpose(1,2,0);valid=sample['valid'].numpy()
            counts+=int(valid.sum());pinhole+=int(sample['K_valid'])
            norm_error=max(norm_error,float(np.abs(np.linalg.norm(normal[valid],axis=-1)-1).max()))
            assert sample['condition'].shape==(3,256,256)
        summary['partitions'][split]={'images':len(data),'scenes':len({r['scene'] for r in data.records}),
                                      'valid_pixels':counts,'max_unit_length_error':norm_error,'pinhole_K_images':pinhole}
    write_json(ROOT/'outputs'/args.run_id/'summary.json',summary)
    print(__import__('json').dumps(summary))
if __name__=='__main__':main()
