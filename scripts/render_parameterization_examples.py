"""Private qualitative previews of fixed pilot checkpoints; no error-based selection."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib,json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.models.model_unet import FaceNormalUNet_B
from src.noise_protocol import image_noise_seed
from src.parameterization import one_step_endpoint
from src.research_io import ROOT,output_path,write_json,now,sha256


def main():
    out=output_path(ROOT/'figures/qualitative_pilot_v1');out.mkdir(parents=True,exist_ok=True)
    roots={'Hypersim':ROOT/'data/hypersim_trainval_v1',
           'Arb':ROOT/'data/arb_mixed_pilot_v1',
           'NYUv2':ROOT/'data/nyu_validation_packaged_v1'}
    chosen=[]
    for domain,folder in roots.items():
        records=json.loads((folder/'private_manifest.json').read_text())['records']
        records=[r for r in records if r['split']=='val']
        records.sort(key=lambda r:hashlib.sha256((domain+'/'+Path(r['path']).name).encode()).digest())
        groups=set()
        for record in records:
            if record['scene'] in groups:continue
            groups.add(record['scene'])
            with np.load(record['path']) as data:
                rgb=data['rgb'].copy();normal=data['normal'].copy();valid=data['valid'].astype(bool)
            chosen.append({'domain':domain,'row':len(groups),'record':record,'rgb':rgb,'normal':normal,'valid':valid})
            if len(groups)==2:break
        if len(groups)!=2:raise RuntimeError('Need two distinct validation identities per domain')
    write_json(out/'selection_private.json',{'created_at':now(),'selection':'first two distinct held-out identities under fixed SHA256 order; no model/error selection',
        'base_inference_seed':42,'training_seed':1247,'optimizer_updates':100,
        'records':[{'domain':r['domain'],**r['record']} for r in chosen]})
    torch.set_num_threads(4);device=torch.device('cuda');model=FaceNormalUNet_B().to(device).eval()
    rows=[]
    for dropout in [0.,.5]:
        predictions={};metrics={}
        for method in ['R','V','Xv','Xn']:
            run=ROOT/f'outputs/P2_parameterization_v1/pilot_{method}_p{dropout:g}_s1247'
            summary=json.loads((run/'summary.json').read_text())
            if summary.get('status')!='completed' or summary['optimizer_updates']!=100:raise RuntimeError('Pilot checkpoint not complete')
            checkpoint=run/'final_local.pth';state=torch.load(checkpoint,map_location='cpu')
            model.load_state_dict(state['model']);panels=[];scores=[]
            with torch.inference_mode():
                for sample in chosen:
                    condition=torch.from_numpy(sample['rgb'].transpose(2,0,1).copy()).float()[None].to(device)/127.5-1
                    key=sample['record'].get('image_key',sample['domain'].lower()+'/'+Path(sample['record']['path']).name)
                    generator=torch.Generator(device=device).manual_seed(image_noise_seed(42,key))
                    noise=torch.randn((1,3,256,256),generator=generator,device=device) if method!='R' else torch.zeros_like(condition)
                    with torch.autocast('cuda',dtype=torch.bfloat16):raw=model(noise,torch.zeros(1,device=device),condition)
                    endpoint=one_step_endpoint(method,raw.float(),noise)
                    prediction=F.normalize(endpoint,dim=1)[0].permute(1,2,0).cpu().numpy()
                    valid=sample['valid'];target=sample['normal']
                    score=float(np.degrees(np.arccos(np.clip((prediction*target).sum(-1),-1,1)))[valid].mean())
                    panels.append(np.where(valid[...,None],np.clip((prediction+1)/2,0,1),0))
                    scores.append(score)
            predictions[method]=panels;metrics[method]=scores
            rows.append({'condition_dropout':dropout,'method':method,'checkpoint_sha256':sha256(checkpoint),'single_draw_mae_deg':scores})
        fig,axes=plt.subplots(6,6,figsize=(13,13.8),dpi=120)
        for i,sample in enumerate(chosen):
            panels=[sample['rgb'],np.where(sample['valid'][...,None],np.clip((sample['normal']+1)/2,0,1),0)]
            panels += [predictions[m][i] for m in ['R','V','Xv','Xn']]
            for j,panel in enumerate(panels):
                ax=axes[i,j];ax.imshow(panel);ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
                label=['RGB','Label','R','V','Xv','Xn'][j]
                if j>=2:label+=f"  {metrics[['R','V','Xv','Xn'][j-2]][i]:.1f} deg"
                ax.set_title(label,fontsize=9,pad=4)
                if j==0:ax.set_ylabel(f"{sample['domain']}\nexample {sample['row']}",fontsize=10)
        fig.suptitle(f'PILOT ONLY: 100 updates | RGB dropout p={dropout:g} | training seed 1247\n'
            'One step, conditional inference, fixed image-specific seed 42. Black = invalid label region.',fontsize=11,y=.995)
        fig.tight_layout(rect=[0,0,1,.96],pad=.6)
        filename=f'pilot_p{dropout:g}_comparison.png'
        fig.savefig(out/filename);plt.close(fig)
        print(str(out/filename),flush=True)
    write_json(out/'preview_metrics_private.json',{'rows':rows,'scope':'six fixed validation examples, single inference draw; not aggregate ranking'})
    (out/'README.md').write_text('# Pilot 效果预览\n\n'
        '两张总览分别对应训练 RGB 条件丢弃 p=0 和 p=0.5；测试时均关闭丢弃和 CFG。\n'
        '每行依次为 RGB、标签、R、V、Xv、Xn；涵盖三个验证域各两个不同身份/资产。\n'
        '样本按固定哈希排序选择，不按误差挑图；噪声逐图独立，固定 base seed 42，各随机模型配对。\n'
        '黑色区域为标签无效区。只训练了 100 次更新，预览用于检查，不代表最终效果或方法排名。\n'
        '逐图信息、图片及权重仅留本地，不上传 GitHub。\n',encoding='utf-8')

if __name__=='__main__':main()
