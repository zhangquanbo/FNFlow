"""Private visual audit of preselected contributors; no automatic asset exclusion."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import json,numpy as np,cv2
from PIL import Image,ImageDraw
from src.boundary_audit import angular_band
from src.research_io import ROOT,write_json,now


def main():
    out=ROOT/'outputs/P1_arb_outlier_visual_review3'
    rows=json.loads((ROOT/'outputs/P1_normal_length_texture_v1/details_private.json').read_text())['records']
    assets={r['asset']:r for r in json.loads((ROOT/'outputs/P1_arb_acquisition_v1/pilot_manifest_private.json').read_text())['records']}
    totals={}
    for row in rows:
        a=totals.setdefault(row['asset'],{'group':row['group'],'outside':0,'views':[]})
        count=row['counts']['0.99']['relaxed_direction_plus_silhouette']['unit_rejected_outside_band']
        a['outside']+=count;a['views'].append((count,row['view']))
    chosen=[]
    for group,n in [('present',3),('not_declared_in_gltf',2)]:
        chosen.extend(sorted((k for k,v in totals.items() if v['group']==group),key=lambda k:-totals[k]['outside'])[:n])
    canvas=Image.new('RGB',(1024,len(chosen)*290),'white');draw=ImageDraw.Draw(canvas);details=[]
    for i,uid in enumerate(chosen):
        a=totals[uid];view=max(a['views'])[1];folder=ROOT/assets[uid]['folder']
        source=folder/f'normal_{view}.exr';native=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
        path=next(folder.glob(f'color_*_{view}.png'));rgba=np.asarray(Image.open(path).convert('RGBA'))
        h,w=native.shape[:2];xs=((np.arange(256)+.5)*w/256).astype(int);ys=((np.arange(256)+.5)*h/256).astype(int)
        raw=native[ys[:,None],xs[None,:]]*2-1;rgba=rgba[ys[:,None],xs[None,:]]
        length=np.linalg.norm(raw,axis=-1);finite=np.isfinite(raw).all(-1);fg=rgba[...,3]>252.45
        direction=np.where(finite[...,None],raw/np.maximum(length[...,None],1e-8),0)
        band=angular_band(direction,fg&finite&(length>1e-8),2,fg)
        rejected=fg&finite&(np.abs(length-1)>.05);outside=rejected&~band
        rgb=np.rint(rgba[...,:3]*(rgba[...,3:4]/255)+255-rgba[...,3:4]).clip(0,255).astype('uint8')
        normals=np.rint((direction[...,::-1]+1)*127.5).clip(0,255).astype('uint8');normals[~fg]=255
        heat=np.full((256,256,3),255,np.uint8);heat[fg]=[160,160,160]
        heat[fg&(length<.95)]=[0,80,255];heat[fg&(length>1.05)]=[255,0,100]
        overlay=rgb.copy();overlay[rejected&band]=[255,180,0];overlay[outside]=[255,0,0]
        for j,pixels in enumerate([rgb,normals,heat,overlay]):canvas.paste(Image.fromarray(pixels),(j*256,i*290+34))
        draw.text((5,i*290+3),f'Candidate {i+1}, {a["group"]}, asset outside count {a["outside"]}; worst view {view}',fill='black')
        draw.text((5,i*290+18),'RGB | unit direction | length: blue<.95, pink>1.05 | excluded: orange=edge, red=outside',fill='black')
        values=length[fg&finite]
        details.append({'asset':uid,'group':a['group'],'selected_view':view,'outside_all_views':a['outside'],
            'foreground_pixels':len(values),'length_quantiles':np.quantile(values,[0,.01,.1,.5,.9,.99,1]).tolist(),
            'fraction_length_above1_05':float((values>1.05).mean()),'fraction_length_below0_95':float((values<.95).mean()),
            'disposition':'unresolved_visual_review_required'})
    out.mkdir(parents=True,exist_ok=True);canvas.save(out/'contact_private.png')
    write_json(out/'details_private.json',{'records':details,'created_at':now()})
    print(json.dumps({'status':'ready_for_visual_review','assets':len(chosen),'contact':str(out/'contact_private.png')}))


if __name__=='__main__':main()
