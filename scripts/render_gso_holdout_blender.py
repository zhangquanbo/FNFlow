"""Blender 3.3 renderer: identical-hit face/smooth geometry normals, RGB/depth/K."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,math,argparse,time,hashlib
import numpy as np
import bpy
sys.path.append(str(Path(bpy.app.binary_path).parent/'custom-python-packages/lib/python3.10/site-packages'))
from mathutils import Matrix,Vector
from mathutils.bvhtree import BVHTree
from PIL import Image
from src.research_io import write_json,sha256,now
from src.mesh_normal_gt import geometry_vertex_normals,interpolated_geometry_normals,gt_angular_gap


def camera_pose(azimuth,elevation=30.,radius=3.):
    a,e=np.deg2rad([azimuth,elevation]);position=radius*np.array([np.cos(a)*np.cos(e),-np.sin(a)*np.cos(e),np.sin(e)])
    back=position/np.linalg.norm(position);right=np.cross([0,0,1.],back);right/=np.linalg.norm(right)
    pose=np.eye(4);pose[:3,:3]=np.stack([right,np.cross(back,right),back],axis=1);pose[:3,3]=position
    return pose


def render_asset(asset,target,device):
    folder=target/asset['split']/asset['name'];folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'receipt_private.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['mesh_sha256']!=asset['mesh_sha256'] or saved.get('renderer_sha256')!=sha256(__file__) or saved.get('gt_code_sha256')!=sha256(ROOT/'src/mesh_normal_gt.py'):raise RuntimeError('GSO source/renderer/GT definition code changed')
        for record in saved['records']:
            if not Path(record['path']).exists() or sha256(record['path'])!=record['npz_sha256']:raise RuntimeError('GSO cached view changed')
        return saved['records']
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.obj(filepath=asset['mesh'],axis_forward='Y',axis_up='Z')
    meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
    if not meshes:raise RuntimeError('No imported GSO mesh')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in meshes:obj.select_set(True)
    bpy.context.view_layer.objects.active=meshes[0];bpy.ops.object.join();obj=bpy.context.object
    for image in bpy.data.images:
        if image.source=='FILE' and not image.packed_file and not Path(bpy.path.abspath(image.filepath)).is_file():
            matches=list(Path(asset['mesh']).parents[1].rglob(Path(image.filepath).name))
            if len(matches)!=1:raise RuntimeError('Missing or ambiguous GSO source texture')
            image.filepath=str(matches[0]);image.reload()
    bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
    vertices=np.array([tuple(v.co) for v in obj.data.vertices]);low=vertices.min(0);high=vertices.max(0)
    scale=1/float((high-low).max());center=(low+high)/2
    for vertex in obj.data.vertices:vertex.co=Vector(((np.array(vertex.co)-center)*scale).tolist())
    bpy.context.view_layer.update();depsgraph=bpy.context.evaluated_depsgraph_get()
    obj.data.calc_loop_triangles()
    vertices=np.array([tuple(v.co) for v in obj.data.vertices],np.float64)
    triangles=np.array([tuple(t.vertices) for t in obj.data.loop_triangles],np.int64)
    vertex_normals=geometry_vertex_normals(vertices,triangles)
    tree=BVHTree.FromPolygons(vertices.tolist(),triangles.tolist(),all_triangles=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.samples=32;scene.cycles.use_denoising=False
    scene.render.resolution_x=scene.render.resolution_y=256;scene.render.resolution_percentage=100
    scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
    scene.render.threads_mode='FIXED';scene.render.threads=4;scene.render.film_transparent=True
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
    scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast'
    if device=='GPU':
        preferences=bpy.context.preferences.addons['cycles'].preferences;preferences.compute_device_type='OPTIX'
        preferences.get_devices()
        for d in preferences.devices:d.use=d.type=='OPTIX'
        scene.cycles.device='GPU'
    else:scene.cycles.device='CPU'
    world=bpy.data.worlds.new('World');scene.world=world;world.use_nodes=True
    world.node_tree.nodes['Background'].inputs[0].default_value=(.3,.3,.3,1)
    for i,position in enumerate([(3,-2,4),(-2,1,2)]):
        lamp=bpy.data.lights.new(f'Area{i}','AREA');lamp.energy=180;lamp.size=4
        light=bpy.data.objects.new(f'Area{i}',lamp);scene.collection.objects.link(light)
        light.location=position;light.rotation_euler=(-light.location).to_track_quat('-Z','Y').to_euler()
    cam=bpy.data.cameras.new('Camera');camera=bpy.data.objects.new('Camera',cam);scene.collection.objects.link(camera);scene.camera=camera
    cam.lens_unit='FOV';cam.angle=.691111;cam.clip_start=.01;cam.clip_end=100
    focal=256/(2*math.tan(.691111/2));k=np.array([[focal,0,127.5],[0,focal,127.5],[0,0,1.]])
    u,v=np.meshgrid(np.arange(256),np.arange(256));cv=np.stack([(u-k[0,2])/focal,(v-k[1,2])/focal,np.ones_like(u)],-1)
    gl=cv*np.array([1,-1,-1]);gl/=np.linalg.norm(gl,axis=-1,keepdims=True)
    records=[]
    for view,azimuth in enumerate([0,90,180,270]):
        scene.cycles.seed=100+view;pose=camera_pose(azimuth);camera.matrix_world=Matrix(pose.tolist());bpy.context.view_layer.update()
        path=folder/f'{view:02d}.npz';png=folder/f'{view:02d}_private.png';scene.render.filepath=str(png)
        bpy.ops.render.render(write_still=True)
        rgba=np.asarray(Image.open(png).convert('RGBA')).astype(np.float32)/255
        rgb=np.rint((rgba[...,:3]*rgba[...,3:4]+1-rgba[...,3:4])*255).clip(0,255).astype(np.uint8)
        normal=np.zeros((256,256,3),np.float32);depth=np.zeros((256,256),np.float32);valid=np.zeros((256,256),bool)
        face_ids=np.full((256,256),-1,np.int64);hit_points=np.zeros((256,256,3),np.float64)
        directions=gl@pose[:3,:3].T;origin=Vector(pose[:3,3]);normal_rotation=-np.diag([1,-1,-1])@pose[:3,:3].T
        inv=np.linalg.inv(pose);projection_error=0.
        for y in range(256):
            for x in range(256):
                hit,n,face,distance=tree.ray_cast(origin,Vector(directions[y,x]),100.)
                if hit is None:continue
                transformed=normal_rotation@np.asarray(n);length=np.linalg.norm(transformed)
                if not np.isfinite(length) or length<1e-8:raise RuntimeError('Invalid ray-cast face normal')
                normal[y,x]=transformed/length;valid[y,x]=True
                face_ids[y,x]=face;hit_points[y,x]=hit
                point_gl=inv@np.array([*hit,1]);depth[y,x]=-point_gl[2]
                pixel=k@(point_gl[:3]*np.array([1,-1,-1]))
                projection_error=max(projection_error,float(np.abs(pixel[:2]/pixel[2]-[x,y]).max()))
        if not valid.any() or (depth[valid]<=0).any():raise RuntimeError('Empty/invalid GSO ray-cast view')
        normal_face=normal.copy();normal_smooth=np.zeros_like(normal)
        smooth=interpolated_geometry_normals(vertices,triangles,vertex_normals,face_ids[valid],hit_points[valid])
        smooth=smooth@normal_rotation.T;smooth/=np.linalg.norm(smooth,axis=-1,keepdims=True)
        normal_smooth[valid]=smooth.astype(np.float32);normal=normal_smooth
        gap=gt_angular_gap(normal_face,normal_smooth,valid)
        gap_map=np.zeros((256,256),np.float32);gap_map[valid]=gap.astype(np.float32)
        error=float(np.abs(np.linalg.norm(normal[valid],axis=-1)-1).max())
        np.savez_compressed(path,rgb=rgb,normal=normal,normal_smooth=normal_smooth,normal_face=normal_face,
                            gt_definition_gap_deg=gap_map,valid=valid,depth=depth,K=k.astype(np.float32),K_valid=np.array(True),
                            projection_3x3=k.astype(np.float32),projection_valid=np.array(True),camera_to_world=pose.astype(np.float32))
        records.append({'scene':asset['name'],'family':asset['family'],'split':asset['split'],'path':str(path),
            'image_key':f'gso/{asset["name"]}/{view}','label_source':'barycentric_area_weighted_smooth_geometry_no_material_perturbation',
            'npz_sha256':sha256(path),'valid_pixels':int(valid.sum()),'max_unit_error':error,'scale_from_source':scale,
            'gt_gap_mean_deg':float(gap.mean()),'gt_gap_median_deg':float(np.median(gap)),
            'gt_gap_p90_deg':float(np.quantile(gap,.9)),'gt_gap_sum_deg':float(gap.sum()),
            'positive_ray_dot_fraction':float(((normal*cv).sum(-1)[valid]>0).mean()),'render_device':device,
            'ray_hit_reprojection_max_error_pixels':projection_error})
    write_json(receipt,{'records':records,'mesh_sha256':asset['mesh_sha256'],'renderer_sha256':sha256(__file__),
        'gt_code_sha256':sha256(ROOT/'src/mesh_normal_gt.py')});return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['CPU','GPU'],default='GPU');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--manifest',default='outputs/P1_gso_intake_v1/manifest_private.json')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    assets=json.loads((ROOT/args.manifest).read_text())['records']
    if args.limit:assets=assets[:args.limit]
    registration=ROOT/'provenance/data_quality_review3_preregistration.json'
    rule=ROOT/'configs/precurve_data_quality_review3.json'
    if json.loads(registration.read_text())['rule_sha256']!=sha256(rule):raise RuntimeError('GSO dual-GT registration changed')
    generation=hashlib.sha256((sha256(__file__)+sha256(ROOT/'src/mesh_normal_gt.py')).encode()).hexdigest()[:16]
    target=ROOT/('data/gso_holdout_smoke_dual_v2' if args.limit else 'data/gso_holdout_dual_v2')/generation;target.mkdir(parents=True,exist_ok=True)
    out=ROOT/('outputs/P1_gso_render_smoke_dual_v2' if args.limit else 'outputs/P1_gso_render_dual_v2')
    records=[]
    try:
        for i,asset in enumerate(assets):
            records.extend(render_asset(asset,target,args.device))
            write_json(out/'status.json',{'status':'rendering','completed_assets':i+1,'target_assets':len(assets),'updated_at':now()})
        groups={s:{r['scene'] for r in records if r['split']==s} for s in ['val','test']}
        assert not groups['val']&groups['test']
        family={s:{r['family'] for r in records if r['split']==s} for s in ['val','test']}
        assert not family['val']&family['test']
        write_json(target/'private_manifest.json',{'records':records,'purpose':'candidate object holdout; not used for training or budget selection'})
        result={'status':'completed','created_at':now(),'assets':len(assets),'images':len(records),
            'assets_by_split':{s:len(v) for s,v in groups.items()},'asset_overlap':0,'product_family_overlap':0,
            'normal_source':'main: barycentric area-weighted vertex geometry, no materials; secondary: ray-hit face normal at identical pixel centers',
            'main_gt_registered_before_model_scores':True,'rule_sha256':sha256(rule),'renderer_sha256':sha256(__file__),
            'gt_code_sha256':sha256(ROOT/'src/mesh_normal_gt.py'),'cache_generation':generation,
            'normal_fields':['normal (main alias)','normal_smooth','normal_face'],'same_mask_for_both_gt':True,
            'gt_gap_image_macro_mean_deg':float(np.mean([r['gt_gap_mean_deg'] for r in records])),
            'gt_gap_asset_macro_mean_deg':float(np.mean([np.mean([r['gt_gap_mean_deg'] for r in records if r['scene']==a['name']]) for a in assets])),
            'gt_gap_pixel_pooled_mean_deg':sum(r['gt_gap_sum_deg'] for r in records)/sum(r['valid_pixels'] for r in records),
            'gap_interpretation':'GT-definition discrepancy; neither a proven irreducible noise floor nor an additive correction to model MAE',
            'camera':'controlled 256 square; FOV .691111; K/poses saved per view; object normalized to unit maximum extent',
            'depth':'camera-z in normalized scene units, source scale retained; not an unqualified metric-depth claim',
            'mask':'ray hits, no alpha or near-unit filtering','training_allowed':False,'model_test_scores_computed':False,
            'max_normal_unit_error':max(r['max_unit_error'] for r in records),'render_device':args.device,
            'max_ray_reprojection_error_pixels':max(r['ray_hit_reprojection_max_error_pixels'] for r in records),
            'cross_source_near_duplicate_exhaustive_check':False,'scope':'rendering readiness; cross-source duplicate limitations remain those of the intake report'}
        write_json(out/'generations'/generation/'summary.json',result)
        write_json(out/'summary.json',result);write_json(out/'status.json',result);print(json.dumps(result),flush=True)
    except Exception as e:
        write_json(out/'status.json',{'status':'failed','completed_views':len(records),'error':repr(e)});raise


if __name__=='__main__':
    from src.gpu_job_lock import gpu_job_lock
    with gpu_job_lock():main()
