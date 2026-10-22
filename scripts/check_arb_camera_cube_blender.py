"""Run with Blender 3.3: close the pinned camera defaults on a known cube.

This uses the camera basis construction in the pinned Arb script and the
documented BlenderProc 2.5 DefaultConfig FOV. It never assigns calibration to
downloaded data on the basis of a failed or unexecuted test.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json,math
import numpy as np
import bpy
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
from src.research_io import output_path,write_json,now

if bpy.app.version[:2]!=(3,3):raise RuntimeError('Use Blender 3.3 for this versioned closure')
out=output_path(ROOT/'outputs/P1_arb_camera_cube_v1');out.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
camera=bpy.data.cameras.new('Camera');obj=bpy.data.objects.new('Camera',camera)
scene=bpy.context.scene;scene.collection.objects.link(obj);scene.camera=obj
camera.lens_unit='FOV';camera.angle=.691111;camera.shift_x=camera.shift_y=0
scene.render.resolution_x=scene.render.resolution_y=512;scene.render.resolution_percentage=100
scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
position=np.array([2*math.cos(math.pi/6),0,2*math.sin(math.pi/6)])
forward=position/np.linalg.norm(position);right=np.cross([0,0,1.],forward);right/=np.linalg.norm(right)
up=np.cross(forward,right);up/=np.linalg.norm(up)
pose=np.eye(4);pose[:3,:3]=np.stack([right,up,forward],axis=1);pose[:3,3]=position
obj.matrix_world=Matrix(pose.tolist())
bpy.ops.mesh.primitive_cube_add(size=1)
cube=bpy.context.object;bpy.context.view_layer.update()
focal=512/(2*math.tan(.691111/2));k=np.array([[focal,0,255.5],[0,focal,255.5],[0,0,1]])
corners=np.array([list(cube.matrix_world@Vector(v)) for v in cube.bound_box])
homogeneous=np.concatenate([corners,np.ones((len(corners),1))],axis=1)
gl=(np.linalg.inv(pose)@homogeneous.T).T[:,:3]
cv=gl*np.array([1,-1,-1]);projected=(k@cv.T).T;projected=projected[:,:2]/projected[:,2:]
independent=[]
for corner in corners:
    p=world_to_camera_view(scene,obj,Vector(corner))
    independent.append([512*p.x-.5,512*(1-p.y)-.5])
projection_error=float(np.abs(projected-np.array(independent)).max())
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
scene.render.threads_mode='FIXED';scene.render.threads=2;scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
scene.render.filepath=str(out/'cube_private.png')
scene.world=bpy.data.worlds.new('World');scene.world.use_nodes=True
scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.8,.8,.8,1)
bpy.ops.render.render(write_still=True)
image=bpy.data.images['Render Result'];rgba=np.array(image.pixels[:]).reshape(512,512,4)[::-1]
ys,xs=np.where(rgba[...,3]>.99)
if not len(xs):raise RuntimeError('Empty cube render')
observed=np.array([[xs.min(),ys.min()],[xs.max(),ys.max()]])
expected=np.stack([projected.min(0),projected.max(0)])
bbox_error=float(np.abs(observed-expected).max())
passed=projection_error<1e-3 and bbox_error<=2
summary={'status':'completed' if passed else 'failed','created_at':now(),'blender_version':bpy.app.version_string,
    'default_FOV_radians':.691111,'candidate_K_512':k.tolist(),'projection_max_error_pixels':projection_error,
    'rendered_cube_bbox_max_error_pixels':bbox_error,'camera_closure_passed':passed,
    'scope':'controlled default camera closure; downloaded asset K remains missing until source version/default equivalence is accepted',
    'sources':['https://github.com/DLR-RM/BlenderProc/blob/v2.5.0/blenderproc/python/utility/DefaultConfig.py',
               'https://huggingface.co/datasets/lizb6626/Arb-Objaverse/blob/329f4fc199498264390df36104a3326e8a30c325/rendering/render_normal_orm.py']}
write_json(out/'summary.json',summary)
print(json.dumps(summary))
if not passed:raise RuntimeError('Cube calibration closure failed')
