import blenderproc as bproc
"""Actual pinned BlenderProc defaults + camera_pose extracted from pinned Arb source."""
import sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import ast,json,math,hashlib
import numpy as np
import bpy
from mathutils import Matrix,Vector
from bpy_extras.object_utils import world_to_camera_view
from src.research_io import write_json,sha256,now

if bproc.__version__!='2.5.0' or bpy.app.version[:2]!=(3,3):raise RuntimeError('Pinned BlenderProc 2.5 + Blender 3.3 required')
source=ROOT/'outputs/P0_data_intake_local_v1/render_normal_orm.py'
tree=ast.parse(source.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='camera_pose']
if len(functions)!=1:raise RuntimeError('Pinned camera_pose function missing')
namespace={'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
bproc.init();bproc.camera.set_resolution(512,512)
pose=namespace['camera_pose'](0.,30.,2.)
bproc.camera.add_camera_pose(pose)
k=bproc.camera.get_intrinsics_as_K_matrix()
f=512/(2*math.tan(.691111/2))
candidate=np.array([[f,0,255.5],[0,f,255.5],[0,0,1.]])
scene=bpy.context.scene;camera=scene.camera
bpy.ops.mesh.primitive_cube_add(size=.5);cube=bpy.context.object;bpy.context.view_layer.update()
corners=np.array([list(cube.matrix_world@Vector(v)) for v in cube.bound_box])
gl=(np.linalg.inv(pose)@np.column_stack([corners,np.ones(8)]).T).T[:,:3]
cv=gl*np.array([1,-1,-1]);projected=(k@cv.T).T;projected=projected[:,:2]/projected[:,2:]
reference=[]
for p in corners:
    xy=world_to_camera_view(scene,camera,Vector(p));reference.append([512*xy.x-.5,512*(1-xy.y)-.5])
projection_error=float(np.abs(projected-np.array(reference)).max())
out=ROOT/'outputs/P1_arb_camera_blenderproc_v1';out.mkdir(parents=True,exist_ok=True)
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=8
scene.render.threads_mode='FIXED';scene.render.threads=2;scene.render.film_transparent=True
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA'
scene.render.filepath=str(out/'cube_private.png')
bpy.ops.render.render(write_still=True)
# Reading the saved PNG avoids a released Render Result buffer in headless Blender.
rendered=bpy.data.images.load(scene.render.filepath,check_existing=False)
rgba=np.array(rendered.pixels[:]).reshape(512,512,4)[::-1]
ys,xs=np.where(rgba[...,3]>.99)
if not len(xs):raise RuntimeError('Empty rendered cube')
bbox=np.array([[xs.min(),ys.min()],[xs.max(),ys.max()]])
bbox_error=float(np.abs(bbox-np.stack([projected.min(0),projected.max(0)])).max())
matrix_error=float(np.abs(k-candidate).max());passed=matrix_error<1e-3 and projection_error<1e-3 and bbox_error<=2
result={'status':'completed' if passed else 'failed','created_at':now(),'blenderproc_version':bproc.__version__,
    'blender_version':bpy.app.version_string,'numpy_runtime_version':np.__version__,
    'candidate_K':candidate.tolist(),'actual_default_K':k.tolist(),
    'default_K_max_error':matrix_error,'projection_max_error_pixels':projection_error,'render_bbox_max_error_pixels':bbox_error,
    'camera_closure_passed':passed,'pinned_camera_function_sha256':hashlib.sha256(ast.unparse(functions[0]).encode()).hexdigest(),
    'observed_bbox_pixels':bbox.tolist(),'expected_bbox_pixels':np.stack([projected.min(0),projected.max(0)]).tolist(),
    'cube_side_length':.5,
    'scope':'same pinned camera_pose and actual BlenderProc initialization/resolution; K not inserted into frozen no-ray training manifests'}
write_json(out/'summary.json',result)
write_json(out/'source_provenance_private.json',{'pinned_render_source_sha256':sha256(source)})
print(json.dumps(result))
if not passed:raise RuntimeError('Pinned camera closure failed')
