"""Independent analytic calibration shapes; run with bundled Blender 3.3."""
import sys, math, json, hashlib
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from src.research_io import write_json,sha256
OUT=ROOT/'data/lightrig_shader_check_v2';OUT.mkdir(parents=True,exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.samples=32;s.cycles.device='CPU'
s.render.threads_mode='FIXED';s.render.threads=4
s.render.resolution_x=s.render.resolution_y=512;s.render.resolution_percentage=100
s.render.film_transparent=True;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGBA'
s.view_settings.view_transform='Standard';s.view_settings.look='Medium High Contrast'
s.world=bpy.data.worlds.new('World');s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs[0].default_value=(.15,.15,.15,1)
cam=bpy.data.objects.new('Camera',bpy.data.cameras.new('Camera'));s.collection.objects.link(cam);s.camera=cam
cam.location=(3,-4,3);cam.rotation_euler=(-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.type='ORTHO';cam.data.ortho_scale=3.6
for pos,energy,size in [((3,-4,5),500,4),((-4,-1,2),250,3),((0,4,4),350,3)]:
 l=bpy.data.objects.new('Area',bpy.data.lights.new('Area','AREA'));s.collection.objects.link(l);l.location=pos;l.data.energy=energy;l.data.size=size;l.rotation_euler=(-l.location).to_track_quat('-Z','Y').to_euler()
bpy.context.view_layer.update();pose=np.array(cam.matrix_world);rot=pose[:3,:3]
y,x=np.mgrid[:512,:512];local=np.stack([((x+.5)/512-.5)*3.6,(.5-(y+.5)/512)*3.6,np.zeros_like(x)],axis=-1)
origin=local@rot.T+pose[:3,3];direction=-rot[:,2];records=[]
for name in ['sphere','cube']:
 if name=='sphere':
  bpy.ops.mesh.primitive_uv_sphere_add(segments=192,ring_count=96,radius=1)
  obj=bpy.context.object
  for p in obj.data.polygons:p.use_smooth=True
  b=(origin*direction).sum(-1);c=(origin*origin).sum(-1)-1;disc=b*b-c;valid=disc>=0;t=-b-np.sqrt(np.maximum(disc,0));hit=origin+t[...,None]*direction;nw=hit/np.maximum(np.linalg.norm(hit,axis=-1,keepdims=True),1e-12)
 else:
  bpy.ops.mesh.primitive_cube_add(size=1.8);obj=bpy.context.object
  t0=(-.9-origin)/direction;t1=(.9-origin)/direction
  enter=np.maximum.reduce(np.minimum(t0,t1),axis=-1);leave=np.minimum.reduce(np.maximum(t0,t1),axis=-1);valid=(enter<leave)&(leave>0)
  hit=origin+enter[...,None]*direction;axis=np.argmax(np.abs(hit),axis=-1);nw=np.zeros_like(hit)
  for k in range(3):nw[...,k]=np.where(axis==k,np.sign(hit[...,k]),0)
 normal=nw@rot;normal[~valid]=0
 mat=bpy.data.materials.new(name);mat.use_nodes=True;bs=mat.node_tree.nodes['Principled BSDF'];bs.inputs['Base Color'].default_value=(.55,.55,.55,1);bs.inputs['Roughness'].default_value=.6;obj.data.materials.append(mat)
 # Shader-only first render, independent of prior BSDF compilation
 # Independent source cross-check: exact author-linked MaterialAnything VIEW shader.
 nodes=mat.node_tree.nodes;nodes.clear();g=nodes.new('ShaderNodeNewGeometry');v=nodes.new('ShaderNodeVectorTransform');v.vector_type='NORMAL';v.convert_from='WORLD';v.convert_to='CAMERA'
 sep=nodes.new('ShaderNodeSeparateXYZ');comb=nodes.new('ShaderNodeCombineXYZ');mul=nodes.new('ShaderNodeMath');mul.operation='MULTIPLY';mul.inputs[1].default_value=-1
 add=nodes.new('ShaderNodeVectorMath');add.operation='ADD';add.inputs[1].default_value=(1,1,1)
 div=nodes.new('ShaderNodeVectorMath');div.operation='DIVIDE';div.inputs[1].default_value=(2,2,2)
 emit=nodes.new('ShaderNodeEmission');output=nodes.new('ShaderNodeOutputMaterial');links=mat.node_tree.links
 for a,b in [(g.outputs['Normal'],v.inputs[0]),(v.outputs[0],sep.inputs[0]),(sep.outputs[0],comb.inputs[0]),(sep.outputs[1],comb.inputs[1]),(sep.outputs[2],mul.inputs[0]),(mul.outputs[0],comb.inputs[2]),(comb.outputs[0],add.inputs[0]),(add.outputs[0],div.inputs[0]),(div.outputs[0],emit.inputs[0]),(emit.outputs[0],output.inputs['Surface'])]:links.new(a,b)
 mat.node_tree.update_tag();mat.update_tag();bpy.context.view_layer.update();s.view_settings.view_transform='Raw';s.render.filepath=str(OUT/(name+'_source_shader.png'));bpy.ops.render.render(write_still=True);s.view_settings.view_transform='Standard'
 np.savez_compressed(OUT/(name+'.npz'),normal_gl=normal.astype(np.float32),valid=valid,pose=pose)
 bpy.data.objects.remove(obj,do_unlink=True)
