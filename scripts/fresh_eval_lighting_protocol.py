"""The historical, fixed seven-way Arb light intervention; no model selection."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def apply_lighting(asset,view):
    if os.environ.get('FNFLOW_EVAL_LIGHT','original')=='original':return
    import bpy
    lights=json.loads((ROOT/'data/gso_lighting_inputs_fresh_20261027/example_lighting.json').read_text())
    index=int(hashlib.sha256((asset['name']+str(view)).encode()).hexdigest()[:8],16)%7
    selected=lights[index]
    for obj in list(bpy.data.objects):
        if obj.type=='LIGHT':bpy.data.objects.remove(obj,do_unlink=True)
    world=bpy.context.scene.world;world.use_nodes=True;nodes=world.node_tree.nodes
    for node in list(nodes):nodes.remove(node)
    bg=nodes.new('ShaderNodeBackground');output=nodes.new('ShaderNodeOutputWorld')
    world.node_tree.links.new(bg.outputs['Background'],output.inputs['Surface'])
    if selected['type']=='hdri':
        texture=nodes.new('ShaderNodeTexEnvironment')
        texture.image=bpy.data.images.load(str(ROOT/'data/gso_lighting_inputs_fresh_20261027'/selected['path']),check_existing=True)
        world.node_tree.links.new(texture.outputs['Color'],bg.inputs['Color']);bg.inputs['Strength'].default_value=1.
    else:
        bg.inputs['Color'].default_value=(0,0,0,1);bg.inputs['Strength'].default_value=0.
        for i,(position,power) in enumerate(zip(selected['locations'],selected['energys'])):
            data=bpy.data.lights.new('ArbPoint'+str(i),'POINT');data.energy=power;data.shadow_soft_size=.25
            obj=bpy.data.objects.new(data.name,data);bpy.context.scene.collection.objects.link(obj);obj.location=position
    bpy.context.scene['arb_light_index']=index
