from scripts.inspect_gltf_normal_textures import material_summary

def test_nested_normal_texture_slots_and_missing_metadata_are_distinguished():
    result=material_summary({'materials':[{'normalTexture':{'index':0},'extensions':{'KHR_materials_clearcoat':{'clearcoatNormalTexture':{'index':1}}},'alphaMode':'BLEND'}]})
    assert result['explicit_normal_texture_slots']==2
    assert result['normal_texture_group']=='present' and result['alpha_modes']=={'BLEND':1}
    assert material_summary({})['normal_texture_group']=='not_declared_in_gltf'
