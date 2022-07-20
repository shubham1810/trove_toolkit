import blenderproc as bproc
import bpy
import numpy as np
import mathutils
import os
import json

object_path = "PATH to 3d_objects"
texture_path = "PATH to PBR_textures"

def load_object(obj_name):   
    # Just get the name of the object
    name = obj_name[:-4]
    category = obj_name.split("_")[0]
    # Get list of old objects
    old_objects = set(bpy.context.scene.objects)

    # Import the object
    bpy.ops.import_scene.fbx(filepath=object_path + obj_name)

    # Get the new object
    obj = (set(bpy.context.scene.objects) - old_objects).pop()

    # Rename
    assigned_name = "added_" + name
    obj.name = assigned_name
    
    return assigned_name

object_items = sorted(os.listdir(object_path))

all_items = {}

for ob_name in object_items[:]:
    if not ("fbx" in ob_name):
        continue
    
    # Load the fbx object
    category = ob_name.split("_")[0]
    name  = load_object(ob_name)
    obj = bpy.data.objects[name]
    
    data = {
        "category": category,
        "dimensions": (obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]),
        "rotation": (obj.rotation_euler[0], obj.rotation_euler[1], obj.rotation_euler[2]),
        "location": (obj.location[0], obj.location[1], obj.location[2]),
        "name": ob_name,
    }
    
    listed = all_items.get(category, [])
    listed.append(data)
    all_items[category] = listed
    
    # Delete the object
    obj.select_set(True)
    bpy.ops.object.delete()

object_config = json.dumps(all_items, indent=4, sort_keys=True)

f = open(object_path + "object_config.json", 'w')
f.write(object_config)
f.close()


# Get list of all textures
textures_available = os.listdir(texture_path)
all_material = {}
texture_category_map = {
    'asphalt': 'road',
    'facade': 'building',
    'grass': 'base',
    'gravel': 'road',
    'ground': 'base',
    'paving': 'sidewalk'
}

mat_types = ['emission', 'normal', 'opacity', 'displacement', 'rough', 'metal', 'color', 'ambientocc']

# Iterate through each texture
for ix in textures_available:
    if "json" in ix:
        continue
    folder_path = os.path.join(texture_path, ix)
    mat_name = ix.split('_')[0]
    tex_items = os.listdir(folder_path)
    
    # Create material paths template
    mat_data = {
        'name': mat_name,
        'base_path': os.path.abspath(texture_path),
        'texture_dir': ix,
    }
    
    for tx in tex_items:
        f_type, f_ext = tx.split('_')[-1].split('.')
        if (not ('preview' in f_type.lower())) and (f_ext.lower() in ['png', 'jpg', 'jpeg']):
            for mtx in mat_types:
                if mtx in f_type.lower():
                    mat_data[mtx] = tx
            if f_type == 'AO':
                mat_data['ambientocc'] = tx
    
    # Check mapping
    for tmap in texture_category_map.keys():
        if tmap in mat_name.lower():
            # Get the list
            current_mats = all_material.get(texture_category_map[tmap], [])
            current_mats.append(mat_data)
            all_material[texture_category_map[tmap]] = current_mats

# print(all_material)

material_config = json.dumps(all_material, indent=4, sort_keys=True)

f = open(os.path.join(texture_path, "mat_config.json"), 'w')
f.write(material_config)
f.close()
print("Completed")