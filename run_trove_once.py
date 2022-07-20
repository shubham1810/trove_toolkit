import blenderproc as bproc

import numpy as np
import argparse
import os
import cv2
import bpy
import mathutils
from mathutils import Vector, Matrix
import json

bproc.init()

parser = argparse.ArgumentParser()
parser.add_argument('--scene_config', default="../config_dump/scene_0001/sample_000.json")
parser.add_argument('--render_path', default="/ssd_scratch/cvit/shubham/render")
args = parser.parse_args()

print(args.scene_config)


RENDER_PATH = args.render_path

# Blender globals
C = bpy.context
D = bpy.data

conf_path = args.scene_config

# Get scene name
scene_name = conf_path.split("/")[-2]
parent_folder = os.path.dirname(conf_path)

def load_config(config_path):
    f = open(config_path, 'r')
    data = json.loads(f.read())
    f.close()
    
    return data

scene_config = load_config(conf_path)
print(scene_config.keys())


# CONFIGURATION VARIABLES
osm_path = "../osm_files/"
latlon = scene_config['geo']
road_width = 2.6
object_path = "../3d_objects/"
texture_config_path = "../PBR_textures/mat_config.json"

config_name = "object_config.json"
vegetation_plane_size = scene_config['scene_width']
base_plane_size = 600
vegetation_mask_path = os.path.join(parent_folder, "vegetation.png")
vegetation_particle_size = 1.1
vegetation_particle_count = 5000 + int(np.random.random()*5000)
num_bushes = 2
num_trees = 3
vegetation_template_underground_offset = -100
num_traffic_signs = 10
traffic_sign_particle_size = 1.0
traffic_sign_particle_count = 80 + int(np.random.random()*40)

# Render image size related variables
num_car_cams = 3
nusc_img_size = [1600, 900]
render_img_size = [1600, 900]
data_scale_x, data_scale_y = render_img_size[0]/nusc_img_size[0], render_img_size[1]/nusc_img_size[1]

# Load configs and corresponding data
obj_conf = load_config(object_path + "object_config.json")
texture_config = load_config(texture_config_path)
fix_barrier = np.random.choice(obj_conf["barrier"])


# Check OSM file availability
if not os.path.exists(os.path.join(osm_path, "{}.osm".format(scene_name))):
    print("OSM file does not exist. Quitting.....")
    exit(0)


# Before we start, major bug fix in location for all scenes
# TODO: Remember global min should be center for offsetting not actual x_min in the config code
for ix in range(len(scene_config['boxes'])):
    scene_config['boxes'][ix]["location"][0] -= scene_config['scene_width']/2
    scene_config['boxes'][ix]["location"][1] -= scene_config['scene_width']/2

# Apply the fix to camera locations as well
for kx in scene_config["cam"].keys():
    scene_config["cam"][kx]["extrinsic"][3] -= scene_config['scene_width']/2
    scene_config["cam"][kx]["extrinsic"][7] -= scene_config['scene_width']/2


# First add the buildings
bpy.context.scene.blosm.dataType = 'osm'
bpy.context.scene.blosm.mode = '3Dsimple'
bpy.context.scene.blosm.singleObject = False
bpy.context.scene.blosm.ignoreGeoreferencing = True

bpy.context.scene.blosm.buildings = True
bpy.context.scene.blosm.water = False
bpy.context.scene.blosm.forests = False
bpy.context.scene.blosm.vegetation = False
bpy.context.scene.blosm.highways = False
bpy.context.scene.blosm.railways = False

bpy.context.scene.blosm.minLat = latlon[0][0]
bpy.context.scene.blosm.minLon = latlon[0][1]
bpy.context.scene.blosm.maxLat = latlon[1][0]
bpy.context.scene.blosm.maxLon = latlon[1][1]

bpy.context.scene.blosm.osmSource = 'file'
bpy.context.scene.blosm.osmFilepath = os.path.join(osm_path, "{}.osm".format(scene_name))

bpy.ops.blosm.import_data()

# Next import roads
bpy.context.scene.blosm.dataType = 'osm'
bpy.context.scene.blosm.mode = '3Dsimple'
bpy.context.scene.blosm.singleObject = True
bpy.context.scene.blosm.ignoreGeoreferencing = True

bpy.context.scene.blosm.buildings = False
bpy.context.scene.blosm.water = False
bpy.context.scene.blosm.forests = False
bpy.context.scene.blosm.vegetation = False
bpy.context.scene.blosm.highways = True
bpy.context.scene.blosm.railways = False

bpy.context.scene.blosm.osmSource = 'file'
bpy.context.scene.blosm.osmFilepath = os.path.join(osm_path, "{}.osm".format(scene_name))

bpy.ops.blosm.import_data()

# Fix the road mesh
objects = bpy.data.objects

# Iterate through the objects
for obj in objects:
    if ("road" in obj.name or "path" in obj.name) and obj.type == "CURVE":
        # Ignore way profiles
        if "profile" in obj.name:
            continue
        # Get all splines for this object
        for spline in obj.data.splines:
            # Get all points in the spline
            for point in spline.points:
                # Update the radius for the width of the road
                point.radius = road_width
        
        # Convert the curve to mesh
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.context.object.data.bevel_mode = 'OBJECT'
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target='MESH') # Main conversion stuff
        obj.select_set(False)
        bpy.context.view_layer.objects.active = None


bpy.ops.object.select_all(action='DESELECT')

# Join all the roads objects into one
for obj in bpy.data.objects:
    active_set = False
    if "road" in obj.name and obj.type == "MESH":
        obj.select_set(True)
        if not active_set:
            bpy.context.view_layer.objects.active = obj
            active_set = True
    
bpy.ops.object.join()
bpy.context.view_layer.objects.active.name = "roads"

# Set road height to almost ground level
bpy.data.objects["roads"].location[2] = 0.0005

# Similarly create a joint mesh for the pathways (sidewalk etc.)
bpy.context.view_layer.objects.active = None
bpy.ops.object.select_all(action='DESELECT')

# Join all the sidewalk objects into one
for obj in bpy.data.objects:
    active_set = False
    if "path" in obj.name and obj.type == "MESH":
        obj.select_set(True)
        if not active_set:
            bpy.context.view_layer.objects.active = obj
            active_set = True
    
bpy.ops.object.join()
bpy.context.view_layer.objects.active.name = "sidewalk"

# Set Sidewalk height to almost ground level but above roads
bpy.data.objects["sidewalk"].location[2] = 0.0015


def origin_to_center(ob, matrix=Matrix()):
    me = ob.data
    mw = ob.matrix_world
    local_verts = [matrix @ Vector(v[:]) for v in ob.bound_box]
    o = sum(local_verts, Vector()) / 8
    o = matrix.inverted() @ o
    me.transform(Matrix.Translation(-o))

    mw.translation = mw @ o


# Orient buildings and their boxes
def reorient_floor(obj_name, mode='3d', name=0):
    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')
    
    # Get object from name
    obj = bpy.data.objects[obj_name]
    obj.select_set(True)
    # mode can be either 2d or 3d
    # bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    origin_to_center(obj)
    
    coords = np.array([v.co for v in obj.data.vertices])
    # default param for bounding box
    default_box_height = np.random.choice([20, 30, 40, 50])
    # If 3D object, only take the bottom set of points
    if mode.lower() == '3d':
        # Also compute some height info
        min_z, max_z = coords[:, -1].min(), coords[:, -1].max()
        default_box_height = np.abs(max_z - min_z)
        coords = coords[:len(coords)//2]        

    # Arrange in circular coordinates to complete polygon
    circular_coords = np.concatenate([coords, coords[:1]], axis=0)

    # Get all angles of the edges
    orientations = []
    object_lengths = []
    for ix in range(coords.shape[0]):
        # For each pair, compute the angle and length
        p1, p2 = circular_coords[ix, :2], circular_coords[ix+1, :2]
        diff = p2-p1
        dist = np.sqrt(np.sum(diff**2))
        angle = np.rad2deg(np.arctan2(diff[1], diff[0]))
        if angle < 0:
            angle = 180 + angle
        # Repeat the angle multiple times based on distance
        # to have better weight in distribution
        orientations += [angle]*int(dist)
        object_lengths.append((angle, dist))

    # Compute the histogram of angles
    angle_dist = np.histogram(orientations, bins=10)
    num_dominant_angles = np.sum(angle_dist[0] > 0)
    if num_dominant_angles <=4:
        angle_orders = np.argsort(angle_dist[0])[::-1][:num_dominant_angles]
        major_angles = []
        for ix in angle_orders:
            maj_ang = 0.5*(angle_dist[1][ix] + angle_dist[1][ix+1])
            major_angles.append((maj_ang, angle_dist[0][ix]))
        
        # placeholder for min error and angles
        min_error, best_pair = 100, []
        # Get all pairs of angles and differences
        for a1 in range(len(major_angles)):
            for a2 in range(a1+1, len(major_angles)):
                # get the corresponding angles
                ax1, ax2 = major_angles[a1], major_angles[a2]
                # Get the angle and weights
                val1, w1 = ax1
                val2, w2 = ax2
                
                # Compute the difference and error
                diff = np.abs(val1 - val2)
                # error = np.abs((90 - diff))/(w1 + w2)
                error = np.abs((90 - diff))/(w1 + w2)
                if error < min_error:
                    min_error = error
                    best_pair = [ax1, ax2]
        
        # Compute the side lengths of the rectangle
        ang1, ang2 = best_pair[0][0], best_pair[1][0]        
        sides = [0, 0, default_box_height]
        for obx in object_lengths:
            ob_angle, ob_len = obx
            # Compute the differences of angles
            diffs = np.array([np.abs(ob_angle - ang1), np.abs(ob_angle - ang2)])
            index = diffs.argmin()
            # check which side the current length belongs to
            if diffs[index] < 10:
                sides[index] += ob_len*0.5
            else:
                # If neither, take a projection for both sides
                sides[0] += np.cos(np.deg2rad(diffs[0]))*ob_len*0.5
                sides[1] += np.sin(np.deg2rad(diffs[1]))*ob_len*0.5
        # Update the dimensions of the object
        new_dims = mathutils.Vector(sides)
        # Compute the oritation angle for better alignment
        rot_angle = object_lengths[np.array(object_lengths)[:, -1].argmax()][0]
        
        # Add a cube object for placeholder
        bpy.ops.mesh.primitive_cube_add()
        bound_box = bpy.context.active_object
        bound_box.dimensions = new_dims
        bound_box.location = obj.location
        # Set the object to ground
        bound_box.location.z = default_box_height/2
        bound_box.rotation_euler = obj.rotation_euler
        bound_box.rotation_euler.rotate_axis('Z', np.deg2rad(rot_angle))
        
        # Set object name
        bound_box.name = "placeholder_build_" + str(name).zfill(2)
        
        # delete the original object
        bound_box.select_set(False)
        obj.select_set(True)
        bpy.ops.object.delete()
    else:
        # Set object name
        obj.name = "texture_build_" + str(name).zfill(2)

for bx in bpy.data.objects:
    if "building" in bx.name and bx.type == "EMPTY":
        # Iterate over the parent object
        building_counter = 0
        for child in bx.children:
            reorient_floor(child.name, mode='3d', name=building_counter)
            building_counter += 1
        bx.name = "original_build_collection"


# Add building objects
f = open(object_path + config_name, 'r')
df = f.read()
f.close()

config = json.loads(df)['building']

def move_to_floor(object_name):
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.context.scene.objects[object_name]
    # Also move the box to the ground properly
    minz =  min((obj.matrix_world @ v.co)[2] for v in obj.data.vertices)
    obj.location[2] -= minz

def load_object(obj_name):    
    # Get list of old objects
    old_objects = set(bpy.context.scene.objects)

    # Import the object
    bpy.ops.import_scene.fbx(filepath=object_path + obj_name)

    # Get the new object
    obj = (set(bpy.context.scene.objects) - old_objects).pop()

    # Rename
    assigned_name = "added_" + obj_name
    obj.name = assigned_name
    
    return assigned_name

# placeholder_build_01
def match_building(box_name):
    # Get the box properties
    bbox = bpy.data.objects[box_name]
    box_ratio = bbox.dimensions[0]/bbox.dimensions[1]
    box_vol = bbox.dimensions[0]*bbox.dimensions[1]*bbox.dimensions[2]

    errors = []

    for bx in range(len(config)):
        # Randomize some building associations
        if np.random.random() < 0.25:
            continue
        
        b = config[bx]
        name = b['name']
        dims = b['dimensions']
        vol = dims[0]*dims[1]*dims[2]
        vol_ratio = np.power(box_vol/(vol + 1e-05), 1/3)
        
        aspect_ratio = dims[0]/(dims[1] + 1e-05)
        diff1, diff2 = aspect_ratio - box_ratio, 1/aspect_ratio - box_ratio
        error1 = 0.8*diff1 + 0.2*vol_ratio
        error2 = 0.8*diff2 + 0.2*vol_ratio
        
        errors.append([error1, bx, True])
        errors.append([error2, bx, False])

    errors = np.abs(np.array(errors))
    idx = np.argsort(errors[:, 0])
    selected_idx = int(errors[idx[0]][1])
    same_orientation = errors[idx[0]][2]

    return config[selected_idx], same_orientation


for box in bpy.data.objects:
    if not (("placeholder" in box.name) and ("build" in box.name)):
        continue

    b, same_orient = match_building(box.name)
    # box = bpy.data.objects['placeholder_build_01']
    dims = box.dimensions

    # Import the building object
    loaded_name = load_object(b['name'])
    ob = bpy.data.objects[loaded_name]

    ob.select_set(True)
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_VOLUME', center='MEDIAN')
    origin_to_center(ob)

    imported_scale = np.array(ob.scale)
    neg_scale_mask = imported_scale < 0
    if neg_scale_mask.sum() > 0:
        ob.scale = mathutils.Vector(imported_scale * neg_scale_mask * -1)

    # make location same
    ob.location = box.location
    ob.rotation_euler = box.rotation_euler

    if same_orient:
        scale_factor = np.divide(box.dimensions, ob.dimensions)[:2]
    else:
        scale_factor = np.divide(mathutils.Vector([dims[1], dims[0], dims[2]]), ob.dimensions)[:2]
        ob.rotation_euler.rotate_axis('Z', np.deg2rad(90))

    ob.dimensions = ob.dimensions * scale_factor.min()

    move_to_floor(ob.name)
    
    # delete the bounding box
    ob.select_set(False)
    box.select_set(True)
    bpy.ops.object.delete()


# Add vegetation data

def generate_random_name(category="car"):
    rad = "".join([chr(np.random.randint(0, 26) + 97)  for ix in range(6)])
    name = "{}_{}".format(category, rad)
    return name

def load_category_object(selected_category):
    # Select data for a random object of this category
    random_object_data = np.random.choice(obj_conf[selected_category])
    
    # Get list of old objects
    old_objects = set(bpy.context.scene.objects)

    # Import the object
    bpy.ops.import_scene.fbx(filepath=object_path + random_object_data['name'])

    # Get the new object
    obj = (set(bpy.context.scene.objects) - old_objects).pop()

    # Rename
    assigned_name = generate_random_name(selected_category)
    obj.name = assigned_name
    
    return assigned_name

def load_specific_category_object(selected_category, item_id=0):
    # Select data for a random object of this category
    random_object_data = obj_conf[selected_category][item_id]
    
    # Get list of old objects
    old_objects = set(bpy.context.scene.objects)

    # Import the object
    bpy.ops.import_scene.fbx(filepath=object_path + random_object_data['name'])

    # Get the new object
    obj = (set(bpy.context.scene.objects) - old_objects).pop()

    # Rename
    assigned_name = generate_random_name(selected_category)
    obj.name = assigned_name
    
    return assigned_name

def origin_to_bottom(ob, matrix=Matrix()):
    me = ob.data
    mw = ob.matrix_world
    local_verts = [matrix @ Vector(v[:]) for v in ob.bound_box]
    o = sum(local_verts, Vector()) / 8
    o.z = min(v.z for v in local_verts)
    o = matrix.inverted() @ o
    me.transform(Matrix.Translation(-o))

    mw.translation = mw @ o

# Function to create any PBR material
def create_new_material(sample):
    # Most of this code is based on node wrangler code
    base_path = sample['base_path']
    texture_dir = sample['texture_dir']
    texture_path = os.path.join(base_path, texture_dir)

    # Create a new material from the sample
    mat = bpy.data.materials.new(name=sample["name"])
    mat.use_nodes = True

    # Select the principaled BSDF as selected/active
    mat.node_tree.nodes.active = mat.node_tree.nodes["Principled BSDF"]

    # Get nodes, links, active node and output node
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    active_node = nodes.active
    output_node = [n for n in nodes if n.bl_idname == 'ShaderNodeOutputMaterial']


    node_input_map = {
        'color': 'Base Color',
        'normal': 'Normal',
        'roughness': 'Roughness',
        'emission': 'Emission',
        'opacity': 'Alpha',
        'metal': 'Metallic',
        
    }

    texture_nodes = []
    disp_texture = None
    normal_node = None
    roughness_node = None

    for mx in sample.keys():
        # Ignode if it's just name attributes
        if mx in ['base_path', 'name', 'texture_dir']:
            continue
        
        # First for displacement
        if 'displacement' in mx:
            # Load displacement image
            img_disp = bpy.data.images.load(os.path.join(texture_path, sample['displacement']))
            disp_texture = nodes.new(type='ShaderNodeTexImage')
            disp_texture.image = img_disp
            disp_texture.image.colorspace_settings.is_data = True
            
            disp_node = nodes.new(type='ShaderNodeDisplacement')
            disp_node.location = active_node.location + Vector((0, -560))
            links.new(disp_node.inputs[0], disp_texture.outputs[0])
            
            # link to output node
            if not output_node[0].inputs[2].is_linked:
                links.new(output_node[0].inputs[2], disp_node.outputs[0])
            continue
        
        if mx not in node_input_map.keys():
            continue
        
        # Load the image data and create a node
        texture_node = nodes.new(type='ShaderNodeTexImage')
        img = bpy.data.images.load(os.path.join(texture_path, sample[mx]))
        texture_node.image = img
        
        # Check if this was a normal node
        if 'normal' in mx:
            normal_node = nodes.new(type='ShaderNodeNormalMap')
            link = links.new(normal_node.inputs[1], texture_node.outputs[0])
            
            link = links.new(active_node.inputs[node_input_map[mx]], normal_node.outputs[0])
            normal_node_texture = texture_node
        elif 'roughness' in mx:
            link = links.new(active_node.inputs[node_input_map[mx]], texture_node.outputs[0])
        else:
            link = links.new(active_node.inputs[node_input_map[mx]], texture_node.outputs[0])
        
        # Use non-color for all but 'Base Color' Textures
        if not node_input_map[mx] in ['Base Color'] and texture_node.image:
            texture_node.image.colorspace_settings.is_data = True
        else:
            texture_node.image.colorspace_settings.is_data = False
        
        # This are all connected texture nodes
        texture_nodes.append(texture_node)
        texture_node.label = node_input_map[mx]
        
    if disp_texture:
        texture_nodes.append(disp_texture)
        
    # Alignment
    for i, texture_node in enumerate(texture_nodes):
        offset = Vector((-550, (i * -280) + 200))
        texture_node.location = active_node.location + offset

    if normal_node:
        # Extra alignment if normal node was added
        normal_node.location = normal_node_texture.location + Vector((300, 0))
        
    # Add texture input + mapping
    mapping = nodes.new(type='ShaderNodeMapping')
    mapping.location = active_node.location + Vector((-1050, 0))
    if len(texture_nodes) > 1:
        # If more than one texture add reroute node in between
        reroute = nodes.new(type='NodeReroute')
        texture_nodes.append(reroute)
        tex_coords = Vector((texture_nodes[0].location.x, sum(n.location.y for n in texture_nodes)/len(texture_nodes)))
        reroute.location = tex_coords + Vector((-50, -120))
        for texture_node in texture_nodes:
            link = links.new(texture_node.inputs[0], reroute.outputs[0])
        link = links.new(reroute.inputs[0], mapping.outputs[0])
    else:
        link = links.new(texture_nodes[0].inputs[0], mapping.outputs[0])

    # Connect texture_coordiantes to mapping node
    texture_input = nodes.new(type='ShaderNodeTexCoord')
    texture_input.location = mapping.location + Vector((-200, 0))
    link = links.new(mapping.inputs[0], texture_input.outputs[2])

    # Create frame around tex coords and mapping
    frame = nodes.new(type='NodeFrame')
    frame.label = 'Mapping'
    mapping.parent = frame
    texture_input.parent = frame
    frame.update()

    # Create frame around texture nodes
    frame = nodes.new(type='NodeFrame')
    frame.label = 'Textures'
    for tnode in texture_nodes:
        tnode.parent = frame
    frame.update()
    return sample['name']


# Prepare UV Scaling functions
# Scale a 2D vector v, considering a scale s and a pivot point p
def Scale2D( v, s, p ):
    return ( p[0] + s[0]*(v[0] - p[0]), p[1] + s[1]*(v[1] - p[1]) )   

def ScaleUV( uvMap, scale, pivot ):
    for uvIndex in range( len(uvMap.data) ):
        uvMap.data[uvIndex].uv = Scale2D( uvMap.data[uvIndex].uv, scale, pivot )

# Function to assign material to objects
def assign_material(object_name, material_name, uv_scale):
    # select objs and set as active objects
    obj = bpy.data.objects[object_name]
    bpy.context.view_layer.objects.active = obj

    # Apply solidify if roads or sidewalk
    if 'road' in object_name:
        bpy.ops.object.modifier_add(type='SOLIDIFY')
        bpy.context.object.modifiers["Solidify"].thickness = 0.01
        bpy.context.object.modifiers["Solidify"].offset = -1
        bpy.ops.object.modifier_apply(modifier="Solidify")


    # Remove current materials
    num_materials = len(obj.data.materials)
    for ix in range(num_materials):
        bpy.ops.object.material_slot_remove()

    # Create a new material for the obj
    mat = bpy.data.materials.get(material_name)

    if obj.data.materials:
        # assign to 1st material slot
        obj.data.materials[0] = mat
    else:
        # no slots
        obj.data.materials.append(mat)


    bpy.ops.object.editmode_toggle()
    # Select all the vertices! This is important
    bpy.ops.mesh.select_mode(type="VERT")
    bpy.ops.mesh.select_all(action = 'SELECT')
    bpy.ops.uv.smart_project()
    bpy.ops.object.editmode_toggle()


    uvMap = obj.data.uv_layers[0]

    #Scale a 2D vector v, considering a scale s and a pivot point p
    def Scale2D( v, s, p ):
        return ( p[0] + s[0]*(v[0] - p[0]), p[1] + s[1]*(v[1] - p[1]) )   

    def ScaleUV( uvMap, scale, pivot ):
        for uvIndex in range( len(uvMap.data) ):
            uvMap.data[uvIndex].uv = Scale2D( uvMap.data[uvIndex].uv, scale, pivot )

    #Defines the pivot and scale
    pivot = Vector( (0.5, 0.5) )
    scale = Vector( uv_scale )

    ScaleUV( uvMap, scale, pivot )


# Assign the material to all important items
# [TODO]: generalize this
material_selection = np.random.choice(texture_config['road'])
material_name = create_new_material(material_selection)
assign_material('roads', material_name, uv_scale=(10, 10))

material_selection = np.random.choice(texture_config['sidewalk'])
material_name = create_new_material(material_selection)
assign_material('sidewalk', material_name, uv_scale=(30, 30))

# For buildings now
for ix in bpy.data.objects:
    if "texture_build" in ix.name and ix.type == "MESH":
        material_selection = np.random.choice(texture_config['building'])
        material_name = create_new_material(material_selection)
        assign_material(ix.name, material_name, uv_scale=(1.5, 1.5))


# Finally, just add the base ground plane
bpy.ops.mesh.primitive_plane_add(size=base_plane_size, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1,1,1))
bpy.context.active_object.name = "base_plane"

# Add material to base plane
material_selection = np.random.choice(texture_config['base'])
material_name = create_new_material(material_selection)
assign_material('base_plane', material_name, uv_scale=(8, 8))


# Finally add the dynamic sky
# [TODO]: Replace with HDRI as well for release code also
bpy.ops.sky.dyn()

dyn_sky = D.worlds.get(C.scene.dynamic_sky_name)
C.scene.world = dyn_sky

# possible horizons and sky colors 
# (Careful, some colors are just weird for fun experiments!)
horizons_possible = [
    (0.0185591, 0, 0.184465, 1),
    (0.123326, 0.155568, 1, 1),
    (0.0469195, 0.615312, 1, 1),
    (0.567353, 0.746357, 1, 1)
]

sky_possible = [
    (0.00074523, 0, 0.0290996, 1),
    (0.00365506, 0, 0.854979, 1),
    (0.0719103, 0.0527483, 1, 1)
]

# Get the settings
scene_brightness = C.scene.world.node_tree.nodes.get("Scene_Brightness")
sky_horizon_color = C.scene.world.node_tree.nodes.get("Sky_and_Horizon_colors") # index 1 for sky and 2 for horizon
cloud_opacity = C.scene.world.node_tree.nodes.get("Cloud_opacity")
cloud_density = C.scene.world.node_tree.nodes.get("Cloud_density")
sky_normal = C.scene.world.node_tree.nodes.get("Sky_normal")

# Set a random color of sky
import random
sky_horizon_color.inputs[1].default_value = random.choice(sky_possible)
sky_horizon_color.inputs[2].default_value = random.choice(horizons_possible)

scene_brightness = 2*np.random.random() + 0.5
cloud_opacity.inputs[0].default_value = np.random.random()*0.4 + 0.6
cloud_density.inputs[0].default_value = np.random.random()*0.4 + 0.6


# Add traffic signs as a particle system
# Create the collection for the traffic signs
ts_collection = bpy.data.collections.new("traffic_signs")
bpy.context.scene.collection.children.link(ts_collection)

# [TODO]: Improve this
selected_ts = np.random.permutation(len(obj_conf['traffic']))[:num_traffic_signs]
for ix in selected_ts:
    ts1 = load_specific_category_object('traffic', ix)
    move_to_floor(ts1)
    origin_to_bottom(bpy.data.objects[ts1])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # rotate along y-axis for proper hair behaviour in particles
    bpy.data.objects[ts1].rotation_euler.rotate_axis('Y', np.deg2rad(90))

    ts_collection.objects.link(bpy.data.objects[ts1])
    bpy.data.objects[ts1].location[2] = vegetation_template_underground_offset
    # bpy.context.scene.collection.objects.unlink(bpy.data.objects[tree1])

# Create a particle system on the sidewalk and add this collection there
# Get the sidewalk object
sidewalk = bpy.data.objects['sidewalk']
if len(sidewalk.particle_systems) == 0:
    sidewalk.modifiers.new("part", type='PARTICLE_SYSTEM')
    part = sidewalk.particle_systems[0]

    settings = part.settings
    settings.type = 'HAIR'
    settings.emit_from = 'FACE'
    
    # viewport display
    settings.display_method = 'RENDER'
    
    # settings.render_type = 'OBJECT'
    # settings.instance_object = bpy.data.objects['bush']
    settings.render_type = 'COLLECTION'
    settings.instance_collection = bpy.data.collections['traffic_signs']
    
    settings.particle_size = traffic_sign_particle_size
    settings.hair_length = 1

    settings.use_rotation_instance = True
    settings.use_advanced_hair = True
    settings.use_rotations = True
    settings.rotation_mode = 'OB_Z'
    # Randomize the orientation of the traffic signs
    settings.phase_factor_random = 2.0

    
    settings.count = traffic_sign_particle_count

# Do all vegetation stuff at the end
# Create the collection for the vegetation stuff
veg_collection = bpy.data.collections.new("vegetation_objects")
bpy.context.scene.collection.children.link(veg_collection)

# [TODO]: Improve this
selected_trees = np.random.permutation(len(obj_conf['tree']))[:num_trees]
for ix in selected_trees:
    tree1 = load_specific_category_object('tree', ix)
    move_to_floor(tree1)
    origin_to_bottom(bpy.data.objects[tree1])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    veg_collection.objects.link(bpy.data.objects[tree1])
    bpy.data.objects[tree1].location[2] = vegetation_template_underground_offset
    # bpy.context.scene.collection.objects.unlink(bpy.data.objects[tree1])

# [TODO]: Improve this
selected_bushes = np.random.permutation(len(obj_conf['bush']))[:num_bushes]
for ix in selected_bushes:
    bush = load_specific_category_object('bush', ix)
    move_to_floor(bush)
    origin_to_bottom(bpy.data.objects[bush])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    veg_collection.objects.link(bpy.data.objects[bush])
    bpy.data.objects[bush].location[2] = vegetation_template_underground_offset
    # bpy.context.scene.collection.objects.unlink(bpy.data.objects[bush])

# Prepare particle system for vegetation
# Create a plane with the correct size
bpy.ops.mesh.primitive_plane_add(size=vegetation_plane_size, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1,1,1))
bpy.context.active_object.name = "vegetation"
# bpy.context.object.rotation_euler[2] = -3.14159


# Load all images
img_rgb = bpy.data.images.load(vegetation_mask_path)

# Create a new material for the road
mat = bpy.data.materials.new(name="base_vegetation")
mat.use_nodes = True

# Select the principaled BSDF as selected/active
mat.node_tree.nodes.active = mat.node_tree.nodes["Principled BSDF"]

# Get nodes, links, active node and output node
nodes = mat.node_tree.nodes
links = mat.node_tree.links
active_node = nodes.active
output_node = [n for n in nodes if n.bl_idname == 'ShaderNodeOutputMaterial']

# Add color node
color_texture = nodes.new(type='ShaderNodeTexImage')
color_texture.image = img_rgb
color_texture.label = "Base Color"

offset = Vector((-550, (2 * -280) + 200))
color_texture.location = active_node.location + offset

link = links.new(active_node.inputs['Base Color'], color_texture.outputs[0])

# Just to be sure
active_node.select = False
nodes.update()
links.update()


# UV Mapping time
# select the plane and set to currently active
veg_plane = bpy.data.objects['vegetation']
bpy.context.view_layer.objects.active = veg_plane

# Remove current materials
num_materials = len(veg_plane.data.materials)
for ix in range(num_materials):
    bpy.ops.object.material_slot_remove()

# Create a new material for the road
mat = bpy.data.materials.get("base_vegetation")

if veg_plane.data.materials:
    # assign to 1st material slot
    veg_plane.data.materials[0] = mat
else:
    # no slots
    veg_plane.data.materials.append(mat)

bpy.context.view_layer.objects.active = veg_plane
bpy.ops.object.editmode_toggle()
# Select all the vertices! This is important
bpy.ops.mesh.select_mode(type="VERT")
bpy.ops.mesh.select_all(action = 'SELECT')
bpy.ops.uv.smart_project()
bpy.ops.object.editmode_toggle()

# [TODO]
import math
# Do it again just to be sure
bpy.ops.object.editmode_toggle()
# Select all the vertices! This is important
bpy.ops.uv.select_all(action='SELECT')
# bpy.ops.uv.smart_project(correct_aspect=True)
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0)
bpy.ops.object.editmode_toggle()

# uvMap = veg_plane.data.uv_layers[0]

# Adding the particle system for vegetation

if len(veg_plane.particle_systems) == 0:
    veg_plane.modifiers.new("part", type='PARTICLE_SYSTEM')
    part = veg_plane.particle_systems[0]

    settings = part.settings
    settings.type = 'HAIR'
    settings.emit_from = 'FACE'
    
    # viewport display
    settings.display_method = 'PATH'
    # settings.display_method = 'RENDER'
    
    # settings.render_type = 'OBJECT'
    # settings.instance_object = bpy.data.objects['bush']
    settings.render_type = 'COLLECTION'
    settings.instance_collection = bpy.data.collections['vegetation_objects']
    settings.use_collection_count = True
    
    settings.particle_size = vegetation_particle_size
    settings.hair_length = 1
    # settings.use_rotation_instance = True
    settings.use_advanced_hair = True
    settings.use_rotations = True
    settings.rotation_mode = 'OB_X'
    
    # Add a texture
    tex = bpy.data.textures.new('veg_tex', type='IMAGE')
    tex.image = img_rgb
    
    # Add to the particle system
    mtex = settings.texture_slots.add()
    mtex.texture = tex
    mtex.use_map_time = False
    mtex.use_map_density = True
    mtex.density_factor = 1.0
    mtex.texture_coords = 'UV'
    mtex.uv_layer = 'UVMap'
    mtex.blend_type = 'MULTIPLY'
    
    settings.count = vegetation_particle_count

# Hide instance from render
bpy.context.object.show_instancer_for_render = False
bpy.context.object.show_instancer_for_viewport = True
bpy.context.object.rotation_euler[2] = -3.14159

# Add rendering information
# improve rendering
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'

#bpy.context.scene.render.tile_x = 200
#bpy.context.scene.render.tile_y = 200

bpy.context.scene.cycles.tile_size = 1024
bpy.context.scene.cycles.samples = 256


bpy.context.scene.cycles.use_adaptive_sampling = True
bpy.context.scene.cycles.use_denoising = True
bpy.context.scene.cycles.denoiser = 'OPENIMAGEDENOISE'


def create_box(loc, size, orientation, cat_name="car"):
    # rescale the boxes a bit to avoid self-intersection
    if "vehicle" in cat_name or "human" in cat_name:
        if not ("construction" in cat_name and "human" not in cat_name):
            box_scale_factor = 0.75
        else:
            box_scale_factor = 1.0
        size[0] *= box_scale_factor
        size[1] *= box_scale_factor
        size[2] *= box_scale_factor
    
    # Get list of old objects
    old_objects = set(bpy.context.scene.objects)
    
    # Create the box
    bpy.ops.mesh.primitive_cube_add(align="WORLD", 
                                    location=loc, 
                                    scale=size)
    bpy.context.object.rotation_euler[2] = orientation
    # [TODO, NOTE]: Remove the ops operation because it cause context error
    # bpy.ops.transform.rotate(value=np.pi, orient_axis='Z', orient_type='GLOBAL')
    
    # Get the new object
    bbox = (set(bpy.context.scene.objects) - old_objects).pop()
    bbox.rotation_euler.rotate_axis('Z', np.deg2rad(180))
    
    for cat_ix in obj_conf.keys():
        # Add check for construction worker so that both vehicle and human are not repeated
        if "human" in cat_name and "construction" in cat_name:
            cat_ix = "human"
        if cat_ix in cat_name:
            place_category_item(cat_ix, bbox)
            bpy.ops.object.select_all(action='DESELECT')
            bbox.select_set(True)
            bpy.ops.object.delete()
            continue
        else:
            pass
    


def generate_random_name(category="car"):
    rad = "".join([chr(np.random.randint(0, 26) + 97)  for ix in range(6)])
    name = "{}_{}".format(category, rad)
    return name

def load_category_object(selected_category):
    # Select data for a random object of this category
    random_object_data = np.random.choice(obj_conf[selected_category])
    if selected_category == 'barrier':
        random_object_data = fix_barrier
    
    # Get list of old objects
    old_objects = set(bpy.context.scene.objects)

    # Import the object
    bpy.ops.import_scene.fbx(filepath=object_path + random_object_data['name'])

    # Get the new object
    obj = (set(bpy.context.scene.objects) - old_objects).pop()

    # Rename
    assigned_name = generate_random_name(selected_category)
    obj.name = assigned_name
    
    return assigned_name

def place_category_item(category_name, box):
    obj_name = load_category_object(category_name)
    obj = bpy.context.scene.objects[obj_name]
    
    obj.rotation_euler.rotate(box.rotation_euler)

    if category_name == 'barrier':
        obj.rotation_euler.rotate_axis('Z', np.deg2rad(90))
    
    obj_size = np.array(obj.dimensions)
    box_size = np.array(box.dimensions)
    ratio = box_size.max()/obj_size.max()

    if category_name == 'construction':
        ratio *= 1.5
    obj.dimensions = mathutils.Vector(ratio * obj_size)

    obj.location[0] = box.location[0]
    obj.location[1] = box.location[1]
    
    box.hide_viewport = True
    box.hide_render = True
    
    move_to_floor(obj_name)

def move_to_floor(object_name):
    bpy.ops.object.select_all(action='DESELECT')
    obj = bpy.context.scene.objects[object_name]
    # Also move the box to the ground properly
    minz =  min((obj.matrix_world @ v.co)[2] for v in obj.data.vertices)
    obj.location[2] -= minz


for ix in range(len(scene_config['boxes'])):
    box = scene_config['boxes'][ix]
    create_box(box['location'], box['box_size'], box['orientation'], box['cat_name'])


category_dict = {"car": 1, "bus": 2, "jeep": 3,
                "truck": 4, "van": 5, "human": 6,
                "build": 7, "road": 8, "barrier": 9, 
                "plane": 10, "cycle": 11, "construction": 12, 
                "bush": 13, "tree": 14, "motorcycle": 15, 
                "cone": 16, "trafficcone": 16, "traffic": 17, # traffic is for traffic signs
                "sidewalk": 18, "_void_": 19} 


category_items = {}
for kx in category_dict.keys():
    category_items[kx] = 0

print("-"*100)
for ix in bpy.data.objects:
    if ix.type == "MESH":
        assigned = False
        for cat_x in category_dict:
            # Check for confusion between construction vehicle and worker
            if "human" in ix.name.lower() and "construction" in ix.name.lower():
                cat_x = "human"
            # check for confusion in traffic cone and sign
            if "traffic" in ix.name.lower() and "cone" in ix.name.lower():
                cat_x = "cone"
            # check for confusion in bus and bush
            if "bus" in ix.name.lower() and "bush" in ix.name.lower():
                cat_x = "bush"
            # Check for confusion in motorcycle and cycle
            if "cycle" in ix.name.lower() and "motorcycle" in ix.name.lower():
                cat_x = "motorcycle"
            if cat_x in ix.name.lower():
                # print(cat_x, ix)
                ix["category_id"] = category_dict[cat_x]
                category_items[cat_x] += 1
                assigned = True
                break
        if not assigned:
            # Assign the void category
            cat_x = "_void_"
            ix["category_id"] = category_dict[cat_x]
            category_items[cat_x] += 1
            assigned = True

print(category_items)

# 960, 540
set_intrinsic_flag = True
for cam_key in scene_config['cam']:
    cam = scene_config['cam'][cam_key]
    if set_intrinsic_flag and cam_key == "CAM_FRONT":
        intrinsics = np.array(cam['intrinsic']).reshape((3,3))
        intrinsics[0, 0] *= data_scale_x
        intrinsics[1, 1] *= data_scale_y
        bproc.camera.set_intrinsics_from_K_matrix(intrinsics, render_img_size[0], render_img_size[1])
        set_intrinsic_flag = False

    extrinsic = np.array(cam['extrinsic']).reshape((4,4))
    updated_extrinsic = bproc.math.change_source_coordinate_frame_of_transformation_matrix(extrinsic, ["X", "-Y", "-Z"])
    bproc.camera.add_camera_pose(updated_extrinsic)


# Add camera poses from some more locations in the scene
# Select car objects
car_objects = []
for ix in bpy.data.objects:
    if "car" in ix.name:
        car_objects.append(ix.name)

selected_cars = np.random.permutation(car_objects)[:num_car_cams]
for sx in selected_cars:
    car_obj = bpy.data.objects[sx]
    for cam_key in scene_config['cam']:
        cam = scene_config['cam'][cam_key]
        extrinsic = np.array(cam['extrinsic']).reshape((4,4))
        extrinsic[0, 3] = car_obj.location[0]
        extrinsic[1, 3] = car_obj.location[1]
        extrinsic[2, 3] = 3.5
        updated_extrinsic = bproc.math.change_source_coordinate_frame_of_transformation_matrix(extrinsic, ["X", "-Y", "-Z"])
        bproc.camera.add_camera_pose(updated_extrinsic)

# Add camera
bproc.camera.set_resolution(render_img_size[0], render_img_size[1])


# bproc.renderer.enable_distance_output(activate_antialiasing=False)
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_normals_output()
bproc.renderer.set_noise_threshold(0.01)  # this is the default value


# Render the image
data = bproc.renderer.render()

# Change node name of Dynamic Sky to background to fix instance seg error
bpy.data.worlds["Dynamic_1"].node_tree.nodes["Scene_Brightness"].name = "Background"

# Render segmentation masks (per class and per instance)
data.update(bproc.renderer.render_segmap(map_by=["class", "instance"]))

chars = [ix for ix in 'abcdefghijklmnopqrstuvwxyz0123456789']
random_name = ''.join(np.random.choice(chars, 8))

bproc.writer.write_hdf5("{}/output_{}/".format(RENDER_PATH, random_name), data)
print("Done")
