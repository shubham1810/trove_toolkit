import blenderproc as bproc

import bpy
import addon_utils

# Get list of all addons and active addons
all_addons = addon_utils.modules()
act_addons = bpy.context.preferences.addons

blosm_path = "PATH TO blosm.zip"
mapbox_token = "MAPBOX token"
osm_data_dir = '/ssd_scratch/shubham/tmp/'

required_default = ['lighting_dynamic_sky', 'node_wrangler']


# Get the names of the addons
all_addons_names = [ix.bl_info['name'] for ix in all_addons]
act_addons_names = [ix.module for ix in act_addons]


for ix in required_default:
    addon_utils.enable(ix, default_set=True, persistent=False, handle_error=None)

bpy.ops.preferences.addon_install(overwrite=True,filepath=blosm_path)
bpy.ops.preferences.addon_enable(module='blender-osm')

# Get the blender osm module and update properties
for ix in act_addons:
    if 'osm' in ix.module:
        blosm = ix
        blosm.preferences.mapboxAccessToken = mapbox_token
        blosm.preferences.dataDir = osm_data_dir

bpy.ops.wm.save_userpref()
print(act_addons_names)