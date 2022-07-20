import numpy as np
import os
import json

# Set the path to the configuration files and the output OSM save path
config_path = "./config_dump/"
osm_path = "./osm_files/"
os.makedirs(osm_path, exist_ok=True)

# Map download API
api = "http://overpass-api.de/api/map?bbox={},{},{},{}"

download_command = 'wget --header="User-Agent: overpass-api-python-wrapper (Linux x86_64)" --header="From: somebody@website.xyz" {} -O {}'
cmd = []


for ix in os.listdir(config_path):
    if os.path.exists(os.path.join(osm_path, "{}.osm".format(ix))):
        continue
    # Get the first json from each scene
    scene_conf_path = os.path.join(config_path, ix, "sample_000.json")
    f = open(scene_conf_path, 'r')
    data = f.read()
    f.close()

    scene_conf = json.loads(data)
    georef = scene_conf["geo"]
    url = api.format(georef[0][1], georef[0][0], georef[1][1], georef[1][0])
    save_path = os.path.join(osm_path, "{}.osm".format(ix))
    cmd.append(download_command.format(url, save_path))

# Add command to delete all empty files
cmd.append(" find {} -size 0 -delete".format(osm_path))


script = "\n".join(cmd)
f = open("map_data.sh", 'w')
f.write(script)
f.close()