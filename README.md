# TRoVE

This repository provides code for the paper TRoVE accepted to ECCV 2022.

## Requirements

Please make sure the following are installed and setup correctly to run the code provided in this repository correctly.

**Python dependencies**

1. Setup [blenderproc](https://github.com/DLR-RM/BlenderProc) for Blender 3.0 and Python 3.6.9 (Not tested for different versions yet)
2. skimage, tqdm, pyquaternion, opencv-python, h5py
3. nuscenes-devkit (for using nuscenes dataset)

**Assets (3D Objects and Textures)**

The 3D objects used in the paper can be downloaded at [3D Objects (drive)](https://drive.google.com/drive/folders/1WUz1g2IxJYSXKB4HiqOifhx-DjF8OZ-9?usp=sharing) and saved in a directory `3d_objects`, and the texture data from [Textures (drive)](https://drive.google.com/drive/folders/1Z4LYMW1FFHxf80p0kZw_3yCf2reQqXp_?usp=sharing) and saved in `PBR_textures`.

**nuScenes setup**

To run the code and experiments from the paper, the nuscenes dataset annotations are required only. The actual images and LiDAR/Radar is optional. The dataset annotations shall be downloaded to a root folder in the nuscenes dataset format and the path will be used in the configuration of the scripts.

**Blender setup**

Blender 3.0 was used in the development of this project using [Blenderproc](https://github.com/DLR-RM/BlenderProc) and additional addons such as [blender-osm](https://github.com/vvoovv/blender-osm).

## Data generation

Make sure to check each script for configurations and setup data paths. First, we prepare to parse the data from nuscenes and prepare the scene meta-data, and then after that we want to generate the OSM data needed for map preparation. For this, run the follwing two scripts:

```
python generate_nuscenes_config.py

python generate_osm_map_script.py
```

Once the data is prepared, we shall have 4 main folders: nuscenes config, osm data, `3d_objects`, and `PBR_textures`. Once this is available, prepare blender and the addons by running:

```
blenderproc run configure_blender_addons.py
```

Blenderproc will download a version of blender if one is not available or set already. After addon setup, update the asset configurations (especially perform this step is paths or machines have been changed)

```
blenderproc run update_3d_asset_config.py
```

And finally, to generate data from a scene sample, run the following command:

```
blenderproc run run_trove_once.py
```

The above command will generate data for the default scene. To generate data for another scene (say scene 5, sample 7), run:

```
blenderproc run run_trove_once.py --scene_config ./config_dump/scene_0005/sample_007.json
```

The command will generate 24 outputs on each run. 6 images from the ego-vehicle in the dataset for the cameras and 18 from 3 random vehicles. The generated data is available in HDF5 format.