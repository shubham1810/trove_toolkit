import copy
import json
import os
import os.path as osp

import cv2
import numpy as np
import skimage.io as sio
import tqdm

from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.bitmap import BitMap
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from pyquaternion import Quaternion

# Add path to nuscenes dataroot and output directory
DATA_ROOT = "/50tb/public_data/nuScenes/"
MAP_ROOT = "/50tb/public_data/nuScenes/"
OUTPUT_PATH = "./config_dump/"

if not OUTPUT_PATH:
    os.makedirs(OUTPUT_PATH)


# Add maps origin location for gps config
# Taken from https://github.com/nutonomy/nuscenes-devkit/issues/144#issuecomment-505377023
georef = {
    'boston-seaport': [42.336849169438615, -71.05785369873047],
    'singapore-onenorth': [1.2882100868743724, 103.78475189208984],
    'singapore-hollandvillage': [1.2993652317780957, 103.78217697143555],
    'singapore-queenstown': [1.2782562240223188, 103.76741409301758],
}

def translation2transform(vec):
    i = np.eye(4)
    i[:3, -1] = vec
    return i

def get_geoloc(loc, offset):
    # //Position, decimal degrees
    lat, lon = loc

    #  //Earth’s radius, sphere
    R=6378137

    #  //offsets in meters
    de, dn = offset # 375, 1112 (example)

    #  //Coordinate offsets in radians
    dLat = dn/R
    dLon = de/(R*np.cos(np.pi*lat/180))

    #  //OffsetPosition, decimal degrees
    latO = lat + dLat * 180/np.pi
    lonO = lon + dLon * 180/np.pi

    return (latO, lonO)

nusc = NuScenes(version='v1.0-trainval', dataroot=DATA_ROOT, verbose=True)

start_idx = 74
num_scenes = 850 # len(nusc.list_scenes())

for get_scene_id in tqdm.tqdm(range(start_idx, num_scenes)):
    # Get scene information
    sample_scene_idx = get_scene_id
    my_scene = nusc.scene[sample_scene_idx]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    map_location = nusc.get('log', my_scene['log_token'])['location']

    # all camera info
    all_cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT']
    relevant_category = ["vehicle", "human", "barrier", "trafficcone"]

    # placeholders for the main data elements
    pts = []
    labels = []
    scene_boxes = []
    camera_info = []

    for ix in range(my_scene['nbr_samples']):
        # Get the main tokens for this sample
        my_sample_token = my_sample['token']
        sample_record = nusc.get('sample', my_sample_token)

        # get the lidar sensor token and the point cloud data along with lidar segmentation annotations
        pointsensor_token = sample_record['data']['LIDAR_TOP']
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = os.path.join(nusc.dataroot, pointsensor['filename'])
        lidarseg_path = os.path.join(nusc.dataroot, 'lidarseg/v1.0-trainval', pointsensor_token + '_lidarseg.bin')
        pc_data = LidarPointCloud.from_file(pcl_path)
        lseg = np.fromfile(lidarseg_path, dtype=np.uint8)

        # Get the annotation boxes for relevant classes
        my_annot_token = my_sample['anns']
        selected_items = []
        for each_token in my_annot_token:
            annot_meta = nusc.get('sample_annotation', each_token)
            # Search for match in required classes
            for rx in relevant_category:
                if rx in annot_meta.get('category_name'):
                    selected_items.append(each_token)
                    break
        
        # Get actual bounding box locations
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(pointsensor_token, selected_anntokens=selected_items)

        # Commpute the calibration matrices (Don't save right now)
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        R1 = Quaternion(cs_record['rotation']).transformation_matrix
        T1 = translation2transform(np.array(cs_record['translation']))

        # Apply first transformation to the point cloud data
        pc_data.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc_data.translate(np.array(cs_record['translation']))

        # Transform the bounding boxes as well
        for i in range(len(boxes)):
            boxes[i].rotate(Quaternion(cs_record['rotation']))
            boxes[i].translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        R2 = Quaternion(poserecord['rotation']).transformation_matrix
        T2 = translation2transform(np.array(poserecord['translation']))

        # Apply second transformation to the point cloud
        pc_data.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc_data.translate(np.array(poserecord['translation']))

        # Transform the bounding boxes as well (to global frame of reference)
        for i in range(len(boxes)):
            boxes[i].rotate(Quaternion(poserecord['rotation']))
            boxes[i].translate(np.array(poserecord['translation']))
        
        # Now we want to iterate through all available cameras
        camera_data = {}
        for cam_ix in all_cams:
            # Get the camera token
            camera_token = sample_record['data'][cam_ix]
            cam = nusc.get('sample_data', camera_token)

            # Third transform: transform from global into the ego vehicle frame for the timestamp of the image.
            poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
            T3 = translation2transform(-np.array(poserecord['translation']))
            R3 = Quaternion(poserecord['rotation']).transformation_matrix.T

            # Fourth transform: transform from ego into the camera.
            cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
            T4 = translation2transform(-np.array(cs_record['translation']))
            R4 = Quaternion(cs_record['rotation']).transformation_matrix.T

            intrinsic = (np.array(cs_record['camera_intrinsic'])).flatten()
            extrinsic = (np.linalg.inv(R4 @ T4 @ R3 @ T3)).flatten()
            
            camera_data[cam_ix] = {
                'intrinsic': list(intrinsic),
                'extrinsic': list(extrinsic),
            }

        # Store the point cloud data in the placeholders (convert to consistent arrays later)
        pts_ = pc_data.points
        pts.append(pts_)
        labels.append(lseg)

        # Store bounding boxes in the placeholder as well
        scene_boxes.append(boxes)

        # Save camera information data in placeholder
        camera_info.append(camera_data)

        if not (my_sample['next'] == ""):
            next_token = my_sample['next']
            my_sample = nusc.get('sample', next_token)

    # Convert all lidar points and segmentation labels to proper numpy arrays
    pts = np.concatenate(pts, axis=-1)
    lbs = np.concatenate(labels, axis=-1)

    x_min, y_min = pts.min(1)[:2].astype(int)
    x_max, y_max = pts.max(1)[:2].astype(int)

    # Compute the center, width and height
    xc = (x_min + x_max)/2
    yc = (y_min + y_max)/2
    w = np.abs(x_max - x_min)
    h = np.abs(y_max - y_min)

    fixed_size = max(w, h)
    # make the coordinate system square for convenience
    x_min = xc - fixed_size/2
    x_max = xc + fixed_size/2
    y_min = yc - fixed_size/2
    y_max = yc + fixed_size/2

    gps_origin = georef[map_location]

    gps_min = get_geoloc(gps_origin, [x_min, y_min])
    gps_max = get_geoloc(gps_origin, [x_max, y_max])

    nusc_map = NuScenesMap(dataroot=MAP_ROOT, map_name=map_location)
    bitmap = BitMap(nusc_map.dataroot, nusc_map.map_name, 'basemap')

    xc = (x_min + x_max)/2
    yc = (y_min + y_max)/2
    w = np.abs(x_max - x_min)
    h = np.abs(y_max - y_min)

    mask_size = 500

    # (xc, yc, w, h): center coords, and (w,h) are for the whole box
    # (400, 1100, 200, 200) mean origin at (300, 1000)
    patch = (xc, yc, w, h)
    img = nusc_map.get_map_mask(patch, 0, ['drivable_area'], (mask_size, mask_size))[0]

    # img = np.flip(img, 0) *  255
    img = img * 255
    # img = np.stack([img, img, img], axis=2)


    # filter the vegetation from the lidar points and shuffle
    vegetation_filter = (lbs == 30)
    vegetation_points = pts[:3, vegetation_filter]
    vegetation_points = vegetation_points[:, np.random.permutation(np.arange(vegetation_points.shape[1]))]

    # Select points we want and apply the min offset
    veg =  vegetation_points[:3, ::10].T - np.array([x_min, y_min, 0])
    z_filter = veg[:, 2] < 4
    ground_veg = veg[z_filter]

    scale_ratio = mask_size/fixed_size

    imc = img.copy()*0

    new_coords = ground_veg*scale_ratio

    for ix in range(new_coords.shape[0]):
        coords = new_coords[ix].astype(np.int16)
        cv2.circle(imc, (coords[0], coords[1]), 7, (255), -1)

    # Mask out the actual driveable area
    imc[img == 255] = 0

    # Also add some trees to the border area
    imc[:10, :] = 255
    imc[-10:, :] = 255
    imc[:, :10] = 255
    imc[:, -10:] = 255

    imc = np.stack([imc, imc, imc], axis=2)

    # simple mapping function to easily make coordinate changes
    fixcord = lambda x: (x[0], x[1], x[2])
    global_min = np.array([x_min, y_min, 0])

    parsed_scene_boxes = {}

    # Iterate over all samples of the scene
    for sx in range(len(scene_boxes)):
        boxes = scene_boxes[sx]

        # Save the centers and sizes of the box
        centers = [box.center for box in boxes]
        sizes = [box.wlh for box in boxes]

        # Iterate over the boxes and store specific proeprties
        bbox_data = []
        for ix in range(len(boxes)):
            bbox_data.append({
                'location': fixcord(centers[ix] - global_min),
                'box_size': tuple(sizes[ix]),
                'orientation': np.pi/2 + (boxes[ix].orientation.axis[-1] * boxes[ix].orientation.angle),
                'cat_name': boxes[ix].name
            })
        
        parsed_scene_boxes[sx] = bbox_data

    # Iterate through all camera info
    for ix in range(len(camera_info)):
        # Get the current camera
        cam_box = camera_info[ix]
        for kx in cam_box.keys():
            # Get location information on global coords
            val_x = cam_box[kx]['extrinsic'][3]
            val_y = cam_box[kx]['extrinsic'][7]
            # Update with the data offset
            camera_info[ix][kx]['extrinsic'][3] = val_x - x_min
            camera_info[ix][kx]['extrinsic'][7] = val_y - y_min

    # Exportable data curation
    # vegetation heatmap
    veg_heatmap_path = "{}/scene_{}/vegetation.png".format(OUTPUT_PATH, str(sample_scene_idx).zfill(4))

    # Create a dump path for the config
    dump_path = "{}/scene_{}/".format(OUTPUT_PATH, str(sample_scene_idx).zfill(4))
    os.makedirs(dump_path, exist_ok=True)

    # Save the vegetation mask at the location
    veg_mask_path = os.path.join(dump_path, "vegetation.png")
    sio.imsave(veg_mask_path, imc)

    # Iterate over each sample from the scene
    for ix in range(len(camera_info)):
        sample_config = {
            'geo': [gps_min, gps_max],
            'bounds': [x_min, y_min, x_max, y_max],
            'vegetation_path': veg_mask_path,
            'scene_width': int(fixed_size), # bug fix to change from int64 (not serializable)
            'boxes': parsed_scene_boxes[ix],
            'cam': camera_info[ix]
        }
        # Generate the config in json format
        sample_conf = json.dumps(sample_config, indent=4, sort_keys=True)
        
        # Save the generated config
        filename = "sample_{}.json".format(str(ix).zfill(3))
        f = open(os.path.join(dump_path, filename), 'w')
        f.write(sample_conf)
        f.close()