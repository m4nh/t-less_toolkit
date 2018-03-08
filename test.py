from pytless import inout, renderer, misc
import os
import numpy as np
import scipy.misc
import cv2
import sys
import math
import json
import PyKDL
import numpy
import glob


def KDLFromArray(chunks, fmt='XYZQ'):
    if fmt == 'RPY':
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            chunks[0], chunks[1], chunks[2]
        )
        frame.M = PyKDL.Rotation.RPY(
            chunks[3],
            chunks[4],
            chunks[5]
        )
    if fmt == 'XYZQ':
        frame = PyKDL.Frame()
        frame.p = PyKDL.Vector(
            chunks[0], chunks[1], chunks[2]
        )
        q = np.array([chunks[3],
                      chunks[4],
                      chunks[5],
                      chunks[6]])
        q = q / np.linalg.norm(q)
        frame.M = PyKDL.Rotation.Quaternion(q[0], q[1], q[2], q[3])
    return frame


def KLDtoNumpyMatrix(frame):
    M = frame.M
    R = numpy.array([
        [M[0, 0], M[0, 1], M[0, 2]],
        [M[1, 0], M[1, 1], M[1, 2]],
        [M[2, 0], M[2, 1], M[2, 2]],
    ])
    P = numpy.transpose(
        numpy.array([
            frame.p.x(),
            frame.p.y(),
            frame.p.z()
        ])
    )
    P = P.reshape(3, 1)
    T = numpy.concatenate([R, P], 1)
    T = numpy.concatenate([T, numpy.array([0, 0, 0, 1]).reshape(1, 4)], 0)
    return T


images_path = '/Users/daniele/Desktop/to_delete/roars_dataset/indust_scene_1_dome/camera_rgb_image_raw_compressed'
dataset_path = '/Users/daniele/Desktop/to_delete/roars_dataset/indust_scene_1_dome.roars'
camera_extrinsics_path = '/Users/daniele/Desktop/to_delete/roars_dataset/indust_scene_1_dome/camera_extrinsics.txt'
camera_intrinsics_path = '/Users/daniele/Desktop/to_delete/roars_dataset/indust_scene_1_dome/camera_intrisics.txt'
poses_path = '/Users/daniele/Desktop/to_delete/roars_dataset/indust_scene_1_dome/robot_poses.txt'
model_path = '/Users/daniele/Downloads/industrial_part1.ply'


#######################################
# Model
#######################################
model = inout.load_ply(model_path)


#######################################
# Dataset data
#######################################
json_data = json.load(open(dataset_path))


#######################################
# Intrinsics
#######################################
K_raw = np.loadtxt(camera_intrinsics_path)
K = np.array([
    [K_raw[2], 0, K_raw[4]],
    [0, K_raw[3], K_raw[5]],
    [0, 0, 1.0]
])

#######################################
# Extrinsics
#######################################
camera_extrinsics = KDLFromArray(np.loadtxt(camera_extrinsics_path))
print("CAMERA EXTRINSICS", camera_extrinsics)

#######################################
# Poses
#######################################
poses = []
raw_poses = np.loadtxt(poses_path)
for p in raw_poses:
    frame = KDLFromArray(p)*camera_extrinsics
    poses.append(frame)

#######################################
# Images
#######################################
images = sorted(glob.glob(os.path.join(images_path, "*.jpg")))
print(len(poses), len(images))

instances_poses = []
for instance in json_data['classes']['5']['instances']:
    frame = KDLFromArray(instance['frame'])
    instances_poses.append(frame)
    print(frame)


index = 10
dindex = 20

roll = 0.0
pitch = 0.0

while True:
    image = cv2.imread(images[index])

    transform = PyKDL.Frame()
    transform.M.DoRotX(roll*math.pi/180.0)
    transform.M.DoRotY(pitch*math.pi/180.0)

    instance_frame = instances_poses[0]*transform
    instance_frame = poses[index].Inverse() * instance_frame

    instance_frame_matrix = KLDtoNumpyMatrix(instance_frame)

    R = instance_frame_matrix[:3, :3]

    t = instance_frame_matrix[:3, 3]*1000.0
    im_size = (640, 480)

    ren_rgb = renderer.render(model, im_size, K, R, t, mode='rgb')
    #image[ren_rgb != 0] = ren_rgb[ren_rgb != 0]

    image = cv2.addWeighted(image, 1, ren_rgb, 0.85, 0)
    cv2.imshow("model", image)
    c = cv2.waitKey(100)
    if c == 100:
        index += dindex
    if c == 97:
        index -= dindex

    #######################################
    # Roll
    #######################################
    if c == 114:
        roll += 1.0
    if c == 102:
        roll -= 1.0

    #######################################
    # Pitch
    #######################################
    if c == 116:
        pitch += 1.0
    if c == 103:
        pitch -= 1.0
    print(c)
