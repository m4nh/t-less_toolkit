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
from scipy.optimize import minimize, differential_evolution


class LineOptimizer(object):
    DEFAULT_FORMAT = 'RPY'

    def __init__(self, K, image, model, instance_frame):
        self.K = K
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.img_size = (640, 480)
        self.model = model
        self.instance_frame = instance_frame
        self.lsd = cv2.createLineSegmentDetector(0)
        self.gains = [5, 5, 5, 200, 200, 200.0]

    def getEdgesWithThreshold(self, model, th=1.0):
        for edge, faces in self.model['edges_graph'].iteritems():
            if len(faces) >= 2:
                f1 = model['faces'][faces[0]]
                f2 = model['faces'][faces[1]]
                n1 = model['normals'][faces[0]]
                n2 = model['normals'][faces[1]]

                angle = math.acos(np.clip(np.dot(n1, n2), -1.0, 1.0))
                if angle > th:
                    pass

    def drawSegments(self, image, segments, line_width=2, min_length=1):

        cop = image.copy()
        if segments is not None:
            for i in range(0, segments.shape[0]):
                line = segments[i, 0, :]
                p1 = line[: 2]
                p2 = line[2: 4]
                dist = np.linalg.norm(p1-p2)
                if dist < min_length:
                    continue
                cv2.line(
                    cop,
                    tuple(line[:2].astype(int)),
                    tuple(line[2:4].astype(int)),
                    (255, 0, 0), line_width
                )
        return cop

    def runOptimization(self, opt_type="RGB"):

        # map(float, np.array(args['initial_guess']))
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        max_disp = 0.02
        bounds = [
            (-max_disp, max_disp),
            (-max_disp, max_disp),
            (-max_disp, max_disp),
            (-math.pi, math.pi),
            (-math.pi, math.pi),
            (-math.pi, math.pi)
        ]

        def cb(x):
            print("Minimization", x)

        if opt_type == "LINES":
            # res = minimize(self.optimize, x0,
            #                method='L-BFGS-B',
            #                bounds=bounds,

            #                options={'maxiter': 100000, 'disp': True, 'eps': 0.001})

            res = differential_evolution(self.optimize,
                                         bounds,
                                         mutation=(0, 0.001)
                                         )

        if opt_type == "RGB":
            res = minimize(self.optimizeRGBDifference, x0,
                           method='Nelder-Mead',
                           bounds=bounds,

                           options={'maxiter': 100000, 'disp': True})

            # res = differential_evolution(self.optimizeRGBDifference,
            #                              bounds,
            #                              mutation=(0, 0.001)
            #                              )

        print("RESULT", res.x)
        return np.multiply(res.x, self.gains)

    def optimize(self, x):

        x = x.reshape(6,).copy()
        print("CURRENT X", x)
        base_frame = self.instance_frame
        frame = KDLFromArray(x, fmt=LineOptimizer.DEFAULT_FORMAT)
        frame = base_frame*frame

        matrix = KLDtoNumpyMatrix(frame)
        R = matrix[:3, :3]
        t = matrix[:3, 3]*1000.0

        ren_rgb = renderer.render(
            self.model,
            self.img_size,
            self.K,
            R, t, mode='rgb', surf_color=(0.1, 0.1, 0.1))

        rgb_lines = lsd.detect(gray)[0]
        model_lines = lsd.detect(cv2.cvtColor(ren_rgb, cv2.COLOR_BGR2GRAY))[0]

        zeros = np.zeros(self.gray.shape)
        rgb_lines_img = self.drawSegments(zeros, rgb_lines)
        model_lines_img = self.drawSegments(zeros, model_lines)

        rendered_image = cv2.addWeighted(image, 1, ren_rgb, 0.85, 0)

        # rgb_lines_img = lsd.drawSegments(zeros, rgb_lines)
        # model_lines_img = lsd.drawSegments(zeros, model_lines)

        diff = rgb_lines_img - model_lines_img

        count = np.count_nonzero(diff.ravel())

        cv2.imshow("model", rendered_image)
        cv2.imshow("opt_rgb", rgb_lines_img)
        cv2.imshow("opt_model", model_lines_img)
        cv2.imshow("opt_diff", diff)
        cv2.waitKey(10)
        return count

    def optimizeRGBDifference(self, x):

        x = x.reshape(6,).copy()
        base_frame = self.instance_frame
        x = np.multiply(x, self.gains)

        if np.linalg.norm(x[:3]) > 0.2:
            return np.inf
        # x[3:] = x[3:]*180/math.pi

        frame = KDLFromArray(x, fmt=LineOptimizer.DEFAULT_FORMAT)
        frame = base_frame*frame

        matrix = KLDtoNumpyMatrix(frame)
        R = matrix[:3, :3]
        t = matrix[:3, 3]*1000.0

        ren_rgb = renderer.render(
            self.model,
            self.img_size,
            self.K,
            R, t, mode='rgb', surf_color=(76.0/255.0, 72.0/255.0, 82.0/255.0))

        gray = self.gray
        rgb_lines = lsd.detect(gray)[0]
        model_lines = lsd.detect(cv2.cvtColor(ren_rgb, cv2.COLOR_BGR2GRAY))[0]

        zeros = np.zeros(self.gray.shape)
        rgb_lines_img = self.drawSegments(zeros, rgb_lines)
        model_lines_img = self.drawSegments(zeros, model_lines)

        ren_rgb = cv2.cvtColor(ren_rgb, cv2.COLOR_BGR2GRAY)

        rendered_image = gray.copy()
        rendered_image[ren_rgb != 0] = ren_rgb[ren_rgb != 0]

        grayf = gray.astype(float)
        rendered_imagef = rendered_image.astype(float)

        diff = np.abs(grayf - rendered_imagef)/255.0

        # rgb_lines_img = lsd.drawSegments(zeros, rgb_lines)
        # model_lines_img = lsd.drawSegments(zeros, model_lines)

        count = np.sum(diff)
        print("MAXMIN", np.min(diff), np.max(diff))

        cv2.imshow("model", rendered_image)
        cv2.imshow("diff", diff)
        cv2.waitKey(10)
        return count


def KDLtoArray(frame, fmt='RPY'):
    if fmt == 'XYZQ':
        p = frame.p
        q = frame.M.GetQuaternion()
        return numpy.array([
            p.x(), p.y(), p.z(), q[0], q[1], q[2], q[3]
        ]).reshape(1, 7)
    elif fmt == 'RPY':
        p = frame.p
        roll, pitch, yaw = frame.M.GetRPY()
        return numpy.array([
            p.x(), p.y(), p.z(), roll, pitch, yaw
        ]).reshape(1, 6)


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
        return frame
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


index = int(sys.argv[1])
dindex = 20

roll = -85
pitch = -85
lsd = cv2.createLineSegmentDetector(0)
optimized_correction = None
while True:
    image = cv2.imread(images[index])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Position 0 of the returned tuple are the detected lines

    transform = PyKDL.Frame()
    transform.M.DoRotX(roll*math.pi/180.0)
    transform.M.DoRotY(pitch*math.pi/180.0)

    instance_frame = instances_poses[0]*transform
    instance_frame = poses[index].Inverse() * instance_frame

    if optimized_correction is not None:
        print("CORRECTING WIGH", optimized_correction)
        instance_frame = instance_frame*optimized_correction

    instance_frame_matrix = KLDtoNumpyMatrix(instance_frame)

    R = instance_frame_matrix[:3, :3]

    t = instance_frame_matrix[:3, 3]*1000.0
    im_size = (640, 480)

    ren_rgb = renderer.render(model, im_size, K, R, t,
                              mode='rgb', surf_color=[0, 1.0, 0])

    # rgb_lines = lsd.detect(gray)[0]
    # model_lines = lsd.detect(cv2.cvtColor(ren_rgb, cv2.COLOR_BGR2GRAY))[0]

    # zeros = np.zeros(gray.shape)
    # rgb_lines_img = lsd.drawSegments(zeros, rgb_lines)
    # model_lines_img = lsd.drawSegments(zeros, model_lines)

    if optimized_correction is None:
        lopt = LineOptimizer(K, image, model, instance_frame)

        eye = PyKDL.Frame()

        cv2.imshow("model", image)
        cv2.waitKey(0)
        optimized_correction = KDLFromArray(lopt.runOptimization(), fmt="RPY")
    # lopt.optimize(KDLtoArray(eye, fmt='RPY'))
    image[ren_rgb != 0] = ren_rgb[ren_rgb != 0]

    # image = cv2.addWeighted(image, 1, ren_rgb, 0.85, 0)
    cv2.imshow("model", image)
    c = cv2.waitKey(0)
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

    print(c, "Pitch", pitch, "Roll", roll)
    if c == 113:
        break
