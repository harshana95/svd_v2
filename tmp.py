"""NYU Depth Dataset V2"""

import glob
import cv2
import numpy as np
import h5py
from collections import Counter
import datasets
from datasets import BuilderConfig, Features, Value, SplitGenerator, Array2D, Image, Sequence
import hashlib
import os
import json
from os.path import join
import random

import skimage
from tqdm import tqdm
def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points
def read_calib_file(path):
    """Read KITTI calibration file
    (from https://github.com/hunse/kitti)
    """
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass
    return data

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(calib_dir, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # get image shape
    im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

class KITTI(datasets.GeneratorBasedBuilder):
    """KITTI Depth Dataset V2"""
    fpath = os.path.realpath(__file__)
    js = json.loads(open(str(fpath)[:-3]+'.json').read())
    data_dir = os.path.dirname(js['original file path'])
    
    VERSION = datasets.Version("1.2.1")

    BUILDER_CONFIGS = [
        BuilderConfig(name="default", version=VERSION, description="Default configuration for KITTI dataset"),
    ]

    DEFAULT_CONFIG_NAME = "default"

    # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
    # To normalize you need to scale the first row by 1 / image_width and the second row
    # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
    # If your principal point is far from the center you might need to disable the horizontal
    # flip augmentation.
    K = np.array([[0.58, 0, 0.5, 0],
                        [0, 1.92, 0.5, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], dtype=np.float32)


    def _info(self):
        features = Features({
            "image": Image(decode=True),
            "depth": Image(decode=True),
        })

        return datasets.DatasetInfo(
            description="Image depth and annotation dataset",
            features=features,
        )

    def _split_generators(self, dl_manager):
        if os.path.exists(os.path.join(self.data_dir, "files_1.npy")):
            print("Loading file list from npy file")
            files_1 = np.load(os.path.join(self.data_dir, "files_1.npy"))
            files_2 = np.load(os.path.join(self.data_dir, "files_2.npy"))
        else:
            print("Scanning folders... ", f'{self.data_dir}/**/**/image_02/data/*.jpg')
            # files_1 = sorted(glob.glob(f'{self.data_dir}/**/**/image_02/data/*.jpg', recursive=True))
            # files_2 = sorted(glob.glob(f'{self.data_dir}/**/**/velodyne_points/data/*.bin', recursive=True))

            files_1 = []
            for fol1 in os.listdir(self.data_dir):
                p1 = join(self.data_dir,fol1)
                if os.path.isdir(p1):
                    for fol2 in os.listdir(p1):
                        p2 = join(p1, fol2)
                        if os.path.isdir(p2):
                            print(fol2)
                            p3 = join(p2, 'image_02', 'data')
                            for fol3 in os.listdir(p3):
                                if fol3.endswith('.jpg'):
                                    files_1.append(join(p3, fol3))

            files_1.sort()

            files_2 = []
            to_remove = []
            for f in files_1:
                f2 = f.replace("image_02", "velodyne_points").replace(".jpg", ".bin")
                if os.path.exists(f2):
                    files_2.append(f2)
                else:
                    to_remove.append(f)
            print("remove", len(to_remove))
            for f in to_remove:
                files_1.remove(f)
            print(f"Done. Total={len(files_1)}")
            files_1 = np.array(files_1)
            files_2 = np.array(files_2)
            np.save(os.path.join(self.data_dir, "files_1.npy"), files_1)
            np.save(os.path.join(self.data_dir, "files_2.npy"), files_2)
        
        # idx = np.arange(len(files_1))
        # np.random.shuffle(idx)
        # idx = idx[:1]
        # print("random sample 1000")
        # files_1 = files_1[idx]
        # files_2 = files_2[idx]
        self.files_1 = files_1
        self.files_2 = files_2

        return [
            SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": datasets.Split.TRAIN,
                },
            ),
            SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": datasets.Split.VALIDATION,
                },
            ),
        ]
    
    @staticmethod
    def get_first_file(path):
        return os.listdir(path)[0]
    
    def _generate_examples(self, split):
        train_percent = 0.996
        full_res_shape = (1242, 375)
        
        
        train_len = int(len(self.files_1)*train_percent)

        def get_dict(p1, p2):
            depth = generate_depth_map(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p2)))),
                                       p2,
                                       2)
            depth = skimage.transform.resize(depth, full_res_shape[::-1], order=0, preserve_range=True, mode='constant')
            # print(depth.min(), depth.max())  # 0 - 80
            return {
                    "image": cv2.cvtColor(cv2.imread(p1), cv2.COLOR_BGR2RGB),
                    "depth": depth,
                }

        if split == datasets.Split.TRAIN:
            for idx in range(train_len):
                yield idx, get_dict(self.files_1[idx], self.files_2[idx])
        else:
            for idx in range(train_len, len(self.files_1)):
                yield idx, get_dict(self.files_1[idx], self.files_2[idx])
        