from glob import glob
from itertools import combinations
from random import shuffle

import numpy as np
import pickle
import cv2
import open3d as o3d
import os
import sys
import yaml
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.append(PACKAGE_DIR)
CABLE_OBSERVER_DIR = os.path.join(PACKAGE_DIR, "cable_observer", "src", "cable_observer", "utils")
sys.path.append(CABLE_OBSERVER_DIR)

from cable_observer.src.cable_observer.utils.tracking import track
from utils.bspline import BSpline
from utils.constants import BSplineConstants
from utils.geometry import euler2rotmat, invert


def uvz2xyz(uvz, f, c):
    z = uvz[:, 2]
    z = np.stack([z, z, np.ones_like(z)], axis=-1)
    xyz = z * (uvz - c) / f / 1000
    return xyz


# ds_name = "short"
# ds_name = "short_ds"
# ds_name = "xy_test"
#ds_name = "xy"
#ds_name = "yz_big"
ds_name = "xyzrpy/l50cm_all_sep_03_01"
# ds_name = "dummy_dataset"
#ds_type = "train"
#ds_type = ""
#ds_type = "val"
#ds_type = "filtered_data"
#ds_type = "filtered_data/val"
#ds_type = "02_21_nominal_l50cm"
#ds_type = "02_21_thin_l50cm"
#ds_type = "02_21_thin_l50cm/train"
#ds_type = "02_21_thin_l50cm/val"
#ds_type = "02_21_thin_l50cm/test"
#ds_type = "02_21_rigid_l50cm"
#ds_type = "02_21_rigid_l50cm/val"
#ds_type = "02_21_rigid_l50cm/train"
#ds_type = "02_21_l40cm"
#ds_type = "02_21_l45cm"
#ds_type = "02_21_l45cm/train"
#ds_type = "02_21_l45cm/val"
#ds_type = "02_21_l45cm/test"
#ds_type = "l50cm_sep/val"
#ds_type = "l50cm_all_sep/val"
#ds_type = "l50cm_all_sep/train"
#ds_type = "l50cm_all_sep_03_01/test"
#ds_type = "l50cm_all_sep_03_01/train"
#ds_type = "l50cm_all_sep_03_01/val"
#ds_type = "02_24_nominal_l50cm_test"
#ds_type = "l50cm_sep/train"
ds_type = "train"
N = 100
# plots = True
plots = False
neural_dlo_model_dataset = []
file_debug = []

# tracking utilities
last_spline_coords = None
stream = open(os.path.join(SCRIPT_DIR, "params.yaml"), 'r')
params = yaml.load(stream, Loader=yaml.FullLoader)

R_base2base = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])

color = ['r', 'g', 'b', 'c', 'm', 'k']
#bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)
#bsp = BSpline(BSplineConstants.n, 3, num_T_pts=1021)
#bsp = BSpline(16, 3, num_T_pts=1021)
bsp = BSpline(16, 3, num_T_pts=BSplineConstants.n)

for k, pkl in enumerate(sorted(glob(os.path.join(SCRIPT_DIR, ds_name, ds_type, "*.pkl")))):
    print(pkl)
    data = pickle.load(open(pkl, 'rb'))
    a = 0
    dataset = data["dataset"]
    if len(dataset) < 2:
        continue
    dataset_info = data["dataset_info"]
    fx = dataset_info["camera_params"]["fx"]
    fy = dataset_info["camera_params"]["fy"]
    cx = dataset_info["camera_params"]["cx"]
    cy = dataset_info["camera_params"]["cy"]
    c = np.array([cx, cy, 0.0])[np.newaxis]
    f = np.array([fx, fy, 1.0])[np.newaxis]
    R_right_base_in_camera, t_right_base_in_camera = dataset_info['right_arm_2_camera']
    R_left_base_in_camera, t_left_base_in_camera = dataset_info['left_arm_2_camera']
    gripper_in_flange = dataset_info["gripper_xyz"][:, np.newaxis]
    t_left_base_in_camera = t_left_base_in_camera[:, np.newaxis]
    t_right_base_in_camera = t_right_base_in_camera[:, np.newaxis]
    R_camera_in_right_base, t_camera_in_right_base = invert(R_right_base_in_camera, t_right_base_in_camera)
    R_camera_in_left_base, t_camera_in_left_base = invert(R_left_base_in_camera, t_left_base_in_camera)
    R_left_flange_in_tcp, t_left_flange_in_tcp = invert(np.eye(3), gripper_in_flange)
    R_right_flange_in_tcp, t_right_flange_in_tcp = invert(np.eye(3), gripper_in_flange)
    t_left_base_in_right_base = (t_camera_in_right_base - t_camera_in_left_base)
    states = []
    #k = 0
    for i, d in enumerate(dataset):
        img = d['rgb_img']
        depth = d['depth_img']
        success, spline_coords, spline_params, skeleton, mask, lower_bound, upper_bound, t = track(frame=img,
                                                                                          depth=depth,
                                                                                          last_spline_coords=last_spline_coords,
                                                                                          params=params)
        if not success:
            continue
        last_spline_coords = spline_coords
        spline_coords = np.stack([spline_coords[:, 1], spline_coords[:, 0], spline_coords[:, 2]], axis=-1)
        spline_control_pts = spline_params['coeffs']
        spline_control_pts = np.stack([spline_control_pts[1], spline_control_pts[0], spline_control_pts[2]], axis=-1)
        #plt.plot(spline_coords[:, 0], spline_coords[:, 1], '.')
        #plt.show()
        xyz_spline = uvz2xyz(spline_coords, f, c)[..., np.newaxis]
        xyz_spline_control_pts = uvz2xyz(spline_control_pts, f, c)[..., np.newaxis]
        # transform cable to right arm coordinate system
        t_left_flange_in_base = d['left_arm_pose'][:3][:, np.newaxis]
        euler_left_flange_in_base = d['left_arm_pose'][3:]
        R_left_flange_in_base = euler2rotmat(euler_left_flange_in_base)
        R_left_base_in_flange, t_left_base_in_flange = invert(R_left_flange_in_base, t_left_flange_in_base)
        R_left_flange_in_base = R_base2base @ R_left_flange_in_base
        t_left_flange_in_base = R_base2base @ t_left_flange_in_base
        t_right_flange_in_base = d['right_arm_pose'][:3][:, np.newaxis]
        euler_right_flange_in_base = d['right_arm_pose'][3:]
        R_right_flange_in_base = euler2rotmat(euler_right_flange_in_base)
        R_right_flange_in_base = R_base2base @ R_right_flange_in_base
        t_right_flange_in_base = R_base2base @ t_right_flange_in_base
        R_right_base_in_flange, t_right_base_in_flange = invert(R_right_flange_in_base, t_right_flange_in_base)
        xyz_spline_in_right_base = R_camera_in_right_base @ xyz_spline + t_camera_in_right_base
        xyz_spline_control_pts_in_right_base = R_camera_in_right_base @ xyz_spline_control_pts + t_camera_in_right_base
        t_right_tcp_in_base_norot = t_right_flange_in_base + R_right_flange_in_base @ gripper_in_flange
        t_left_tcp_in_base_norot = t_left_flange_in_base + R_left_flange_in_base @ gripper_in_flange

        xyz_spline_in_right_tcp_norot = xyz_spline_in_right_base - t_right_tcp_in_base_norot
        xyz_spline_control_pts_in_right_tcp_norot = xyz_spline_control_pts_in_right_base - t_right_tcp_in_base_norot
        xyz_left_tcp_in_right_tcp_norot = t_left_tcp_in_base_norot + t_left_base_in_right_base - t_right_tcp_in_base_norot

        #print(xyz_spline.shape)
        z = xyz_spline[:, 2, 0]
        dz = np.diff(z)
        #print(dz)
        #print(np.max(np.abs(dz)))
        if np.max(np.abs(dz)) > 0.005:
            print("TOO BIG Z DEV")
            print(np.max(np.abs(dz)))
            continue
            #pass
        #else: continue

        dxyz = np.diff(xyz_spline[:, :, 0], axis=0)
        length = np.sum(np.linalg.norm(dxyz, axis=-1))
        print("LENGTH:", length)
        L = 0.45
        if length < L:
            print("TOO SHORT:", length)
            continue

        #dist_from_right_tcp = np.linalg.norm(xyz_spline_control_pts_in_right_tcp_norot[..., 0] - np.array([-0.14, 0., 0.])[np.newaxis], axis=-1)
        #dist_from_right_tcp = np.linalg.norm(xyz_spline_control_pts_in_right_tcp_norot[..., 0] - np.array([0., 0., 0.02])[np.newaxis], axis=-1)
        dist_from_right_tcp = np.linalg.norm(xyz_spline_control_pts_in_right_tcp_norot[..., 0], axis=-1)
        d = 0.07
        if dist_from_right_tcp[0] > d and dist_from_right_tcp[-1] > d:
            print("WRONG TCP")
            print(dist_from_right_tcp[0])
            print(dist_from_right_tcp[-1])
            #continue
        if dist_from_right_tcp[-1] < dist_from_right_tcp[0]:
            print("REVERSE")
            xyz_spline_control_pts_in_right_tcp_norot = xyz_spline_control_pts_in_right_tcp_norot[::-1]

        #print(xyz_spline_control_pts_in_right_tcp_norot.shape)
        #print(bsp.N.shape)
        xyz_bsp = bsp.N[0] @ xyz_spline_control_pts_in_right_tcp_norot[..., 0]
        length = np.sum(np.linalg.norm(np.diff(xyz_bsp, axis=0), axis=-1))
        #length = np.sum(np.linalg.norm(np.diff(xyz_spline[..., 0], axis=0), axis=-1))
        #skip = 68
        #plt.subplot(121)
        #plt.plot(xyz_bsp[:, 0], xyz_bsp[:, 1], 'r')
        #plt.plot(xyz_spline_in_right_tcp_norot[:, 0], xyz_spline_in_right_tcp_norot[:, 1], 'b')
        #plt.subplot(122)
        #plt.plot(xyz_bsp[:, 0], xyz_bsp[:, 2], 'r')
        #plt.plot(xyz_spline_in_right_tcp_norot[:, 0], xyz_spline_in_right_tcp_norot[:, 2], 'b')
        #plt.show()
        states.append(np.concatenate([R_left_flange_in_base.reshape(-1), R_right_flange_in_base.reshape(-1),
                                      xyz_left_tcp_in_right_tcp_norot[:, 0],
                                      # xyz_spline_in_right_tcp_norot[::10].reshape(-1)
                                      #xyz_spline_in_right_tcp_norot.reshape(-1)
                                      #xyz_bsp[::skip].reshape(-1),
                                      xyz_bsp.reshape(-1),
                                      #xyz_spline_control_pts_in_right_tcp_norot.reshape(-1),
                                      np.array([length])
                                      ], axis=0))

        # xyz_spline_in_right_flange = R_right_base_in_flange @ xyz_spline_in_right_base + t_right_base_in_flange
        # xyz_spline_in_right_tcp = R_right_flange_in_tcp @ xyz_spline_in_right_flange + t_right_flange_in_tcp
        #xyz_bsp = bsp.N[0] @ xyz_spline_control_pts_in_right_tcp_norot[..., 0]

        if plots:
            plt.subplot(231)
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.plot(spline_coords[:, 0], spline_coords[:, 1], 'r.')
            plt.subplot(232)
            plt.imshow(mask)
            plt.plot(spline_coords[:, 0], spline_coords[:, 1], 'r')
            plt.subplot(233)
            plt.imshow(depth)
            plt.plot(spline_coords[:, 0], spline_coords[:, 1], 'r')
            mod = 10 / 1280 * 2 * (1280 / 2 - spline_coords[:, 0])
            plt.plot(spline_coords[:, 0] + mod, spline_coords[:, 1], 'b')
            plt.subplot(234)
            xrange = 1280 / fx / 2
            yrange = 720 / fy / 2
            plt.xlim(-xrange, xrange)
            plt.ylim(-yrange, yrange)
            plt.plot(xyz_spline[:, 0], -xyz_spline[:, 1], 'r.')
            plt.subplot(235)
            plt.plot(xyz_spline[:, 2])
            plt.show()
            # break
        mod = 10 / 1280 * 2 * (1280 / 2 - spline_coords[:, 0])
        #k = i
        plt.subplot(221)
        plt.plot(xyz_spline_control_pts_in_right_tcp_norot[:, 0], xyz_spline_control_pts_in_right_tcp_norot[:, 1], color[k % len(color)] + "x")
        plt.subplot(222)
        plt.plot(xyz_spline_control_pts_in_right_tcp_norot[:, 1], xyz_spline_control_pts_in_right_tcp_norot[:, 2], color[k % len(color)] + "x")
        plt.subplot(223)
        plt.plot(xyz_bsp[:, 0], xyz_bsp[:, 1], color[k % len(color)])
        #plt.plot(xyz_spline[:, 0], xyz_spline[:, 1], color[k % len(color)])
        plt.subplot(224)
        plt.plot(xyz_bsp[:, 1], xyz_bsp[:, 2], color[k % len(color)])
        #plt.plot(xyz_spline[:, 1], xyz_spline[:, 2], color[k % len(color)])
        a = 0
        #k += 1
        #plt.show()

    print(pkl)
    #plt.show()
    if len(states) < 2:
        continue
    states = np.stack(states, axis=0)
    c = np.array(list(combinations(range(states.shape[0]), 2)))
    neural_dlo_model_dataset.append(np.concatenate([states[c[:, 0]], states[c[:, 1]]], axis=-1))
    neural_dlo_model_dataset.append(np.concatenate([states[c[:, 1]], states[c[:, 0]]], axis=-1))

    #neural_dlo_model_dataset.append(np.concatenate([states[:-1], states[1:]], axis=-1))

    #neural_dlo_model_dataset.append(states)
    for e in neural_dlo_model_dataset[-1]:
        file_debug.append(pkl)
#plt.show()
#assert False
neural_dlo_model_dataset = np.concatenate(neural_dlo_model_dataset, axis=0)

#idx1 = np.random.randint(0, neural_dlo_model_dataset.shape[0] - 1, N)
#idx2 = np.random.randint(0, neural_dlo_model_dataset.shape[0] - 1, N)
#
#ds1 = neural_dlo_model_dataset[idx1]
#ds2 = neural_dlo_model_dataset[idx2]
#
#neural_dlo_model_dataset = np.concatenate([ds1, ds2], axis=-1)

# filter not moving pairs
neural_dlo_model_dataset_filtered = []
for i, el in enumerate(neural_dlo_model_dataset):
    print(i)
    ncp = BSplineConstants.ncp
    R_l_0 = el[:9].reshape((3, 3))
    R_r_0 = el[9:18].reshape((3, 3))
    xyz_l_0 = el[18:21]
    cp_0 = el[21:21 + ncp]
    len0 = el[21 + ncp: 22 + ncp]
    R_l_1 = el[22 + ncp:31 + ncp].reshape((3, 3))
    R_r_1 = el[31 + ncp:40 + ncp].reshape((3, 3))
    xyz_l_1 = el[40 + ncp:43 + ncp]
    cp_1 = el[43 + ncp:43 + 2 * ncp]
    len1 = el[43 + 2 * ncp:44 + 2 * ncp]
    dRl = np.linalg.norm(R_l_1 - R_l_0)
    dRr = np.linalg.norm(R_r_1 - R_r_0)
    dxyzl = np.linalg.norm(xyz_l_1 - xyz_l_0)
    dcp_ = cp_1 - cp_0
    dcp = np.linalg.norm(dcp_)
    cp_0_ = np.reshape(cp_0, (BSplineConstants.n, BSplineConstants.dim))
    cp_1_ = np.reshape(cp_1, (BSplineConstants.n, BSplineConstants.dim))
    #len0 = np.sum(np.linalg.norm(np.diff(cp_0_, axis=0), axis=-1))
    #len1 = np.sum(np.linalg.norm(np.diff(cp_1_, axis=0), axis=-1))
    dlen = np.abs(len1 - len0)
    if (dRl > 1e-2 or dRr > 1e-2 or dxyzl > 1e-3) and dcp > 0.01 and np.abs(len1 - len0) < 0.01:# and dxyzl > 0.05:
    #if (dRl > 1e-2 or dRr > 1e-2 or dxyzl > 1e-3) and dcp > 0.01 and np.abs(len1 - len0) < 0.02:# and dxyzl > 0.05:
        el = np.concatenate([el[:21 + ncp], el[22+ncp:-1]], axis=-1)
        neural_dlo_model_dataset_filtered.append(el)
    else:
        print("error")
        print(dRl, dRr, dxyzl, dcp, dlen, dxyzl)
        #print(file_debug[i])

neural_dlo_model_dataset = np.stack(neural_dlo_model_dataset_filtered, axis=0)
print(neural_dlo_model_dataset.shape[0])

os.makedirs(f"prepared_datasets/{ds_name}/", exist_ok=True)
np.savetxt(f"prepared_datasets/{ds_name}/{ds_type}.tsv", neural_dlo_model_dataset, delimiter="\t")
