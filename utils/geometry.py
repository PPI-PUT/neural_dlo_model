import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space


def rotmat2rotvec(rm):
    return R.from_matrix(rm).as_rotvec(degrees=False)


def rotvec2rotmat(rotvec):
    return R.from_rotvec(rotvec).as_matrix()


def euler2rotmat(euler):
    return R.from_euler("xyz", euler).as_matrix()


def rotvec2euler(rotvec):
    return R.from_rotvec(rotvec).as_euler("xyz", degrees=False)


def quat2rotmat(quat):
    return R.from_quat(quat).as_matrix()


def invert(R, t):
    R = R.T
    return R, -R @ t


def compute_curve_energy(dr, ddr):
    n = np.linalg.norm(np.cross(dr, ddr), axis=-1) ** 2
    d = np.linalg.norm(dr, axis=-1) ** 5
    energy = n / (d + 1e-10)
    # print("MAX ENERGY:", np.max(energy))
    return np.mean(energy, axis=-1)


def calculateL3(coords_ground_truth, coords_prediction):
    assert len(coords_ground_truth.shape) == 2
    assert len(coords_prediction.shape) == 2
    assert coords_ground_truth.shape[0] in [2, 3]
    assert coords_prediction.shape[0] in [2, 3]
    def calculate_accumulated_length(coords):
        length_vec = np.linalg.norm(np.diff(coords, axis=-1), axis=0)
        length_vec = np.concatenate([[0.], np.cumsum(length_vec)])
        length_vec /= length_vec[-1]
        return length_vec

    def calculate_distances(coords_from, lengths_from, coords_to, lengths_to):
        k = 0
        i = 0
        dists = []
        while i < len(lengths_to):
            if lengths_to[k] <= lengths_from[i] <= lengths_to[k + 1]:
                # define a point and calculate distance
                t = (lengths_from[i] - lengths_to[k]) / (lengths_to[k + 1] - lengths_to[k])
                point_to = coords_to[:, k] * (1 - t) + t * coords_to[:, k + 1]
                dist = np.linalg.norm(point_to - coords_from[:, i])
                dists.append(dist)
                i += 1
            else:
                k += 1
        return dists

    gt_length_vec = calculate_accumulated_length(coords_ground_truth)
    pred_length_vec = calculate_accumulated_length(coords_prediction)

    L3_pred2gt_dists = calculate_distances(coords_prediction, pred_length_vec, coords_ground_truth, gt_length_vec)
    L3_gt2pred_dists = calculate_distances(coords_ground_truth, gt_length_vec, coords_prediction, pred_length_vec)

    L3_pred2gt = np.mean(L3_pred2gt_dists)
    L3_gt2pred = np.mean(L3_gt2pred_dists)

    return (L3_gt2pred + L3_pred2gt) / 2.
