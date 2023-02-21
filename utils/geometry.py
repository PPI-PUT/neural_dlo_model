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
    #print("MAX ENERGY:", np.max(energy))
    return np.mean(energy, axis=-1)
