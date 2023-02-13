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
