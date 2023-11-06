import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.constants import BSplineConstants
import matplotlib.pyplot as plt


def prepare_dataset_cond(path, rot, n=0, diff=False, augment=False, norm=False, scale=1.):
    assert rot in ["quat", "rotmat", "rotvec", "euler"]
    data = np.load(path)
    if n > 0:
        idx = np.random.randint(0, data.shape[0] - 1, n)
        data = data[idx]
    ncp = BSplineConstants.ncp
    R_l_0 = data[:, :-1, :9].reshape((data.shape[0], data.shape[1] - 1, 3, 3))
    R_r_0 = data[:, :-1, 9:18].reshape((data.shape[0], data.shape[1] - 1, 3, 3))
    xyz_l_0 = data[:, :-1, 18:21] * scale
    cp_0 = data[:, :-1, 21:21 + ncp].reshape(
        (data.shape[0], data.shape[1] - 1, BSplineConstants.n, BSplineConstants.dim)) * scale
    # for i in range(cp_0.shape[0]):
    # for i in range(100):
    #    plt.subplot(211)
    #    plt.plot(cp_0[i, ..., 1], cp_0[i, ..., 2])
    #    plt.subplot(212)
    #    plt.plot(cp_0[i, ..., 1], cp_0[i, ..., 0])
    # plt.gca().set_aspect('equal')
    # plt.show()
    R_l_1 = data[:, -1, :9].reshape((-1, 3, 3))
    R_r_1 = data[:, -1, 9:18].reshape((-1, 3, 3))
    xyz_l_1 = data[:, -1, 18:21] * scale
    cp_1 = data[:, -1, 21:21 + ncp].reshape((-1, BSplineConstants.n, BSplineConstants.dim)) * scale
    diff_R_l = np.transpose(R_l_0[:, -1], (0, 2, 1)) @ R_l_1
    diff_R_r = np.transpose(R_r_0[:, -1], (0, 2, 1)) @ R_r_1
    mul = 1.

    if norm:
        c0_len = np.sum(np.linalg.norm(cp_0[:, -1, 1:] - cp_0[:, -1, :-1], axis=-1), axis=-1)
        c1_len = np.sum(np.linalg.norm(cp_1[:, 1:] - cp_1[:, :-1], axis=-1), axis=-1)
        # normalize translations by length
        xyz_l_0 = xyz_l_0 / c0_len[:, np.newaxis, np.newaxis]
        xyz_l_1 = xyz_l_1 / c1_len[:, np.newaxis]

    if diff:
        if rot == "quat":
            X1 = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_quat() for i in range(R_l_0.shape[1])],
                                 R.from_matrix(diff_R_l).as_quat(),
                                 *[R.from_matrix(R_r_0[:, i]).as_quat() for i in range(R_r_0.shape[1])],
                                 R.from_matrix(diff_R_r).as_quat(),
                                 ], axis=-1).astype(np.float32)
        elif rot == "rotmat":
            X1 = np.concatenate([*[R_l_0[:, i].reshape((-1, 9)) for i in range(R_l_0.shape[1])],
                                 diff_R_l.reshape((-1, 9)),
                                 *[R_r_0[:, i].reshape((-1, 9)) for i in range(R_r_0.shape[1])],
                                 diff_R_r.reshape((-1, 9)),
                                 ], axis=-1).astype(np.float32)
        elif rot == "rotvec":
            X1 = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_rotvec() for i in range(R_l_0.shape[1])],
                                 R.from_matrix(diff_R_l).as_rotvec(),
                                 *[R.from_matrix(R_r_0[:, i]).as_rotvec() for i in range(R_r_0.shape[1])],
                                 R.from_matrix(diff_R_r).as_rotvec(),
                                 ], axis=-1).astype(np.float32)
        elif rot == "euler":
            X1 = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_euler("xyz") for i in range(R_l_0.shape[1])],
                                 R.from_matrix(diff_R_l).as_euler("xyz"),
                                 *[R.from_matrix(R_r_0[:, i]).as_euler("xyz") for i in range(R_r_0.shape[1])],
                                 R.from_matrix(diff_R_r).as_euler("xyz"),
                                 ], axis=-1).astype(np.float32)
        X2 = np.concatenate([*[xyz_l_0[:, i] * mul for i in range(xyz_l_0.shape[1])],
                             (xyz_l_1 - xyz_l_0[:, -1]) * mul,
                             ], axis=-1)
    else:
        if rot == "quat":
            X1 = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_quat() for i in range(R_l_0.shape[1])],
                                 R.from_matrix(R_l_1).as_quat(),
                                 *[R.from_matrix(R_r_0[:, i]).as_quat() for i in range(R_r_0.shape[1])],
                                 R.from_matrix(R_r_1).as_quat(),
                                 ], axis=-1).astype(np.float32)
        elif rot == "rotmat":
            X1 = np.concatenate([*[R_l_0[:, i].reshape((-1, 9)) for i in range(R_l_0.shape[1])],
                                 R_l_1.reshape((-1, 9)),
                                 *[R_r_0[:, i].reshape((-1, 9)) for i in range(R_r_0.shape[1])],
                                 R_r_1.reshape((-1, 9)),
                                 ], axis=-1).astype(np.float32)
        elif rot == "rotvec":
            X1 = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_rotvec() for i in range(R_l_0.shape[1])],
                                 R.from_matrix(R_l_1).as_rotvec(),
                                 *[R.from_matrix(R_r_0[:, i]).as_rotvec() for i in range(R_r_0.shape[1])],
                                 R.from_matrix(R_r_1).as_rotvec(),
                                 ], axis=-1).astype(np.float32)
        elif rot == "euler":
            X1 = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_euler("xyz") for i in range(R_l_0.shape[1])],
                                 R.from_matrix(R_l_1).as_euler("xyz"),
                                 *[R.from_matrix(R_r_0[:, i]).as_euler("xyz") for i in range(R_r_0.shape[1])],
                                 R.from_matrix(R_r_1).as_euler("xyz"),
                                 ], axis=-1).astype(np.float32)
        X2 = np.concatenate([*[xyz_l_0[:, i] * mul for i in range(xyz_l_0.shape[1])],
                             xyz_l_1 * mul,
                             ], axis=-1)
    X3 = cp_0.astype(np.float32) * mul
    X3d = np.concatenate([X3[:, :, :1], np.diff(X3, axis=-2)], axis=-2)
    X3 = np.concatenate([X3, X3d], axis=-1)
    X3 = X3.reshape((X3.shape[0], -1, 6))
    Y = cp_1.astype(np.float32) * mul

    if augment:
        if diff:
            bs = Y.shape[0]
            identity = np.tile(np.eye(3)[np.newaxis], (bs, 1, 1))
            if rot == "quat":
                X1aug = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_quat() for i in range(R_l_0.shape[1])],
                                        R.from_matrix(identity).as_quat(),
                                        *[R.from_matrix(R_r_0[:, i]).as_quat() for i in range(R_r_0.shape[1])],
                                        R.from_matrix(identity).as_quat(),
                                        ], axis=-1).astype(np.float32)
            elif rot == "rotmat":
                X1aug = np.concatenate([*[R_l_0[:, i].reshape((-1, 9)) for i in range(R_l_0.shape[1])],
                                        identity.reshape((-1, 9)),
                                        *[R_r_0[:, i].reshape((-1, 9)) for i in range(R_r_0.shape[1])],
                                        identity.reshape((-1, 9)),
                                        ], axis=-1).astype(np.float32)
            elif rot == "rotvec":
                X1aug = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_rotvec() for i in range(R_l_0.shape[1])],
                                        R.from_matrix(identity).as_rotvec(),
                                        *[R.from_matrix(R_r_0[:, i]).as_rotvec() for i in range(R_r_0.shape[1])],
                                        R.from_matrix(identity).as_rotvec(),
                                        ], axis=-1).astype(np.float32)
            elif rot == "euler":
                X1aug = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_euler("xyz") for i in range(R_l_0.shape[1])],
                                        R.from_matrix(identity).as_euler("xyz"),
                                        *[R.from_matrix(R_r_0[:, i]).as_euler("xyz") for i in range(R_r_0.shape[1])],
                                        R.from_matrix(identity).as_euler("xyz"),
                                        ], axis=-1).astype(np.float32)
            X2aug = np.concatenate([*[xyz_l_0[:, i] * mul for i in range(xyz_l_0.shape[1])],
                                    np.zeros_like(xyz_l_1) * mul,
                                    ], axis=-1).astype(np.float32)
        else:
            if rot == "quat":
                X1aug = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_quat() for i in range(R_l_0.shape[1])],
                                        R.from_matrix(R_l_0[:, -1]).as_quat(),
                                        *[R.from_matrix(R_r_0[:, i]).as_quat() for i in range(R_r_0.shape[1])],
                                        R.from_matrix(R_r_0[:, -1]).as_quat(),
                                        ], axis=-1).astype(np.float32)
            elif rot == "rotmat":
                X1aug = np.concatenate([*[R_l_0[:, i].reshape((-1, 9)) for i in range(R_l_0.shape[1])],
                                        R_l_0[:, -1].reshape((-1, 9)),
                                        *[R_r_0[:, i].reshape((-1, 9)) for i in range(R_r_0.shape[1])],
                                        R_r_0[:, -1].reshape((-1, 9)),
                                        ], axis=-1).astype(np.float32)
            elif rot == "rotvec":
                X1aug = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_rotvec() for i in range(R_l_0.shape[1])],
                                        R.from_matrix(R_l_0[:, -1]).as_rotvec(),
                                        *[R.from_matrix(R_r_0[:, i]).as_rotvec() for i in range(R_r_0.shape[1])],
                                        R.from_matrix(R_r_0[:, -1]).as_rotvec(),
                                        ], axis=-1).astype(np.float32)
            elif rot == "euler":
                X1aug = np.concatenate([*[R.from_matrix(R_l_0[:, i]).as_euler("xyz") for i in range(R_l_0.shape[1])],
                                        R.from_matrix(R_l_0[:, -1]).as_euler("xyz"),
                                        *[R.from_matrix(R_r_0[:, i]).as_euler("xyz") for i in range(R_r_0.shape[1])],
                                        R.from_matrix(R_r_0[:, -1]).as_euler("xyz"),
                                        ], axis=-1).astype(np.float32)
            X2aug = np.concatenate([*[xyz_l_0[:, i] * mul for i in range(xyz_l_0.shape[1])],
                                    xyz_l_0[:, -1] * mul,
                                    ], axis=-1)

        X3aug = cp_0.astype(np.float32) * mul
        X3daug = np.concatenate([X3aug[:, :, :1], np.diff(X3aug, axis=-2)], axis=-2)
        X3aug = np.concatenate([X3aug, X3daug], axis=-1)
        X3aug = X3aug.reshape((X3aug.shape[0], -1, 6))
        Yaug = cp_0[:, -1].astype(np.float32) * mul

        X1 = np.concatenate([X1, X1aug], axis=0)
        X2 = np.concatenate([X2, X2aug], axis=0)
        X3 = np.concatenate([X3, X3aug], axis=0)
        Y = np.concatenate([Y, Yaug], axis=0)

    X1 = X1.astype(np.float32)
    X2 = X2.astype(np.float32)
    X3 = X3.astype(np.float32)
    Y = Y.astype(np.float32)

    ds_size = data.shape[0]
    ds = tf.data.Dataset.from_tensor_slices({"x1": X1, "x2": X2, "x3": X3, "y": Y})
    return ds, ds_size, X1, X2, X3, Y  # , data


def unpack_rotation(rotation, n=2):
    size = rotation.shape[-1]
    step = int(size / ((n + 1) * 2))
    R_l_0 = rotation[:, :n*step]
    R_l_1 = rotation[:, n*step:(n+1)*step]
    R_r_0 = rotation[:, (n+1)*step:(2*n+1)*step]
    R_r_1 = rotation[:, (2*n+1)*step:]
    return R_l_0, R_l_1, R_r_0, R_r_1
