import tensorflow as tf
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from utils.constants import BSplineConstants

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data["x1"], data["x2"], data["x3"], data["y"])
            pbar.update(batch_size)


def prepare_dataset(path, n=0):
    data = np.loadtxt(path, delimiter='\t').astype(np.float32)
    if n > 0:
        idx = np.random.randint(0, data.shape[0] - 1, n)
        data = data[idx]
    ncp = BSplineConstants.ncp
    R_l_0 = data[:, :9].reshape((-1, 3, 3))
    R_r_0 = data[:, 9:18].reshape((-1, 3, 3))
    xyz_l_0 = data[:, 18:21]
    cp_0 = data[:, 21:21 + ncp].reshape((-1, BSplineConstants.n, BSplineConstants.dim))
    R_l_1 = data[:, 21 + ncp:30 + ncp].reshape((-1, 3, 3))
    R_r_1 = data[:, 30 + ncp:39 + ncp].reshape((-1, 3, 3))
    xyz_l_1 = data[:, 39 + ncp:42 + ncp]
    cp_1 = data[:, 42 + ncp:42 + 2 * ncp].reshape((-1, BSplineConstants.n, BSplineConstants.dim))
    diff_R_l = np.transpose(R_l_0, (0, 2, 1)) @ R_l_1
    diff_R_r = np.transpose(R_r_0, (0, 2, 1)) @ R_r_1
    mul = 1.
    # X1 = np.concatenate([R_l_0.reshape((-1, 9)), diff_R_l.reshape((-1, 9)),
    #                     R_r_0.reshape((-1, 9)), diff_R_r.reshape((-1, 9)),
    #                     ], axis=-1)
    # X2 = np.concatenate([xyz_l_0 * mul,
    #                     (xyz_l_1 - xyz_l_0) * mul,
    #                     ], axis=-1)
    #X1 = np.concatenate([R_l_0.reshape((-1, 9)), R_l_1.reshape((-1, 9)),
    #                     R_r_0.reshape((-1, 9)), R_r_1.reshape((-1, 9)),
    #                     ], axis=-1)
    #a = R.from_matrix(R_l_0)
    X1 = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(R_l_1).as_quat(),
                         R.from_matrix(R_r_0).as_quat(), R.from_matrix(R_r_1).as_quat(),
                         ], axis=-1).astype(np.float32)
    X2 = np.concatenate([xyz_l_0 * mul,
                         xyz_l_1 * mul,
                         ], axis=-1).astype(np.float32)
    X3 = cp_0.astype(np.float32) * mul
    Y = cp_1.astype(np.float32) * mul
    ds_size = data.shape[0]
    ds = tf.data.Dataset.from_tensor_slices({"x1": X1, "x2": X2, "x3": X3, "y": Y})
    return ds, ds_size, X1, X2, X3, Y


def whiten(t, v):
    m = np.mean(t, axis=0)
    s = np.std(t, axis=0)
    t = (t - m) / (s + 1e-8)
    v = (v - m) / (s + 1e-8)
    return t, v, m, s


def mix_datasets(tX1, tX2, tX3, tY, vX1, vX2, vX3, vY):
    X1 = np.concatenate([tX1, vX1], axis=0)
    X2 = np.concatenate([tX2, vX2], axis=0)
    X3 = np.concatenate([tX3, vX3], axis=0)
    Y = np.concatenate([tY, vY], axis=0)

    N = Y.shape[0]
    train_size = int(0.8 * N)
    val_size = N - train_size
    idx = np.arange(N)
    np.random.shuffle(idx)

    tX1 = X1[idx[:train_size]]
    tX2 = X2[idx[:train_size]]
    tX3 = X3[idx[:train_size]]
    tY = Y[idx[:train_size]]
    vX1 = X1[idx[train_size:]]
    vX2 = X2[idx[train_size:]]
    vX3 = X3[idx[train_size:]]
    vY = Y[idx[train_size:]]

    # dttX = np.linalg.norm(tX[np.newaxis] - tX[:, np.newaxis], axis=-1)
    # dtvX = np.linalg.norm(tX[np.newaxis] - vX[:, np.newaxis], axis=-1)
    return tX1, tX2, tX3, tY, vX1, vX2, vX3, vY, train_size, val_size


def compute_ds_stats(ds):
    x1s = []
    x2s = []
    x3s = []
    ys = []
    for data in ds:
        x1s.append(data["x1"])
        x2s.append(data["x2"])
        x3s.append(data["x3"])
        ys.append(data["y"])
    x1s = np.concatenate(x1s, axis=0)
    x2s = np.concatenate(x2s, axis=0)
    x3s = np.concatenate(x3s, axis=0)
    ys = np.concatenate(ys, axis=0)
    ds_stats = dict(
        m1=np.mean(x1s, axis=0, keepdims=True),
        m2=np.mean(x2s, axis=0, keepdims=True),
        m3=np.mean(x3s, axis=0, keepdims=True),
        my=np.mean(ys, axis=0, keepdims=True),
        s1=np.std(x1s, axis=0, keepdims=True),
        s2=np.std(x2s, axis=0, keepdims=True),
        s3=np.std(x3s, axis=0, keepdims=True),
        sy=np.std(ys, axis=0, keepdims=True),
    )
    return ds_stats


def whitening(x1, x2, x3, y, ds_stats):
    x1 = (x1 - ds_stats["m1"]) / (ds_stats["s1"] + 1e-8)
    x2 = (x2 - ds_stats["m2"]) / (ds_stats["s2"] + 1e-8)
    x3 = (x3 - ds_stats["m3"]) / (ds_stats["s3"] + 1e-8)
    y = (y - ds_stats["my"]) / (ds_stats["sy"] + 1e-8)
    return x1, x2, x3, y


def unpack_rotation(rotation):
    #R_l_0 = rotation[:, :9]
    #R_l_1 = rotation[:, 9:18]
    #R_r_0 = rotation[:, 18:27]
    #R_r_1 = rotation[:, 27:]
    R_l_0 = rotation[:, :4]
    R_l_1 = rotation[:, 4:8]
    R_r_0 = rotation[:, 8:12]
    R_r_1 = rotation[:, 12:]
    return R_l_0, R_l_1, R_r_0, R_r_1


def unpack_translation(translation):
    t_l_0 = translation[:, :3]
    t_r_0 = translation[:, 3:]
    return t_l_0, t_r_0