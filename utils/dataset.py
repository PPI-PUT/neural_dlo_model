import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.constants import BSplineConstants

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data["x1"], data["x2"],data["x3"], data["y"])
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
    cp_0 = data[:, 21:21 + ncp]  # .reshape((-1, ncp / 3, 3))
    R_l_1 = data[:, 21 + ncp:30 + ncp].reshape((-1, 3, 3))
    R_r_1 = data[:, 30 + ncp:39 + ncp].reshape((-1, 3, 3))
    xyz_l_1 = data[:, 39 + ncp:42 + ncp]
    cp_1 = data[:, 42 + ncp:42 + 2 * ncp]  # .reshape((-1, ncp / 3, 3))
    diff_R_l = np.transpose(R_l_0, (0, 2, 1)) @ R_l_1
    diff_R_r = np.transpose(R_r_0, (0, 2, 1)) @ R_r_1
    mul = 1.
    X = np.concatenate([R_l_0.reshape((-1, 9)), diff_R_l.reshape((-1, 9)),
                        R_r_0.reshape((-1, 9)), diff_R_r.reshape((-1, 9)),
                        xyz_l_0 * mul,
                        (xyz_l_1 - xyz_l_0) * mul,
                        cp_0 * mul,
                        ], axis=-1)
    X1 = np.concatenate([R_l_0.reshape((-1, 9)), diff_R_l.reshape((-1, 9)),
                        R_r_0.reshape((-1, 9)), diff_R_r.reshape((-1, 9)),
                        ], axis=-1)
    X2 = np.concatenate([xyz_l_0 * mul,
                         (xyz_l_1 - xyz_l_0) * mul,
                         ], axis=-1)
    X3 = cp_0 * mul
    Y = (cp_1 - cp_0) * mul
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

    #dttX = np.linalg.norm(tX[np.newaxis] - tX[:, np.newaxis], axis=-1)
    #dtvX = np.linalg.norm(tX[np.newaxis] - vX[:, np.newaxis], axis=-1)
    return tX1, tX2, tX3, tY, vX1, vX2, vX3, vY, train_size, val_size

