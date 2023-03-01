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


def prepare_dataset(path, n=0, augment=False):
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

    if augment:
        #X1aug = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(R_l_1).as_quat(),
        #                     R.from_matrix(R_r_0).as_quat(), R.from_matrix(R_r_1).as_quat(),
        #                     ], axis=-1).astype(np.float32)
        #X2aug = np.concatenate([xyz_l_0 * mul,
        #                     xyz_l_1 * mul,
        #                     ], axis=-1).astype(np.float32)
        #X3aug = cp_0.astype(np.float32) * mul
        #Yaug = cp_1.astype(np.float32) * mul
        X1aug = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(R_l_0).as_quat(),
                                R.from_matrix(R_r_0).as_quat(), R.from_matrix(R_r_0).as_quat(),
                                ], axis=-1).astype(np.float32)
        X2aug = np.concatenate([xyz_l_0 * mul,
                                xyz_l_0 * mul,
                                ], axis=-1).astype(np.float32)
        X3aug = cp_0.astype(np.float32) * mul
        Yaug = cp_0.astype(np.float32) * mul
        #zdev = np.eye(BSplineConstants.n)[np.random.randint(0, 16, (Yaug.shape[0]))]
        #zdev = 0.03 * np.random.random() * np.stack([np.zeros_like(zdev), np.zeros_like(zdev), zdev], axis=-1)
        #X3aug += zdev
        #zdev = np.eye(BSplineConstants.n)[np.random.randint(0, 16, (Yaug.shape[0]))]
        #zdev = 0.03 * np.random.random() * np.stack([np.zeros_like(zdev), np.zeros_like(zdev), zdev], axis=-1)
        #Yaug += zdev

        X1 = np.concatenate([X1, X1aug], axis=0)
        X2 = np.concatenate([X2, X2aug], axis=0)
        X3 = np.concatenate([X3, X3aug], axis=0)
        Y = np.concatenate([Y, Yaug], axis=0)

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
    x1s = np.stack([data["x1"] for data in ds], axis=0)
    x2s = np.stack([data["x2"] for data in ds], axis=0)
    x3s = np.stack([data["x3"] for data in ds], axis=0)
    ys = np.stack([data["y"] for data in ds], axis=0)
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
    size = rotation.shape[-1]
    step = int(size / 4)
    R_l_0 = rotation[:, :step]
    R_l_1 = rotation[:, step:2*step]
    R_r_0 = rotation[:, 2*step:3*step]
    R_r_1 = rotation[:, 3*step:]
    return R_l_0, R_l_1, R_r_0, R_r_1


def unpack_translation(translation):
    t_l_0 = translation[:, :3]
    t_r_0 = translation[:, 3:]
    return t_l_0, t_r_0


def prepare_dataset_cond(path, n=0, quat=False, diff=False, augment=False):
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
    if diff:
        if quat:
            X1 = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(diff_R_l).as_quat(),
                                 R.from_matrix(R_r_0).as_quat(), R.from_matrix(diff_R_r).as_quat(),
                                 ], axis=-1).astype(np.float32)
        else:
            X1 = np.concatenate([R_l_0.reshape((-1, 9)), diff_R_l.reshape((-1, 9)),
                                 R_r_0.reshape((-1, 9)), diff_R_r.reshape((-1, 9)),
                                 ], axis=-1)
        X2 = np.concatenate([xyz_l_0 * mul,
                             (xyz_l_1 - xyz_l_0) * mul,
                             ], axis=-1)
    else:
        if quat:
            X1 = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(R_l_1).as_quat(),
                                 R.from_matrix(R_r_0).as_quat(), R.from_matrix(R_r_1).as_quat(),
                                 ], axis=-1).astype(np.float32)
        else:
            X1 = np.concatenate([R_l_0.reshape((-1, 9)), R_l_1.reshape((-1, 9)),
                                 R_r_0.reshape((-1, 9)), R_r_1.reshape((-1, 9)),
                                 ], axis=-1)
        X2 = np.concatenate([xyz_l_0 * mul,
                             xyz_l_1 * mul,
                             ], axis=-1).astype(np.float32)
    X3 = cp_0.astype(np.float32) * mul
    Y = cp_1.astype(np.float32) * mul

    if augment:
        if diff:
            bs = Y.shape[0]
            identity = np.tile(np.eye(3)[np.newaxis], (bs, 1, 1))
            if quat:
                X1aug = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(identity).as_quat(),
                                        R.from_matrix(R_r_0).as_quat(), R.from_matrix(identity).as_quat(),
                                        ], axis=-1).astype(np.float32)
            else:
                X1aug = np.concatenate([R_l_0.reshape((-1, 9)), identity.reshape((-1, 9)),
                                        R_r_0.reshape((-1, 9)), identity.reshape((-1, 9)),
                                        ], axis=-1)
            X2aug = np.concatenate([xyz_l_0 * mul,
                                    np.zeros_like(xyz_l_1) * mul,
                                    ], axis=-1).astype(np.float32)
        else:
            if quat:
                X1aug = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(R_l_0).as_quat(),
                                        R.from_matrix(R_r_0).as_quat(), R.from_matrix(R_r_0).as_quat(),
                                        ], axis=-1).astype(np.float32)
            else:
                X1aug = np.concatenate([R_l_0.reshape((-1, 9)), R_l_0.reshape((-1, 9)),
                                        R_r_0.reshape((-1, 9)), R_r_0.reshape((-1, 9)),
                                        ], axis=-1)
            X2aug = np.concatenate([xyz_l_0 * mul,
                                    xyz_l_0 * mul,
                                    ], axis=-1).astype(np.float32)

        X3aug = cp_0.astype(np.float32) * mul
        Yaug = cp_0.astype(np.float32) * mul

        X1 = np.concatenate([X1, X1aug], axis=0)
        X2 = np.concatenate([X2, X2aug], axis=0)
        X3 = np.concatenate([X3, X3aug], axis=0)
        Y = np.concatenate([Y, Yaug], axis=0)

    ds_size = data.shape[0]
    ds = tf.data.Dataset.from_tensor_slices({"x1": X1, "x2": X2, "x3": X3, "y": Y})
    return ds, ds_size, X1, X2, X3, Y


def prepare_dataset_cond_ref(path, n=0, quat=False, diff=False, augment=False):
    data = np.loadtxt(path, delimiter='\t').astype(np.float32)
    if n > 0:
        idx = np.random.randint(0, data.shape[0] - 1, n)
        data = data[idx]
    ncp = BSplineConstants.ncp
    npts = ncp
    R_l_0 = data[:, :9].reshape((-1, 3, 3))
    R_r_0 = data[:, 9:18].reshape((-1, 3, 3))
    xyz_l_0 = data[:, 18:21]
    cp_0 = data[:, 21:21 + ncp].reshape((-1, BSplineConstants.n, BSplineConstants.dim))
    pts_0 = data[:, 21 + ncp:21 + ncp + npts].reshape((-1, BSplineConstants.n, BSplineConstants.dim))
    R_l_1 = data[:, 21 + ncp + npts:30 + ncp + npts].reshape((-1, 3, 3))
    R_r_1 = data[:, 30 + ncp + npts:39 + ncp + npts].reshape((-1, 3, 3))
    xyz_l_1 = data[:, 39 + ncp + npts:42 + ncp + npts]
    cp_1 = data[:, 42 + ncp + npts:42 + 2 * ncp + npts].reshape((-1, BSplineConstants.n, BSplineConstants.dim))
    pts_1 = data[:, 42 + 2 * ncp + npts: 42 + 2 * ncp + 2 * npts].reshape((-1, BSplineConstants.n, BSplineConstants.dim))
    diff_R_l = np.transpose(R_l_0, (0, 2, 1)) @ R_l_1
    diff_R_r = np.transpose(R_r_0, (0, 2, 1)) @ R_r_1
    mul = 1.
    if diff:
        if quat:
            X1 = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(diff_R_l).as_quat(),
                                 R.from_matrix(R_r_0).as_quat(), R.from_matrix(diff_R_r).as_quat(),
                                 ], axis=-1).astype(np.float32)
        else:
            X1 = np.concatenate([R_l_0.reshape((-1, 9)), diff_R_l.reshape((-1, 9)),
                                 R_r_0.reshape((-1, 9)), diff_R_r.reshape((-1, 9)),
                                 ], axis=-1)
        X2 = np.concatenate([xyz_l_0 * mul,
                             (xyz_l_1 - xyz_l_0) * mul,
                             ], axis=-1)
    else:
        if quat:
            X1 = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(R_l_1).as_quat(),
                                 R.from_matrix(R_r_0).as_quat(), R.from_matrix(R_r_1).as_quat(),
                                 ], axis=-1).astype(np.float32)
        else:
            X1 = np.concatenate([R_l_0.reshape((-1, 9)), R_l_1.reshape((-1, 9)),
                                 R_r_0.reshape((-1, 9)), R_r_1.reshape((-1, 9)),
                                 ], axis=-1)
        X2 = np.concatenate([xyz_l_0 * mul,
                             xyz_l_1 * mul,
                             ], axis=-1).astype(np.float32)
    X3 = cp_0.astype(np.float32) * mul
    X4 = pts_0.astype(np.float32) * mul
    Y = cp_1.astype(np.float32) * mul
    Y1 = pts_1.astype(np.float32) * mul

    if augment:
        if diff:
            bs = Y.shape[0]
            identity = np.tile(np.eye(3)[np.newaxis], (bs, 1, 1))
            if quat:
                X1aug = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(identity).as_quat(),
                                        R.from_matrix(R_r_0).as_quat(), R.from_matrix(identity).as_quat(),
                                        ], axis=-1).astype(np.float32)
            else:
                X1aug = np.concatenate([R_l_0.reshape((-1, 9)), identity.reshape((-1, 9)),
                                        R_r_0.reshape((-1, 9)), identity.reshape((-1, 9)),
                                        ], axis=-1)
            X2aug = np.concatenate([xyz_l_0 * mul,
                                    np.zeros_like(xyz_l_1) * mul,
                                    ], axis=-1).astype(np.float32)
        else:
            if quat:
                X1aug = np.concatenate([R.from_matrix(R_l_0).as_quat(), R.from_matrix(R_l_0).as_quat(),
                                        R.from_matrix(R_r_0).as_quat(), R.from_matrix(R_r_0).as_quat(),
                                        ], axis=-1).astype(np.float32)
            else:
                X1aug = np.concatenate([R_l_0.reshape((-1, 9)), R_l_0.reshape((-1, 9)),
                                        R_r_0.reshape((-1, 9)), R_r_0.reshape((-1, 9)),
                                        ], axis=-1)
            X2aug = np.concatenate([xyz_l_0 * mul,
                                    xyz_l_0 * mul,
                                    ], axis=-1).astype(np.float32)

        X3aug = cp_0.astype(np.float32) * mul
        X4aug = pts_0.astype(np.float32) * mul
        Yaug = cp_0.astype(np.float32) * mul
        Y1aug = pts_0.astype(np.float32) * mul

        X1 = np.concatenate([X1, X1aug], axis=0)
        X2 = np.concatenate([X2, X2aug], axis=0)
        X3 = np.concatenate([X3, X3aug], axis=0)
        X4 = np.concatenate([X4, X4aug], axis=0)
        Y = np.concatenate([Y, Yaug], axis=0)
        Y1 = np.concatenate([Y1, Y1aug], axis=0)

    ds_size = data.shape[0]
    ds = tf.data.Dataset.from_tensor_slices({"x1": X1, "x2": X2, "x3": X3, 'x4': X4, "y": Y, "y1": Y1})
    return ds, ds_size, X1, X2, X3, X4, Y, Y1


def compute_ds_stats_ref(ds):
    x1s = []
    x2s = []
    x3s = []
    ys = []
    y1s = []
    for data in ds:
        x1s.append(data["x1"])
        x2s.append(data["x2"])
        x3s.append(data["x3"])
        ys.append(data["y"])
        y1s.append(data["y1"])
    x1s = np.concatenate(x1s, axis=0)
    x2s = np.concatenate(x2s, axis=0)
    x3s = np.concatenate(x3s, axis=0)
    ys = np.concatenate(ys, axis=0)
    y1s = np.concatenate(y1s, axis=0)
    ds_stats = dict(
        m1=np.mean(x1s, axis=0, keepdims=True),
        m2=np.mean(x2s, axis=0, keepdims=True),
        m3=np.mean(x3s, axis=0, keepdims=True),
        my=np.mean(ys, axis=0, keepdims=True),
        my1=np.mean(y1s, axis=0, keepdims=True),
        s1=np.std(x1s, axis=0, keepdims=True),
        s2=np.std(x2s, axis=0, keepdims=True),
        s3=np.std(x3s, axis=0, keepdims=True),
        sy=np.std(ys, axis=0, keepdims=True),
        sy1=np.std(y1s, axis=0, keepdims=True),
    )
    return ds_stats


def whitening_ref(x1, x2, x3, y, y1, ds_stats):
    x1 = (x1 - ds_stats["m1"]) / (ds_stats["s1"] + 1e-8)
    x2 = (x2 - ds_stats["m2"]) / (ds_stats["s2"] + 1e-8)
    x3 = (x3 - ds_stats["m3"]) / (ds_stats["s3"] + 1e-8)
    y = (y - ds_stats["my"]) / (ds_stats["sy"] + 1e-8)
    y1 = (y1 - ds_stats["my1"]) / (ds_stats["sy1"] + 1e-8)
    return x1, x2, x3, y, y1


