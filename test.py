import os

from losses.cable import CableBSplineLoss
from models.separated_neural_predictor import SeparatedNeuralPredictor
from utils.bspline import BSpline
from utils.constants import BSplineConstants

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataset import _ds, prepare_dataset, whiten, mix_datasets
from utils.execution import ExperimentHandler
from models.basic_neural_predictor import BasicNeuralPredictor

np.random.seed(444)

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
#def mix_datasets(tX, tY, vX, vY):
#    X = np.concatenate([tX, vX], axis=0)
#    Y = np.concatenate([tY, vY], axis=0)
#
#    N = X.shape[0]
#    train_size = int(0.8 * N)
#    val_size = N - train_size
#    idx = np.arange(N)
#    np.random.shuffle(idx)
#
#    tX = X[idx[:train_size]]
#    tY = Y[idx[:train_size]]
#    vX = X[idx[train_size:]]
#    vY = Y[idx[train_size:]]
#
#    dttX = np.linalg.norm(tX[np.newaxis] - tX[:, np.newaxis], axis=-1)
#    dtvX = np.linalg.norm(tX[np.newaxis] - vX[:, np.newaxis], axis=-1)
#    return tX, tY, vX, vY, train_size, val_size

class args:
    batch_size = 1
    #batch_size = 16
    #dataset_path = "./data/prepared_datasets/xy_bs/train.tsv"
    #dataset_path = "./data/prepared_datasets/xy_pts256/train.tsv"
    #dataset_path = "./data/prepared_datasets/xy_pts256/val.tsv"
    #dataset_path = "./data/prepared_datasets/xy_bs_keq/train.tsv"
    #dataset_path = "./data/prepared_datasets/yz_big_keq/val.tsv"
    #dataset_path = "./data/prepared_datasets/yz_big_keq_sorted/val.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy/val.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_02_09__10_30/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_09__13_30/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_09__14_00/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_09__14_30_dxyzl5cm/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_09__15_00_dxyzlg5cm/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__09_15/train.tsv"
    dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__10_40/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_all2all_02_10__14_00/train.tsv"
    #dataset_path = "./data/prepared_datasets/xy/val.tsv"


train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset(args.dataset_path)  # , n=10)
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset(args.dataset_path.replace("train", "val"))


tX1, vX1, m1, s1 = whiten(tX1, vX1)
tX2, vX2, m2, s2 = whiten(tX2, vX2)
tX3, vX3, m3, s3 = whiten(tX3, vX3)
tY, vY, my, sy = whiten(tY, vY)

train_ds = tf.data.Dataset.from_tensor_slices({"x1": tX1, "x2": tX2, "x3": tX3, "y": tY})
val_ds = tf.data.Dataset.from_tensor_slices({"x1": vX1, "x2": vX2, "x3": vX3, "y": vY})

#tX1, tX2, tX3, tY, vX1, vX2, vX3, vY, train_size, val_size = mix_datasets(tX1, tX2, tX3, tY, vX1, vX2, vX3, vY)
#train_ds = tf.data.Dataset.from_tensor_slices({"x1": tX1, "x2": tX2, "x3": tX3, "y": tY})
#val_ds = tf.data.Dataset.from_tensor_slices({"x1": vX1, "x2": vX2, "x3": vX3, "y": vY})

#ds, size = val_ds, val_size
ds, size = train_ds, train_size

#m1 = m2 = m3 = my = 0.
#s1 = s2 = s3 = sy = 1.

#train_ds, train_size, tX, tY = prepare_dataset(args.dataset_path)
#val_ds, val_size, vX, vY = prepare_dataset(args.dataset_path.replace("val", "train"))
#
#tX, tY, vX, vY, _, size = mix_datasets(tX, tY, vX, vY)
#
#ds = tf.data.Dataset.from_tensor_slices({"x": vX, "y": vY})

#loss = tf.keras.losses.mean_squared_error
#loss = tf.keras.losses.mean_absolute_error
#def loss(gt, pred):
#    gt = tf.reshape(gt, (-1, BSplineConstants.n, BSplineConstants.dim))
#    pred = tf.reshape(pred, (-1, BSplineConstants.n, BSplineConstants.dim))
#    diff = tf.reduce_sum(tf.square(gt[:, tf.newaxis] - pred[:, :, tf.newaxis]), axis=-1)
#    loss_1 = tf.reduce_min(diff, axis=-1)
#    loss_2 = tf.reduce_min(diff, axis=-2)
#    return tf.reduce_mean(loss_1 + loss_2, axis=-1)
loss = CableBSplineLoss()

#model = BasicNeuralPredictor()
model = SeparatedNeuralPredictor()

ckpt = tf.train.Checkpoint(model=model)
#ckpt.restore("./trainings/short_ds_bs16_lr5em5_run2/checkpoints/best-115")
#ckpt.restore("./trainings/xy_bs8_lr5em5_l5x256_scaledincms_regloss1em3_fixedmul/checkpoints/best-521")
#ckpt.restore("./trainings/xy_bs8_lr5em5_l5x256_scaledincms_regloss1em4_cleaned/checkpoints/best-195")
#ckpt.restore("./trainings/xy_bs8_lr5em5_l5x256_scaledincms_regloss1em4_cleaned_pts256_lossall2all/checkpoints/last_n-308")
#ckpt.restore("./trainings//xy_bs8_lr5em5_l5x256_scaledincms_regloss1em4_cleaned_bs_keq/checkpoints/last_n-155")
#ckpt.restore("./trainings/yz_bs8_lr5em5_l5x256_cm_regloss1em4_bs_keq/checkpoints/last_n-11")
#ckpt.restore("./trainings/yz_bs16_lr5em5_l5x256_cm_regloss1em4_bs_keq_dsnotseparated/checkpoints/best-152")
#ckpt.restore("./trainings/yz_bs16_lr5em5_l5x256_cm_regloss1em4_bs_keq_dsseparatedbutsorted/checkpoints/best-196")
#ckpt.restore("./trainings/xyzrpy_02_09__09_25_bs16_lr5em5_l5x256_cm_regloss1em4_bs_keq/checkpoints/last_n-33")
#ckpt.restore("./trainings/xyzrpy_02_09__10_30_bs16_lr5em5_l5x256_m_regloss1em4_bs_keq/checkpoints/best-508")
#ckpt.restore("./trainings/xyzrpy_02_09__10_30_bs16_lr5em5_separated_l1x256_l3x256_dm_regloss1em4_bs_keq/checkpoints/best-344")
#ckpt.restore("./trainings/xyzrpy_02_09__10_30_bs16_lr5em5_separated_l1x256_l3x256_diffs_mwhithened_regloss1em4_bs_keq/checkpoints/best-60")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_09__13_30_bs64_lr5em5_l5x256_mwhithened_regloss1em4_bs_keq/checkpoints/last_n-15")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_09__14_00_bs64_lr5em4_l5x256_mwhithened_regloss1em4_bs_keq/checkpoints/last_n-2")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_09__14_30_dxyzl5cm_bs64_lr5em4_l5x256_mwhithened_regloss1em4_bs_keq/checkpoints/best-6")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_09__14_30_dxyzl5cm_bs64_lr5em6_l5x256_mwhithened_regloss1em4_bs_keq/checkpoints/best-257")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_09__15_00_dxyzlg5cm_bs64_lr5em5_l5x256_mwhithened_regloss1em4_bs_keq/checkpoints/best-8")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_09__15_00_dxyzlg5cm_bs64_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq/checkpoints/best-47")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_10__09_15_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsmixed/checkpoints/best-59")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_10__10_40_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed/checkpoints/best-25")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_10__10_40_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed_absloss/checkpoints/best-21")
#ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_10__10_40_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed_absloss_dcurvloss1em2/checkpoints/best-21")
ckpt.restore("./trainings/xyzrpy_episodic_all2all_02_10__10_40_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed_absloss_withened/checkpoints/last_n-40")
#ckpt.restore("./trainings/xyzrpy_all2all_02_10__14_00_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed_absloss_withened/checkpoints/best-24")


bsp = BSpline(25, 3)

dataset_epoch = ds.shuffle(size)
dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
epoch_loss = []
#bsp = BSpline()
for i, rot, trans, cable, y_gt in _ds('Train', dataset_epoch, size, 0, args.batch_size):
    #y_base = x[:, -BSplineConstants.ncp:]
    y_base = cable
    y_pred = model(rot, trans, cable)
    model_loss = loss(y_gt + y_base, y_pred + y_base)
    print("MODEL:", model_loss[0])
    ncp = BSplineConstants.ncp
    x = tf.concat([rot * s1 + m1, trans * s2 + m2, cable * s3 + m3], axis=-1)

    k = 0
    x = x.numpy()
    y_gt = y_gt.numpy() * sy + my
    y_pred = y_pred.numpy() * sy + my

    R_l_0 = x[k, :9].reshape((3, 3))
    diffR_l = x[k, 9:18].reshape((3, 3))
    R_r_0 = x[k, 18:27].reshape((3, 3))
    diffR_r = x[k, 27:36].reshape((3, 3))
    xyz_l_0 = x[k, 36:39]
    diffxyz_l = x[k, 39:42]
    cp0 = x[k, 42:42+ncp]
    cp1 = y_gt + cp0

    model_loss = loss(cp0, cp1)
    #print("DIFF:", model_loss)
    cp0 = cp0.reshape((BSplineConstants.n, BSplineConstants.dim))
    cp1 = cp1.reshape((BSplineConstants.n, BSplineConstants.dim))

    cp_pred = y_pred.reshape((BSplineConstants.n, BSplineConstants.dim)) + cp0
    cp_gt = y_gt.reshape((BSplineConstants.n, BSplineConstants.dim)) + cp0

    xl = -0.2
    xh = 0.5
    yl = -0.3
    yh = 0.4
    zl = -0.35
    zh = 0.35
    plt.subplot(221)
    plt.xlim(xl, xh)
    plt.ylim(yl, yh)
    plt.plot(cp_gt[:, 1], cp_gt[:, 2], 'rx')
    plt.plot(cp_pred[:, 1], cp_pred[:, 2], 'bo')
    plt.plot(cp0[:, 1], cp0[:, 2], 'g^')
    plt.subplot(223)
    plt.xlim(zl, zh)
    plt.ylim(yl, yh)
    plt.plot(cp_gt[:, 0], cp_gt[:, 1], 'rx')
    plt.plot(cp_pred[:, 0], cp_pred[:, 1], 'bo')
    plt.plot(cp0[:, 0], cp0[:, 1], 'g^')
    v_pred = (bsp.N[0] @ cp_pred)
    v_gt = (bsp.N[0] @ cp_gt)
    v_base = (bsp.N[0] @ cp0)
    plt.subplot(222)
    plt.xlim(xl, xh)
    plt.ylim(yl, yh)
    plt.plot(v_pred[:, 1], v_pred[:, 2], label="pred")
    plt.plot(v_gt[:, 1], v_gt[:, 2], label="gt")
    plt.plot(v_base[:, 1], v_base[:, 2], label="base")
    plt.subplot(224)
    plt.xlim(zl, zh)
    plt.ylim(yl, yh)
    plt.plot(v_pred[:, 0], v_pred[:, 1], label="pred")
    plt.plot(v_gt[:, 0], v_gt[:, 1], label="gt")
    plt.plot(v_base[:, 0], v_base[:, 1], label="base")
    plt.xlim(-0.1, 0.7)
    plt.ylim(-0.4, 0.4)
    plt.legend()
    plt.show()

    #k = 0
    #x = x.numpy()
    #R_l_0 = x[k, :9].reshape((3, 3))
    #R_r_0 = x[k, 9:18].reshape((3, 3))
    #xyz_l_0 = x[k, 18:21]
    #cp_0 = x[k, 21:21 + ncp]  # .reshape((-1, ncp / 3, 3))
    #R_l_1 = x[k, 21 + ncp:30 + ncp].reshape((3, 3))
    #R_r_1 = x[k, 30 + ncp:39 + ncp].reshape((3, 3))
    #xyz_l_1 = x[k, 39 + ncp:42 + ncp]
    #cp_1 = x[k, 42 + ncp:42 + 2 * ncp]  # .reshape((-1, ncp / 3, 3))
    #diff_R_l = np.transpose(R_l_0, (1, 0)) @ R_l_1
    #diff_R_r = np.transpose(R_r_0, (1, 0)) @ R_r_1
    #Y = cp_1 - cp_0

    epoch_loss.append(model_loss)

epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
print(epoch_loss)
