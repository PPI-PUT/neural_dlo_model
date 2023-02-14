import os

from losses.cable import CableBSplineLoss, CableSeqLoss
from models.separated_cnn_neural_predictor import SeparatedCNNNeuralPredictor
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


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    # batch_size = 128
    #batch_size = 64
    batch_size = 32
    working_dir = './trainings'
    # out_name = 'short_ds_bs16_lr5em5_l5x1024_scaledincms_regloss1em4'
    # out_name = 'yz_bs16_lr5em5_l5x256_cm_regloss1em4_bs_keq_dsseparatedbutsorted_traindata10'
    #out_name = 'xyzrpy_02_09__10_30_bs16_lr5em5_separated_l1x256_l3x256_diffs_mwhithened_regloss1em4_bs_keq'
    # out_name = 'xyzrpy_02_09__10_30_bs16_lr5em5_l5x256_m_regloss1em4_bs_keq'
    #out_name = 'xyzrpy_episodic_all2all_02_09__15_00_dxyzlg5cm_bs64_lr5em5_l5x256_m_regloss0em4_bs_keq_dsmixed'
    #out_name = 'xyzrpy_episodic_all2all_02_09__15_00_dxyzlg5cm_bs64_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_len_yz_curv_yz'
    #out_name = 'xyzrpy_episodic_all2all_02_10__09_15_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed_length_loss_1em1'
    #out_name = 'xyzrpy_episodic_all2all_02_10__10_00_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed'
    #out_name = 'xyzrpy_episodic_all2all_02_10__10_20_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed'
    out_name = 'xyzrpy_episodic_all2all_02_10__14_00_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed_absloss_withened_'
    #out_name = 'xyzrpy_episodic_all2all_02_10__14_00_bs32_lr5em5_separated_cablecnn_l1x128_l2x128_outputcnn_m_regloss0em4_bs_keq_dsnotmixed_absloss_withened'
    #out_name = 'xyzrpy_all2all_02_10__14_00_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsmixed_absloss_withened'
    #out_name = 'test'
    log_interval = 100
    learning_rate = 5e-5
    l2reg = 0e-4
    len_loss = 0
    acc_loss = 0e-1
    # dataset_path = "./data/prepared_datasets/xy_pts256/train.tsv"
    # dataset_path = "./data/prepared_datasets/xy_bs/train.tsv"
    # dataset_path = "./data/prepared_datasets/yz_big_pts256/train.tsv"
    # dataset_path = "./data/prepared_datasets/yz_big_keq_sorted/train.tsv"
    # dataset_path = "./data/prepared_datasets/xyzrpy/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_02_09__10_30/train.tsv"
    # dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_09__11_00/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_09__13_30/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_09__15_00_dxyzlg5cm/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__09_00/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__09_15/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__10_00/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__10_20/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__10_40/train.tsv"
    dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__14_00/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_all2all_02_10__14_00/train.tsv"
    # dataset_path = "./data/prepared_datasets/yz_big_keq/train.tsv"
    # dataset_path = "./data/prepared_datasets/yz_big_keq_n1000/train.tsv"
    # dataset_path = "./data/prepared_datasets/xy_bs_keq/train.tsv"
    # dataset_path = "./data/prepared_datasets/short_ds/train.tsv"


train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset(args.dataset_path)  # , n=10)
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset(args.dataset_path.replace("train", "val"))

#tX1, tX2, tX3, tY, vX1, vX2, vX3, vY, train_size, val_size = mix_datasets(tX1, tX2, tX3, tY, vX1, vX2, vX3, vY)

tX1, vX1, m1, s1 = whiten(tX1, vX1)
tX2, vX2, m2, s2 = whiten(tX2, vX2)
tX3, vX3, m3, s3 = whiten(tX3, vX3)
tY, vY, my, sy = whiten(tY, vY)

train_ds = tf.data.Dataset.from_tensor_slices({"x1": tX1, "x2": tX2, "x3": tX3, "y": tY})
val_ds = tf.data.Dataset.from_tensor_slices({"x1": vX1, "x2": vX2, "x3": vX3, "y": vY})

bsp = BSpline(25, 3)
bsp32 = BSpline(25, 3, num_T_pts=32)

# dttX = np.abs(tX[np.newaxis] - tX[:, np.newaxis])
# dtvX = np.abs(tX[np.newaxis] - vX[:, np.newaxis])
# nttX = np.linalg.norm(tX[np.newaxis] - tX[:, np.newaxis], axis=-1)
# ntvX = np.linalg.norm(tX[np.newaxis] - vX[:, np.newaxis], axis=-1)
#
# ncp = BSplineConstants.ncp
# vid = 0
# closest_idx = np.argmin(ntvX[vid])
# cpt = tX[closest_idx, 42:42 + ncp]
# cpt = cpt.reshape((BSplineConstants.n, BSplineConstants.dim))
# cpv = vX[vid, 42:42 + ncp]
# cpv = cpv.reshape((BSplineConstants.n, BSplineConstants.dim))
#
# plt.subplot(121)
# plt.plot(cpv[:, 1], cpv[:, 2], 'bo')
# plt.plot(cpt[:, 1], cpt[:, 2], 'g^')
# plt.subplot(122)
# v_t = (bsp.N[0] @ cpt)
# v_v = (bsp.N[0] @ cpv)
# plt.plot(v_t[:, 1], v_t[:, 2], label="train")
# plt.plot(v_v[:, 1], v_v[:, 2], label="val")
# plt.legend()
# plt.show()
#
#tX, tY, vX, vY, train_size, val_size = mix_datasets(tX, tY, vX, vY)

#train_ds = tf.data.Dataset.from_tensor_slices({"x": tX, "y": tY})
#val_ds = tf.data.Dataset.from_tensor_slices({"x": vX, "y": vY})

#tX1, tX2, tX3, tY, vX1, vX2, vX3, vY, train_size, val_size = mix_datasets(tX1, tX2, tX3, tY, vX1, vX2, vX3, vY)
#train_ds = tf.data.Dataset.from_tensor_slices({"x1": tX1, "x2": tX2, "x3": tX3, "y": tY})
#val_ds = tf.data.Dataset.from_tensor_slices({"x1": vX1, "x2": vX2, "x3": vX3, "y": vY})

opt = tf.keras.optimizers.Adam(args.learning_rate)

#loss = tf.keras.losses.mean_squared_error
#def loss(gt, pred, mul=1.):
#   gt = mul * tf.reshape(gt, (-1, BSplineConstants.n, BSplineConstants.dim))
#   pred = mul * tf.reshape(pred, (-1, BSplineConstants.n, BSplineConstants.dim))
#   #plt.plot(pred[0, :, 0], pred[0, :, 1])
#   #plt.plot(gt[0, :, 0], gt[0, :, 1])
#   #plt.show()
#   diff = tf.reduce_sum(tf.square(gt - pred), axis=-1)
#   control_pts_loss = tf.reduce_mean(diff, axis=-1)
#   vgt = bsp.N @ gt
#   dvgt = bsp.dN @ gt
#   ddvgt = bsp.ddN @ gt
#   vpred = bsp.N @ pred
#   dvpred = bsp.dN @ pred
#   ddvpred = bsp.ddN @ pred
#   len_gt = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(vgt[:, 1:, 1:] - vgt[:, :-1, 1:]), axis=-1)), axis=-1)
#   len_pred = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(vpred[:, 1:, 1:] - vpred[:, :-1, 1:]), axis=-1)), axis=-1)
#   length_loss = tf.square(len_gt - len_pred)
#   curv_yz_gt = (dvgt[..., 1] * ddvgt[..., 2] - ddvgt[..., 1] * dvgt[..., 2]) / tf.sqrt(dvgt[..., 1] ** 2 + dvgt[..., 2] ** 2 + 1e-8)
#   curv_yz_pred = (dvpred[..., 1] * ddvpred[..., 2] - ddvpred[..., 1] * dvpred[..., 2]) / tf.sqrt(dvpred[..., 1] ** 2 + dvpred[..., 2] ** 2 + 1e-8)
#   dyz_gt = tf.sqrt(tf.reduce_sum(tf.square(vgt[:, 1:, 1:] - vgt[:, :-1, 1:]), axis=-1))
#   dyz_pred = tf.sqrt(tf.reduce_sum(tf.square(vpred[:, 1:, 1:] - vpred[:, :-1, 1:]), axis=-1))
#   acccurv_yz_gt = tf.reduce_sum(tf.abs(curv_yz_gt[:, :-1]) * dyz_gt, axis=-1)
#   acccurv_yz_pred = tf.reduce_sum(tf.abs(curv_yz_pred[:, :-1]) * dyz_pred, axis=-1)
#   acccurv_yz_loss = tf.square(acccurv_yz_gt - acccurv_yz_pred)
#   return control_pts_loss, length_loss, acccurv_yz_loss

loss = CableBSplineLoss()
loss_pts = CableSeqLoss()


# model = BasicNeuralPredictor()
model = SeparatedNeuralPredictor()
#model = SeparatedCNNNeuralPredictor()

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)
# experiment_handler.restore("./trainings/short_ds_bs16_lr5em5_run2/checkpoints/best-115")
# experiment_handler.restore("./trainings/short_ds_bs16_lr5em5_fixed_small_cable_changes/checkpoints/best-95")
# experiment_handler.restore("./trainings/short_ds_bs16_lr5em5_l5x1024_scaledincms_regloss1em4/checkpoints/last_n-9")
# experiment_handler.restore("./trained_models/last_n-274")
# experiment_handler.restore("./trainings/xy_bs8_lr5em5_l5x256_scaledincms_regloss1em4/checkpoints/last_n-34")
# experiment_handler.restore("./trainings/xy_bs8_lr5em5_l5x256_scaledincms_regloss1em4_fixedmul/checkpoints/last_n-162")

train_step = 0
val_step = 0
best_epoch_loss = 1e10
best_unscaled_epoch_loss = 1e10
mul = 1.
for epoch in range(30000):
    # training
    dataset_epoch = train_ds.shuffle(train_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    prediction_losses = []
    diff_ratios = []
    pts_losses = []
    experiment_handler.log_training()
    for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            y_base = cable * s3 + m3
            y_pred = model(rotation, translation, cable, training=True)
            y_pred = y_pred * sy + my
            y_gt = y_gt * sy + my
            prediction_loss, length_loss, accurv_yz_loss = loss(y_gt + y_base, y_pred + y_base)
            gt_pts = bsp32.N @ tf.reshape(y_gt + y_base, (-1, BSplineConstants.n, BSplineConstants.dim))
            pred_pts = bsp32.N @ tf.reshape(y_pred + y_base, (-1, BSplineConstants.n, BSplineConstants.dim))
            pts_loss = loss_pts(gt_pts, pred_pts)
            #prediction_loss_, length_loss = loss(y_gt + y_base, y_pred + y_base, 100.)
            #prediction_loss = loss(mul * y_gt, mul * y_pred)
            diff_ratio = tf.reduce_mean(tf.abs(y_gt - y_pred) / tf.abs(y_gt), axis=-1)
            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                 if 'bias' not in v.name])
            model_loss = prediction_loss + args.l2reg * reg_loss + args.len_loss * length_loss + args.acc_loss * accurv_yz_loss
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss.append(model_loss)
        prediction_losses.append(prediction_loss)
        pts_losses.append(pts_loss)
        diff_ratios.append(diff_ratio)

        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/pts_loss', tf.reduce_mean(pts_loss), step=train_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=train_step)
            tf.summary.scalar('metrics/length_loss', tf.reduce_mean(length_loss), step=train_step)
            tf.summary.scalar('metrics/accurv_yz_loss', tf.reduce_mean(accurv_yz_loss), step=train_step)
            tf.summary.scalar('metrics/diff_ratio', tf.reduce_mean(diff_ratio), step=train_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=train_step)
        train_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
    pts_losses = tf.reduce_mean(tf.concat(pts_losses, -1))
    diff_ratios = tf.reduce_mean(tf.concat(diff_ratios, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)
        tf.summary.scalar('epoch/pts_loss', pts_losses, step=epoch)
        tf.summary.scalar('epoch/diff_ratio', diff_ratios, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    prediction_losses = []
    diff_ratios = []
    pts_losses = []
    experiment_handler.log_validation()
    for i, rotation, translation, cable, y_gt in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        y_base = cable * s3 + m3
        y_pred = model(rotation, translation, cable, training=False)
        y_gt = y_gt * sy + my
        y_pred = y_pred * sy + my
        prediction_loss, length_loss, accurv_yz_loss = loss(y_gt + y_base, y_pred + y_base)
        gt_pts = bsp32.N @ tf.reshape(y_gt + y_base, (-1, BSplineConstants.n, BSplineConstants.dim))
        pred_pts = bsp32.N @ tf.reshape(y_pred + y_base, (-1, BSplineConstants.n, BSplineConstants.dim))
        pts_loss = loss_pts(gt_pts, pred_pts)
        #prediction_loss_, length_loss = loss(y_gt + y_base, y_pred + y_base, 100.)
        #prediction_loss, length_loss = loss(y_gt + y_base, y_pred + y_base)
        # prediction_loss = loss(y_gt, y_pred)
        #prediction_loss = loss(mul * y_gt, mul * y_pred)
        diff_ratio = np.mean(np.abs(y_gt - y_pred) / np.abs(y_gt), axis=-1)
        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                             if 'bias' not in v.name])
        model_loss = prediction_loss + args.l2reg * reg_loss + args.len_loss * length_loss + args.acc_loss * accurv_yz_loss

        epoch_loss.append(model_loss)
        prediction_losses.append(prediction_loss)
        pts_losses.append(pts_loss)
        diff_ratios.append(diff_ratio)

        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/pts_loss', tf.reduce_mean(pts_loss), step=train_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=val_step)
            tf.summary.scalar('metrics/length_loss', tf.reduce_mean(length_loss), step=val_step)
            tf.summary.scalar('metrics/accurv_yz_loss', tf.reduce_mean(accurv_yz_loss), step=val_step)
            tf.summary.scalar('metrics/diff_ratio', tf.reduce_mean(diff_ratio), step=val_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
    pts_losses = tf.reduce_mean(tf.concat(pts_losses, -1))
    diff_ratios = tf.reduce_mean(tf.concat(diff_ratios, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)
        tf.summary.scalar('epoch/pts_loss', pts_losses, step=epoch)
        tf.summary.scalar('epoch/diff_ratio', diff_ratios, step=epoch)

    w = 20
    if epoch % w == w - 1:
        experiment_handler.save_last()
    if best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        experiment_handler.save_best()
