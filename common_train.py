import os

from losses.cable import CableBSplineLoss
from models.cnn import CNN
from models.cnn_sep import CNNSep
from models.inbilstm import INBiLSTM
from models.separated_cnn_neural_predictor import SeparatedCNNNeuralPredictor
from models.separated_neural_predictor import SeparatedNeuralPredictor
from utils.bspline import BSpline
from utils.constants import BSplineConstants

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataset import _ds, prepare_dataset, whiten, mix_datasets, whitening, compute_ds_stats, unpack_rotation, \
    unpack_translation, prepare_dataset_cond
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
    #out_name = 'xyzrpy_episodic_all2all_02_10__14_00_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsnotmixed_absloss_withened_'
    #out_name = 'xyzrpy_episodic_all2all_02_10__14_00_p16_bs32_lr5em5_cnn_sep_dsnotmixed_absloss_withened'
    #out_name = 'xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16_bs32_lr5em5_inbilstm_absloss_withened_quat'
    out_name = 'xyzrpy_episodic_semisep_all2all_02_21__12_30_cp16_bs32_lr5em5_sep_absloss_withened_quat'
    #out_name = 'xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_pts16_bs32_lr5em5_cnn_sep_dsnotmixed_absloss_withened'
    #out_name = 'xyzrpy_episodic_02_10__14_00_p16_bs32_lr5em5_cnn_dsmixed_absloss_withened'
    #out_name = 'xyzrpy_episodic_all2all_02_10__14_00_bs32_lr5em5_separated_cablecnn_l1x128_l2x128_outputcnn_m_regloss0em4_bs_keq_dsnotmixed_absloss_withened'
    #out_name = 'xyzrpy_all2all_02_10__14_00_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsmixed_absloss_withened'
    #out_name = 'test'
    log_interval = 100
    learning_rate = 5e-5
    l2reg = 0e-4
    len_loss = 0
    acc_loss = 0e-1
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_02_10__14_00_p16/train.tsv"
    dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_02_21__12_30_cp16/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_pts16/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__14_00_p16/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_all2all_02_10__14_00/train.tsv"
    # dataset_path = "./data/prepared_datasets/yz_big_keq/train.tsv"
    # dataset_path = "./data/prepared_datasets/yz_big_keq_n1000/train.tsv"
    # dataset_path = "./data/prepared_datasets/xy_bs_keq/train.tsv"
    # dataset_path = "./data/prepared_datasets/short_ds/train.tsv"


diff = True
quat = True
train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset_path, quat=quat, diff=diff)  # , n=10)
#train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset_path, quat=quat, diff=diff, augment=True)  # , n=10)
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset_cond(args.dataset_path.replace("train", "val"), quat=quat, diff=diff)

#tX1, tX2, tX3, tY, vX1, vX2, vX3, vY, train_size, val_size = mix_datasets(tX1, tX2, tX3, tY, vX1, vX2, vX3, vY)
#train_ds = tf.data.Dataset.from_tensor_slices({"x1": tX1, "x2": tX2, "x3": tX3, "y": tY})
#val_ds = tf.data.Dataset.from_tensor_slices({"x1": vX1, "x2": vX2, "x3": vX3, "y": vY})

ds_stats = compute_ds_stats(train_ds)

#tX1, vX1, m1, s1 = whiten(tX1, vX1)
#X2, vX2, m2, s2 = whiten(tX2, vX2)
#X3, vX3, m3, s3 = whiten(tX3, vX3)
#m3, s3 = 0., 1.
#Y, vY, my, sy = whiten(tY, vY)

#train_ds = tf.data.Dataset.from_tensor_slices({"x1": tX1, "x2": tX2, "x3": tX3, "y": tY})
#val_ds = tf.data.Dataset.from_tensor_slices({"x1": vX1, "x2": vX2, "x3": vX3, "y": vY})

#bsp = BSpline(25, 3)


opt = tf.keras.optimizers.Adam(args.learning_rate)

loss = CableBSplineLoss()


#model = BasicNeuralPredictor()
#model = SeparatedCNNNeuralPredictor()
model = SeparatedNeuralPredictor()
#model = INBiLSTM()
#model = CNN()
#model = CNNSep()

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)

def inference(rotation, translation, cable):
    rotation_, translation_, cable_, y_gt_ = whitening(rotation, translation, cable, y_gt, ds_stats)
    R_l_0, R_l_1, R_r_0, R_r_1 = unpack_rotation(rotation_)
    t_l_0, t_l_1 = unpack_translation(translation_)
    y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), cable_)
    #y_pred_ = model(rotation_, translation_, cable_, training=True)
    y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
    #y_pred = model(rotation, translation, cable, training=True)
    return y_pred


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
    cp_losses_abs = []
    pts_losses_euc = []
    experiment_handler.log_training()
    for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = inference(rotation, translation, cable)
            cp_loss_abs, cp_loss_euc, cp_loss_l2, \
            pts_loss_abs, pts_loss_euc, pts_loss_l2, \
            length_loss, accurv_yz_loss, pred_energy, gt_energy = loss(y_gt, y_pred)

            prediction_loss = cp_loss_abs

            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                 if 'bias' not in v.name])
            model_loss = prediction_loss + args.l2reg * reg_loss + args.len_loss * length_loss + args.acc_loss * accurv_yz_loss
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss.append(model_loss)
        cp_losses_abs.append(cp_loss_abs)
        prediction_losses.append(prediction_loss)
        pts_losses_euc.append(pts_loss_euc)

        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=train_step)
            tf.summary.scalar('metrics/cp_loss_abs', tf.reduce_mean(cp_loss_abs), step=train_step)
            tf.summary.scalar('metrics/pts_loss_euc', tf.reduce_mean(pts_loss_euc), step=train_step)
            tf.summary.scalar('metrics/length_loss', tf.reduce_mean(length_loss), step=train_step)
            tf.summary.scalar('metrics/accurv_yz_loss', tf.reduce_mean(accurv_yz_loss), step=train_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=train_step)
        train_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
    cp_losses_abs = tf.reduce_mean(tf.concat(cp_losses_abs, -1))
    pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)
        tf.summary.scalar('epoch/cp_loss_abs', cp_losses_abs, step=epoch)
        tf.summary.scalar('epoch/pts_loss_euc', pts_losses_euc, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    prediction_losses = []
    cp_losses_abs = []
    pts_losses_euc = []
    experiment_handler.log_validation()
    for i, rotation, translation, cable, y_gt in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        y_pred = inference(rotation, translation, cable)
        cp_loss_abs, cp_loss_euc, cp_loss_l2,\
        pts_loss_abs, pts_loss_euc, pts_loss_l2,\
        length_loss, accurv_yz_loss, pred_energy, gt_energy = loss(y_gt, y_pred)

        prediction_loss = cp_loss_abs

        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                             if 'bias' not in v.name])
        model_loss = prediction_loss + args.l2reg * reg_loss + args.len_loss * length_loss + args.acc_loss * accurv_yz_loss

        epoch_loss.append(model_loss)
        cp_losses_abs.append(cp_loss_abs)
        prediction_losses.append(prediction_loss)
        pts_losses_euc.append(pts_loss_euc)

        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=train_step)
            tf.summary.scalar('metrics/pts_loss_euc', tf.reduce_mean(pts_loss_euc), step=train_step)
            tf.summary.scalar('metrics/cp_loss_abs', tf.reduce_mean(cp_loss_abs), step=val_step)
            tf.summary.scalar('metrics/length_loss', tf.reduce_mean(length_loss), step=val_step)
            tf.summary.scalar('metrics/accurv_yz_loss', tf.reduce_mean(accurv_yz_loss), step=val_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    cp_losses_abs = tf.reduce_mean(tf.concat(cp_losses_abs, -1))
    pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)
        tf.summary.scalar('epoch/cp_loss_abs', cp_losses_abs, step=epoch)
        tf.summary.scalar('epoch/pts_loss_euc', pts_losses_euc, step=epoch)

    w = 20
    if epoch % w == w - 1:
        experiment_handler.save_last()
    if best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        experiment_handler.save_best()
