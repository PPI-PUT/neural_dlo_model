import os

from losses.cable import CableBSplineLoss
from losses.cable_pts import CablePtsLoss
from models.cnn import CNN
from models.cnn_sep import CNNSep
from models.inbilstm import INBiLSTM
from models.scale_neural_predictor import ScaleNeuralPredictor, ScaleNeuralPredictor1, ScaleNeuralPredictor2
from models.separated_cnn_neural_predictor import SeparatedCNNNeuralPredictor
from models.separated_neural_predictor import SeparatedNeuralPredictor
from utils.bspline import BSpline
from utils.constants import BSplineConstants

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataset import _ds, prepare_dataset, whiten, mix_datasets, whitening, compute_ds_stats, unpack_rotation, \
    unpack_translation, prepare_dataset_cond, unpack_cable, prepare_dataset_left_right_cond, \
    unpack_translation_left_right
from utils.execution import ExperimentHandler
from models.basic_neural_predictor import BasicNeuralPredictor

np.random.seed(444)


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    batch_size = 128
    working_dir = './trainings'
    out_name = 'left_right_mb_03_30_test'
    log_interval = 100
    learning_rate = 5e-4
    l2reg = 0e-4
    len_loss = 0
    acc_loss = 0e-1
    dataset_path = "./data/prepared_datasets/left_right_mb_03_30/train.tsv"


#diff = True
diff = False
#rot = "quat"
rot = "rotmat"
#rot = "rotvec"
#ifdcable = True
ifdcable = False
#train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_left_right_cond(args.dataset_path, rot=rot, diff=diff)  # , n=10)
train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_left_right_cond(args.dataset_path, rot=rot, diff=diff, augment=True)  # , n=10)
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset_left_right_cond(args.dataset_path.replace("train", "val"), rot=rot, diff=diff)

ds_stats = compute_ds_stats(train_ds)

opt = tf.keras.optimizers.Adam(args.learning_rate)

loss = CablePtsLoss()

# model = BasicNeuralPredictor()
# model = SeparatedCNNNeuralPredictor()
model = SeparatedNeuralPredictor()
#model = ScaleNeuralPredictor1()
#model = ScaleNeuralPredictor2()
#model = INBiLSTM()
# model = CNN()
# model = CNNSep()

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)


def inference(rotation, translation, cable):
    rotation_, translation_, cable_ = whitening(rotation, translation, cable, ds_stats)
    R_l_0, R_l_1, R_r_0, R_r_1 = unpack_rotation(rotation_)
    t_l_0, t_l_1, t_r_0, t_r_1 = unpack_translation_left_right(translation_)
    cable_, dcable_ = unpack_cable(cable_)
    y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1, t_r_0, t_r_1), dcable_ if ifdcable else cable_,
                    unpack_rotation(rotation), unpack_translation(translation))
    #y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), dcable_ if ifdcable else cable_)
    #y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
    y_pred = y_pred_
    return y_pred + cable[..., :3]


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
    pts_losses_abs = []
    pts_losses_euc = []
    pts_losses_l2 = []
    experiment_handler.log_training()
    for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            y_pred = inference(rotation, translation, cable)
            pts_loss_abs, pts_loss_euc, pts_loss_l2 = loss(y_gt, y_pred)
            prediction_loss = pts_loss_abs

            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                 if 'bias' not in v.name])
            model_loss = prediction_loss + args.l2reg * reg_loss
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss.append(model_loss)
        prediction_losses.append(prediction_loss)
        pts_losses_abs.append(pts_loss_abs)
        pts_losses_euc.append(pts_loss_euc)
        pts_losses_l2.append(pts_loss_l2)

        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=train_step)
            tf.summary.scalar('metrics/pts_loss_abs', tf.reduce_mean(pts_loss_abs), step=train_step)
            tf.summary.scalar('metrics/pts_loss_euc', tf.reduce_mean(pts_loss_euc), step=train_step)
            tf.summary.scalar('metrics/pts_loss_l2', tf.reduce_mean(pts_loss_l2), step=train_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=train_step)
        train_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
    pts_losses_abs = tf.reduce_mean(tf.concat(pts_losses_abs, -1))
    pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))
    pts_losses_l2 = tf.reduce_mean(tf.concat(pts_losses_l2, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)
        tf.summary.scalar('epoch/pts_loss_abs', pts_losses_abs, step=epoch)
        tf.summary.scalar('epoch/pts_loss_euc', pts_losses_euc, step=epoch)
        tf.summary.scalar('epoch/pts_loss_l2', pts_losses_l2, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    prediction_losses = []
    pts_losses_abs = []
    pts_losses_euc = []
    pts_losses_l2 = []
    experiment_handler.log_validation()
    for i, rotation, translation, cable, y_gt in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        y_pred = inference(rotation, translation, cable)
        pts_loss_abs, pts_loss_euc, pts_loss_l2 = loss(y_gt, y_pred)
        prediction_loss = pts_loss_abs

        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                             if 'bias' not in v.name])
        model_loss = prediction_loss + args.l2reg * reg_loss

        epoch_loss.append(model_loss)
        prediction_losses.append(prediction_loss)
        pts_losses_abs.append(pts_loss_abs)
        pts_losses_euc.append(pts_loss_euc)
        pts_losses_l2.append(pts_loss_l2)

        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=train_step)
            tf.summary.scalar('metrics/pts_loss_abs', tf.reduce_mean(pts_loss_abs), step=val_step)
            tf.summary.scalar('metrics/pts_loss_euc', tf.reduce_mean(pts_loss_euc), step=val_step)
            tf.summary.scalar('metrics/pts_loss_l2', tf.reduce_mean(pts_loss_l2), step=val_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
    pts_losses_abs = tf.reduce_mean(tf.concat(pts_losses_abs, -1))
    pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))
    pts_losses_l2 = tf.reduce_mean(tf.concat(pts_losses_l2, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)
        tf.summary.scalar('epoch/pts_loss_abs', pts_losses_abs, step=epoch)
        tf.summary.scalar('epoch/pts_loss_euc', pts_losses_euc, step=epoch)
        tf.summary.scalar('epoch/pts_loss_l2', pts_losses_l2, step=epoch)

    w = 20
    if epoch % w == w - 1:
        experiment_handler.save_last()
    if best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        experiment_handler.save_best()
