import os

from losses.cable import CableSeqLoss
from models.inbilstm import INBiLSTM
from models.separated_cnn_neural_predictor import SeparatedCNNNeuralPredictor
from models.separated_neural_predictor import SeparatedNeuralPredictor
from utils.bspline import BSpline
from utils.constants import BSplineConstants

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    out_name = 'xyzrpy_episodic_all2all_02_10__14_00_bs32_lr5em5_inbilstm_m_regloss0em4_bs_keq_dsnotmixed_absloss_notwithened_'
    #out_name = 'xyzrpy_all2all_02_10__14_00_bs32_lr5em5_separated_l1x256_l3x256_m_regloss0em4_bs_keq_dsmixed_absloss_withened'
    #out_name = 'test'
    log_interval = 100
    learning_rate = 5e-5
    l2reg = 0e-4
    len_loss = 0
    acc_loss = 0e-1
    dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__14_00/train.tsv"


train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset(args.dataset_path)  # , n=10)
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset(args.dataset_path.replace("train", "val"))

#tX1, tX2, tX3, tY, vX1, vX2, vX3, vY, train_size, val_size = mix_datasets(tX1, tX2, tX3, tY, vX1, vX2, vX3, vY)

#tX1, vX1, m1, s1 = whiten(tX1, vX1)
#tX2, vX2, m2, s2 = whiten(tX2, vX2)
#tX3, vX3, m3, s3 = whiten(tX3, vX3)
#tY, vY, my, sy = whiten(tY, vY)

#train_ds = tf.data.Dataset.from_tensor_slices({"x1": tX1, "x2": tX2, "x3": tX3, "y": tY})
#val_ds = tf.data.Dataset.from_tensor_slices({"x1": vX1, "x2": vX2, "x3": vX3, "y": vY})

bsp = BSpline(25, 3, num_T_pts=32)

opt = tf.keras.optimizers.Adam(args.learning_rate)

loss = CableSeqLoss()

# model = BasicNeuralPredictor()
#model = SeparatedNeuralPredictor()
model = INBiLSTM()

experiment_handler = ExperimentHandler(args.working_dir, args.out_name, args.log_interval, model, opt)

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
    experiment_handler.log_training()
    for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
        with tf.GradientTape(persistent=True) as tape:
            gt_points = bsp.N @ tf.reshape(y_gt + cable, (-1, BSplineConstants.n, BSplineConstants.dim))
            points = bsp.N @ tf.reshape(cable, (-1, BSplineConstants.n, BSplineConstants.dim))
            a_left = tf.concat([rotation[:, :18], translation], axis=-1)
            a_right = rotation[:, 18:]
            y_pred = model(points, a_left, a_right, training=True)
            prediction_loss = loss(gt_points, y_pred)

            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                 if 'bias' not in v.name])
            model_loss = prediction_loss + args.l2reg * reg_loss
        grads = tape.gradient(model_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        epoch_loss.append(model_loss)
        prediction_losses.append(prediction_loss)

        with tf.summary.record_if(train_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=train_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=train_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=train_step)
        train_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)

    # validation
    dataset_epoch = val_ds.shuffle(val_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    prediction_losses = []
    experiment_handler.log_validation()
    for i, rotation, translation, cable, y_gt in _ds('Val', dataset_epoch, val_size, epoch, args.batch_size):
        gt_points = bsp.N @ tf.reshape(y_gt + cable, (-1, BSplineConstants.n, BSplineConstants.dim))
        points = bsp.N @ tf.reshape(cable, (-1, BSplineConstants.n, BSplineConstants.dim))
        a_left = tf.concat([rotation[:, :18], translation], axis=-1)
        a_right = rotation[:, 18:]
        y_pred = model(points, a_left, a_right, training=True)
        prediction_loss = loss(gt_points, y_pred)

        reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                             if 'bias' not in v.name])
        model_loss = prediction_loss + args.l2reg * reg_loss

        epoch_loss.append(model_loss)
        prediction_losses.append(prediction_loss)

        with tf.summary.record_if(val_step % args.log_interval == 0):
            tf.summary.scalar('metrics/model_loss', tf.reduce_mean(model_loss), step=val_step)
            tf.summary.scalar('metrics/prediction_loss', tf.reduce_mean(prediction_loss), step=val_step)
            tf.summary.scalar('metrics/reg_loss', tf.reduce_mean(reg_loss), step=val_step)
        val_step += 1

    epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
    prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))

    with tf.summary.record_if(True):
        tf.summary.scalar('epoch/loss', epoch_loss, step=epoch)
        tf.summary.scalar('epoch/prediction_loss', prediction_losses, step=epoch)

    w = 20
    if epoch % w == w - 1:
        experiment_handler.save_last()
    if best_epoch_loss > epoch_loss:
        best_epoch_loss = epoch_loss
        experiment_handler.save_best()
