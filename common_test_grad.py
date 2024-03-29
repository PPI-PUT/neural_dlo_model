import os
from copy import copy

from losses.cable import CableBSplineLoss
from models.cnn import CNN
from models.cnn_sep import CNNSep
from models.inbilstm import INBiLSTM
from models.separated_cnn_neural_predictor import SeparatedCNNNeuralPredictor
from models.separated_neural_predictor import SeparatedNeuralPredictor
from utils.bspline import BSpline
from utils.constants import BSplineConstants

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataset import _ds, prepare_dataset, whiten, mix_datasets, whitening, compute_ds_stats, unpack_rotation, \
    unpack_translation
from utils.execution import ExperimentHandler
from models.basic_neural_predictor import BasicNeuralPredictor

np.random.seed(444)


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

class args:
    batch_size = 1
    working_dir = './trainings'
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__14_00_p16/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_02_10__14_00_p16/train.tsv"
    dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16/train.tsv"


train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset(args.dataset_path)  # , n=10)
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset(args.dataset_path.replace("train", "val"))

ds_stats = compute_ds_stats(train_ds)

ds, ds_size = train_ds, train_size
#ds, ds_size = val_ds, val_size

bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)

loss = CableBSplineLoss()

# model = BasicNeuralPredictor()
# model = SeparatedCNNNeuralPredictor()
# model = SeparatedNeuralPredictor()
#model = INBiLSTM()
model = CNN()
#model = CNNSep()


ckpt = tf.train.Checkpoint(model=model)
#ckpt.restore("./trained_models/xyzrpy_episodic_all2all_02_10__14_00_p16_bs32_lr5em5_cnn_noskip_add_dsnotmixed_absloss_withened_cablediff/checkpoints/best-15")
#ckpt.restore("./trained_models/xyzrpy_episodic_all2all_02_10__14_00_p16_bs32_lr5em5_inbilstm_dsnotmixed_absloss_withened/checkpoints/best-33")
#ckpt.restore("./trainings/xyzrpy_episodic_semisep_all2all_02_10__14_00_p16_bs32_lr5em5_cnn_sep_dsnotmixed_absloss_withened/checkpoints/best-41")
#ckpt.restore("./trainings/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16_bs32_lr5em5_cnn_absloss_withened_quat/checkpoints/best-35")
ckpt.restore("./trainings/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16_bs32_lr5em5_cnn_absloss_withened_quat/checkpoints/best-48")

def inference(rotation, translation, cable):
    rotation_, translation_, cable_, y_gt_ = whitening(rotation, translation, cable, y_gt, ds_stats)
    y_pred_ = model(rotation_, translation_, cable_, training=True)
    y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
    # y_pred = model(rotation, translation, cable, training=True)
    return y_pred

plot = True
#plot = False

dataset_epoch = ds.shuffle(ds_size)
dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
epoch_loss = []
prediction_losses = []
cp_losses_abs = []
pts_losses_euc = []
for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, ds_size, 0, args.batch_size):
    rotation_, translation_, cable_, y_gt_ = whitening(rotation, translation, cable, y_gt, ds_stats)
    R_l_0, R_l_1_, R_r_0, R_r_1_ = unpack_rotation(rotation_)
    t_l_0, t_l_1_ = unpack_translation(translation_)
    t_l_1 = copy(t_l_0)
    R_l_1 = copy(R_l_0)
    R_r_1 = copy(R_r_0)
    opt = tf.keras.optimizers.Adam(1e-1)
    t_l_s = [copy(t_l_1)]
    R_l_s = [copy(R_l_1)]
    for i in range(100):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t_l_1)
            tape.watch(R_l_1)
            tape.watch(R_r_1)
            y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), cable_)
            y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
            cp_loss_abs, cp_loss_euc, cp_loss_l2, \
            pts_loss_abs, pts_loss_euc, pts_loss_l2, \
            length_loss, accurv_yz_loss = loss(y_gt, y_pred)

            prediction_loss = cp_loss_abs

            reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                                 if 'bias' not in v.name])
            model_loss = prediction_loss

        tl_grads = tape.gradient(model_loss, t_l_1)
        Rl_grads = tape.gradient(model_loss, R_l_1)
        Rr_grads = tape.gradient(model_loss, R_r_1)
        lr = 1e-1
        t_l_1 -= lr * tl_grads
        R_l_1 -= lr * Rl_grads
        R_r_1 -= lr * Rr_grads
        t_l_s.append(copy(t_l_1))
        R_l_s.append(copy(R_l_1))

    t_l_s = np.concatenate(t_l_s, axis=0)
    for i in range(3):
        plt.subplot(131 + i)
        plt.plot(t_l_s[:, i])
        plt.plot([0], [t_l_0[0, i]], 'go')
        plt.plot([t_l_s.shape[0]], [t_l_1_[0, i]], 'rx')
    plt.show()
    R_l_s = np.concatenate(R_l_s, axis=0)
    for i in range(4):
        plt.subplot(331 + i)
        plt.plot(R_l_s[:, i])
        plt.plot([0], [R_l_0[0, i]], 'go')
        plt.plot([R_l_s.shape[0]], [R_l_1_[0, i]], 'rx')
    plt.show()
    cp_pred = y_pred[0]
    cp_gt = y_gt[0]
    cp0 = cable[0]

    if plot:
        xl = -0.2
        xh = 0.5
        yl = -0.3
        yh = 0.4
        zl = -0.35
        zh = 0.35
        plt.subplot(221)
        # plt.xlim(xl, xh)
        # plt.ylim(yl, yh)
        plt.plot(cp_gt[:, 1], cp_gt[:, 2], 'rx')
        plt.plot(cp_pred[:, 1], cp_pred[:, 2], 'bo')
        plt.plot(cp0[:, 1], cp0[:, 2], 'g^')
        plt.subplot(223)
        # plt.xlim(zl, zh)
        # plt.ylim(yl, yh)
        plt.plot(cp_gt[:, 0], cp_gt[:, 1], 'rx')
        plt.plot(cp_pred[:, 0], cp_pred[:, 1], 'bo')
        plt.plot(cp0[:, 0], cp0[:, 1], 'g^')

        v_pred = (bsp.N[0] @ cp_pred)
        v_gt = (bsp.N[0] @ cp_gt)
        v_base = (bsp.N[0] @ cp0)
        plt.subplot(222)
        # plt.xlim(xl, xh)
        # plt.ylim(yl, yh)
        plt.plot(v_pred[:, 1], v_pred[:, 2], label="pred")
        plt.plot(v_gt[:, 1], v_gt[:, 2], label="gt")
        plt.plot(v_base[:, 1], v_base[:, 2], label="base")
        plt.subplot(224)
        # plt.xlim(zl, zh)
        # plt.ylim(yl, yh)
        plt.plot(v_pred[:, 0], v_pred[:, 1], label="pred")
        plt.plot(v_gt[:, 0], v_gt[:, 1], label="gt")
        plt.plot(v_base[:, 0], v_base[:, 1], label="base")
        # plt.xlim(-0.1, 0.7)
        # plt.ylim(-0.4, 0.4)
        plt.legend()
        plt.show()

    epoch_loss.append(model_loss)
    cp_losses_abs.append(cp_loss_abs)
    prediction_losses.append(prediction_loss)
    pts_losses_euc.append(pts_loss_euc)

epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
cp_losses_abs = tf.reduce_mean(tf.concat(cp_losses_abs, -1))
pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))
print("EPOCH LOSS:", epoch_loss)
print("PREDICTION LOSS:", prediction_losses)
print("CP ABS LOSS:", cp_losses_abs)
print("PTS EUC LOSS:", pts_losses_euc)
