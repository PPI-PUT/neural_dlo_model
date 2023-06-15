import os
from copy import copy

from losses.cable import CableBSplineLoss
from losses.cable_pts import CablePtsLoss
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
    unpack_translation, unpack_cable, prepare_dataset_cond
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
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16/train.tsv"
    dataset_path = "./data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"

model_type = "sep"
diff = False
rot = "rotmat"
ifdcable = True
aug = True

train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset_path, rot, diff=diff, augment=False)
#train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset(args.dataset_path)  # , n=10)
#val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset(args.dataset_path.replace("train", "val"))
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset_cond(args.dataset_path.replace("train", "val"), rot, diff=diff, augment=False)
test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset_path.replace("train", "test"), rot, diff=diff, augment=False)

ds_stats = compute_ds_stats(train_ds)

ds, ds_size = train_ds, train_size
#ds, ds_size = val_ds, val_size
#ds, ds_size = test_ds, test_size

bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)

#loss = CableBSplineLoss()
loss = CablePtsLoss()

#model = BasicNeuralPredictor()

# model = BasicNeuralPredictor()
# model = SeparatedCNNNeuralPredictor()
#model = SeparatedNeuralPredictor()
#model = INBiLSTM()
#model = CNN()
#model = CNNSep()
model = None
if model_type == "sep":
    model = SeparatedNeuralPredictor()
elif model_type == "inbilstm":
    model = INBiLSTM()
else:
    print(f"MODEL TYPE {model_type} NOT KNOWN")
    assert False


ckpt_path = f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_" \
            f"{model_type}_" \
            f"{'diff' if diff else 'nodiff'}_{rot}_" \
            f"{'dcable' if ifdcable else 'cable'}_" \
            f"{'augwithzeros' if aug else 'noaug'}/checkpoints/best-41"
ckpt = tf.train.Checkpoint(model=model)
#ckpt.restore("./trained_models/xyzrpy_episodic_all2all_02_10__14_00_p16_bs32_lr5em5_cnn_noskip_add_dsnotmixed_absloss_withened_cablediff/checkpoints/best-15")
#ckpt.restore("./trained_models/xyzrpy_episodic_all2all_02_10__14_00_p16_bs32_lr5em5_inbilstm_dsnotmixed_absloss_withened/checkpoints/best-33")
#ckpt.restore("./trainings/xyzrpy_episodic_semisep_all2all_02_10__14_00_p16_bs32_lr5em5_cnn_sep_dsnotmixed_absloss_withened/checkpoints/best-41")
#ckpt.restore("./trainings/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16_bs32_lr5em5_cnn_absloss_withened_quat/checkpoints/best-35")
#ckpt.restore("./trainings/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16_bs32_lr5em5_cnn_absloss_withened_quat/checkpoints/best-48")
ckpt.restore(ckpt_path)

#def inference(rotation, translation, cable):
#    rotation_, translation_, cable_, y_gt_ = whitening(rotation, translation, cable, y_gt, ds_stats)
#    y_pred_ = model(rotation_, translation_, cable_, training=True)
#    y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
#    # y_pred = model(rotation, translation, cable, training=True)
#    return y_pred

plot = True
#plot = False

dataset_epoch = ds.shuffle(ds_size)
dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
epoch_loss = []
prediction_losses = []
pts_losses_abs = []
pts_losses_euc = []
K = 64
C = 8
for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, ds_size, 0, args.batch_size):
    #rotation_, translation_, cable_, y_gt_ = whitening(rotation, translation, cable, y_gt, ds_stats)
    rotation_, translation_, cable_ = whitening(rotation, translation, cable, ds_stats)
    R_l_0, R_l_1_, R_r_0, R_r_1_ = unpack_rotation(rotation_)
    t_l_0, t_l_1_ = unpack_translation(translation_)
    cable_, dcable_ = unpack_cable(cable_)
    t_l_1 = copy(t_l_0)[0]
    R_l_1 = copy(R_l_0)[0]
    R_r_1 = copy(R_r_0)[0]
    t_l_s = [copy(t_l_1)]
    R_l_s = [copy(R_l_1)]
    R_r_s = [copy(R_r_1)]
    t_l_0 = tf.tile(t_l_0, (K, 1))
    R_l_0 = tf.tile(R_l_0, (K, 1))
    R_r_0 = tf.tile(R_r_0, (K, 1))
    cable = tf.tile(cable, (K, 1, 1))
    cable_ = tf.tile(cable_, (K, 1, 1))
    dcable_ = tf.tile(dcable_, (K, 1, 1))
    t_l_1_std = np.ones_like(t_l_1)
    R_l_1_std = np.ones_like(R_l_1)
    R_r_1_std = np.ones_like(R_l_1)
    for i in range(10):
        t_l_1_samples = np.random.normal(t_l_1, t_l_1_std, (K, t_l_1_std.shape[0]))
        R_l_1_samples = np.random.normal(R_l_1, R_l_1_std, (K, R_l_1_std.shape[0]))
        R_r_1_samples = np.random.normal(R_r_1, R_r_1_std, (K, R_r_1_std.shape[0]))
        #y_pred_ = model((R_l_0, R_l_1_samples, R_r_0, R_r_1_samples),
        #                (t_l_0, t_l_1_samples), cable_)
        y_pred_ = model((R_l_0, R_l_1_samples, R_r_0, R_r_1_samples), (t_l_0, t_l_1_samples),
                        dcable_ if ifdcable else cable_,
                        unpack_rotation(rotation), unpack_translation(translation))
        #y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
        y_pred = y_pred_ + cable[..., :3]
        #cp_loss_abs, cp_loss_euc, cp_loss_l2, \
        #pts_loss_abs, pts_loss_euc, pts_loss_l2, \
        #length_loss, accurv_yz_loss = loss(y_gt, y_pred)
        pts_loss_abs, pts_loss_euc, pts_loss_l2 = loss(y_gt, y_pred)

        prediction_loss = pts_loss_abs
        idx = np.argsort(prediction_loss)
        vals = prediction_loss.numpy()[idx[:C]]
        chosen_t_l_1 = t_l_1_samples[idx[:C]]
        chosen_R_l_1 = R_l_1_samples[idx[:C]]
        chosen_R_r_1 = R_r_1_samples[idx[:C]]
        t_l_1 = np.mean(chosen_t_l_1, axis=0)
        t_l_1_std = np.std(chosen_t_l_1, axis=0)
        R_l_1 = np.mean(chosen_R_l_1, axis=0)
        R_l_1_std = np.std(chosen_R_l_1, axis=0)
        R_r_1 = np.mean(chosen_R_r_1, axis=0)
        R_r_1_std = np.std(chosen_R_r_1, axis=0)
        t_l_s.append(copy(t_l_1))
        R_l_s.append(copy(R_l_1))
        R_r_s.append(copy(R_r_1))
        a = 0


    #t_l_s = np.stack(t_l_s, axis=0)
    #for i in range(3):
    #    plt.subplot(131 + i)
    #    plt.plot(t_l_s[:, i])
    #    plt.plot([0], [t_l_0[0, i]], 'go')
    #    plt.plot([t_l_s.shape[0]], [t_l_1_[0, i]], 'rx')
    #plt.show()
    #R_l_s = np.stack(R_l_s, axis=0)
    #for i in range(4):
    #    plt.subplot(331 + i)
    #    plt.plot(R_l_s[:, i])
    #    plt.plot([0], [R_l_0[0, i]], 'go')
    #    plt.plot([R_l_s.shape[0]], [R_l_1_[0, i]], 'rx')
    #plt.show()
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

    #epoch_loss.append(model_loss)
    pts_losses_abs.append(pts_loss_abs)
    prediction_losses.append(prediction_loss)
    pts_losses_euc.append(pts_loss_euc)

#epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
pts_losses_abs = tf.reduce_mean(tf.concat(pts_losses_abs, -1))
pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))
#print("EPOCH LOSS:", epoch_loss)
print("PREDICTION LOSS:", prediction_losses)
print("CP ABS LOSS:", pts_losses_abs)
print("PTS EUC LOSS:", pts_losses_euc)
