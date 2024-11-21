import os
from time import perf_counter

from losses.cable import CableBSplineLoss
from losses.cable_all2all import CableBSplineAll2AllLoss, CableAll2AllLoss
from losses.cable_pts import CablePtsLoss
from models.cnn import CNN
from models.inbilstm import INBiLSTM
from models.jacobian_neural_predictor import JacobianNeuralPredictor
from models.separated_cnn_neural_predictor import SeparatedCNNNeuralPredictor
from models.separated_neural_predictor import SeparatedNeuralPredictor
from models.transformer import Transformer
from utils.bspline import BSpline
from utils.constants import BSplineConstants

from utils.geometry import calculateL3

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataset import _ds, prepare_dataset, whiten, mix_datasets, whitening, compute_ds_stats, unpack_translation, \
    unpack_rotation, prepare_dataset_cond, unpack_cable, prepare_dataset_cond_lennorm, normalize_cable
from utils.execution import ExperimentHandler
from models.basic_neural_predictor import BasicNeuralPredictor

np.random.seed(444)
tf.random.set_seed(444)


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#plot = False
plot = True

class args:
    #batch_size = 128
    batch_size = 1
    #batch_size = 8
    #batch_size = 64
    working_dir = './trainings'
    # dataset_path = "./data/prepared_datasets/xyzrpy_episodic_all2all_02_10__14_00_p16/train.tsv"
    # dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_02_10__14_00_p16/train.tsv"
    # dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_fixed_02_10__14_00_p16/train.tsv"
    # dataset_path = "./data/prepared_datasets/xyzrpy_episodic_semisep_all2all_02_21__12_30_cp16/train.tsv"

    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_sep_all2all_02_23__01_00_cp16/train.tsv"
    #dataset_path = "./data/prepared_datasets/xyzrpy_episodic_sep_all2all_03_01__08_00_cp16/train.tsv"

    #dataset_path = "./data/prepared_datasets/new_mb_03_27_poc64/train.tsv"
    #dataset_path = "./data/prepared_datasets/new_mb_45cm_04_03/train.tsv"
    #dataset_path = "./data/prepared_datasets/new_mb_zoval_04_25/train.tsv"
    #dataset_path = "./data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"
    #dataset_path = "./data/prepared_datasets/06_09_final_off3cm_p2/train.tsv"
    #dataset_path = "./data/prepared_datasets/06_09_final_off3cm_p3/train.tsv"
    #dataset_path = "./data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"
    base_dataset_path = "./data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"
    #dataset_path = "./data/prepared_datasets/06_09_final_off3cm_40cm/train.tsv"
    #dataset_path = "./data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"
    dataset_path = "./data/prepared_datasets/04_17_final_off3cm_55cm/train.tsv"


#rot = "quat"
rot = "rotvec"
#rot = "rotmat"
diff = True
#diff = False
#ifdcable = False
ifdcable = True
#norm = True
norm = False
#train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset_path, rot, diff=diff, augment=True, norm=norm)
train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.base_dataset_path, rot, diff=diff, augment=True, norm=norm)
#train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset(args.dataset_path, augment=True)  # , n=10)
#train_ds_, train_size_, tX1_, tX2_, tX3_, tY_ = prepare_dataset(args.dataset_path)  # , n=10)
#val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset(args.dataset_path.replace("train", "val"))
val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset_cond(args.dataset_path.replace("train", "val"), rot, diff=diff, augment=False, norm=norm)
#test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset_path.replace("train", "test_filtered"), rot, diff=diff, augment=False)
#test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset_path.replace("train", "test"), rot, diff=diff, augment=False, norm=norm, scale=50./40.)
test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset_path.replace("train", "test"), rot, diff=diff, augment=False, norm=norm, scale=50./55.)
test_data = np.loadtxt(args.dataset_path.replace("train", "test"), delimiter='\t').astype(np.float32)

ds_stats = compute_ds_stats(train_ds, norm=norm)
#ds_stats_ = compute_ds_stats(train_ds_)

#ds, ds_size = train_ds, train_size
#ds, ds_size = val_ds, val_size
ds, ds_size = test_ds, test_size

bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)

#loss = CableBSplineLoss()
loss_all2all = CableAll2AllLoss()
loss = CablePtsLoss()

#model = BasicNeuralPredictor()
#model = SeparatedCNNNeuralPredictor()
#model = SeparatedNeuralPredictor()
#model = INBiLSTM()
# model = CNN()
model = Transformer(num_layers=2, num_heads=8, dff=256, d_model=64, dropout_rate=0.1, target_size=3)
#model = JacobianNeuralPredictor(rot, diff)


ckpt = tf.train.Checkpoint(model=model)
#ckpt.restore("./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_nodiff_rotvec_cable_augwithzeros/last_n-293")
#ckpt.restore("./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_nodiff_rotmat_dcable_augwithzeros/checkpoints/best-127")
#ckpt.restore("./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_diff_rotmat_dcable_augwithzeros_/checkpoints/best-91")
#ckpt.restore("./trained_models/all_mb_zoval_04_25/new_mb_zoval_04_25_poc64_lr5em4_bs128_sep_diff_rotvec_cable_augwithzeros/checkpoints/best-154")
#ckpt.restore("./trained_models//new_mb_zoval_04_25_poc64_lr5em4_bs128_sep_diff_rotvec_cable_augwithzeros/checkpoints/best-154")

# sep
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_dcable_augwithzeros/checkpoints/best-41")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_dcable_noaug/checkpoints/best-20")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_sep_diff_rotmat_dcable_noaug/checkpoints/best-45")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_sep_diff_rotmat_dcable_augwithzeros/checkpoints/best-58")
# inbilstm
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_inbilstm_nodiff_rotmat_dcable_augwithzeros/checkpoints/best-50")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_inbilstm_nodiff_rotmat_dcable_noaug/checkpoints/best-30")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_inbilstm_diff_rotmat_dcable_noaug/checkpoints/best-32")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_inbilstm_diff_rotmat_dcable_augwithzeros/checkpoints/best-31")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_inbilstm_nodiff_rotvec_dcable_noaug_/checkpoints/best-42")
# transformer
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_nodiff_rotmat_dcable_noaug/checkpoints/best-30")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_nodiff_rotmat_dcable_augwithzeros/checkpoints/best-24") # diff - to be retrained on nodiff
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotmat_dcable_augwithzeros/checkpoints/best-38")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotmat_dcable_noaug/checkpoints/best-51")

# jacobian
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_jac_nodiff_rotmat_dcable_augwithzeros/checkpoints/best-16")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_jac_nodiff_rotmat_dcable_noaug/checkpoints/best-13")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_jac_diff_rotmat_dcable_augwithzeros/checkpoints/best-30")

# p2
#ckpt.restore("./trained_models/06_09_final_off3cm_p2/best-1059")
# p3
#ckpt.restore("./trained_models/06_09_final_off3cm_p3/best-163")

# len
#ckpt.restore("./trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_cable_augwithzeros_lennormfix/checkpoints/best-35")
#ckpt.restore("./trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_cable_augwithzeros_lennormtransfixed2/checkpoints/best-79")
#ckpt.restore("./trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_cable_augwithzeros/checkpoints/best-83")
#ckpt.restore("./trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotvec_dcable_augwithzeros/checkpoints/best-59")

# nowhithening
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm_nowhithening/best-125")
#ckpt.restore("./trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_cable_augwithzeros/checkpoints/best-83")
ckpt.restore("./trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotvec_dcable_augwithzeros/checkpoints/best-59")
#ckpt.restore("./trained_models/06_09_final_off3cm_50cm_nowhithening/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotvec_dcable_augwithzeros_nowhithening/checkpoints/best-27")


#ckpt.restore("./trained_models/all/xyzrpy_episodic_sep_all2all_02_23__01_00_cp16_bs128_lr5em4_sep_absloss_withened_quat_augwithzeros/checkpoints/last_n-70")

# ckpt.restore("./trained_models/all/xyzrpy_episodic_sep_all2all_02_23__01_00_cp16_bs128_lr5em4_inbilstm_absloss_withened_quat_augwithzeros/checkpoints/best-19")


#def inference(rotation, translation, cable):
#    rotation_, translation_, cable_ = whitening(rotation, translation, cable, ds_stats)
#    R_l_0, R_l_1, R_r_0, R_r_1 = unpack_rotation(rotation_)
#    t_l_0, t_l_1 = unpack_translation(translation_)
#    t0 = perf_counter()
#    y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), cable_)
#    t1 = perf_counter()
#    # print("INFERENCE TIME:", t1 - t0)
#    # y_pred_ = model(rotation_, translation_, cable_, training=True)
#    y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
#    # y_pred = model(rotation, translation, cable, training=True)
#    return y_pred

def inference(rotation, translation, cable):
    cable_ = cable
    if norm:
        cable_ = normalize_cable(cable)
    rotation_, translation_, cable_ = whitening(rotation, translation, cable_, ds_stats)
    #rotation_, translation_, cable_ = rotation, translation, cable_
    R_l_0, R_l_1, R_r_0, R_r_1 = unpack_rotation(rotation_)
    t_l_0, t_l_1 = unpack_translation(translation_)
    cable_, dcable_ = unpack_cable(cable_)
    y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), dcable_ if ifdcable else cable_,
                    unpack_rotation(rotation), unpack_translation(translation))
    #y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), dcable_ if ifdcable else cable_)
    #y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
    y_pred = y_pred_
    return y_pred + cable[..., :3]


def compute_length(cp):
    pts = bsp.N @ cp
    diff = np.linalg.norm(np.diff(pts, axis=-2), axis=-1)
    length = np.sum(diff, axis=-1)
    return length


dataset_epoch = ds#.shuffle(ds_size)
dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
epoch_loss = []
prediction_losses = []
pts_losses_abs = []
pts_losses_euc = []
ratio_losses = []
l12_losses = []
l12ref = []
L3_gtpreds = []
L3_gtcables = []
test_data_filtered = []
data = []
for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, ds_size, 0, args.batch_size):
    #c0_len = np.sum(np.linalg.norm(cable[:, 1:, :3] - cable[:, :-1, :3], axis=-1), axis=-1)
    #c1_len = np.sum(np.linalg.norm(y_gt[:, 1:] - y_gt[:, :-1], axis=-1), axis=-1)

    #translation = translation / c0_len[:, np.newaxis]

    t0 = perf_counter()
    y_pred = inference(rotation, translation, cable)
    t1 = perf_counter()
    print(t1 - t0)
    pts_loss_abs, pts_loss_euc, pts_loss_l2 = loss(y_gt, y_pred)

    cable, dcable_ = unpack_cable(cable)

    # ratio = np.linalg.norm(y_gt - y_pred, axis=-1) / (np.linalg.norm(y_gt - cable, axis=-1) + 1e-8)
    # ratio_loss = tf.reduce_mean(ratio, axis=-1)
    #ratio_loss = np.mean(np.linalg.norm(y_gt - y_pred, axis=-1), axis=-1) / (
    #            np.mean(np.linalg.norm(y_gt - cable, axis=-1), axis=-1) + 1e-8)
    for k in range(y_pred.shape[0]):
        #frechet_dist_gtpred = frdist(y_gt[i], y_pred[i])
        #frechet_dist_gtcable = frdist(y_gt[i], cable[i])
        #dtw_gtpred = dtw(y_gt[i], y_pred[i])
        #dtw_gtcable = dtw(y_gt[i], cable[i])
        L3_gtpred = calculateL3(y_gt[k].numpy().T, y_pred[k].numpy().T)
        L3_gtcable = calculateL3(y_gt[k].numpy().T, cable[k].numpy().T)
        #if L3_gtcable < 0.02:
        #    continue
        L3_gtpreds.append(L3_gtpred)
        L3_gtcables.append(L3_gtcable)
        a = 0
        ratio_loss = L3_gtpred / (L3_gtcable + 1e-8)
        print("L3 PRED:", L3_gtpred)
        print("L3 DIFF:", L3_gtcable)
        print("RATIO:", ratio_loss)
        ratio_losses.append(ratio_loss)
        if ratio_loss < 0.3:
            test_data_filtered.append(test_data[i * args.batch_size + k])
        data.append((cable[k], y_gt[k], y_pred[k], ratio_loss))


    _, _, l12_gtpred = loss_all2all(y_gt, y_pred)
    _, _, l12_gtcable = loss_all2all(y_gt, cable)
    #ratio_loss = l12_gtpred / (l12_gtcable + 1e-8)
    #print(ratio_loss)
    # ratio_loss = tf.reduce_sum(np.abs(y_gt - y_pred) / (np.abs(y_gt - cable) + 1e-8), axis=(-2, -1))

    prediction_loss = pts_loss_abs

    reg_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables
                         if 'bias' not in v.name])
    model_loss = prediction_loss

    cp_pred = y_pred[0]
    cp_gt = y_gt[0]
    cp0 = cable[0]

    if plot:
        xl = -0.1
        xh = 0.5
        yl = -0.3
        yh = 0.3
        zl = -0.3
        zh = 0.3
        plt.clf()
        plt.subplot(221)
        plt.xlim(xl, xh)
        plt.ylim(yl, yh)
        plt.plot(cp_gt[:, 1], cp_gt[:, 2], 'rx')
        plt.plot(cp_pred[:, 1], cp_pred[:, 2], 'bo')
        plt.plot(cp0[:, 1], cp0[:, 2], 'g^')
        plt.subplot(223)
        plt.xlim(zl, zh)
        plt.ylim(xl, xh)
        plt.plot(cp_gt[:, 0], cp_gt[:, 1], 'rx')
        plt.plot(cp_pred[:, 0], cp_pred[:, 1], 'bo')
        plt.plot(cp0[:, 0], cp0[:, 1], 'g^')

        v_pred = (bsp.N[0] @ cp_pred)
        v_gt = (bsp.N[0] @ cp_gt)
        v_base = (bsp.N[0] @ cp0)
        plt.subplot(222)
        #plt.xlim(xl, xh)
        #plt.ylim(yl, yh)
        plt.plot(v_pred[:, 1], v_pred[:, 2], label="pred")
        plt.plot(v_gt[:, 1], v_gt[:, 2], label="gt")
        plt.plot(v_base[:, 1], v_base[:, 2], label="base")
        plt.axis('scaled')
        #plt.legend()
        plt.subplot(224)
        #plt.xlim(zl, zh)
        #plt.ylim(xl, xh)
        plt.plot(v_pred[:, 1], v_pred[:, 0], label="pred")
        plt.plot(v_gt[:, 1], v_gt[:, 0], label="gt")
        plt.plot(v_base[:, 1], v_base[:, 0], label="base")
        plt.axis('scaled')
        # plt.xlim(-0.1, 0.7)
        # plt.ylim(-0.4, 0.4)
        plt.legend()
        #plt.show()
        plt.savefig(f"pred_imgs_55/{ratio_loss:.5f}.pdf")
        a = 0

    epoch_loss.append(model_loss)
    pts_losses_abs.append(pts_loss_abs)
    prediction_losses.append(prediction_loss)
    pts_losses_euc.append(pts_loss_euc)
    l12_losses.append(l12_gtpred)
    l12ref.append(l12_gtcable)

ratio_losses = np.array(ratio_losses)
L3_gtpreds = np.array(L3_gtpreds)
L3_gtcables = np.array(L3_gtcables)
l12_losses = np.concatenate(l12_losses, -1)
l12ref = np.concatenate(l12ref, -1)


test_data_filtered = np.stack(test_data_filtered, axis=0)
np.savetxt(args.dataset_path.replace("train", "test_filtered"), test_data_filtered.astype(np.float32), delimiter='\t')

# plt.subplot(121)
# plt.hist(gt_energies, bins=50, color='r')
# plt.hist(pred_energies, bins=50, color='b')
# plt.subplot(122)
# plt.hist(diff_energies, bins=50)
# plt.show()
plt.subplot(121)
plt.hist(ratio_losses, bins=50)
plt.subplot(122)
plt.scatter(L3_gtcables, L3_gtpreds)
plt.show()

epoch_loss = tf.reduce_mean(tf.concat(epoch_loss, -1))
prediction_losses = tf.reduce_mean(tf.concat(prediction_losses, -1))
pts_losses_abs = tf.reduce_mean(tf.concat(pts_losses_abs, -1))
pts_losses_euc = tf.reduce_mean(tf.concat(pts_losses_euc, -1))
ratio_losses_mean = np.mean(ratio_losses)
ratio_losses_med = np.median(ratio_losses)
l12_losses_mean = np.mean(l12_losses)
l12_losses_med = np.median(l12_losses)
l12ref_mean = np.mean(l12ref)
l12ref_med = np.median(l12ref)
print("EPOCH LOSS:", epoch_loss)
print("PREDICTION LOSS:", prediction_losses)
print("PTS ABS LOSS:", pts_losses_abs)
print("PTS EUC LOSS:", pts_losses_euc)
print("RATIO LOSS MEAN:", ratio_losses_mean)
print("RATIO LOSS MED:", ratio_losses_med)
print("L12 LOSS MEAN:", l12_losses_mean)
print("L12 LOSS MED:", l12_losses_med)
print("L12REF LOSS MEAN:", l12ref_mean)
print("L12REF LOSS MED:", l12ref_med)
