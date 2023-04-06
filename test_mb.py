import os
from glob import glob
from itertools import product
from time import perf_counter

from losses.cable import CableBSplineLoss
from losses.cable_pts import CableBSplinePtsLoss, CablePtsLoss
from models.cnn import CNN
from models.inbilstm import INBiLSTM
from models.separated_cnn_neural_predictor import SeparatedCNNNeuralPredictor
from models.separated_neural_predictor import SeparatedNeuralPredictor
from utils.bspline import BSpline
from utils.constants import BSplineConstants
from utils.geometry import calculateL3

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.dataset import _ds, prepare_dataset, whiten, mix_datasets, whitening, compute_ds_stats, unpack_translation, \
    unpack_rotation, prepare_dataset_cond, unpack_cable
from utils.execution import ExperimentHandler
from models.basic_neural_predictor import BasicNeuralPredictor

np.random.seed(444)


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

#mod = ["sep", "inbilstm"]
#mod = ["sep"]
mod = ["inbilstm"]
qr = ["quat", "rotmat", "rotvec"]
diff = ["nodiff", "diff"]
aug = ["noaug", "augwithzeros"]
cab = ["cable", "dcable"]

results = {}

#for m, p, q, d in product(mod, pts, qr, diff):
#    print(m, p, q, d)
for m, q, c, d, a in product(mod, qr, cab, diff, aug):
    print(m, q, c, d, a)

    class args:
        batch_size = 128
        working_dir = './trainings'
        dataset_path = f"./data/prepared_datasets/new_mb_03_27_poc64/train.tsv"


    train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset_path, rot=q, diff=(d == "diff"), augment=(a == "augwithzeros"))
    val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset_cond(args.dataset_path.replace("train", "val"), rot=q, diff=(d == "diff"))
    test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset_path.replace("train", "test"), rot=q, diff=(d == "diff"))

    ds_stats = compute_ds_stats(train_ds)

    #ds, ds_size = train_ds, train_size
    #ds, ds_size = val_ds, val_size
    ds, ds_size = test_ds, test_size

    loss = CablePtsLoss()

    if m == "sep":
        model = SeparatedNeuralPredictor()
    elif m == "cnn":
        model = CNN()
    elif m == "inbilstm":
        model = INBiLSTM()
    else:
        print("WRONG MODEL NAME")
        assert False

    ckpt = tf.train.Checkpoint(model=model)
    name = f"new_mb_03_27_poc64_lr5em4_bs128_{m}_{d}_{q}_{c}_{a}"
    dirname = "all_mb_03_27"
    path = f"./trained_models/{dirname}/{name}/checkpoints"
    print(path)
    best_list = list(glob(os.path.join(path, "best-*.index")))
    if not best_list:
        continue
    best = best_list[0][:-6]
    ckpt.restore(best).expect_partial()
    #ckpt.restore(best)


    def inference(rotation, translation, cable):
        rotation_, translation_, cable_ = whitening(rotation, translation, cable, ds_stats)
        R_l_0, R_l_1, R_r_0, R_r_1 = unpack_rotation(rotation_)
        t_l_0, t_l_1 = unpack_translation(translation_)
        cable_, dcable_ = unpack_cable(cable_)
        y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), dcable_ if c == "dcable" else cable_,
                        unpack_rotation(rotation), unpack_translation(translation))
        # y_pred_ = model((R_l_0, R_l_1, R_r_0, R_r_1), (t_l_0, t_l_1), dcable_ if ifdcable else cable_)
        # y_pred = y_pred_ * ds_stats["sy"] + ds_stats["my"]
        y_pred = y_pred_
        return y_pred + cable[..., :3]


    dataset_epoch = ds.shuffle(ds_size)
    dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
    epoch_loss = []
    pts_losses_abs = []
    pts_losses_euc = []
    ratio_losses = []
    for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, ds_size, 0, args.batch_size):
        y_pred = inference(rotation, translation, cable)
        pts_loss_abs, pts_loss_euc, pts_loss_l2 = loss(y_gt, y_pred)

        cable, dcable_ = unpack_cable(cable)
        for i in range(y_pred.shape[0]):
            # frechet_dist_gtpred = frdist(y_gt[i], y_pred[i])
            # frechet_dist_gtcable = frdist(y_gt[i], cable[i])
            # dtw_gtpred = dtw(y_gt[i], y_pred[i])
            # dtw_gtcable = dtw(y_gt[i], cable[i])
            L3_gtpred = calculateL3(y_gt[i].numpy().T, y_pred[i].numpy().T)
            L3_gtcable = calculateL3(y_gt[i].numpy().T, cable[i].numpy().T)
            ratio_loss = L3_gtpred / (L3_gtcable + 1e-8)
            print("L3:", L3_gtpred)
            print("RATIO:", ratio_loss)
            ratio_losses.append(ratio_loss)

        pts_losses_abs.append(pts_loss_abs)
        pts_losses_euc.append(pts_loss_euc)

    pts_loss_abs = tf.concat(pts_losses_abs, -1).numpy()
    pts_loss_euc = tf.concat(pts_losses_euc, -1).numpy()
    ratio_losses = np.array(ratio_losses)

    #results[m + "_" + q + "_" + c + "_" + d + "_" + a] = {
    results = {
        #"mean_ratio_loss": mean_ratio_losses, "std_ratio_loss": std_ratio_losses,
        #"mean_pts_loss_abs": mean_pts_losses_abs, "std_pts_loss_abs": std_pts_losses_abs,
        #"mean_pts_loss_euc": mean_pts_losses_euc, "std_pts_loss_euc": std_pts_losses_euc,
        "ratio_loss": ratio_losses,
        "pts_loss_abs": pts_losses_abs,
        "pts_loss_euc": pts_losses_euc,
    }

    np.save(f"results/{dirname}/{name}.npy", results)

#np.save("results_all_new_mb.npy", results)