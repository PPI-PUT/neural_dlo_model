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
from models.transformer import Transformer
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
from models.linear_neural_predictor import LinearNeuralPredictor
from models.jacobian_neural_predictor import JacobianNeuralPredictor, JacobianRBFN

np.random.seed(444)

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

results = {}

#dlo_len = 40
#dlo_len = 45
#scaled = False
#scaled = True


class args:
    batch_size = 128
    working_dir = '../trainings'
    dataset45_path = f"../data/prepared_datasets/06_09_final_off3cm_45cm/train.tsv"
    dataset40_path = f"../data/prepared_datasets/06_09_final_off3cm_40cm/train.tsv"
    dataset50_path = f"../data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"
    dataset55_path = f"../data/prepared_datasets/04_17_final_off3cm_55cm/train.tsv"


def test(dlo_len, scaled):
    # for path in glob(f"../trained_models/lengths2_mb_03_04/new_mb_*100percent40cm_04_03_poc64_lr5em4_bs128_sep_nodiff_rotmat_dcable_augwithzeros"):
    #for path in glob(f"../trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_40cm_lr5em4_bs128_transformer_diff_rotvec_dcable_augwithzeros"):
    #for path in glob(f"../trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_50cm*"):
    for path in glob(f"../trained_models/06_09_final_off3cm_lengths/06_09_final_off3cm_4*"):
        if not "jactheir" in path: continue # only jactheir
        if "transformer01drop" in path: continue
        #if not "transformer_" in path: continue
        #for path in glob(f"../trained_models/06_09_final_off3cm_lengths/*"):
        name = path.split("/")[-1]
        print(name)
        if "lennorm" in name:
            continue
        if "40cm50cm" in name:
            continue
        if dlo_len == 40 and "_45cm_" in name:
            continue
        if dlo_len == 45 and "_40cm_" in name:
            continue
        fields = name.split("_")

        q = fields[9]
        d = fields[8] == "diff"
        m = fields[7]
        c = fields[10]

        model_len = dlo_len
        train_ds_type = fields[4]
        if train_ds_type == "40cm":
            train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset40_path, rot=q, diff=d, augment=True)
            model_len = 40.
        elif train_ds_type == "45cm":
            train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset45_path, rot=q, diff=d, augment=True)
            model_len = 45.
        elif train_ds_type == "50cm":
            train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset50_path, rot=q, diff=d, augment=True)
            model_len = 50.
        elif train_ds_type == "40cm45cm50cm":
           train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset40_path, rot=q, diff=d, augment=True)
           train1_ds, train1_size, t1X1, t1X2, t1X3, t1Y = prepare_dataset_cond(args.dataset45_path, rot=q, diff=d, augment=True)
           train2_ds, train2_size, t2X1, t2X2, t2X3, t2Y = prepare_dataset_cond(args.dataset50_path, rot=q, diff=d, augment=True)
           train_ds = train_ds.concatenate(train1_ds).concatenate(train2_ds)
           train_size = train_size + train1_size + train2_size
        # elif train_ds_type == "40cm50cm":
        #    train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset40_path, rot=q, diff=d, augment=True)
        #    train1_ds, train1_size, t1X1, t1X2, t1X3, t1Y = prepare_dataset_cond(args.dataset50_path, rot=q, diff=d,
        #                                                                         augment=True)
        #    train_ds = train_ds.concatenate(train1_ds)
        #    train_size = train_size + train1_size
        else:
            print("WRONG TRAINING DATASET TYPE")
            assert False

        # continue
        # train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset_path, rot=q, diff=d)
        # val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset_cond(args.dataset_path.replace("train", "val"), rot=q, diff=d)
        scale = 1.
        if scaled:
            scale = float(model_len) / float(dlo_len)
        #continue
        if dlo_len == 45:
            test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset45_path.replace("train", "test"),
                                                                             rot=q, diff=d, scale=scale)
        elif dlo_len == 40:
            test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset40_path.replace("train", "test"),
                                                                             rot=q, diff=d, scale=scale)
        elif dlo_len == 50:
            test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset50_path.replace("train", "test"),
                                                                             rot=q, diff=d, scale=scale)
        elif dlo_len == 55:
            test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset55_path.replace("train", "test"),
                                                                             rot=q, diff=d, scale=scale)

        ds_stats = compute_ds_stats(train_ds)

        # ds, ds_size = train_ds, train_size
        # ds, ds_size = val_ds, val_size
        ds, ds_size = test_ds, test_size

        loss = CablePtsLoss()

        if m == "sep":
            model = SeparatedNeuralPredictor()
        #elif m == "transformer":
        elif "transformer" in m:
            if "drop" in m:
                model = Transformer(num_layers=2, num_heads=8, dff=256, d_model=64, dropout_rate=0.1, target_size=3)
            else:
                model = Transformer(num_layers=2, num_heads=8, dff=256, d_model=64, dropout_rate=0.0, target_size=3)
        elif m == "inbilstm":
            model = INBiLSTM()
        elif m == "lin":
            model = LinearNeuralPredictor()
        elif m == "jactheir":
            model = JacobianNeuralPredictor(q, d)
        else:
            print("WRONG MODEL NAME")
            assert False

        ckpt = tf.train.Checkpoint(model=model)
        print(path)
        best_list = list(glob(os.path.join(path, "checkpoints", "best-*.index")))
        if not best_list:
            continue
        best = best_list[0][:-6]
        ckpt.restore(best).expect_partial()


        # ckpt.restore(best)

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

        # results[m + "_" + q + "_" + c + "_" + d + "_" + a] = {
        results = {
            # "mean_ratio_loss": mean_ratio_losses, "std_ratio_loss": std_ratio_losses,
            # "mean_pts_loss_abs": mean_pts_losses_abs, "std_pts_loss_abs": std_pts_losses_abs,
            # "mean_pts_loss_euc": mean_pts_losses_euc, "std_pts_loss_euc": std_pts_losses_euc,
            "ratio_loss": ratio_losses,
            "pts_loss_abs": pts_losses_abs,
            "pts_loss_euc": pts_losses_euc,
        }

        dirname = "06_09_final_off3cm_lengths"
        #dirname = "06_09_final_off3cm_lengths_scaled"
        os.makedirs(dirname, exist_ok=True)
        np.save(f"{dirname}/test{dlo_len}{'scaled' if scaled else ''}_{name}.npy", results)

    # np.save("results_all_new_mb.npy", results)

#dlo_len = 40
#dlo_len = 45
#scaled = False
#scaled = True

for dlo_len in [40, 45]:
#for dlo_len in [55]:
    for scaled in [True, False]:
    #for scaled in [False]:
        test(dlo_len, scaled)