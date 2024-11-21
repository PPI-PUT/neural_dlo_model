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
from models.jacobian_neural_predictor import JacobianNeuralPredictor, JacobianRBFN

np.random.seed(444)

# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_diff_rotmat_dcable_augwithzeros_/checkpoints"
# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_nodiff_rotvec_dcable_augwithzeros/checkpoints"
# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_diff_rotvec_cable_augwithzeros/checkpoints"
# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_nodiff_rotmat_dcable_noaug/checkpoints"
# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_nodiff_rotmat_dcable_augwithzeros/checkpoints"
# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_inbilstm_nodiff_rotvec_cable_noaug/checkpoints"
# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_inbilstm_nodiff_rotvec_cable_augwithzeros/checkpoints"

# path = f"../trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_sep_nodiff_rotmat_dcable_augwithzeros/checkpoints"
# path = f"./trained_models/all_mb_03_27/new_mb_03_27_poc64_lr5em4_bs128_inbilstm_nodiff_rotvec_cable_augwithzeros/checkpoints"

# path = f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_cable_augwithzeros/checkpoints"
# path = f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_inbilstm_nodiff_rotvec_cable_augwithzeros/checkpoints"
# path = f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotvec_dcable_augwithzeros/checkpoints"
#paths = [
#    f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_cable_augwithzeros/checkpoints",
#    f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_inbilstm_nodiff_rotvec_cable_augwithzeros/checkpoints",
#    f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotvec_dcable_augwithzeros/checkpoints",
#]
paths = [
    #"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_jactheir_diff_rotmat_dcable_augwithzeros/checkpoints"
    #"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_sep_nodiff_rotmat_cable_augwithzeros/checkpoints",
    f"./trained_models/06_09_final_off3cm_50cm/06_09_final_off3cm_50cm_lr5em4_bs128_transformer_diff_rotvec_dcable_augwithzeros/checkpoints",
]


class args:
    working_dir = '../trainings'
    # dataset_path = f"../data/prepared_datasets/new_mb_03_27_poc64/train.tsv"
    dataset_path = f"./data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"
    # dataset_path = "./data/prepared_datasets/06_09_final_off3cm_50cm/train.tsv"


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


for path in paths:
    name = path.split("/")[-2]
    name_fields = name.split("_")

    m = name_fields[7]
    d = name_fields[8]
    q = name_fields[9]
    c = name_fields[10]
    a = name_fields[11]

    time_stats = {}

    train_ds, train_size, tX1, tX2, tX3, tY = prepare_dataset_cond(args.dataset_path, rot=q, diff=(d == "diff"),
                                                                   augment=(a == "augwithzeros"))
    val_ds, val_size, vX1, vX2, vX3, vY = prepare_dataset_cond(args.dataset_path.replace("train", "val"), rot=q,
                                                               diff=(d == "diff"))
    test_ds, test_size, teX1, teX2, teX3, teY = prepare_dataset_cond(args.dataset_path.replace("train", "test"), rot=q,
                                                                     diff=(d == "diff"))

    ds_stats = compute_ds_stats(train_ds)

    # ds, ds_size = train_ds, train_size
    # ds, ds_size = val_ds, val_size
    ds, ds_size = test_ds, test_size

    if m == "sep":
        model = SeparatedNeuralPredictor()
    elif m == "cnn":
        model = CNN()
    elif m == "inbilstm":
        model = INBiLSTM()
    elif m == "transformer":
        model = Transformer(num_layers=2, num_heads=8, dff=256, d_model=64, dropout_rate=0.1, target_size=3)
    elif m == "jactheir":
        model = JacobianNeuralPredictor(q, d == "diff")
    else:
        print("WRONG MODEL NAME")
        assert False

    ckpt = tf.train.Checkpoint(model=model)
    best_list = list(glob(os.path.join(path, "best-*.index")))
    assert best_list
    best = best_list[0][:-6]
    ckpt.restore(best).expect_partial()
    # ckpt.restore(best)

    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        dataset_epoch = ds.shuffle(ds_size)
        dataset_epoch = dataset_epoch.batch(bs).prefetch(bs)
        times = []
        for i, rotation, translation, cable, y_gt in _ds('Train', dataset_epoch, ds_size, 0, bs):
            t0 = perf_counter()
            y_pred = inference(rotation, translation, cable)
            t1 = perf_counter()
            times.append(t1 - t0)

        print()
        print(bs)
        print(len(times))
        print(np.mean(times))
        print()
        time_stats[bs] = np.array(times)

    os.makedirs(f"timings_cpu", exist_ok=True)
    np.save(f"timings_cpu/{name}.npy", time_stats)
    #os.makedirs(f"timings_gpu", exist_ok=True)
    #np.save(f"timings_gpu/{name}.npy", time_stats)
