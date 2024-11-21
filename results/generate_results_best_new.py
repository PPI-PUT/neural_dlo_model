from glob import glob
from itertools import product
import matplotlib.pyplot as plt

import numpy as np

# data = np.load("results_all_new_aug_.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb_sep.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb_inbilstm_.npy", allow_pickle=True)[()]
from matplotlib.lines import Line2D

data = {}
# results_dir = "all_mb_03_27"
# results_dir = "all_mb_zoval_04_25"
# prefix = "new_mb_03_27_poc64_lr5em4_bs128"
# prefix = f"new_mb_zoval_04_25_poc64_lr5em4_bs128"
results_dir = "06_09_final_off3cm_50cm"
prefix = f"06_09_final_off3cm_50cm_lr5em4_bs128"
for r in glob(f"{results_dir}/{prefix}*.npy"):
    d = np.load(r, allow_pickle=True)[()]
    name = r.split("/")[-1][len(prefix) + 1:-4]
    data[name] = d


def get_minimum(data, field):
    min_value = 1e10
    min_name = ""
    for k, v in data.items():
        if v[field] < min_value:
            min_name = k
            min_value = v[field]
    return min_name, min_value


def boxplot(data, field):
    keys = list(sorted(data.keys()))
    v = [data[k][field] * 100. for k in keys]

    medians = [np.median(x) for x in v]
    idx = np.argsort(medians)
    results = [(keys[i], medians[i]) for i in idx]

    sep_noaug_results = [(k, v) for k, v in results if k.startswith("sep") and k.endswith("noaug")]
    sep_aug_results = [(k, v) for k, v in results if k.startswith("sep") and k.endswith("augwithzeros")]
    inbilstm_noaug_results = [(k, v) for k, v in results if k.startswith("inbilstm") and k.endswith("noaug")]
    inbilstm_aug_results = [(k, v) for k, v in results if k.startswith("inbilstm") and k.endswith("augwithzeros")]
    transformer_noaug_results = [(k, v) for k, v in results if k.startswith("transformer") and k.endswith("noaug")]
    transformer_aug_results = [(k, v) for k, v in results if k.startswith("transformer") and k.endswith("augwithzeros")]
    jac_noaug_results = [(k, v) for k, v in results if k.startswith("jactheir") and k.endswith("noaug")]
    jac_aug_results = [(k, v) for k, v in results if k.startswith("jactheir") and k.endswith("augwithzeros")]
    lin_noaug_results = [(k, v) for k, v in results if k.startswith("lin") and k.endswith("noaug")]
    lin_aug_results = [(k, v) for k, v in results if k.startswith("lin") and k.endswith("augwithzeros")]

    sep_results = [x for x in zip(sorted(sep_noaug_results, key=lambda x: x[0]), sorted(sep_aug_results, key=lambda x: x[0]))]
    inbilstm_results = [x for x in zip(sorted(inbilstm_noaug_results, key=lambda x: x[0]), sorted(inbilstm_aug_results, key=lambda x: x[0]))]
    transformer_results = [x for x in zip(sorted(transformer_noaug_results, key=lambda x: x[0]), sorted(transformer_aug_results, key=lambda x: x[0]))]
    jac_results = [x for x in zip(sorted(jac_noaug_results, key=lambda x: x[0]), sorted(jac_aug_results, key=lambda x: x[0]))]
    lin_results = [x for x in zip(sorted(lin_noaug_results, key=lambda x: x[0]), sorted(lin_aug_results, key=lambda x: x[0]))]

    sep_results = sorted(sep_results, key=lambda x: x[1][1])
    inbilstm_results = sorted(inbilstm_results, key=lambda x: x[1][1])
    transformer_results = sorted(transformer_results, key=lambda x: x[1][1])
    jac_results = sorted(jac_results, key=lambda x: x[1][1])
    lin_results = sorted(lin_results, key=lambda x: x[1][1])

    sep_improvements = [(x[0][1] - x[1][1]) / x[0][1] for x in sep_results]
    inbilstm_improvements = [(x[0][1] - x[1][1]) / x[0][1] for x in inbilstm_results]
    transformer_improvements = [(x[0][1] - x[1][1]) / x[0][1] for x in transformer_results]
    jac_improvements = [(x[0][1] - x[1][1]) / x[0][1] for x in jac_results]
    lin_improvements = [(x[0][1] - x[1][1]) / x[0][1] for x in lin_results]

    #keys_noaug = [x[0][0] for x in [transformer_noaug_results, inbilstm_noaug_results, sep_noaug_results, jac_noaug_results]]
    #keys_aug = [x[0][0] for x in [transformer_aug_results, inbilstm_aug_results, sep_aug_results, jac_aug_results]]
    keys_noaug = [x[0][0] for x in [transformer_noaug_results, lin_noaug_results, inbilstm_noaug_results, sep_noaug_results, jac_noaug_results]]
    keys_aug = [x[0][0] for x in [transformer_aug_results, lin_aug_results, inbilstm_aug_results, sep_aug_results, jac_aug_results]]

    mul = 100.
    noaug_v = [data[k][field] * mul for k in keys_noaug]
    aug_v = [data[k][field] * mul for k in keys_aug]

    v = noaug_v + aug_v
    keys = keys_noaug + keys_aug

    colors = []
    a = 0

    #positions = list(range(9))
    #del positions[4]

    positions = list(range(11))
    del positions[5]

    # plt.figure(figsize=(10, 5))
    # plt.figure(figsize=(10, 4))
    plt.figure(figsize=(12, 4))
    bp = plt.boxplot(v, positions=positions,
                     showmeans=True, showfliers=False,
                     # patch_artist=True, notch=True,
                     meanprops=dict(marker="+", markeredgecolor="black"),
                     # medianprops=dict(color="magenta"),
                     boxprops=dict(linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2),
                     # meanprops=dict(marker="+", markeredgecolor="tab:cyan"),
                     # boxprops=dict(facecolor=c, color=c),
                     # capprops=dict(color=c),
                     # whiskerprops=dict(color=c),
                     # flierprops=dict(color=c, markeredgecolor=c),
                     # medianprops=dict(color=c),
                     widths=0.75)
    # bp = ax.boxplot(collection[i], positions=[0, spacing])
    for i in range(len(bp["boxes"])):
        lw = 2.
        if "transformer" in keys[i]:
            c = 'tab:blue'
        elif "inbilstm" in keys[i]:
            c = 'tab:orange'
        elif "sep" in keys[i]:
            c = 'tab:green'
        elif "jactheir" in keys[i]:
            c = 'tab:red'
        elif "lin" in keys[i]:
            c = 'tab:purple'
        else:
            c = [0., 0., 1.]

        bp["boxes"][i].set_color(c)
        bp["boxes"][i].set_fillstyle("full")
        bp["boxes"][i].set_markerfacecolor("g")
        bp["boxes"][i].set_linewidth(lw)
        # bp["fliers"][i].set_color(c)
        bp["whiskers"][2 * i].set_color(c)
        bp["whiskers"][2 * i + 1].set_color(c)
        bp["caps"][2 * i].set_color(c)
        # bp["caps"][2 * i].set_linestyle(cs)
        bp["caps"][2 * i + 1].set_color(c)
        # bp["caps"][2 * i + 1].set_linestyle(cs)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    custom_lines = [
        Line2D([0], [0], color="tab:blue", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:purple", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:orange", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:green", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:red", lw=2, linestyle="-"),
    ]
    #ax.legend(custom_lines, ['Transformer', "INBiLSTM", 'MLP', 'JacMLP'],
    ax.legend(custom_lines, ['Transformer', "Lin", "INBiLSTM", 'MLP', 'JacMLP'],
              bbox_to_anchor=(0.95, 0.0), frameon=False, ncol=5)
    plt.subplots_adjust(bottom=0.2)
    #plt.ylim(0., 60.)

    plt.grid(axis="y")
    #ax.set_xticks([])
    ax.set_xticks([1.5, 6.5])
    #ax.set_xticklabels(["without augmentation", "with augmentation"])
    ax.set_xticklabels(["baseline experiment", "with augmentation"])
    ax.tick_params(axis='x', which='both', length=0)
    ax.xaxis.tick_top()
    #plt.setp(ax.get_xticks(), visible=False)
    # ax.set_xticks(positions)
    # ax.set_xticklabels(names)
    # plt.xticks(rotation=45)
    plt.show()


def sort(data, field):
    l = [(k, np.mean(v[field])) for k, v in data.items()]
    l = sorted(l, key=lambda x: x[1])
    return l


def row_noaug(data, method, metric):
    row = f"{method} "
    for q, d, c in product(["quat", "rotmat", "rotvec"],
                           ["diff", "nodiff"],
                           ["cable", "dcable"]):
        # for c, q, d in product(["cp", "poc"],
        #                       ["quat", "rot"],
        #                       ["diff", ""]):
        key = f'{method}_{d}_{q}_{c}_augwithzeros'
        # key = f'{method}_{d}_{q}_{c}_noaug'
        if key not in data.keys():
            row += "& XXX "
            continue
        e = data[key]
        mul = 100.
        row += f"& ${mul * np.mean(e[metric]):.1f}\pm{mul * np.std(e[metric]):.1f}$ "
    row += "\\\\"
    return row


boxplot(data, "ratio_loss")
