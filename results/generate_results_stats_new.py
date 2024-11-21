from glob import glob
from itertools import product
import matplotlib.pyplot as plt

import numpy as np

#data = np.load("results_all_new_aug_.npy", allow_pickle=True)[()]
#data = np.load("results_all_new_mb.npy", allow_pickle=True)[()]
#data = np.load("results_all_new_mb_sep.npy", allow_pickle=True)[()]
#data = np.load("results_all_new_mb_inbilstm_.npy", allow_pickle=True)[()]
from matplotlib.lines import Line2D

data = {}
#results_dir = "all_mb_03_27"
#results_dir = "all_mb_zoval_04_25"
#prefix = "new_mb_03_27_poc64_lr5em4_bs128"
#prefix = f"new_mb_zoval_04_25_poc64_lr5em4_bs128"
results_dir = "06_09_final_off3cm_50cm"
prefix = f"06_09_final_off3cm_50cm_lr5em4_bs128"
for r in glob(f"{results_dir}/{prefix}*.npy"):
    d = np.load(r, allow_pickle=True)[()]
    name = r.split("/")[-1][len(prefix)+1:-4]
    data[name] = d



def get_minimum(data, field):
    min_value = 1e10
    min_name = ""
    for k, v in data.items():
        if v[field] < min_value:
            min_name = k
            min_value = v[field]
    return min_name, min_value


def boxplot(data, field, aug):
    keys = sorted(data.keys())
    aug = "augwithzeros" if aug else "noaug"
    keys = [k for k in keys if aug in k]
    v = [data[k][field] * 100. for k in keys]

    medians = [np.median(x) for x in v]
    idx = np.argsort(medians)
    results = [(keys[i], medians[i]) for i in idx]

    #valid = data[results[0][0]]["L3_loss"] < 0.07
    #v = [x[valid] for x in v]
    #medians = [np.median(x) for x in v]
    #idx = np.argsort(medians)
    #results = [(keys[i], medians[i]) for i in idx]


    #a = data[results[0][0]]
    #plt.subplot(121)
    #plt.plot(a["ratio_loss"], '.')
    #plt.subplot(122)
    #plt.plot(a["L3_loss"], '.')
    #plt.show()

    v = [v[i] for i in idx]
    keys = [keys[i] for i in idx]
    colors = []
    a = 0
    positions = list(range(len(v)))
    #del positions[6]
    #plt.figure(figsize=(10, 5))
    #plt.figure(figsize=(10, 4))
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
        bs = "-"
        cs = "-"
        cc = False
        lw = 2.
        if "inbilstm" in keys[i]:
            c = 'tab:blue'
        elif "sep" in keys[i]:
            c = 'tab:orange'
        elif "transformer" in keys[i]:
            c = 'tab:green'
        elif "jactheir" in keys[i]:
            c = 'tab:red'
        else:
            c = [0., 0., 1.]

        if "dcable" in keys[i]:
            bs = ":"

        if "_diff_" in keys[i]:
            #cs = ":"
            cc = "tab:purple"

        if "quat" in keys[i]:
            ws = "-"
        elif "rotmat" in keys[i]:
            ws = "--"
        elif "rotvec" in keys[i]:
            ws = "-."
        bp["boxes"][i].set_color(c)
        bp["boxes"][i].set_fillstyle("full")
        bp["boxes"][i].set_markerfacecolor("g")
        bp["boxes"][i].set_linewidth(lw)
        bp["boxes"][i].set_linestyle(bs)
        # bp["fliers"][i].set_color(c)
        bp["whiskers"][2 * i].set_color(c)
        bp["whiskers"][2 * i + 1].set_color(c)
        bp["whiskers"][2 * i].set_linestyle(ws)
        bp["whiskers"][2 * i + 1].set_linestyle(ws)
        bp["caps"][2 * i].set_color(c if not cc else cc)
        #bp["caps"][2 * i].set_linestyle(cs)
        bp["caps"][2 * i + 1].set_color(c if not cc else cc)
        #bp["caps"][2 * i + 1].set_linestyle(cs)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    custom_lines = [
        Line2D([0], [0], color="tab:blue", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:orange", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:green", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:red", lw=2, linestyle="-"),
        Line2D([0], [0], color="tab:purple", lw=2, linestyle="-"),
        Line2D([0], [0], color="k", lw=2, linestyle="-"),
        Line2D([0], [0], color="k", lw=2, linestyle="--"),
        Line2D([0], [0], color="k", lw=2, linestyle="-."),
        Line2D([0], [0], color="k", lw=2, linestyle=":"),
    ]
    if aug == "noaug":
        ax.legend(custom_lines, ["INBiLSTM", 'MLP', 'Transformer', 'JacMLP', "diff",
                                 'quaternion', "rotation matrix", "axis angle", "DLO directions"],
                  bbox_to_anchor=(0.9, 0.0), frameon=False, ncol=5)
    plt.subplots_adjust(bottom=0.2)
    plt.ylim(0., 70.)

    ax.set_xticks([])
    #ax.set_xticks(positions)
    #ax.set_xticklabels(names)
    #plt.xticks(rotation=45)
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
    #for c, q, d in product(["cp", "poc"],
    #                       ["quat", "rot"],
    #                       ["diff", ""]):
        key = f'{method}_{d}_{q}_{c}_augwithzeros'
        #key = f'{method}_{d}_{q}_{c}_noaug'
        if key not in data.keys():
            row += "& XXX "
            continue
        e = data[key]
        mul = 100.
        row += f"& ${mul * np.mean(e[metric]):.1f}\pm{mul * np.std(e[metric]):.1f}$ "
    row += "\\\\"
    return row


#sd = sort(data, "mean_pts_loss_euc")
sd = sort(data, "ratio_loss")
#boxplot(data, "ratio_loss", False)
boxplot(data, "ratio_loss", True)
print(sd)
print(row_noaug(data, "sep", "ratio_loss"))
#print(row_noaug(data, "sep", "pts_loss_euc"))
#print(row_noaug(data, "cnn", "pts_loss_euc"))
#print(row_noaug(data, "inbilstm", "pts_loss_euc"))
