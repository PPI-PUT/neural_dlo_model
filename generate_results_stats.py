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
prefix = "new_mb_03_27_poc64_lr5em4_bs128"
for r in glob(f"results/all_mb_03_27/{prefix}*.npy"):
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


def boxplot_noaug(data, field):
    keys = sorted(data.keys())
    aug = "augwithzeros"
    #aug = "noaug"
    keys = [k for k in keys if aug in k]
    v = [data[k][field] * 100. for k in keys]
    names = []
    for k in keys:
        ks = k.split("_")
        d = "DR" if ks[1] == "diff" else "R"
        r = "Q" if ks[2] == "quat" else ("RM" if ks[2] == "rotmat" else "RV")
        c = "C" if ks[3] == "cable" else "DC"
        name = f"{d}_{r}_{c}"
        names.append(name)
    colors = []
    a = 0
    positions = list(range(len(v)+1))
    del positions[6]
    #plt.figure(figsize=(10, 5))
    plt.figure(figsize=(10, 4))
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
        lw = 2.
        if "inbilstm" in keys[i]:
            c = [1., 0., 0.]
            bs = ":"
        else:
            #c = "b"
            c = [0., 0., 1.]
            if "dcable" in keys[i]:
                #c[0] += 0.5
                #c[2] -= 0.6
                #lw = 3.5
                #bs = "--"
                bs = ":"
        #if "nodiff" in keys[i]:
        #    lw = 3.
        #else:
        #    lw = 2.
        if "nodiff" in keys[i]:
            c[1] += 0.5

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
        bp["caps"][2 * i].set_color(c)
        bp["caps"][2 * i + 1].set_color(c)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    custom_lines = [
        Line2D([0], [0], color=[1., 0., 0.], lw=2, linestyle="-"),
        Line2D([0], [0], color=[1., 0.5, 0.], lw=2, linestyle="-"),
        Line2D([0], [0], color=[0., 0., 1.], lw=2, linestyle="-"),
        Line2D([0], [0], color=[0., 0.5, 1.], lw=2, linestyle="-"),
        Line2D([0], [0], color="k", lw=2, linestyle="-"),
        Line2D([0], [0], color="k", lw=2, linestyle="--"),
        Line2D([0], [0], color="k", lw=2, linestyle="-."),
        Line2D([0], [0], color="k", lw=2, linestyle=":"),
    ]
    if aug == "noaug":
        ax.legend(custom_lines, ["INBiLSTM Diff", 'INBiLSTM IE', 'FC Diff', 'FC IE',
                                 'quaternion', "rotation matrix", "axis angle", "DLO directions"],
                  bbox_to_anchor=(0.9, 0.0), frameon=False, ncol=4)
    plt.subplots_adjust(bottom=0.2)

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
boxplot_noaug(data, "ratio_loss")
print(sd)
print(row_noaug(data, "sep", "ratio_loss"))
#print(row_noaug(data, "sep", "pts_loss_euc"))
#print(row_noaug(data, "cnn", "pts_loss_euc"))
#print(row_noaug(data, "inbilstm", "pts_loss_euc"))
