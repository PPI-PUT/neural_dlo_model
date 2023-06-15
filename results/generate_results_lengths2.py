from glob import glob
from itertools import product
import matplotlib.pyplot as plt

import numpy as np

from matplotlib.lines import Line2D

data = []
paths = []
prefix = "new_mb_"
for r in sorted(glob(f"lengths2_mb_04_03/{prefix}*.npy")):
    d = np.load(r, allow_pickle=True)[()]
    data.append(d)
    paths.append(r)



def boxplot(data, field):
    data = [x[field] for x in data]
    #baseline = [data[2]]
    #nopretrain = [data[1], data[0]] + data[3:5]
    #pretrain = [data[6], data[5]] + data[7:9]
    #baseline = [data[1]]
    #nopretrain = [data[0], data[4], data[3], data[2]]
    #pretrain = [data[5], data[8], data[7], data[6]]

    baseline40 = [data[2]]
    baseline45 = [data[3]]
    nopretrain40 = [data[i] for i in [0, 8, 6, 4]]
    nopretrain45 = [data[i] for i in [1, 9, 7, 5]]
    pretrain40 = [data[i] for i in [10, 16, 14, 12]]
    pretrain45 = [data[i] for i in [11, 17, 15, 13]]

    def bp(v, positions, c, ws="-"):
        bp = plt.boxplot(v, positions=positions,
                        showmeans=True, showfliers=False,
                        # patch_artist=True, notch=True,
                        meanprops=dict(marker="+", markeredgecolor="black"),
                        # medianprops=dict(color="magenta"),
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        widths=0.15)
        # bp = ax.boxplot(collection[i], positions=[0, spacing])
        ci = 0
        for i in range(len(bp["boxes"])):
            bp["boxes"][i].set_color(c)
            bp["boxes"][i].set_fillstyle("full")
            bp["boxes"][i].set_markerfacecolor("g")
            # bp["fliers"][i].set_color(c)
            bp["whiskers"][2 * i].set_color(c)
            bp["whiskers"][2 * i + 1].set_color(c)
            bp["whiskers"][2 * i].set_linestyle(ws)
            bp["whiskers"][2 * i + 1].set_linestyle(ws)
            bp["caps"][2 * i].set_color(c)
            bp["caps"][2 * i + 1].set_color(c)
    bp(baseline40, [-0.4], "tab:blue", ws="--")
    bp(baseline45, [-0.2], "tab:orange", ws="--")
    bp(nopretrain40, np.linspace(0.3, 3.5, 4), "tab:blue")
    bp(pretrain40, np.linspace(0.5, 3.7, 4), "tab:blue", ws="--")
    bp(nopretrain45, np.linspace(0.7, 3.9, 4), "tab:orange")
    bp(pretrain45, np.linspace(0.9, 4.1, 4), "tab:orange", ws="--")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.set_yscale("log")
    ax.set_ylabel("Relative error [%]")
    ax.set_xlabel("Part of the training set [%]")
    plt.subplots_adjust(bottom=0.2)

    custom_lines = [
        Line2D([0], [0], color="tab:blue", lw=2),
        Line2D([0], [0], color="tab:orange", lw=2),
        Line2D([0], [0], color="black", lw=2, linestyle="-"),
        Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    ax.legend(custom_lines, ["40cm", "45cm", "not pretrained", "pretrained"])
    cpx_map, cpy_map = xy_to_local_map(cp[:, 0], cp[:, 1])
    #bbox_to_anchor=(1.0, 0.0), frameon=False, ncol=4)

    #ax.set_xticks([])
    positions = [-0.3] + np.linspace(0.65, 3.8, 4).tolist()
    names = [str(x) for x in [0, 0.1, 1, 10, 100]]
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    #plt.xticks(rotation=45)
    plt.show()

boxplot(data, "ratio_loss")
