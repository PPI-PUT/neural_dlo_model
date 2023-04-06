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
    baseline = [data[1]]
    nopretrain = [data[0], data[4], data[3], data[2]]
    pretrain = [data[5], data[8], data[7], data[6]]
    def bp(v, positions, c):
        bp = plt.boxplot(v, positions=positions,
                        showmeans=True, showfliers=False,
                        # patch_artist=True, notch=True,
                        meanprops=dict(marker="+", markeredgecolor="black"),
                        # medianprops=dict(color="magenta"),
                        boxprops=dict(linewidth=2),
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        widths=0.25)
        # bp = ax.boxplot(collection[i], positions=[0, spacing])
        ci = 0
        for i in range(len(bp["boxes"])):
            bp["boxes"][i].set_color(c)
            bp["boxes"][i].set_fillstyle("full")
            bp["boxes"][i].set_markerfacecolor("g")
            # bp["fliers"][i].set_color(c)
            bp["whiskers"][2 * i].set_color(c)
            bp["whiskers"][2 * i + 1].set_color(c)
            bp["caps"][2 * i].set_color(c)
            bp["caps"][2 * i + 1].set_color(c)
    bp(baseline, [-0.2], "tab:green")
    bp(nopretrain, np.linspace(0.5, 3.5, 4), "tab:orange")
    bp(pretrain, np.linspace(0.8, 3.8, 4), "tab:green")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    ax.set_yscale("log")
    ax.set_ylabel("Relative error [%]")
    ax.set_xlabel("Part of the training set [%]")
    plt.subplots_adjust(bottom=0.2)

    custom_lines = [
        Line2D([0], [0], color="tab:green", lw=2),
        Line2D([0], [0], color="tab:orange", lw=2),
    ]
    ax.legend(custom_lines, ["pretrained", 'not pretrained'])
              #bbox_to_anchor=(1.0, 0.0), frameon=False, ncol=4)

    #ax.set_xticks([])
    positions = [-0.2] + np.linspace(0.65, 3.65, 4).tolist()
    names = [str(x) for x in [0, 0.1, 1, 10, 100]]
    ax.set_xticks(positions)
    ax.set_xticklabels(names)
    #plt.xticks(rotation=45)
    plt.show()

boxplot(data, "ratio_loss")
