from glob import glob
from itertools import product
import matplotlib.pyplot as plt

import numpy as np

# data = np.load("results_all_new_aug_.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb_sep.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb_inbilstm_.npy", allow_pickle=True)[()]
from matplotlib.lines import Line2D

data= {}
prefix = "06_09_final_off3cm_"
for r in sorted(glob(f"06_09_final_off3cm_lengths/*.npy")):
    d = np.load(r, allow_pickle=True)[()]
    name = r.split("/")[-1][:-4]
    data[name] = d

def boxplot(data, field):
    data_50_test40 = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test40_") and "_50cm_" in k}
    data_50_test45 = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test45_") and "_50cm_" in k}

    data_50_test40scaled = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test40scaled_") and "_50cm_" in k}
    data_50_test45scaled = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test45scaled_") and "_50cm_" in k}

    data_40_test40 = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test40_") and "_40cm_" in k}
    data_45_test45 = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test45_") and "_45cm_" in k}

    data_404550_test40 = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test40_") and "_40cm45cm50cm_" in k}
    data_404550_test45 = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test45_") and "_40cm45cm50cm_" in k}

    data_404550_test50 = {k.split("_")[8]: v for k, v in data.items() if k.startswith("test50_") and "_40cm45cm50cm_" in k}

    #order = ["transformer", "inbilstm", "sep"]
    order = ["transformer", "inbilstm", "sep", "jactheir"]
    #order = ["transformer", "inbilstm", "sep"]
    mul = 100.
    v40_t40 = [mul * data_40_test40[k][field] for k in order]
    v50_t40 = [mul * data_50_test40[k][field] for k in order]
    v50_t40s = [mul * data_50_test40scaled[k][field] for k in order]
    v404550_t40 = [mul * data_404550_test40[k][field] for k in order]

    v45_t45 = [mul * data_45_test45[k][field] for k in order]
    v50_t45 = [mul * data_50_test45[k][field] for k in order]
    v50_t45s = [mul * data_50_test45scaled[k][field] for k in order]
    v404550_t45 = [mul * data_404550_test45[k][field] for k in order]

    #v404550_t50 = [mul * data_404550_test50[k][field] for k in order]

    v = v45_t45 + v50_t45 + v50_t45s + v404550_t45 + v40_t40 + v50_t40 + v50_t40s + v404550_t40
    n_groups = 8
    keys = order * n_groups
    width = 0.25
    shift = 3 * width
    pos = np.linspace(0., (len(order) - 1) * (1.25 * width), len(order))
    gap = 3 * width
    positions = np.concatenate([pos + (pos[-1] + gap) * i for i in range(n_groups)])
    positions[int(len(positions)/2.):] += shift
    positions = positions #+ width
    #positions = list(range(len(v)))
    midline = (positions[int(len(positions)/2.)-1] + positions[int(len(positions)/2.)]) / 2.
    plt.figure(figsize=(7, 3.5))
    plt.plot([midline, midline], [0., 80.], "k--")
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
                     widths=width)
    # bp = ax.boxplot(collection[i], positions=[0, spacing])
    ci = 0
    for i in range(len(bp["boxes"])):
        if "inbilstm" in keys[i]:
            c = 'tab:orange'
        elif "sep" in keys[i]:
            c = 'tab:green'
        elif "transformer" in keys[i]:
            c = 'tab:blue'
        elif "jactheir" in keys[i]:
            c = 'tab:red'
        elif "lin" in keys[i]:
            c = 'tab:purple'
        else:
            c = [0., 0., 1.]
        bp["boxes"][i].set_color(c)
        bp["boxes"][i].set_fillstyle("full")
        bp["boxes"][i].set_markerfacecolor("g")
        # bp["fliers"][i].set_color(c)
        bp["whiskers"][2 * i].set_color(c)
        bp["whiskers"][2 * i + 1].set_color(c)
        bp["caps"][2 * i].set_color(c)
        bp["caps"][2 * i + 1].set_color(c)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    custom_lines = [
       Line2D([0], [0], color='tab:blue', lw=2, linestyle="-"),
       Line2D([0], [0], color='tab:orange', lw=2, linestyle="-"),
       Line2D([0], [0], color='tab:green', lw=2, linestyle="-"),
       Line2D([0], [0], color='tab:red', lw=2, linestyle="-"),
    ]
    #ax.legend(custom_lines, ['Transformer', "IN-biLSTM", 'MLP', 'Lin'],
    #ax.legend(custom_lines, ['Transformer', "IN-biLSTM", 'MLP'],
    ax.legend(custom_lines, ['Transformer', "IN-biLSTM", 'MLP', "JacMLP"],
              #bbox_to_anchor=(0.85, -0.02), frameon=False, ncol=3)
              bbox_to_anchor=(0.95, -0.02), frameon=False, ncol=4)

    plt.ylim(-4., 75.)
    #ax.set_xticks([])
    # ax.set_xticks(positions)
    # ax.set_xticklabels(names)
    # plt.xticks(rotation=45)
    plt.grid(axis="y")
    ax.set_xticks([])
    #ax.set_xticks([positions[4], positions[13]])
    #ax.set_xticklabels(["Tested on 45cm", "Tested on 40cm"])
    #ax.tick_params(axis='x', which='both', length=0)
    #ax.xaxis.tick_top()

    ax.text(0.24, 0.87, 'Tested on 45cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.79, 0.87, 'Tested on 40cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.08, -0.01, '45cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.18, -0.01, '50cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.30, -0.07, '50cm\n+scale', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.41, -0.07, '40+45\n+50cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.59, -0.01, '40cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.70, -0.01, '50cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.82, -0.07, '50cm\n+scale', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    ax.text(0.93, -0.07, '40+45\n+50cm', verticalalignment='bottom', horizontalalignment='center', transform=ax.transAxes)
    plt.show()


boxplot(data, "ratio_loss")
