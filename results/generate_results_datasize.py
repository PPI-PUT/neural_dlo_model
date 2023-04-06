from glob import glob
from itertools import product
import matplotlib.pyplot as plt

import numpy as np

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

data = {}
prefix = "new_mb_03_27_poc64_lr5em4_bs128"
for r in glob(f"percent_mb_03_27/{prefix}*.npy"):
    d = np.load(r, allow_pickle=True)[()]
    name = r.split("/")[-1][len(prefix)+1:-4]
    data[name] = d

def boxplot(data, field):
    keys = sorted(data.keys())
    #p = ["00", "001", "01", "1", "10", "100"]
    p = ["001", "01", "1", "10", "100"]
    keys = [k for k in keys if np.any([x + "percent" in k for x in p])]
    keys_sep_aug = [k for k in keys if "augwithzeros" in k and "sep" in k]
    keys_sep_noaug = [k for k in keys if "noaug" in k and "sep" in k]
    keys_in_aug = [k for k in keys if "augwithzeros" in k and "inbilstm" in k]
    keys_in_noaug = [k for k in keys if "noaug" in k and "inbilstm" in k]
    def sort_keys(keys):
        percents = []
        for k in keys:
            share = k.split("_")[-1][:-13]
            if share.startswith("0"):
                percent = float("." + share[1:])
            else:
                percent = float(share.replace("d", "."))
            percents.append(percent)

        idxs = argsort(percents)
        percents = [percents[i] for i in idxs]
        keys = [keys[i] for i in idxs]
        return keys, percents

    keys_sep_aug, percents = sort_keys(keys_sep_aug)
    keys_sep_noaug, percents = sort_keys(keys_sep_noaug)
    keys_in_aug, percents = sort_keys(keys_in_aug)
    keys_in_noaug, percents = sort_keys(keys_in_noaug)
    v_sep_aug = [data[k][field] * 100. for k in keys_sep_aug]
    v_sep_noaug = [data[k][field] * 100. for k in keys_sep_noaug]
    v_in_aug = [data[k][field] * 100. for k in keys_in_aug]
    v_in_noaug = [data[k][field] * 100. for k in keys_in_noaug]

    x = np.arange(len(percents))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    #for attribute, measurement in penguin_means.items():
    #    offset = width * multiplier
    #    rects = ax.bar(x + offset, measurement, width, label=attribute)
    #    ax.bar_label(rects, padding=3)
    #    multiplier += 1

    ax = plt.gca()
    for i, (v, name) in enumerate([(v_sep_noaug, "FC noaug"), (v_sep_aug, "FC aug"),
                                   (v_in_noaug, "INBiLSTM noaug"), (v_in_aug, "INBiLSTM aug")]):
        a = ax.bar(x + width * i, np.mean(np.array(v), axis=-1), width, label=name)
        #ax.bar_label(a, padding=3, fmt="%.1f")

    #a = ax.bar(x, v_sep_noaug, width, label="FC noaug")
    #ax.bar_label(a, padding=3)
    #a = ax.bar(x + width, v_sep_aug, width, label="FC aug")
    #ax.bar_label(a, padding=3)
    #a = ax.bar(x + 2 * width, v_in_noaug, width, label="INBiLSTM noaug")
    #ax.bar_label(a, padding=3)
    #a = ax.bar(x + 3 * width, v_in_aug, width, label="INBiLSTM aug")
    #ax.bar_label(a, padding=3)


    # Add some text for labels, title and custom x-axis tick labels, etc.
    #ax.set_ylabel('Length (mm)')
    #ax.set_title('Penguin attributes by species')
    ax.set_ylabel("Mean relative error [%]")
    ax.set_xlabel("Part of the training set [%]")
    ax.set_xticks(x + width, percents)
    ax.legend(ncols=1)
    #ax.set_ylim(0, 250)

    plt.show()

    #positions = list(range(len(v_sep_aug)))
    #plt.subplot(211)
    #bp = plt.boxplot(v_sep_noaug, positions=positions,
    #                showmeans=True, showfliers=False,
    #                meanprops=dict(marker="+", markeredgecolor="black"),
    #                boxprops=dict(linewidth=2),
    #                whiskerprops=dict(linewidth=2),
    #                capprops=dict(linewidth=2),
    #                widths=0.75)

    #bp = plt.boxplot(v_sep_aug, positions=[p + positions[-1] + 1 for p in positions],
    #                 showmeans=True, showfliers=False,
    #                 meanprops=dict(marker="+", markeredgecolor="black"),
    #                 boxprops=dict(linewidth=2),
    #                 whiskerprops=dict(linewidth=2),
    #                 capprops=dict(linewidth=2),
    #                 widths=0.75)

    #plt.yscale("log")
    #ax = plt.gca()
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ##ax.spines['bottom'].set_visible(False)

    ##ax.set_xticks([])
    #ax.set_xticks(positions)
    #ax.set_xticklabels(percents)
    #plt.xticks(rotation=45)
    #plt.show()


boxplot(data, "ratio_loss")
