from glob import glob
from itertools import product
import matplotlib.pyplot as plt

import numpy as np

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

data = {}
prefix = "06_09_final_off3cm_50cm_lr5em4_bs128_"
for r in glob(f"timings_cpu/{prefix}*.npy"):
    d = np.load(r, allow_pickle=True)[()]
    name = r.split("/")[-1][len(prefix):-4]
    data["cpu_" + name.split("_")[0]] = d

def boxplot(data):
    keys = sorted(data["cpu_sep"].keys())
    #p = ["00", "001", "01", "1", "10", "100"]

    sep_cpu = [data["cpu_sep"][k] for k in keys]
    transformer_cpu = [data["cpu_transformer"][k] for k in keys]
    inbilstm_cpu = [data["cpu_inbilstm"][k] for k in keys]
    jac_cpu = [data["cpu_jactheir"][k] for k in keys]
    #sep_gpu = [data["gpu_sep"][k] for k in keys]
    #inbilstm_gpu = [data["gpu_inbilstm"][k] for k in keys]

    plt.figure(figsize=(7, 3))
    ax = plt.gca()
    x = np.arange(len(keys))  # the label locations
    width = 0.2  # the width of the bars
    color = {"MLP": "tab:green", "Transformer": "tab:blue", "IN-biLSTM": "tab:orange", "JacMLP": "tab:red"}
    for i, (v, name) in enumerate([(sep_cpu, "MLP"), (transformer_cpu, "Transformer"),
                                   (inbilstm_cpu, "IN-biLSTM"), (jac_cpu, "JacMLP")]):
    #for i, (v, name) in enumerate([(sep_cpu, "CPU FC"), (sep_gpu, "GPU FC"),
    #                               (inbilstm_cpu, "CPU INBiLSTM"), (inbilstm_gpu, "GPU INBiLSTM")]):
        means = np.array([np.mean(x) for x in v])
        a = ax.bar(x + width * i, means * 1000., width, label=name, color=color[name])

    ax.set_ylabel("Mean inference time [ms]")
    ax.set_xlabel("Batch size")
    ax.set_xticks(x + width, keys)
    #ax.legend(ncols=2)
    ax.legend(ncols=4)
    #ax.legend()
    plt.subplots_adjust(bottom=0.2)
    #ax.set_ylim(0, 250)
    ax.set_yscale("log")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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


boxplot(data)
