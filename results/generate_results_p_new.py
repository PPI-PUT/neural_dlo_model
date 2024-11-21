from glob import glob
from itertools import product
import matplotlib.pyplot as plt

import numpy as np

# data = np.load("results_all_new_aug_.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb_sep.npy", allow_pickle=True)[()]
# data = np.load("results_all_new_mb_inbilstm_.npy", allow_pickle=True)[()]
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, LogFormatter, NullFormatter, StrMethodFormatter

data_p2 = {}
data_p3 = {}
#cable_type = "p2"
cable_type = "p3"
prefix = "06_09_final_off3cm_"
#for r in sorted(glob(f"06_09_final_off3cm_{cable_type}/{prefix}*.npy")):
#for r in sorted(glob(f"06_09_final_off3cm_{cable_type}_wrongwhithening/{prefix}*.npy")):
for r in sorted(glob(f"06_09_final_off3cm_p2/{prefix}*.npy")):
    d = np.load(r, allow_pickle=True)[()]
    name = r.split("/")[-1][len(prefix):-4]
    data_p2[name] = d
for r in sorted(glob(f"06_09_final_off3cm_p3/{prefix}*.npy")):
    d = np.load(r, allow_pickle=True)[()]
    name = r.split("/")[-1][len(prefix):-4]
    data_p3[name] = d


def plot(data, field):
    data_sep = {k: v for k, v in data.items() if "sep" in k}
    data_inbilstm = {k: v for k, v in data.items() if "inbilstm" in k}
    data_transf = {k: v for k, v in data.items() if "transformer" in k}
    data_jac = {k: v for k, v in data.items() if "jactheir" in k}

    mul = 100.

    data_sep_pretrained = {k: v for k, v in data_sep.items() if "pretrained" in k}
    data_sep_fromscratch = {k: v for k, v in data_sep.items() if "pretrained" not in k and "50cm" not in k}
    data_sep_baseline = [mul * v[field] for k, v in data_sep.items() if k.startswith("50cm")]
    data_inbilstm_pretrained = {k: v for k, v in data_inbilstm.items() if "pretrained" in k}
    data_inbilstm_fromscratch = {k: v for k, v in data_inbilstm.items() if "pretrained" not in k and "50cm" not in k}
    data_inbilstm_baseline = [mul * v[field] for k, v in data_inbilstm.items() if k.startswith("50cm")]
    data_transf_pretrained = {k: v for k, v in data_transf.items() if "pretrained" in k}
    data_transf_fromscratch = {k: v for k, v in data_transf.items() if "pretrained" not in k and "50cm" not in k}
    data_transf_baseline = [mul * v[field] for k, v in data_transf.items() if k.startswith("50cm")]
    data_jac_pretrained = {k: v for k, v in data_jac.items() if "pretrained" in k}
    data_jac_fromscratch = {k: v for k, v in data_jac.items() if "pretrained" not in k and "50cm" not in k}
    data_jac_baseline = [mul * v[field] for k, v in data_jac.items() if k.startswith("50cm")]

    percents = {"01percent": 0.1, "1percent": 1., "10percent": 10.}
    results = []
    for d in [data_sep_pretrained, data_sep_fromscratch,
              data_inbilstm_pretrained, data_inbilstm_fromscratch,
              data_transf_pretrained, data_transf_fromscratch,
              data_jac_pretrained, data_jac_fromscratch]:
        result = [(percents[k.split("_")[1]] if k.split("_")[1] in percents.keys() else 100., np.median(mul * v[field]))
                  for k, v in d.items()]
        result = sorted(result, key=lambda x: x[0])
        results.append(np.array(result))
    sep_pretrained, sep_fromscratch,\
    inbilstm_pretrained, inbilstm_fromscratch,\
    transf_pretrained, transf_fromscratch, \
    jac_pretrained, jac_fromscratch = results

    sep_baseline = (0., np.median(data_sep_baseline))
    inbilstm_baseline = (0., np.median(data_inbilstm_baseline))
    transf_baseline = (0., np.median(data_transf_baseline))
    jac_baseline = (0., np.median(data_jac_baseline))

    plt.plot(sep_fromscratch[:, 0], sep_fromscratch[:, 1], color="tab:green", marker="o", linestyle="-")
    plt.plot(sep_pretrained[:, 0], sep_pretrained[:, 1], color="tab:green", marker="o", linestyle="--")
    plt.plot(inbilstm_fromscratch[:, 0], inbilstm_fromscratch[:, 1], color="tab:orange", marker="o", linestyle="-")
    plt.plot(inbilstm_pretrained[:, 0], inbilstm_pretrained[:, 1], color="tab:orange", marker="o", linestyle="--")
    plt.plot(transf_fromscratch[:, 0], transf_fromscratch[:, 1], color="tab:blue", marker="o", linestyle="-")
    plt.plot(transf_pretrained[:, 0], transf_pretrained[:, 1], color="tab:blue", marker="o", linestyle="--")
    plt.plot(jac_fromscratch[:, 0], jac_fromscratch[:, 1], color="tab:red", marker="o", linestyle="-")
    plt.plot(jac_pretrained[:, 0], jac_pretrained[:, 1], color="tab:red", marker="o", linestyle="--")
    plt.plot([0.1, 100.], [sep_baseline[1], sep_baseline[1]], color="tab:green", linestyle="-.")
    plt.plot([0.1, 100.], [inbilstm_baseline[1], inbilstm_baseline[1]], color="tab:orange", linestyle="-.")
    plt.plot([0.1, 100.], [transf_baseline[1], transf_baseline[1]], color="tab:blue", linestyle="-.")
    plt.plot([0.1, 100.], [jac_baseline[1], jac_baseline[1]], color="tab:red", linestyle="-.")
    plt.xscale("log")
    plt.yscale("log")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    #plt.ticklabel_format(axis='x', style='plain', scilimits=(-5, 8))
    plt.xticks([0.1, 1., 10., 100.])
    #plt.gca().xaxis.set_minor_formatter(NullFormatter())
    #formatter = LogFormatter(labelOnlyBase=False)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(formatter)
    #ax.yaxis.get_major_formatter().set_scientific(False)
    #plt.ticklabel_format(axis='y', style='plain')





    # custom_lines = [
    #    Line2D([0], [0], color=[1., 0., 0.], lw=2, linestyle="-"),
    #    Line2D([0], [0], color=[1., 0.5, 0.], lw=2, linestyle="-"),
    #    Line2D([0], [0], color=[0., 0., 1.], lw=2, linestyle="-"),
    #    Line2D([0], [0], color=[0., 0.5, 1.], lw=2, linestyle="-"),
    #    Line2D([0], [0], color="k", lw=2, linestyle="-"),
    #    Line2D([0], [0], color="k", lw=2, linestyle="--"),
    #    Line2D([0], [0], color="k", lw=2, linestyle="-."),
    #    Line2D([0], [0], color="k", lw=2, linestyle=":"),
    # ]
    # ax.legend(custom_lines, ["INBiLSTM Diff", 'INBiLSTM IE', 'FC Diff', 'FC IE',
    #                         'quaternion', "rotation matrix", "rotation vector", "DLO directions"],
    #          bbox_to_anchor=(1.0, 0.0), frameon=False, ncol=4)

    #ax.set_xticks([])
    # ax.set_xticks(positions)
    # ax.set_xticklabels(names)
    # plt.xticks(rotation=45)
    #plt.show()

plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Braided cable")
plot(data_p2, "ratio_loss")
plt.subplot(122)
plt.title("Solar cable")
plot(data_p3, "ratio_loss")

custom_lines = [
    Line2D([0], [0], color='tab:blue', lw=2, linestyle="-"),
    Line2D([0], [0], color='tab:orange', lw=2, linestyle="-"),
    Line2D([0], [0], color='tab:green', lw=2, linestyle="-"),
    Line2D([0], [0], color='tab:red', lw=2, linestyle="-"),
]
plt.legend(custom_lines, ['Transformer', "IN-biLSTM", 'MLP', 'JacMLP'])
# plt.legend()
plt.show()
