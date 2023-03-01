from utils.bspline import BSpline
from utils.constants import BSplineConstants
import numpy as np

from utils.geometry import compute_curve_energy


class CableBSplineAll2AllLoss:
    def __init__(self):
        self.bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)

    def __call__(self, gt, pred):
        vgt = (self.bsp.N @ gt).numpy()[:, np.newaxis]
        vpred = (self.bsp.N @ pred).numpy()[:, :, np.newaxis]

        diff = (vgt - vpred)

        l1 = np.linalg.norm(diff, axis=-1)
        l2 = np.linalg.norm(diff, axis=-1)

        l1 = np.min(l1, axis=1)
        l2 = np.min(l2, axis=2)

        l1 = np.mean(l1, axis=-1)
        l2 = np.mean(l2, axis=-1)

        l12 = (l1 + l2) / 2.

        return l1, l2, l12

class CableBSplineAll2AllXYZLoss:
    def __init__(self):
        self.bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)

    def __call__(self, gt, pred):
        vgt = (self.bsp.N @ gt).numpy()[:, np.newaxis]
        vpred = (self.bsp.N @ pred).numpy()[:, :, np.newaxis]

        diff = (vgt - vpred)

        l1 = np.linalg.norm(diff, axis=-1)
        l2 = np.linalg.norm(diff, axis=-1)

        l1idx = np.argmin(l1, axis=1)
        l2idx = np.argmin(l2, axis=2)

        s = np.arange(1024)
        diffs_l1 = []
        diffs_l2 = []
        for i in range(diff.shape[0]):
            diffs_l1.append(diff[i, l1idx[i, s], s])
            diffs_l2.append(diff[i, s, l2idx[i, s]])

        diffs_l1 = np.array(diffs_l1)
        diffs_l2 = np.array(diffs_l2)

        dl1 = np.mean(np.abs(diffs_l1), axis=1)
        dl2 = np.mean(np.abs(diffs_l2), axis=1)

        l12 = (dl1 + dl2) / 2.

        return l1, l2, l12
