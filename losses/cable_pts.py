from utils.bspline import BSpline
from utils.constants import BSplineConstants
import tensorflow as tf

from utils.geometry import compute_curve_energy


class CablePtsLoss:
    def __init__(self):
        pass

    def __call__(self, gt, pred, bspline=True):
        mabs = lambda x: tf.reduce_mean(tf.reduce_sum(tf.abs(x), axis=-1), axis=-1)
        meuc = lambda x: tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)), axis=-1)
        ml2 = lambda x: tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=-1), axis=-1)

        pts_loss_mabs = mabs(gt - pred)
        pts_loss_meuc = meuc(gt - pred)
        pts_loss_ml2 = ml2(gt - pred)

        return pts_loss_mabs, pts_loss_meuc, pts_loss_ml2

class CableBSplinePtsLoss:
    def __init__(self):
        self.bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)
        self.bsp_pts = BSpline(BSplineConstants.n, BSplineConstants.dim, num_T_pts=1021)

    def __call__(self, gt, pred, bspline=True):
        mabs = lambda x: tf.reduce_mean(tf.reduce_sum(tf.abs(x), axis=-1), axis=-1)
        meuc = lambda x: tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)), axis=-1)
        ml2 = lambda x: tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=-1), axis=-1)

        control_pts_loss_mabs = mabs(gt - pred)
        control_pts_loss_meuc = meuc(gt - pred)
        control_pts_loss_ml2 = ml2(gt - pred)
        if bspline:
            vgt = (self.bsp_pts.N @ gt)[:, ::68]
            vpred = (self.bsp_pts.N @ pred)[:, ::68]
        else:
            vgt = gt
            vpred = pred

        pts_loss_mabs = mabs(vgt - vpred)
        pts_loss_meuc = meuc(vgt - vpred)
        pts_loss_ml2 = ml2(vgt - vpred)



        return control_pts_loss_mabs, control_pts_loss_meuc, control_pts_loss_ml2,\
               pts_loss_mabs, pts_loss_meuc, pts_loss_ml2