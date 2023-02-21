from utils.bspline import BSpline
from utils.constants import BSplineConstants
import tensorflow as tf

from utils.geometry import compute_curve_energy


class CableBSplineLoss:
    def __init__(self):
        self.bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)

    def __call__(self, gt, pred, mul=1.):
        mabs = lambda x: tf.reduce_mean(tf.reduce_sum(tf.abs(x), axis=-1), axis=-1)
        meuc = lambda x: tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)), axis=-1)
        ml2 = lambda x: tf.reduce_mean(tf.reduce_sum(tf.square(x), axis=-1), axis=-1)
        #gt = mul * tf.reshape(gt, (-1, BSplineConstants.n, BSplineConstants.dim))
        #pred = mul * tf.reshape(pred, (-1, BSplineConstants.n, BSplineConstants.dim))
        # plt.plot(pred[0, :, 0], pred[0, :, 1])
        # plt.plot(gt[0, :, 0], gt[0, :, 1])
        # plt.show()
        control_pts_loss_mabs = mabs(gt - pred)
        control_pts_loss_meuc = meuc(gt - pred)
        control_pts_loss_ml2 = ml2(gt - pred)
        vgt = self.bsp.N @ gt
        dvgt = self.bsp.dN @ gt
        ddvgt = self.bsp.ddN @ gt
        vpred = self.bsp.N @ pred
        dvpred = self.bsp.dN @ pred
        ddvpred = self.bsp.ddN @ pred

        pts_loss_mabs = mabs(vgt - vpred)
        pts_loss_meuc = meuc(vgt - vpred)
        pts_loss_ml2 = ml2(vgt - vpred)

        len_gt = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(vgt[:, 1:, 1:] - vgt[:, :-1, 1:]), axis=-1)), axis=-1)
        len_pred = tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(vpred[:, 1:, 1:] - vpred[:, :-1, 1:]), axis=-1)),
                                 axis=-1)
        length_loss = tf.square(len_gt - len_pred)

        curv_yz_gt = (dvgt[..., 1] * ddvgt[..., 2] - ddvgt[..., 1] * dvgt[..., 2]) / tf.sqrt(
            dvgt[..., 1] ** 2 + dvgt[..., 2] ** 2 + 1e-8)
        curv_yz_pred = (dvpred[..., 1] * ddvpred[..., 2] - ddvpred[..., 1] * dvpred[..., 2]) / tf.sqrt(
            dvpred[..., 1] ** 2 + dvpred[..., 2] ** 2 + 1e-8)
        dcurv_yz_gt = curv_yz_gt[:, 1:] - curv_yz_gt[:, :-1]
        dcurv_yz_pred = curv_yz_pred[:, 1:] - curv_yz_pred[:, :-1]
        dyz_gt = tf.sqrt(tf.reduce_sum(tf.square(vgt[:, 1:, 1:] - vgt[:, :-1, 1:]), axis=-1))
        dyz_pred = tf.sqrt(tf.reduce_sum(tf.square(vpred[:, 1:, 1:] - vpred[:, :-1, 1:]), axis=-1))
        acccurv_yz_gt = tf.reduce_sum(tf.abs(dcurv_yz_gt) * dyz_gt, axis=-1)
        acccurv_yz_pred = tf.reduce_sum(tf.abs(dcurv_yz_pred) * dyz_pred, axis=-1)
        acccurv_yz_loss = tf.abs(acccurv_yz_gt - acccurv_yz_pred)

        pred_energy = compute_curve_energy(dvpred, ddvpred)
        gt_energy = compute_curve_energy(dvgt, ddvgt)

        return control_pts_loss_mabs, control_pts_loss_meuc, control_pts_loss_ml2,\
               pts_loss_mabs, pts_loss_meuc, pts_loss_ml2,\
               length_loss, acccurv_yz_loss, pred_energy, gt_energy