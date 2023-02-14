from utils.bspline import BSpline
from utils.constants import BSplineConstants
import tensorflow as tf


class CableBSplineLoss:
    def __init__(self):
        self.bsp = BSpline(BSplineConstants.n, BSplineConstants.dim)

    def __call__(self, gt, pred, mul=1.):
        gt = mul * tf.reshape(gt, (-1, BSplineConstants.n, BSplineConstants.dim))
        pred = mul * tf.reshape(pred, (-1, BSplineConstants.n, BSplineConstants.dim))
        # plt.plot(pred[0, :, 0], pred[0, :, 1])
        # plt.plot(gt[0, :, 0], gt[0, :, 1])
        # plt.show()
        diff = tf.reduce_sum(tf.abs(gt - pred), axis=-1)
        control_pts_loss = tf.reduce_mean(diff, axis=-1)
        vgt = self.bsp.N @ gt
        dvgt = self.bsp.dN @ gt
        ddvgt = self.bsp.ddN @ gt
        vpred = self.bsp.N @ pred
        dvpred = self.bsp.dN @ pred
        ddvpred = self.bsp.ddN @ pred
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
        return control_pts_loss, length_loss, acccurv_yz_loss


class CableSeqLoss:
    def __init__(self):
        pass

    def __call__(self, gt, pred):
        diff = tf.reduce_sum(tf.abs(gt - pred), axis=-1)
        pts_loss = tf.reduce_mean(diff, axis=-1)
        return pts_loss
