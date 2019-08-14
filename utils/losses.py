import tensorflow as tf
from keras import backend as K
def dice_coe(output, target, loss_type='sorensen', axis=(1, 2, 3), smooth=1e-5):
    # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
    # y_pred = nn.Sigmoid()(y_pred)
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def focal_loss(y_true, y_pred,gamma=2., alpha=0.25):
    # 如果模型最后没有 nn.Sigmoid()，那么这里就需要对预测结果计算一次 Sigmoid 操作
    # y_pred = nn.Sigmoid()(y_pred)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    focal_loss_fixed = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def mixedLoss(y_ture,y_pred,alpha):
    return alpha * focal_loss(y_ture,y_pred) - K.log(dice_coe(y_pred,y_ture))