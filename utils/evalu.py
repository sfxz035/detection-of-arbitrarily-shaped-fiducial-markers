import tensorflow as tf
import sklearn as sk
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils.postproce import *
def dice_coef_theoretical(y_pred, y_true,threvalu=0.5):
    """Define the dice coefficient
        Args:
        y_pred: Prediction
        y_true: Ground truth Label
        Returns:
        Dice coefficient
        """

    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.nn.sigmoid(y_pred)
    # y_pred_f = tf.cast(tf.greater(y_pred_f, threvalu), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred_f, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection) / (union + 0.00001)

    if (tf.reduce_sum(y_pred) == 0) and (tf.reduce_sum(y_true) == 0):
        dice = 1
    return dice

def pix_RePre(y_pred, y_true,threvalu=0.5):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    # y_pred_f = tf.cast(tf.greater(y_pred, threvalu), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    P = tf.reduce_sum(y_true_f)
    P_pre = tf.reduce_sum(y_pred_f)
    recall = intersection/P
    precison = intersection/P_pre

    return recall,precison
def Iou_tf(y_pred,y_true,threvalu=0.5):
    ### 1
    # y_true = (y_true-np.min(y_true))/(np.max(y_true)-np.min(y_true))
    # y_true = tf.cast(y_true,tf.bool)
    # y_pred_f = tf.cast(tf.greater(y_pred, threvalu), tf.bool)
    # intersection = y_true&y_pred_f
    # union = y_true|y_pred_f
    # intersection = tf.reduce_sum(tf.cast(intersection,tf.float32))
    # union = tf.reduce_sum(tf.cast(union,tf.float32))

    ####2
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)-intersection

    iou = intersection/union
    if (tf.reduce_sum(y_pred) == 0) and (tf.reduce_sum(y_true) == 0):
        iou = 1
    return iou
def Iou_np(y_pred,y_true):
    y_true_f = np.reshape(y_true,[-1]).astype(np.float32)
    y_pred_f = np.reshape(y_pred,[-1]).astype(np.float32)
    intersection = np.sum(y_pred_f*y_true_f)
    union = np.sum(y_true_f)+np.sum(y_pred_f)-intersection

    iou = intersection/union
    if(np.sum(y_pred)==0)and(np.sum(y_true==0)):
        iou=1
    return iou

def calcu(y_pre,y_ture):
    arrArea_pre = connectComp(y_pre)
    arrArea_true = connectComp(y_ture)
    nub1 = np.shape(arrArea_true)[0]
    nub2 = np.shape(arrArea_pre)[0]
    # if nub1==1 and nub2 == 0:
    #     return 1,10000
    if nub1 != nub2:
        print('nub != nub2')
        nub = max(nub1,nub2)
    else:
        nub = nub1
    predic = []
    iouList = []
    for i in range(nub):
        try:
            area_true = arrArea_true[i]
            area_pre = arrArea_pre[i]
            # plt.imshow(area_true)
            # plt.show()
            # plt.imshow(area_pre)
            # plt.show()
            iou = Iou_np(area_pre,area_true)
            iouList.append(iou)
            if iou >0.4:
                predic.append(1)
            else:
                predic.append(0)
                print('0!!!!!!!!!!!')
        except(IndexError):
            return -1,-1
    return predic,iouList

def calcu2(y_pre,y_ture):
    maskFilt_pre = filterFewPoint(y_pre)
    maskFilt_pre = maskFilt_pre.astype(np.uint8)
    contours_pre, hierarchy_pre = cv.findContours(maskFilt_pre, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    maskFilt_true = filterFewPoint(y_ture)
    maskFilt_true = maskFilt_true.astype(np.uint8)
    contours_true, hierarchy_true = cv.findContours(maskFilt_true, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    nub1,nub2 = len(contours_pre),len(contours_true)
    if nub1 != nub2:
        print('nub != nub2')
        nub = max(nub1,nub2)
    else:
        nub = nub1
    predic = []
    iouList = []
    for i in range(0, nub):
        try:
            x1L, y1L, w1, h1 = cv.boundingRect(contours_pre[i])
            x1R, y1R = x1L+w1, y1L+h1
            x2L, y2L, w2, h2 = cv.boundingRect(contours_true[i])
            x2R, y2R = x2L+w2, y2L+h2
            xL,yL = max(x1L,x2L),max(y1L,y2L)
            xR,yR = min(x1R,x2R),min(y1R,y2R)
            intersection1 = max(xR-xL,0)
            intersection2 = max(yR-yL,0)
            inter = intersection1*intersection2

            # mask = np.zeros([1024, 1024], dtype='uint8')
            # cv.rectangle(mask, (x1L, y1L), (x1R, y1R), (153, 153, 0), 1)
            # cv.rectangle(mask, (x2L, y2L), (x2R, y2R), (153, 153, 0), 1)
            # cv.namedWindow('imgrec',0)
            # cv.resizeWindow('imgrec', 500, 500)
            # cv.imshow('imgrec',mask)
            # cv.waitKey(0)

            union_square = w1*h1+w2*h2-inter
            iou = inter/union_square
            iouList.append(iou)
            if iou >0.4:
                predic.append(1)
            else:
                predic.append(0)
                print('0!!!!!!!!!!!')
        except(IndexError):
            return -1,-1
    return predic,iouList
#### 待处理
def tf_confusion_metrics(predict, real, session, feed_dict):
    predictions = tf.argmax(predict, 1)
    actuals = tf.argmax(real, 1)

    ones_like_actuals = tf.ones_like(actuals)
    zeros_like_actuals = tf.zeros_like(actuals)
    ones_like_predictions = tf.ones_like(predictions)
    zeros_like_predictions = tf.zeros_like(predictions)

    tp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    tn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )

    fp_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, zeros_like_actuals),
                tf.equal(predictions, ones_like_predictions)
            ),
            "float"
        )
    )

    fn_op = tf.reduce_sum(
        tf.cast(
            tf.logical_and(
                tf.equal(actuals, ones_like_actuals),
                tf.equal(predictions, zeros_like_predictions)
            ),
            "float"
        )
    )
    tp, tn, fp, fn = session.run([tp_op, tn_op, fp_op, fn_op], feed_dict)

    tpr = float(tp) / (float(tp) + float(fn))
    fpr = float(fp) / (float(fp) + float(tn))
    fnr = float(fn) / (float(tp) + float(fn))

    accuracy = (float(tp) + float(tn)) / (float(tp) + float(fp) + float(fn) + float(tn))

    recall = tpr
    precision = float(tp) / (float(tp) + float(fp))

    f1_score = (2 * (precision * recall)) / (precision + recall)


