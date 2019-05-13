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
    # y_true = (y_true-np.min(y_true))/(np.max(y_true)-np.min(y_true))
    # y_pred = (y_pred-np.min(y_pred))/(np.max(y_pred)-np.min(y_pred))
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
            if iou >0.5:
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
# tf_confusion_metrics(y, realLabel, sess, feed_dict={predict: predictLabel , real:test_y})
# predictLabel = tf.constant(y)
# predictLabel =  predictLabel.eval()         # 将tensor转为ndarray
# realLabel = tf.convert_to_tensor(test_y)    # 将ndarray转为tensor
# tf_confusion_metrics(y, realLabel, sess, feed_dict={predict: predictLabel , real:test_y})
# def sk_Mertrics():
    # pred = multilayer_perceptron(x, weights, biases)
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #
    # with tf.Session() as sess:
    #     init = tf.initialize_all_variables()
    # sess.run(init)
    # for epoch in range(150):
    #     for i in range(total_batch):
    #         train_step.run(feed_dict={x: train_arrays, y: train_labels})
    #         avg_cost += sess.run(cost, feed_dict={x: train_arrays, y: train_labels}) / total_batch
    #     if epoch % display_step == 0:
    #         print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
    #
    # # metrics
    # y_p = tf.argmax(pred, 1)
    # val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_arrays, y: test_label})
    #
    # print("validation accuracy:", val_accuracy)
    # y_true = np.argmax(test_label, 1)
    # print("Precision", sk.metrics.precision_score(y_true, y_pred))
    # print("Recall", sk.metrics.recall_score(y_true, y_pred))
    # print("f1_score", sk.metrics.f1_score(y_true, y_pred))
    # print("confusion_matrix")
    # print(sk.metrics.confusion_matrix(y_true, y_pred))
    # fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)
    # fpr, tpr, tresholds = sk.metrics.roc_curve(y_true, y_pred)


