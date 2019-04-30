import tensorflow as tf
import sklearn as sk
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def connectComp(img):
    imgPre = np.greater(img, 200)
    imgPre = imgPre.astype(np.uint8)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(imgPre, connectivity=8)
    # plt.imshow(labels)
    # plt.show()
    # cv.namedWindow('imgmask', 0)
    # cv.resizeWindow('imgmask', 500, 500)
    # cv.imshow('imgmask', imgPre)
    # cv.waitKey(0)
    # plt.imshow(imgPre)
    # plt.show()

    ####  滤除掉像素点极少的区域，输出区域数组
    rect_squence = []
    for i in range(ret-1):
        mask = (labels==i+1)
        # ### 1.索引找不同类别的像素个数
        # arr = labels[mask]
        # area = arr.size
        #--------------
        ### 2.stats 取出area面积
        area = stats[i+1][-1]
        if area > 25:
            # plt.imshow(mask)
            # plt.show()
            rect_squence.append(mask)
    rect = np.asarray(rect_squence)
    return rect


def filterFewPoint(mask):
    imgPre = np.greater(mask, 200)
    imgPre = imgPre.astype(np.uint8)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(imgPre, connectivity=8)
    # plt.imshow(labels)
    # plt.show()
    for i in range(ret-1):
        maskzj = (labels==i+1)
        area = stats[i+1][-1]
        if area < 25:
            # plt.imshow(maskzj)
            # plt.show()
            labels[maskzj] = 0
            # plt.imshow(labels)
            # plt.show()
        else:
            labels[maskzj] = 255
    return labels
def contourmask(img,mask):
    maskFilt = filterFewPoint(mask)
    maskFilt = maskFilt.astype(np.uint8)
    contours, hierarchy = cv.findContours(maskFilt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
        cv.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 1)

    # plt.imshow(mask)
    # plt.show()
    #
    # cv.namedWindow('imgmask',0)
    # cv.resizeWindow('imgmask', 500, 500)
    # cv.imshow('imgmask',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
