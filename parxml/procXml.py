import  xml.dom.minidom
import numpy as np
import cv2 as cv
import os

def generadata(readpath,writepath):
    indexPatient = -1
    a = os.listdir(readpath)
    for file1 in os.listdir(readpath):
        file1 = a[15]
        file_dir = readpath+file1+'/'
        file_name = []
        indexPatient += 1
        for file2 in os.listdir(file_dir):
            file_name.append(file_dir + file2)
        nubFile = len(file_name)
        for i in range(nubFile//2):
            i = 2
            path1 = file_name[i*2]
            path2 = file_name[i*2+1]
            img = cv.imread(path1)
            img_shape = np.shape(img)
            mask, areaList = prcxml(path2,img_shape)
            # np.savez(writepath+str(indexPatient)+'.'+str(i)+'.npz',img=img,recmask=mask,areaList=areaList)

def prcxml(xmlPath,im_shape):
    dom = xml.dom.minidom.parse(xmlPath)
    root = dom.documentElement
    xminList = root.getElementsByTagName('xmin')
    xmaxList = root.getElementsByTagName('xmax')
    yminList = root.getElementsByTagName('ymin')
    ymaxList = root.getElementsByTagName('ymax')
    length = len(xminList)
    areaList = []
    mask = np.zeros([im_shape[0],im_shape[1]],dtype='uint8')
    for i in range(length):
        xminObj = xminList[i]
        xmin = int(xminObj.firstChild.data)
        yminObj = yminList[i]
        ymin = int(yminObj.firstChild.data)
        xmaxObj = xmaxList[i]
        xmax = int(xmaxObj.firstChild.data)
        ymaxObj = ymaxList[i]
        ymax = int(ymaxObj.firstChild.data)

        pointMin = (xmin,ymin)
        pointMax = (xmax,ymax)
        areaList.append([pointMin,pointMax])
        cv.rectangle(mask, pointMin, pointMax, 255, -1)
    return mask, areaList
# src = cv.imread(path2)
# cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
# cv.namedWindow('input_image',0)
# cv.resizeWindow('input_image', 500, 500)
# cv.imshow('input_image',src)
# cv.waitKey(0)
# cv.destroyAllWindows()

if __name__ == '__main__':
    readpath = 'E:/code/segment/data/liver cases/'
    writepath = 'E:/code/segment/data/data1/train/data.'
    generadata(readpath,writepath)