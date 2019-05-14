import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# %matplotlib inline

# path
def load(file_path):
    # read all data
    data=open(file_path,'rb').read()
    # del the head part
    image_data = data[320:]
    # get image size
    image_size = np.sqrt((len(image_data) / 2)).astype(int).tolist()
    # define the image
    image = np.empty((image_size, image_size), dtype=float)
    # loop for insert data
    for i in range(image_size):
        for j in range(image_size):
            index = i * image_size + j
            val = int(image_data[2 * index + 1]) * 256 + int(image_data[2 * index])
            image[j, i] = val
    # nomarlize
    # image_std = (image - np.mean(image)) / np.std(image)
    # image_std_clip = np.clip(image_std, -0.75, 0.75)
    ## 添加通道，  映射到0，255
    # image_minmax = (image_std_clip-np.min(image_std_clip))/(np.max(image_std_clip)-np.min(image_std_clip))
    # img = (image_minmax*255).astype(np.uint8)
    # img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    ## 映射到0，1
    image_maxmin = (image-np.min(image))/(np.max(image)-np.min(image))
    print('convert done')

    return image_maxmin

if __name__ == '__main__':
    # file_path = 'C:\\Users\\Administrator\\Desktop\\liver_cases_rawdata\\liver_cases_rawdata\\chendarong\\A_1_LI_1489568403_276000_UNPROCESSED_IBRST_00'
    file_path = 'E:/code/segment/data/liver_cases_rawdata/chendarong/A_101_LI_1489568524_311000_UNPROCESSED_IBRST_00'
    a = load(file_path)
    a = cv.flip(a,0,dst=None)
    plt.imshow(a, cmap=plt.cm.gray)
    plt.show()
