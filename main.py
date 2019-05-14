import tensorflow as tf
import os
import numpy as np
import dataset
import networks.U_net as U_net
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2 as cv
from utils import evalu
from utils import postproce
os.environ["CUDA_VISIBLE_DEVICES"] = '1'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)

epoch = 2000000
batch_size = 8
learning_rate = 0.0001
savenet_path = './libSaveNet/save_unet/'
trainfile_dir = './data/data3/train/'
testfile_dir = './data/data3/test/'
input_name = 'img'
label_name = 'recmask'
channel = 3
x_train,y_train = dataset.get_data(trainfile_dir, input_name, label_name)
x_test,y_test = dataset.get_data(testfile_dir, input_name, label_name)
####
# x_train,y_train = dataset.get_data(trainfile_dir, input_name, label_name,sample_num=10,is_test=True)
# x_test,y_test = dataset.get_data(testfile_dir, input_name, label_name,sample_num=148,is_test=True)

## 1 归一化到0，255的数据
# x_train = x_train/255
# y_train = y_train/255
# x_test = x_test/255
# y_test = y_test/255
## 2 标准化并截取到-0.75，0.75的数据
# x_train = (x_train+0.75)/1.5
# x_test = (x_test+0.75)/1.5

channel = 1
x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)



y_train = np.expand_dims(y_train,-1)
y_test = np.expand_dims(y_test,-1)
def train():
    x = tf.placeholder(tf.float32,shape = [batch_size,1024,1024, channel])
    y_ = tf.placeholder(tf.float32,shape = [batch_size,1024,1024,1])
    # is_training = tf.placeholder(tf.bool)
    # dropout_value = tf.placeholder(tf.float32)  # 参与节点的数目百分比

    y = U_net.inference(x)
    loss = tf.reduce_mean(tf.square(y - y_))
    # loss = tf.abs(evalu.dice_coef_theoretical(y,y_)-1)
    summary_op = tf.summary.scalar('trainloss', loss)
    summary_op2 = tf.summary.scalar('testloss', loss)
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with tf.control_dependencies([batch_norm_updates_op]):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=8)
    writer = tf.summary.FileWriter('./my_graph/train', sess.graph)
    writer2 = tf.summary.FileWriter('./my_graph/test')
    tf.global_variables_initializer().run()
    # last_file = tf.train.latest_checkpoint(savenet_path)
    # if last_file:
    #     tf.logging.info('Restoring model from {}'.format(last_file))
        # saver.restore(sess, last_file)

    count, m = 0, 0
    for ep in range(epoch):
        batch_idxs = len(x_train) // batch_size
        for idx in range(batch_idxs):
            # batch_input = x_train[idx * batch_size: (idx + 1) * batch_size]
            # batch_labels = y_train[idx * batch_size: (idx + 1) * batch_size]
            batch_input, batch_labels = dataset.random_batch(x_train,y_train,batch_size)
            sess.run(train_step, feed_dict={x: batch_input, y_: batch_labels})
            count += 1
            # print(count)
            if count % 50 == 0:
                m += 1
                batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, batch_size)
                # batch_input_test = x_test[0 : batch_size]
                # batch_labels_test = y_test[0 : batch_size]
                loss1 = sess.run(loss, feed_dict={x: batch_input,y_: batch_labels})
                loss2 = sess.run(loss, feed_dict={x: batch_input_test, y_: batch_labels_test})
                print("Epoch: [%2d], step: [%2d], train_loss: [%.8f]" \
                      % ((ep + 1), count, loss1), "\t", 'test_loss:[%.8f]' % (loss2))
                writer.add_summary(sess.run(summary_op, feed_dict={x: batch_input, y_: batch_labels}), m)
                writer2.add_summary(sess.run(summary_op2, feed_dict={x: batch_input_test,
                                                                     y_: batch_labels_test}), m)
            if (count + 1) % 20000 == 0:
                saver.save(sess, os.path.join(savenet_path, 'conv_unet%d.ckpt-done' % (count)))

def test():
    batch_size = 1
    ###------------------数据集
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    savepath = 'E:\code\segment\libSaveNet\save_unetz\conv_unet59999.ckpt-done'
    x = tf.placeholder(tf.float32,shape = [1,1024,1024, channel])
    y_ = tf.placeholder(tf.float32,shape = [1,1024,1024,1])
    y = U_net.inference(x,is_training=True)
    loss = tf.reduce_mean(tf.square(y - y_))
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    nub = np.shape(x_train)[0]
     # mub = np.random.randint(0,nub,10)
    predicList = []
    ycList = []
    for i in range(nub):
        # i = 39
        # inputTest = x_train[i:i+1,:,:,:]
        # labelTest = y_train[i:i+1,:,:,:]
        inputTest = x_test[i:i + 1, :, :, :]
        labelTest = y_test[i:i + 1, :, :, :]

        output = sess.run(y,feed_dict={x: inputTest})
        loss_test = sess.run(loss, feed_dict={x: inputTest, y_: labelTest})

        ##原始
        # out = np.squeeze(output).astype(np.uint8)
        # label = np.squeeze(labelTest).astype(np.uint8)
        # img = np.squeeze(inputTest).astype(np.uint8)

        ## 1 归一化到0，255的数据
        img = (np.squeeze(inputTest)*255).astype(np.uint8)
        out = np.squeeze(output).astype(np.uint8)
        out = out * 255


        label = (np.squeeze(labelTest)*255).astype(np.uint8)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))  # 定义结构元素
        outclosing = cv.morphologyEx(out, cv.MORPH_CLOSE, kernel)  # 闭运算
        ##### 框出目标区域——————————————————
        postproce.contourmask(img,outclosing)
        cv.namedWindow('imgrec',0)
        cv.resizeWindow('imgrec', 500, 500)
        cv.imshow('imgrec',img)
        cv.namedWindow('label',0)
        cv.resizeWindow('label', 500, 500)
        cv.imshow('label',label)
        cv.namedWindow('output',0)
        cv.resizeWindow('output', 500, 500)
        cv.imshow('output',outclosing)
        cv.waitKey(0)
        cv.destroyAllWindows()
        ##### 评价指标——————————————————————
        predic,iouList = evalu.calcu(outclosing,label)
        if predic==-1 and iouList==-1:
            print('loss: %g, wrong index：%g' % (loss_test,i))
            ycList.append(i)
        else:
            for j in range(len(predic)):
                predicList.append(predic[j])
            print(i)
            print('loss: %g' % (loss_test),iouList)
    pos = predicList.count(1)
    all = len(predicList)
    precision = pos / len(predicList)
    print('1-nub: %g, all nub：%g' % (pos,all))
    print(precision)

if __name__ == '__main__':
    train()
    # test()