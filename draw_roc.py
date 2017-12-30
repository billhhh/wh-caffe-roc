# -*- coding: utf-8 -*-

import caffe
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

net_file='deploy_resnet50_CXR14.prototxt'
caffe_model='23_4_resnet50_CXR14_b64_iter_7000.caffemodel'
#mean_file='ilsvrc_2012_mean.npy'
mean_file = np.array([128,128,128])

# load the net with trained weights
net = caffe.Net(net_file, caffe_model, caffe.TEST)
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# add preprocessing
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
transformer.set_mean('data', mean_file) #### subtract mean ####
transformer.set_raw_scale('data', 255) # pixel value range
transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR


# set test batchsize
net.blobs['data'].reshape(1,        # batch 大小
                          3,         # 3-channel (BGR) images
                          224,224)  # 图像大小为:224x224


# 加载图片
im = caffe.io.load_image('D:/dataset/CXR8/images_zip/images/00000003_003.png')
# 用上面的transformer.preprocess来处理刚刚加载图片
transformed_image = transformer.preprocess('data', im)
#plt.imshow(im)
#plt.show()
net.blobs['data'].data[...] = transformed_image

y_score = []
y_true = []
N =1
for i in xrange(N): # assuming you have N validation samples
    #x_i = im # get i-th validation sample
    #y_true.append( y_i )  # y_i is 0 or 1 the TRUE label of x_i

    out = net.forward()  # get prediction for x_i
    output_prob = out['probability_4'][0] #batch中第一张图像的概率值
    y_score.append( output_prob )
    # 打印概率最大的类别代号，argmax()函数是求取矩阵中最大元素的索引
    print 'predicted class is:', output_prob.argmax()

# once you have N y_score and y_true values
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
auc = metrics.roc_auc_score(y_true, y_score)
