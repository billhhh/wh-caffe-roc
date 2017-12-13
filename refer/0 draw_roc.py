# -*- coding: utf-8 -*-

import caffe
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

net_file='deploy_resnet50_CXR14.prototxt'
caffe_model='23_2_resnet50_CXR14_b64_iter_37000.caffemodel'
#mean_file='ilsvrc_2012_mean.npy'
mean_file = np.array([128,128,128])

# load the net with trained weights
net = caffe.Net(net_file, caffe_model, caffe.TEST)
# 得到data的形状，这里的图片是默认matplotlib底层加载的
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

# matplotlib加载的image是像素[0-1],图片的数据格式[weight,high,channels]，RGB
# caffe加载的图片需要的是[0-255]像素，数据格式[channels,weight,high],BGR，那么就需要转换

# channel 放到前面
transformer.set_transpose('data', (2, 0, 1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
transformer.set_mean('data',mean_file)
# 图片像素放大到[0-255]
transformer.set_raw_scale('data', 255)
# RGB-->BGR 转换
transformer.set_channel_swap('data', (2, 1, 0))

'''
# 设置输入图像大小
net.blobs['data'].reshape(1,        # batch 大小
                          3,         # 3-channel (BGR) images
                          224, 224)  # 图像大小为:224x224
'''

# 加载图片
im = caffe.io.load_image('D:/dataset/CXR8/test_set/00022276_000.png')
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
    output_prob = out['prob'][0] #batch中第一张图像的概率值
    y_score.append( output_prob )
    print 'predicted class is:', output_prob.argmax()

# once you have N y_score and y_true values
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
auc = metrics.roc_auc_score(y_true, y_score)
