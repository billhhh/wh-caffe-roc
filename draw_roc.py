import caffe
from sklearn import metrics
# load the net with trained weights
net = caffe.Net('deploy_resnet50.prototxt', '23_2_resnet50_CXR14_b64_iter_37000.caffemodel', caffe.TEST)


y_score = []
y_true = []
for i in xrange(N): # assuming you have N validation samples
    x_i = ... # get i-th validation sample
    y_true.append( y_i )  # y_i is 0 or 1 the TRUE label of x_i
    out = net.forward( data=x_i )  # get prediction for x_i
    y_score.append( out['prob'][1] ) # get score for "1" class
# once you have N y_score and y_true values
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
auc = metrics.roc_auc_score(y_true, y_scores)