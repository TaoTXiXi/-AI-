import pickle
import time
import numpy as np
from sklearn import metrics
from traffic_classification import traffic_classification, traffic_skip
import tensorflow as tf
import tensorlayer as tl
import os
import tensorflow.nn
import seaborn as sns
import matplotlib.pyplot as plt
from config import config
def accuracy(pred, label):  #精确度计算
    predict = tf.nn.softmax(pred)
    correction = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
    return tf.reduce_mean(tf.cast(correction, tf.float32))

def main_Test():
    # ======================== config =================== #
    cnn_name = config.TRAIN.CNN
    class_num = 20  # 流量的总类别
    checkpoint_dir = '.\\model\\'
    test_img = ".\\data\\testing.pickle"             # 测试数据集
    test_label = ".\\data\\testing_label.pickle"     # 测试标签

    dict_20class = {0: 'BitTorrent', 1: 'Cridex', 2: 'Facetime', 3: 'FTP', 4: 'Geodo', 5: 'Gmail', 6: 'Htbot', 7: 'Miuref',
                    8: 'MySQL', 9: 'Neris', 10: 'Nsis-ay', 11: 'Outlook', 12: 'Shifu', 13: 'Skype', 14: 'SMB',
                    15: 'Tinba', 16: 'Virut', 17: 'Weibo', 18: 'WorldOfWarcraft', 19: 'Zeus'}
    list_benign = ['BitTorrent', 'Facetime', 'FTP', 'Gmail', 'MySQL',
                   'Outlook', 'Skype', 'SMB', 'Weibo', 'WorldOfWarcraft']
    # read data
    with open(test_img, 'rb') as f:
        X_test = pickle.load(f)

    with open(test_label, 'rb') as f:
        label_test = pickle.load(f)
    # sample_size = X_test.shape[0]  # 测试样本数量
    # define placeholder
    sample_traffic = tf.placeholder('float32', [X_test.shape[0], 28, 28, 3], name="input")
    sample_label = tf.placeholder('int64', [X_test.shape[0], class_num], name="label")

    # define network train model and test model
    if cnn_name == "simple":
        net_test = traffic_classification(sample_traffic, class_num=class_num, name="traffic", reuse=False, is_train=False)
    else:
        net_test = traffic_skip(sample_traffic, class_num=class_num, name="traffic", reuse=False,
                                          is_train=False)
    # 计算精确度
    pred = tf.nn.softmax(net_test.outputs)
    accura = accuracy(net_test.outputs, sample_label)

    print('[*] define testing options ... ')

    # Testing
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # store the model if model exist
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, cnn_name) + '.npz',
                                 network=net_test)
    X_samples_img = X_test
    X_samples_label = label_test

    label_gen = sess.run(tf.nn.softmax(net_test.outputs), {sample_traffic: X_samples_img})
    print(label_gen.dtype)
    label_gen = np.argmax(label_gen, 1)
    # 计算召回率
    recall = metrics.recall_score(np.argmax(X_samples_label, 1), label_gen, average="macro")
    # 计算f1
    f1 = metrics.f1_score(np.argmax(X_samples_label, 1), label_gen, average="macro")
    # 计算precision
    precision = metrics.precision_score(np.argmax(X_samples_label, 1), label_gen, average="macro")
    # 计算混淆矩阵
    confuse = metrics.confusion_matrix(np.argmax(X_samples_label, 1), label_gen)
    # print(label_gen.shape)
    # 计算精度
    acc = sess.run(accura, feed_dict={sample_traffic: X_samples_img,
                                     sample_label: X_samples_label})
    print("model for testing accuracy:{}".format(round(float(acc), 3)))

    # 绘制图表
    sns.set()
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    sns.heatmap(confuse, annot=False, ax=ax1, xticklabels=dict_20class.values(),
                yticklabels=dict_20class.values())  # 绘制热力图
    ax1.set_title('confusion matrix')  # 标题
    ax1.set_xlabel('predict_label')  # x轴
    ax1.set_ylabel('true_label')  # y轴

    data = [recall, f1, precision, acc]
    ax2.bar(range(len(data)), data,color=['C0', 'C1', 'C2', 'C3'])
    ax2.set(xlabel='Model', title="recall/f1/precision/accuracy", xticks=range(len(data)),
            xticklabels=["recall", "f1 ", "precision", "accuracy"],
            ylim=[0.9, 1.0])
    plt.show()

if __name__ == "__main__":
    main_Test()
