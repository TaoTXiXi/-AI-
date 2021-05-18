import pickle
import time
from traffic_classification import traffic_classification, traffic_skip
import tensorflow as tf
import tensorlayer as tl
import os
import numpy as np
from config import config
def accuracy(pred, label):  #精确度计算
    predict = tf.nn.softmax(pred)
    correction = tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1))
    return tf.reduce_mean(tf.cast(correction, tf.float32))

def main_Train():
    # ======================== config =================== #
    batch_size = config.TRAIN.batch_size         # batch_size
    class_num = config.TRAIN.class_num           # 流量的总类别
    lr = config.TRAIN.lr                         # 学习率
    beta1 = config.TRAIN.beta1                   # Adam
    n_epoch = config.TRAIN.n_epoch               # n_epoch
    decay_every = config.TRAIN.decay_every       # decay_every
    lr_decay = config.TRAIN.lr_decay             # lr_decay
    checkpoint_dir = config.TRAIN.checkpoint_dir  # checkpoint_dir
    train_img = config.TRAIN.train_img           # 训练数据集
    train_label = config.TRAIN.train_label       # 训练标签
    test_img = config.TRAIN.test_img             # 测试数据集
    test_label = config.TRAIN.test_label         # 测试标签
    cnn_name = config.TRAIN.CNN                  # CNN model name

    dict_20class = {0: 'BitTorrent', 1: 'Cridex', 2: 'Facetime', 3: 'FTP', 4: 'Geodo', 5: 'Gmail', 6: 'Htbot',
                    7: 'Miuref',8: 'MySQL', 9: 'Neris', 10: 'Nsis-ay', 11: 'Outlook', 12: 'Shifu', 13: 'Skype', 14: 'SMB',
                    15: 'Tinba', 16: 'Virut', 17: 'Weibo', 18: 'WorldOfWarcraft', 19: 'Zeus'}

    # read data
    with open(train_img, 'rb') as f:
        X_train = pickle.load(f)
    print(X_train.shape)
    with open(train_label, 'rb') as f:
        X_label = pickle.load(f)

    with open(test_img, 'rb') as f:
        X_test = pickle.load(f)

    with open(test_label, 'rb') as f:
        label_test = pickle.load(f)

    # define placeholder
    input_traffic = tf.placeholder('float32', [batch_size, 28, 28, 3], name="input")
    label = tf.placeholder('int32', [batch_size,class_num], name="label")

    # define network train model and test model
    if cnn_name == "simple":
        net = traffic_classification(input_traffic, class_num=class_num, name="traffic", reuse=False, is_train=True)
        net_test = traffic_classification(input_traffic, class_num=class_num, name="traffic", reuse=True,
                                          is_train=False)
    else:
        net = traffic_skip(input_traffic, class_num=class_num, name="traffic", reuse=False, is_train=True)
        net_test = traffic_skip(input_traffic, class_num=class_num, name="traffic", reuse=True,
                                          is_train=False)
    # net = traffic_skip(input_traffic, class_num=class_num, name="traffic", reuse=False, is_train=True)
    # net_test = traffic_skip(input_traffic, class_num=class_num, name="traffic", reuse=True, is_train=False)

    # define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net.outputs, labels=label))
    # 计算精确度
    pred = net.outputs
    accura = accuracy(pred, label)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)

    print('[*] define training options ... ')
    c_vars = tl.layers.get_variables_with_name('traffic', True, True)
    # 优化函数
    c_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=c_vars)

    # TRAINING
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)
    # store the model if model exist
    tl.files.load_and_assign_npz(sess=sess,
                                 name=os.path.join(checkpoint_dir, cnn_name) + '.npz',
                                 network=net)

    for epoch in range(n_epoch):
        # learning rate decay
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)
        # Training model
        i = 0
        for batch in tl.iterate.minibatches(inputs=X_train, targets=X_label, batch_size=batch_size, shuffle=True):
            i += 1
            input_batch, input_label = batch
            start = time.time()
            acc_batch, loss_batch, _ = sess.run([accura, loss, c_optim], feed_dict={input_traffic: input_batch,
                                                                        label: input_label})
            if i%20 == 0:   # 每20个batch_size输出一次结果
                print("[epoch:{:2}\{:2}] acc:{:4} loss:{:4} time:{:4}"
                      .format(epoch+1, n_epoch,
                              np.round(float(acc_batch), 3),
                              np.round(float(loss_batch), 3),
                              np.round(float(time.time()-start), 3)))

        # Test the model
        acc_all = 0
        i = 0
        for batch in tl.iterate.minibatches(inputs=X_test, targets=label_test, batch_size=batch_size, shuffle=True):
            test_, test_label = batch
            y_ = sess.run(net_test.outputs, {input_traffic: test_})
            acc = sess.run(accuracy(y_,test_label), feed_dict={label:test_label})
            acc_all += acc
            i += 1
        print("model for testing accuracy:{}".format(round(float(acc_all/i), 3)))
        print("================== Next epoch =================")

    # save current best model
    tl.files.save_npz(net.all_params,
                      name=os.path.join(checkpoint_dir, cnn_name) + '.npz',
                      sess=sess)
    print("[*] Save checkpoints SUCCESS!")

if __name__ == "__main__":
    main_Train()
