import tensorlayer as tl
from tensorlayer.layers import *

def traffic_classification(input, class_num, name=None, reuse=False, is_train = True):
    """
    Traffic classify through simple CNN
    :param input: 输入的流量数据 batch_size 28 28 3
    :param class_num: 分类数
    :param name: 名称域
    :param reuse: 是否重用
    :param is_train:
    :return: 返回分类的未经softmax结果
    """
    # ===================== Initial prepare ====================== #
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0)

    # =================== define network ========================= #
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        x = InputLayer(input, name="Input_layer")  # x.outputs.shape应为None 28 28 3
        print(x.outputs.shape)
        # Conv1  conv1.outputs.shape: None 28 28 16
        conv1 = Conv2d(x, n_filter=16, filter_size=(5, 5), strides=(1, 1), padding="SAME", act=None,
                       W_init=w_init, b_init=b_init, name="conv1")
        # Conv2  conv2.outputs.shape: None 28 28 8
        conv2 = Conv2d(conv1, n_filter=8, filter_size=(3, 3), strides=(1, 1),padding="SAME", act=None,
                       W_init=w_init, b_init=b_init, name="conv2")
        # average pool  aver_pool.outputs.shape: None 14 14 8
        aver_pool = MeanPool2d(conv2, strides=(2, 2), name="mean_pool")
        # Conv3  conv3.outputs.shape: None 14 14 8
        conv3 = Conv2d(conv2, n_filter=8, filter_size=(3, 3), strides=(1, 1),padding="SAME", act=None,
                       W_init=w_init, b_init=b_init, name="conv3")
        # max pool  max_pool.outputs.shape: None 7 7 8
        max_pool = MaxPool2d(conv3, strides=(2, 2), name="max_pool")
        # Dropout
        drop_out = DropoutLayer(layer=max_pool, keep=0.5, is_train=is_train, is_fix=True, name="dropout")
        # Flatten
        flatten = FlattenLayer(drop_out, name="flat")
        # fully connected
        full = DenseLayer(flatten, n_units=100, act=tf.nn.relu, W_init=w_init, b_init=b_init, name="dense_layer1")
        # fully connected of output
        out = DenseLayer(full, n_units=class_num, W_init=w_init, b_init=b_init,name="output")

        return out
def traffic_skip(input, class_num, name=None, reuse=False, is_train = True):
    """
    Traffic classify through simple CNN
    :param input: 输入的流量数据 batch_size 28 28 3
    :param class_num: 分类数
    :param name: 名称域
    :param reuse: 是否重用
    :param is_train:
    :return: 返回分类的未经softmax结果
    """
    # ===================== Initial prepare ====================== #
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0)

    # =================== define network ========================= #
    with tf.variable_scope(name, reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        x = InputLayer(input, name="Input_layer")  # x.outputs.shape应为None 28 28 3
        print(x.outputs.shape)
        # Conv1  conv1.outputs.shape: None 14 14  8
        conv1 = Conv2d(x, n_filter=8, filter_size=(3, 3), strides=(2, 2), padding="SAME", act=None,
                       W_init=w_init, b_init=b_init, name="conv1")
        # # Conv2  conv2.outputs.shape: None 7 7 4
        conv2 = Conv2d(conv1, n_filter=4, filter_size=(3, 3), strides=(2, 2), padding="SAME", act=None,
                       W_init=w_init, b_init=b_init, name="conv2")
        # Dropout
        drop_out = DropoutLayer(layer=conv2, keep=0.5, is_train=is_train, is_fix=True, name="dropout")
        # fully connected
        flat1 = FlattenLayer(drop_out, name="flatten1")
        full1 = DenseLayer(flat1, n_units=100, act=tf.nn.relu, W_init=w_init, b_init=b_init, name="dense_layer1")
        # skip connect
        flat2 = FlattenLayer(x, name="flatten2")
        full2 = DenseLayer(flat2, n_units=100, act=tf.nn.relu,  W_init=w_init, b_init=b_init, name="dense_layer2")
        # concat
        full = ConcatLayer([full1, full2])
        # fully connected of output
        out = DenseLayer(full, n_units=class_num, W_init=w_init, b_init=b_init, name="output")

        return out
if __name__ == "__main__":
    pass
