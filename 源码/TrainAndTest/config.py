from easydict import EasyDict as edict

config = edict()
config.TRAIN = edict()

config.TRAIN.batch_size = 64  # batch_size
config.TRAIN.class_num = 20  # 流量的总类别
config.TRAIN.lr = 0.001      # 学习率
config.TRAIN.beta1 = 0.5     # Adam
config.TRAIN.n_epoch = 10
config.TRAIN.decay_every = 5
config.TRAIN.lr_decay = 0.5
config.TRAIN.checkpoint_dir = 'model\\'
config.TRAIN.train_img = ".\\data\\training.pickle"           # 训练数据集
config.TRAIN.train_label = ".\\data\\training_label.pickle"   # 训练标签
config.TRAIN.test_img = ".\\data\\testing.pickle"             # 测试数据集
config.TRAIN.test_label = ".\\data\\testing_label.pickle"     # 测试标签
config.TRAIN.CNN = "skip"                                   # CNN model name  "simple" or skip