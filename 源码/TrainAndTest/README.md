## 恶意流量检测代码使用概要

traffic_classification.py：CNN模型代码

config.py：训练时的配置文件   

train.py：训练模型

test.py：测试模型效果

traffic_cnn.py：用于绘制一些图形效果

#### 1.请确保data目录有四个pickle数据文件，分别为training,training_lable,testing,testing_label

#### 2.配置环境

conda install tensorflow=1.15  tensorflow-gpu=1.15

pip install tensorlayer==1.7.0  -i    https://pypi.tuna.tsinghua.edu.cn/simple/ 

**使用conda 需下载miniconda或者anaconda，推荐下载miniconda，怎么使用miniconda请自行百度。**

**注意conda需要换源，通常为清华源或者上海交大源。**

#### 3.运行

打开终端至对应的运行python文件的目录或者直接用pycharm打开TrainAndTest文件



**训练或者测试传统CNN，请将config.py中的最后一行为：config.TRAIN.CNN = "simple"**

**训练或者测试skip CNN，请将config.py中的最后一行为：config.TRAIN.CNN = "skip"**



训练请运行：python train.py  或者pycharm中配置好环境后直接右键选择run train.py

测试请运行： python test.py  或者pycharm中配置好环境后直接右键选择run train.py

