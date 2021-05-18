## 处理数据

前往https://github.com/yungshenglu/USTC-TK2016可以查看工具集的具体使用方法。前往https://github.com/yungshenglu/USTC-TFC2016获取数据集。

### 使用

请先确保python环境下有如下的包：pickle、cv2、os、numpy。

#### 1.下载数据

前往https://github.com/yungshenglu/USTC-TFC2016获取数据集，并且将其放在1_Pcap文件夹下面，重命名为USTC-TFC2016，其中有些文件需要解压，解压包中有两个Pcap文件使用其中一个即可。

#### 2.打开PowerShell并且运行1_Pcap2Session.ps1

运行.\1_Pcap2Session.ps1   , 可能会需要管理员权限，使用Set-ExecutionPolicy  RemoteSigned。

成功后将会看到2_Session下面生成两个目录AllLayers\ , L7\。

#### 3.运行2_ProcessSession.ps1 

运行  .\2_ProcessSession.ps1 ， 会在3_ProcessedSession\目录下生成 FilteredSession\ ,TrimedSession\  

TrimedSession\  目录下的文件已经被处理为784bytes的大小。

#### 4.运行3_Session2Png.py

将会在4_Png目录下生成Test\ 与 Train\ 两个文件夹。

##### 5.运行img2pickle.py

1. 处理训练数据时，确保img2pickle.py文件的第六行代码为:data_dir = ".\\4_Png\\Train",且32与36行中的open中的文件分别为training.pickle 和training_label.pickle
2. 处理训练数据时，确保img2pickle.py文件的第六行代码为:data_dir = ".\\4_Png\\Test",且32与36行中的open中的文件为testing.pickle 和testing_label.pickle
3. 最后会在data目录下生成四个pickle文件格式的数据。

#### 6.将处理完成的pickle数据文件复制到TrainAndTest下的data文件夹下

