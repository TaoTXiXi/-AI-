import pickle
import numpy as np
import cv2
import os

data_dir = ".\\4_Png\\Test"
data_saving_path = ".\\data"
train_img = []
label_img = []
print("[*] 开始处理数据： ")
for class_ in os.listdir(data_dir):
    for file in os.listdir(os.path.join(data_dir, class_)):
        filename = os.path.join(data_dir, class_, file)  # 文件位置
        # 读取图片
        img = cv2.imread(filename=filename)
        img = (img / np.max(img)).astype("float32")
        # print(img.shape, img.dtype, np.max(img), np.min(img))
        train_img.append(img)
        # 保存图片对应的label
        label = np.zeros(20, dtype=int)
        label[int(class_)] = 1
        label_img.append(label)
        # print(label)

# 将列表转换ndarray格式
train_img = np.asarray(train_img)
label_img = np.asarray(label_img)

print("[*] 开始保存数据至pickle格式：")

print("[*] 保存归一化的图片：")
with open(os.path.join(data_saving_path, 'testing.pickle'), 'wb') as f:
    pickle.dump(train_img, f, protocol=4)

print("[*] 保存图片对应的label：")
with open(os.path.join(data_saving_path, 'testing_label.pickle'), 'wb') as f:
    pickle.dump(label_img, f, protocol=4)
