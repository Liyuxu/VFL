import os
import struct
import numpy as np
import pickle
import torch


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


data, labels = load_mnist('./MNIST/')
data = data.astype(float)

row, line = data.shape
# 把矩阵转化为元组列表
dict1 = {}
dict2 = {}
for i in range(0, row):
    for j in range(0, line):
        data[i][j] = data[i][j] / 255

# print(labels[0:20])
# print(type(labels))
# labels1 = torch.from_numpy(labels[0:30000])
# labels2 = torch.from_numpy(labels[30000:60000])
labels_list = labels.tolist()
data_list = data.tolist()
data1 = []
data2 = []
labels1 = []
labels2 = []
# print(labels_list[0:20], type(labels_list))
# labels1 = labels[0:30000].tolist()
# labels2 = labels[30000:60000].tolist()
# data1 = data[0:30000].tolist()
# data2 = data[30000:60000].tolist()

for l in range(10):
    cnt = labels_list.count(l)
    idx = 0
    i = 0
    while i < cnt and idx < 60000 and len(labels_list) > 30000:
        if labels_list[idx] == l:
            labels1.append(labels_list[idx])
            data1.append(data_list[idx])
            del labels_list[idx]
            del data_list[idx]
            idx -= 1
            i += 1
        idx += 1

data2 = data_list
labels2 = labels_list
print('len(labels1):{}      len(labels2):{}'.format(len(labels1), len(labels2)))
for l in range(10):
    print(l, ':', labels1.count(l), end='   ')
print('\n')
for l in range(10):
    print(l, ':', labels2.count(l), end='   ')
print('\n')
print('len(data1):{}      len(data2):{}'.format(len(data1), len(data2)))
print(data1[1])

data1 = torch.Tensor(data1)
data2 = torch.Tensor(data2)
labels1 = torch.Tensor(labels1)
labels2 = torch.Tensor(labels2)

temp1 = {'x': data1, 'y': labels1}
dict1.update(temp1)

temp2 = {'x': data2, 'y': labels2}
dict2.update(temp2)

with open('iid/MNIST_niid0.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(dict1, f)
with open('iid/MNIST_niid1.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(dict2, f)
