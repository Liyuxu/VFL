import os
import struct
import numpy as np
import pickle
import torch
import random

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
print(">> data shape, label shape:", data.shape, labels.shape, type(labels))
row, line = data.shape
# 把矩阵转化为元组列表
# dict1 = {}
# dict2 = {}
# dict3 = {}
for i in range(0, row):
    for j in range(0, line):
        data[i][j] = data[i][j] / 255

# print(labels[0:20])
# print(type(labels))
# labels1 = torch.from_numpy(labels[0:30000])
# labels2 = torch.from_numpy(labels[30000:60000])
# labels_list = labels.tolist()
# data_list = data.tolist()

# data1 = data[:, 0:200].tolist()
# data2 = data[:, 200:400].tolist()
# data3 = data[:, 400:].tolist()


# print(">>>> data1,2,3 shape", np.array(data1).shape, np.array(data2).shape, np.array(data3).shape)
# data1 = data[0:20000].tolist()
# data2 = data[20000:40000].tolist()
# data3 = data[40000:60000].tolist()

# for l in range(10):
#     cnt = labels_list.count(l)
#     idx = 0
#     i = 0
#     if l > 5:
#         cnt += 1
#     while i < cnt//2 and idx < 60000:
#         if labels_list[idx] == l:
#             labels1.append(labels_list[idx])
#             data1.append(data_list[idx])
#             del labels_list[idx]
#             del data_list[idx]
#             idx -= 1
#             i += 1
#         idx += 1

# data2 = data_list
# labels2 = labels_list
# print('len(labels1):{}      len(labels2):{}'.format(len(labels1), len(labels2)))
# for l in range(10):
#     print(l, ':', labels1.count(l), end='   ')
# print('\n')
# for l in range(10):
#     print(l, ':', labels2.count(l), end='   ')
# print('\n')
# print('len(data1):{}      len(data2):{}'.format(len(data1), len(data2)))

# data1 = torch.Tensor(data1)
# data2 = torch.Tensor(data2)
# data3 = torch.Tensor(data3)
#
# # labels1 = torch.Tensor(labels1)
# # labels2 = torch.Tensor(labels2)
# # labels3 = torch.Tensor(labels3)
#
# temp1 = {'x': data1, 'y': labels}
# dict1.update(temp1)
# temp2 = {'x': data2, 'y': labels}
# dict2.update(temp2)
# temp3 = {'x': data3, 'y': labels}
# dict3.update(temp3)
#
# with open('VFLMNIST/MNIST_0.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(dict1, f)
# with open('VFLMNIST/MNIST_1.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(dict2, f)
# with open('VFLMNIST/MNIST_2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(dict3, f)

print(labels[0:20], type(labels))

num_clients = 2
batch = 784 // num_clients
div_data = []
for i in range(num_clients):
    if i == num_clients-1:
        div_data.append(data[:, batch * i:].tolist())
        print(">>>> div_data{} shape:".format(i), np.array(div_data[i]).shape)
        break
    div_data.append(data[:, batch * i:batch * (i+1)].tolist())
    print(">>>> div_data{} shape:".format(i), np.array(div_data[i]).shape)

div_data_tensor, data_label = [], []
for i in range(num_clients):
    div_data_tensor.append(torch.Tensor(div_data[i]))
    print("type", type(div_data_tensor[i]))
    data_label.append({'x': div_data_tensor[i], 'y': torch.Tensor(labels)})
    print("data_label:", data_label[i], type(data_label[i]['y']), data_label[i]['y'])
    file_name = 'VFLMNIST/K' + str(num_clients) + '_' + str(i) + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(data_label[i], f)
    f.close()


