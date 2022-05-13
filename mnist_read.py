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

test_data, test_labels = load_mnist('./MNIST', kind='t10k')
test_data = test_data.astype(float)

print(">> data shape, label shape:", data.shape, labels.shape, type(labels))
print(">> data shape, label shape:", test_data.shape, test_labels.shape, type(test_labels))
row, line = data.shape
for i in range(0, row):
    for j in range(0, line):
        data[i][j] = data[i][j] / 255

t_row, t_line = test_data.shape
for i in range(0, t_row):
    for j in range(0, t_line):
        test_data[i][j] = test_data[i][j] / 255
# Make dataloader successfully

num_clients = 4
batch = 784 // num_clients

# divide dataset
div_data = []
print("type(data), data.shape[1]:", type(data), data.shape[1])
col_rand_array = np.arange(data.shape[1])
np.random.shuffle(col_rand_array)

div_test_data = []
for i in range(num_clients):
    if i == num_clients-1:
        div_data.append(data[:, col_rand_array[batch * i:]].tolist())
        print(">>>> div_data{} shape:".format(i), np.array(div_data[i]).shape)
        div_test_data.append(test_data[:, col_rand_array[batch * i:]].tolist())
        print(">>>> div_test_data{} shape:".format(i), np.array(div_test_data[i]).shape)
        break
    div_data.append(data[:, col_rand_array[batch * i:batch * (i+1)]].tolist())
    print(">>>> div_data{} shape:".format(i), np.array(div_data[i]).shape)
    div_test_data.append(test_data[:, col_rand_array[batch * i:batch * (i + 1)]].tolist())
    print(">>>> div_test_data{} shape:".format(i), np.array(div_test_data[i]).shape)


# list to tensor
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

div_test_data_tensor, test_data_sets = [], []
for i in range(num_clients):
    div_test_data_tensor.append(torch.Tensor(div_test_data[i]))
    print("test_type", type(div_test_data_tensor[i]))
    test_data_sets.append({'x': div_test_data_tensor[i], 'y': torch.Tensor(test_labels)})
    print("test_data_sets:", test_data_sets[i], type(test_data_sets[i]['y']), test_data_sets[i]['y'])
    file_name = 'VFLMNIST/test_K' + str(num_clients) + '_' + str(i) + '.pkl'
    with open(file_name, 'wb') as f:
        pickle.dump(test_data_sets[i], f)
    f.close()

