import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch
import pickle  # for pkl file reading
import os
import sys
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import time
import scipy.io
import torchvision.transforms as transforms
from config import SERVER_ADDR, SERVER_PORT
from utils import recv_msg, send_msg
import socket
import struct
from torchvision import transforms
import math


def read_data(data_dir):
    """Parses data in given train and test data directories

    Assumes:
        1. the data in the input directories are .json files with keys 'users' and 'user_data'
        2. the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data (ndarray)
        test_data: dictionary of test data (ndarray)
    """

    # clients = []
    # groups = []
    data = {}
    print('>>> Read data from:', data_dir)

    # open training dataset pkl files
    with open(data_dir, 'rb') as inf:
        cdata = pickle.load(inf)
    # print('cdata :', cdata)
    data.update(cdata)

    data = MiniDataset(data['x'], data['y'])

    return data


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        self.data = self.data.astype("float32")
        self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            data = Image.fromarray(data)

        if self.transform is not None:
            data = self.transform(data)

        return data, target


file_name = 'VFLMNIST/K4_0.pkl'
train_data = read_data(file_name)
print('data read successfully')
# train_loader = torch.utils.data.DataLoader(dataset=train_data,
#                                            batch_size=batch_size,
#                                            shuffle=False)
# for idx, (x, y) in enumerate(train_loader):
#     print(idx, x.shape, y.shape)
# x, y = next(iter(train_loader))
# print(x, y, x.shape, y.shape)
# x, y = next(iter(train_loader))
# print(x, y, x.shape, y.shape)
# print(np.random.choice(range(1, 600), 10, replace=False).tolist())
# print(np.random.choice(range(1, 600), 10, replace=False).tolist())
# sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)

batch_size = 3
indices = np.random.choice(range(1, 600), batch_size, replace=False).tolist()
print(indices)
sampler_val = torch.utils.data.sampler.SequentialSampler(indices)
validation_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=batch_size,
                                                sampler=sampler_val)
batchsampler_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_sampler=sampler_val)
x, y = next(iter(validation_loader))
print("batchsampler:", batchsampler_loader)
# print(x.shape, y.shape)
# print(x, y)
print("for ----------")
for idx, (x, y) in enumerate(validation_loader):
    print(idx, x.shape, y.shape)
    print(x, y)

train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=1,
                                           shuffle=False)
for idx, (x, y) in enumerate(train_loader):
    if idx in indices:
        print(idx, x, y)
        print(train_loader[idx])

# sampler = list(range(600))
# import random
# random.shuffle(sampler)
# print(sampler)
# print(sampler.index(0))
print('Make dataloader successfully')
