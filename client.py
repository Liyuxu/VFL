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


# read data set from pkl files

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


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    device = 'cuda' if args['gpu'] else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    loss = loss / total
    return accuracy, loss


# Model for MQTT_IOT_IDS dataset
class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


# socket
sock = socket.socket()
sock.connect((SERVER_ADDR, SERVER_PORT))
print('---------------------------------------------------------------------------')
try:
    while True:
        msg = recv_msg(sock, 'MSG_INIT_SERVER_TO_CLIENT')
        print('Received message from server:', msg)
        options = msg[1]
        cid = msg[2]
        n_nodes = msg[3]

        # Training parameters
        lr_rate = options['lr']  # Initial learning rate
        batch_size = options['batch_size']  # Data sample for training per comm. round
        num_round = options['num_round']
        out_dim = options['out_dim']

        # Import the data set
        file_name = './VFLMNIST/K' + str(n_nodes) + '_' + str(cid) + '.pkl'
        train_data = read_data(file_name)

        print(file_name, 'train data read successfully')
        # Init model: make the data loader, divide dataset by batch_size; shuffle = False
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_file_name = './VFLMNIST/test_K' + str(n_nodes) + '_' + str(cid) + '.pkl'
        test_data = read_data(test_file_name)
        print(test_file_name, 'test data read successfully')
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=128,
                                                  shuffle=False)
        trainX, trainY = [], []
        for idx, (x, y) in enumerate(train_loader):
            trainX.append(x)
            trainY.append(y)
        test_x, test_y = next(iter(train_loader))
        # init model
        in_dim = len(test_x[0])
        model = Logistic(in_dim, out_dim)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
        # Learning rate decay , lr = lr * gamma
        gamma = 0.9
        step_size = 100
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch=-1)
        criterion = torch.nn.CrossEntropyLoss()

        cv_acc, cv_loss = [], []

        while True:
            print('---------------------------------------------------------------------------')
            msg = recv_msg(sock, 'MSG_SERVER_TO_CLIENT_SAMPLER')
            indices = msg[1]
            is_last_round = msg[2]
            round_i = msg[3]
            print(">>  Round ", round_i)
            print(">>   lr  =", optimizer.param_groups[0]['lr'])
            if is_last_round:
                saveTitle = './simulationData/client' + str(cid) + '_K' + str(options['clients_per_round']) \
                            + 'T' + str(options['num_round']) + 'B' + str(options['batch_size'])
                saveVariableName = 'client' + str(cid) + 'K' + str(options['clients_per_round']) \
                                   + 'T' + str(options['num_round']) + 'B' + str(options['batch_size'])
                scipy.io.savemat(saveTitle + '_acc' + '.mat', mdict={saveVariableName + '_acc': cv_acc})
                scipy.io.savemat(saveTitle + '_loss' + '.mat', mdict={saveVariableName + '_loss': cv_loss})
                break

            x, y = trainX[indices], trainY[indices]
            print("idx: ", indices, " y:", y)
            print('Make dataloader successfully')

            # calculate predication
            model.train()
            optimizer.zero_grad()
            pred = model(x)
            # send pred to server
            msg = ['MSG_CLIENT_TO_SERVER_PRED', pred]
            send_msg(sock, msg)

            # for backward
            # calculate a loss value casually
            loss = criterion(pred, y)
            # receive global loss
            msg = recv_msg(sock, 'MSG_SERVER_TO_CLIENT_GLOSS')
            global_loss = msg[1]
            # global_loss.data -> loss.data
            loss.data = torch.full_like(loss, global_loss.item())

            loss.backward()
            optimizer.step()
            # learning rate decay
            scheduler.step()

            testaccuracy, testloss = test_inference(options, model, test_loader)
            cv_acc.append(testaccuracy)
            cv_loss.append(testloss)
            print("------ acc =", testaccuracy, "------ loss =", testloss)

except (struct.error, socket.error):
    print('Server has stopped')
    pass
