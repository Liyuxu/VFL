import argparse
import os
import copy
import pandas as pd
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import importlib
from torch.utils.data import DataLoader, Dataset
import scipy.io
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS, BATCH_LIST, SERVER_ADDR, SERVER_PORT
import importlib
import socket
from utils import recv_msg, send_msg
from torchvision import transforms
import math
from PIL import Image


class MiniDataset(Dataset):
    def __init__(self, data, labels):
        super(MiniDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")

        if self.data.ndim == 4 and self.data.shape[3] == 3:
            self.data = self.data.reshape(-1, 16, 16, 3).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, 4),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ]
            )
        elif self.data.ndim == 4 and self.data.shape[3] == 1:
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        elif self.data.ndim == 3:
            self.data = self.data.reshape(-1, 28, 28, 1).astype("uint8")
            self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))
                 ]
            )
        else:
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

    data.update(cdata)

    data = MiniDataset(data['x'], data['y'])

    return data


def calculateLoss(args, preds, y):
    '''

    Args:
        args: options
        preds: [pred1,pred2,...,predn]
        y: labels

    Returns: CrossEntropyLoss(preds, y)

    '''
    print(">> y:", y)
    pred = sum(preds)
    criterion = torch.nn.CrossEntropyLoss()
    batch_loss = criterion(pred, y)

    # Prediction
    for i in range(len(preds)):
        correct = 0
        _, pred_labels = torch.max(preds[i], 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, y)).item()
        acc = correct / len(y)
        print(">>>> ", i, " acc:", acc)

    correct = 0
    _, pred_labels = torch.max(pred, 1)
    pred_labels = pred_labels.view(-1)
    correct += torch.sum(torch.eq(pred_labels, y)).item()
    acc = correct/len(y)
    # print("batch_loss:", batch_loss, batch_loss.grad_fn)
    return batch_loss, acc


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args['gpu'] else 'cpu'
    criterion = torch.nn.CrossEntropyLoss()
    # testloader = DataLoader(test_dataset, batch_size=128,
    #                         shuffle=False)

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
        super(Logistic, self).__init__()  # 继承语法
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=50)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=2)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=5000)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.3)
    parser.add_argument('--out_dim',
                        help='output dimension',
                        type=int,
                        default=10)

    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    return options


def select_clients():
    num_clients = min(options['clients_per_round'], n_nodes)
    return np.random.choice(range(0, len(client_sock_all)), num_clients, replace=False).tolist()


def divideSampler(num_samples, batch_size):
    import random
    lis = list(range(num_samples))
    random.shuffle(lis)
    sampler = []
    if batch_size > num_samples:
        raise Exception("The batch_size should be less than or equal to the number of samples")
    for i in range(num_samples // batch_size):
        sampler.append(lis[batch_size * i:batch_size * (i + 1)])
    if num_samples % batch_size:
        sampler.append(lis[num_samples // batch_size * batch_size:])
    return sampler


if __name__ == '__main__':

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 地址复用
    listening_sock.bind((SERVER_ADDR, SERVER_PORT))
    client_sock_all = []

    options = read_options()

    n_nodes = 2
    aggregation_count = 0
    # Establish connections to each client, up to n_nodes clients, setup for clients
    while len(client_sock_all) < n_nodes:
        listening_sock.listen(5)
        print("Waiting for {} incoming connections...".format(n_nodes))
        (client_sock, (ip, port)) = listening_sock.accept()
        print('Got connection from ', (ip, port))
        print(client_sock)
        client_sock_all.append([ip, port, client_sock])

    for n in range(0, n_nodes):
        msg = ['MSG_INIT_SERVER_TO_CLIENT', options, n]
        send_msg(client_sock_all[n][2], msg)

    print('All clients connected')

    # exp_details(options)
    if options['gpu']:
        torch.cuda.set_device(options['gpu'])
    device = 'cuda' if options['gpu'] else 'cpu'

    # print config informations
    print("Task Config:\n--n_nodes =", n_nodes, "\n--K =", options['clients_per_round'],
          "\n--BatchSize =", options['batch_size'], "\n--NumIter =", options['num_round'],
          "\n--learning rate =", options['lr'])

    # Load Dataset
    test_data = read_data('VFLMNIST/K4_0.pkl')
    batch_size = options['batch_size']
    test_loader = DataLoader(dataset=test_data,
                             batch_size=options['batch_size'],
                             shuffle=False)
    trainX, trainY = [], []
    for idx, (x, y) in enumerate(test_loader):
        trainX.append(x)
        trainY.append(y)

    indices = divideSampler(60000, batch_size)
    len_indices = len(indices)

    cv_acc = []
    
    for i in range(options['num_round']):
        print('---------------------------------------------------------------------------')
        print(f'\n | Global Training Round : {i+1} |')
        # get the index of samples
        idx_indices = i % len_indices
        x, y = trainX[idx_indices], trainY[idx_indices]

        preds = []
        selected_clients = select_clients()

        is_last_round = False
        aggregation_count += 1
        if aggregation_count == options['num_round']:
            is_last_round = True

        msg = ['MSG_SERVER_TO_CLIENT_SAMPLER', idx_indices, is_last_round, aggregation_count]
        for n in selected_clients:
            send_msg(client_sock_all[n][2], msg)

        print('Waiting for local iteration at client')

        for n in selected_clients:
            msg = recv_msg(client_sock_all[n][2], 'MSG_CLIENT_TO_SERVER_PRED')
            pred = msg[1]
            preds.append(copy.deepcopy(pred))

        global_loss, acc = calculateLoss(options, preds, y)
        cv_acc.append(acc)
        print(">>>>  acc: ", acc)

        msg = ['MSG_SERVER_TO_CLIENT_GLOSS', global_loss]
        for n in selected_clients:
            send_msg(client_sock_all[n][2], msg)

    saveTitle = './simulationData/server_' + 'K' + str(options['clients_per_round']) \
                + 'T' + str(options['num_round']) + 'B' + str(options['batch_size'])
    saveVariableName = 'server_' + 'K' + str(options['clients_per_round']) \
                       + 'T' + str(options['num_round']) + 'B' + str(options['batch_size'])
    scipy.io.savemat(saveTitle + '_acc' + '.mat', mdict={saveVariableName + '_acc': cv_acc})
    # save csv
    # name = ['acc']
    # test = pd.DataFrame(columns=name, data=cv_acc)
    # test.to_csv(saveTitle + '_acc' + '.csv', encoding='gbk')

