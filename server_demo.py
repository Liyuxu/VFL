import argparse
import os
import copy
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


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


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

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg9')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='mnist_all_data_0_equal_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='logistic')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0.001)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=False,
                        help='use gpu (default: False)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=500)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=5)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=2)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=10000)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=40)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.01)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    # print('parser is :', parser)
    parsed = parser.parse_args()
    # print('parsed is :', parsed)
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()

    return options


def select_clients():
    num_clients = min(options['clients_per_round'], n_nodes)
    # np.random.seed(seed)
    return np.random.choice(range(0, len(client_sock_all)), num_clients, replace=False).tolist()


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
        print("Waiting for incoming connections...")
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

    test_data = read_data('./MNIST_test.pkl')

    test_loader = DataLoader(dataset=test_data,
                             batch_size=64,
                             shuffle=True)

    global_model = Logistic(784, 10)

    global_model.to(device)
    global_model.train()

    global_weights = global_model.state_dict()

    train_accuracy, train_loss = [], []
    cv_loss, cv_acc = [], []
    print_every = 2

    global_train_time = []
    start1 = time.time()
    for i in range(options['num_round']):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {i + 1} |\n')

        global_weights = global_model.state_dict()

        selected_clients = select_clients()

        is_last_round = False
        print('---------------------------------------------------------------------------')
        aggregation_count += 1
        if aggregation_count == options['num_round']:
            is_last_round = True

        start = time.time()
        for n in selected_clients:
            msg = ['MSG_WEIGHT_TAU_SERVER_TO_CLIENT', is_last_round, global_weights, aggregation_count]
            send_msg(client_sock_all[n][2], msg)

        print('Waiting for local iteration at client')

        for n in selected_clients:
            msg = recv_msg(client_sock_all[n][2], 'MSG_WEIGHT_TIME_SIZE_CLIENT_TO_SERVER')

            w = msg[1]
            local_weights.append(copy.deepcopy(w))
            # local_losses.append(copy.deepcopy(loss))

        global_weights = average_weights(local_weights)

        global_model.load_state_dict(global_weights)

        end = time.time()
        test_acc, test_loss = test_inference(options, global_model, test_loader)
        print(test_acc, test_loss)
        cv_acc.append(test_acc)
        cv_loss.append(test_loss)
        global_train_time.append(end - start)
    end1 = time.time()

    saveTitle = 'K' + str(options['clients_per_round']) + 'T' + str(options['num_round']) + 'E' + str(
        options['num_epoch']) + 'B' + str(options['batch_size'])
    scipy.io.savemat(saveTitle + '_time' + '.mat', mdict={saveTitle + '_time': global_train_time})
    scipy.io.savemat(saveTitle + '_acc' + '.mat', mdict={saveTitle + '_acc': cv_acc})
    scipy.io.savemat(saveTitle + '_loss' + '.mat', mdict={saveTitle + '_loss': cv_loss})
    # Save tracked information
