from config import read_options
import sys
import copy
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import scipy.io
from config import SERVER_ADDR, SERVER_PORT
import socket
from utils import recv_msg, send_msg
from torchvision import transforms
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import tkinter as tk
import threading


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

    pred = sum(preds)/len(preds)
    criterion = torch.nn.CrossEntropyLoss()
    batch_loss = criterion(pred, y)

    # Prediction

    for i in range(len(preds)):
        if isinstance(preds[i], int):
            continue
        correct = 0
        _, pred_labels = torch.max(preds[i], 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, y)).item()
        acc = correct / len(y)
        print(">>>> client_", i, " acc:", acc)

    correct = 0
    _, pred_labels = torch.max(pred, 1)
    pred_labels = pred_labels.view(-1)
    correct += torch.sum(torch.eq(pred_labels, y)).item()
    acc = correct / len(y)
    # print("batch_loss:", batch_loss, batch_loss.grad_fn)
    return batch_loss, acc


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
        super(Logistic, self).__init__()  # 继承语法
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


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


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

global win
global tempGraphLabel
runFlag = True
tempData = []
cv_acc = []
numRound = 0

'''
图表类，定义时参数root为父控件
'''

class tempGraph():
    def __init__(self, root):
        self.root = root  # 主窗体
        self.canvas = tk.Canvas()  # 创建一块显示图形的画布
        self.figure = self.create_matplotlib()  # 返回matplotlib所画图形的figure对象
        self.showGraphIn(self.figure)  # 将figure显示在tkinter窗体上面

    '''生成fig'''

    def create_matplotlib(self):
        # 创建绘图对象f
        f = plt.figure(num=2, figsize=(16, 8), dpi=100, edgecolor='green', frameon=True)
        # 创建一副子图
        self.fig11 = plt.subplot(1, 1, 1)
        self.line11, = self.fig11.plot([], [])

        def setLabel(fig, title, titleColor="red"):
            fig.set_title(title + "Accuracy", color=titleColor)  # 设置标题
            fig.set_xlabel('round')  # 设置x轴标签
            fig.set_ylabel("acc")  # 设置y轴标签
            fig.axis([0, numRound, 0, 1])  # 设置x,y坐标范围

        setLabel(self.fig11, "globalModel")
        return f

    '''把fig显示到tkinter'''

    def showGraphIn(self, figure):
        # 把绘制的图形显示到tkinter窗口上
        self.canvas = FigureCanvasTkAgg(figure, self.root)
        self.canvas.draw()  # 以前的版本使用show()方法，matplotlib 2.2之后不再推荐show（）用draw代替，但是用show不会报错，会显示警告
        self.canvas.get_tk_widget().pack(side=tk.TOP)  # , fill=tk.BOTH, expand=1

        # 把matplotlib绘制图形的导航工具栏显示到tkinter窗口上
        toolbar = NavigationToolbar2Tk(self.canvas,
                                       self.root)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    '''更新fig'''

    def updateMeltGraph(self, meltData):
        x = [i for i in range(len(meltData))]
        self.line11.set_xdata(x)  # x轴也必须更新
        self.line11.set_ydata(meltData)  # 更新y轴数据
        #  更新x数据，但未更新绘图范围。当我把新数据放在绘图上时，它完全超出了范围。解决办法是增加：
        self.fig11.relim()
        self.fig11.autoscale_view()
        plt.draw()


'''
更新窗口
'''


def updateWindow():
    global win
    global tempGraphLabel, tempData, runFlag
    if runFlag:
        tempGraphLabel.updateMeltGraph(cv_acc)
    win.after(200, updateWindow)  # 1000ms更新画布


'''
关闭窗口触发函数，关闭S7连接，置位flag
'''


def closeWindow():
    global runFlag
    runFlag = False
    sys.exit()


'''
创建控件
'''


def createGUI():
    global win
    win = tk.Tk()
    displayWidth = win.winfo_screenwidth()  # 获取屏幕宽度
    displayHeight = win.winfo_screenheight()
    winWidth, winHeight = displayWidth, displayHeight - 70
    winX, winY = -8, 0
    win.title("title1")
    win.geometry(
        '%dx%d-%d+%d' %
        (winWidth,
         winHeight,
         winX, winY))  # %dx%d宽度x 高度+横向偏移量(距左边)+纵向偏移量(距上边)

    win.protocol("WM_DELETE_WINDOW", closeWindow)

    graphFrame = tk.Frame(win)  # 创建图表控件
    graphFrame.place(x=0, y=0)
    global tempGraphLabel
    tempGraphLabel = tempGraph(graphFrame)

    updateWindow()  # 更新画布
    win.mainloop()


if __name__ == '__main__':

    listening_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 地址复用
    listening_sock.bind((SERVER_ADDR, SERVER_PORT))
    client_sock_all = []

    options = read_options()

    n_nodes = 4
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
        msg = ['MSG_INIT_SERVER_TO_CLIENT', options, n, n_nodes]
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

    # global cv_acc

    # 添加子线程用于实时展示精度
    # t1 = threading.Thread(target=createGUI)
    # t1.start()

    numRound = options['num_round']
    cv_loss = []
    preds = [[0 for i in range(n_nodes)] for j in range(len_indices)]
    for i in range(options['num_round']+1):
        print('---------------------------------------------------------------------------')
        print(f'\n | Global Training Round : {i} |')
        # get the index of samples
        idx_indices = i % len_indices
        x, y = trainX[idx_indices], trainY[idx_indices]
        # preds = []
        selected_clients = select_clients()

        is_last_round = False
        if aggregation_count == options['num_round']:
            is_last_round = True
            msg = ['MSG_SERVER_TO_CLIENT_SAMPLER', idx_indices, is_last_round, aggregation_count]
            for n in range(len(client_sock_all)):
                send_msg(client_sock_all[n][2], msg)
            break
        else:
            msg = ['MSG_SERVER_TO_CLIENT_SAMPLER', idx_indices, is_last_round, aggregation_count]
            for n in selected_clients:
                send_msg(client_sock_all[n][2], msg)

        aggregation_count += 1
        print('Waiting for local iteration at client')

        for n in selected_clients:
            msg = recv_msg(client_sock_all[n][2], 'MSG_CLIENT_TO_SERVER_PRED')
            pred = msg[1]
            preds[idx_indices][n] = copy.deepcopy(pred)

        global_loss, acc = calculateLoss(options, preds[idx_indices], y)
        cv_acc.append(acc)
        cv_loss.append(global_loss.item())
        print(">>>>  acc: ", acc)

        msg = ['MSG_SERVER_TO_CLIENT_GLOSS', global_loss]
        for n in selected_clients:
            send_msg(client_sock_all[n][2], msg)

    saveTitle = './simulationData/server_' + 'K' + str(options['clients_per_round']) \
                + 'T' + str(options['num_round']) + 'B' + str(options['batch_size'])
    saveVariableName = 'server_' + 'K' + str(options['clients_per_round']) \
                       + 'T' + str(options['num_round']) + 'B' + str(options['batch_size'])
    scipy.io.savemat(saveTitle + '_acc' + '.mat', mdict={saveVariableName + '_acc': cv_acc})
    scipy.io.savemat(saveTitle + '_loss' + '.mat', mdict={saveVariableName + '_loss': cv_loss})

    # Save tracked information
    # plot
    plotTitle = './jpg/server_' + 'K' + str(options['clients_per_round']) \
                + 'T' + str(options['num_round']) + 'B' + str(options['batch_size'])
    plt.figure(1)
    plt.plot(cv_acc)
    plt.title("Global Accuracy")
    plt.xlabel("round")
    plt.ylabel("accuracy")
    plt.savefig(plotTitle + '_acc.jpg')
    plt.close()

    plt.figure(2)
    plt.plot(cv_loss)
    plt.title("Global Loss")
    plt.xlabel("round")
    plt.ylabel("loss")
    plt.savefig(plotTitle + '_loss.jpg')
    plt.close()
