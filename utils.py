import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
import h5py
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import torch.nn.functional as F


def h5_data_write(matrix, save_path):
    print("h5py文件正在写入磁盘...")
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('d_matrix', data=matrix)
    print("h5py文件保存成功！")

def h5_data_read(filename):
    """
        keys() ： 获取本文件夹下所有的文件及文件夹的名字
        f['key_name'] : 获取对应的对象
    """
    file = h5py.File(filename, 'r')
    matrix = file['d_matrix'][:]
    print("文件读取完毕...")
    return matrix

def get_data(dataset_name='muufl'):
    if dataset_name == 'muufl':
        print('Muufl......')
        hsi = sio.loadmat('../dataset/muufl/HSI.mat')['data']
        labels = sio.loadmat('../dataset/muufl/HSI.mat')['gt']
        lidar = sio.loadmat('../dataset/muufl/LiDAR_DEM.mat')['LiDAR_DEM']

    if dataset_name == 'trento':
        print("trento......")
        hsi = sio.loadmat('./HSI_data.mat')['HSI_data']
        labels = sio.loadmat('./All_Label.mat')['All_Label']
        lidar = sio.loadmat('./LiDAR_data.mat')['LiDAR_data']


    if dataset_name == '2013Houston':
        print("2013Houston......")
        hsi = sio.loadmat('../dataset/houston/HSI_data.mat')['HSI_data']
        labels = sio.loadmat('../dataset/houston/All_Label.mat')['All_Label']
        lidar = sio.loadmat('../dataset/houston/LiDAR_data.mat')['LiDAR_data']

    hsi_ = np.zeros_like(hsi)
    lidar_ = np.zeros_like(lidar)
    scaler = StandardScaler()
    for i in range(hsi.shape[2]):
        hsi_[:,:,i] = scaler.fit_transform(hsi[:,:,i])
    lidar_ = scaler.fit_transform(lidar)

    return hsi_, labels, lidar_

def applyPCA(X, numComponents=48):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca



class Queue:
    def __init__(self, capacity, dim):
        self.capacity = capacity
        self.size = 0
        self.queue = torch.empty((capacity, dim), dtype=torch.float32)
        self.front = 0
        self.rear = -1

    def is_full(self):
        return self.size == self.capacity

    def is_empty(self):
        return self.size == 0

    def enqueue(self, item):
        if self.is_full():
            print("Queue is full")
            return
        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty")
            return
        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item

    def peek(self):
        if self.is_empty():
            print("Queue is empty")
            return
        return self.queue[self.front]

    def display(self):
        if self.is_empty():
            print("Queue is empty")
            return
        if self.front <= self.rear:
            print(self.queue[self.front:self.rear+1])
        else:
            print(torch.cat((self.queue[self.front:], self.queue[:self.rear+1])))

    def get_queue(self):
        return self.queue

    def get_num(self):
        return self.size


def get_adj(hsi,all, lidar, device, edge_num=10):


    distances_hsi = F.pairwise_distance(hsi.unsqueeze(1), hsi)
    distances_lidar = F.pairwise_distance(lidar.unsqueeze(1), lidar)
    distances_all = F.pairwise_distance(all.unsqueeze(1), all)
    corr_matrix = distances_hsi + distances_lidar + distances_all
    _, indices = torch.topk(corr_matrix, edge_num, dim=1)
    corr_matrix_y = torch.zeros_like(corr_matrix)
    corr_matrix_y = corr_matrix_y.scatter_(1, indices, corr_matrix.gather(1, indices)).detach().cpu()
    corr_matrix = sp.coo_matrix(corr_matrix_y)
    indices, values = from_scipy_sparse_matrix(corr_matrix)
    adj = indices

    return adj