# 导入模块
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pandas as pd
from collections import Counter
from copy import deepcopy
import datasets as local_datasets
import numpy as np
import matplotlib.pyplot as plt
def find_cls(inter_sum, rnd):
    for i in range(len(inter_sum)):
        if rnd<inter_sum[i]:
            break
    return i - 1

def get_tag(data_name):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if data_name =='CIFAR10':
        train_dataset = datasets.CIFAR10(
            "./data",
            train=True,
            download=True,
            transform=transform_train)
    elif data_name == 'CIFAR100':
        train_dataset = datasets.cifar.CIFAR100(
            "./data",
            train=True,
            download=True,
            transform=transform_train)
    elif data_name =='EMNIST':
        train_dataset = datasets.EMNIST(
            "./data",
            split='byclass',
            train=True,
            download=True,
            transform=transforms.ToTensor())
    elif data_name =='MNIST':
        train_dataset = datasets.EMNIST(
            "./data",
            #split='mnist',
            split='balanced',
            #split='byclass',
            train=True,
            download=True,
            transform=transforms.ToTensor())
    if data_name == 'tiny-imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
        ])
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join('./data', 'tiny_imagenet'),
            split='train',
            transform=transform_train
        )
    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomRotation(10),  # RandomRotation 추가
            transforms.RandomCrop(64, padding=4),
            transforms.RandomResizedCrop((224, 224)),
            # resize 256_comb_coteach_OpenNN_CIFAR -> random_crop 224 ==> crop 32, padding 4
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2770, 0.2691, 0.2821]),
        ])
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join('./data', 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )
    # print('len(train_dataset)',len(train_dataset)) len(train_dataset) 697932

    id2targets =[train_dataset[i][1] for i in range(len(train_dataset))]
    targets = np.array(id2targets)
    # counter = Counter(targets)
    # print(counter)
    sort_index = np.argsort(targets)

    return id2targets, sort_index

def data_from_dirichlet(data_name, alpha_value, nums_cls, nums_wk, nums_sample ):
    # data_name = 'CIFAR10'
    id2targets, sort_index = get_tag(data_name)
    # print('len(sort_index)',len(sort_index))

    # 生成随机数
    dct = {}
    for idx in sort_index:
        cls = id2targets[idx]
        if not dct.get(cls):
            dct[cls]=[]
        dct[cls].append(idx)
    sort_index = [dct[key] for key in dct.keys()]
    # for i in sort_index:
    #     print(len(i))
    tag_index = deepcopy(sort_index)
    # sort_index = sort_index.reshape((nums_cls,-1))
    # sort_index = list(sort_index)
    # tag_index = [list(i) for i in sort_index]
    # print('len(tag_index)',len(tag_index))

    #类别数个维度。
    alpha = [alpha_value] * nums_cls 
    gamma_rnd = np.zeros([nums_cls, nums_wk])
    dirichlet_rnd = np.zeros([nums_cls, nums_wk])
    for n in range(nums_wk):
        if n%10==0:
            alpha1 = 1
            # alpha1 = 100 
        else:
            alpha1 = 1
        for i in range(nums_cls):
            gamma_rnd[i, n]=np.random.gamma(alpha1 * alpha[i], 1)
        # 逐样本归一化（对维度归一化）
        Z_d = np.sum(gamma_rnd[:, n])
        dirichlet_rnd[:, n] = gamma_rnd[:, n]/Z_d
    # print('dirichlet_rnd',dirichlet_rnd[:,1])

    #对每个客户端
    data_idx = []
    for j in range(nums_wk):
        #q 向量
        inter_sum = [0]
        #计算概率前缀和
        for i in dirichlet_rnd[:,j]:
            inter_sum.append(i+inter_sum[-1])
        sample_index = []
        for i in range(nums_sample):
            rnd = np.random.random()
            sample_cls = find_cls(inter_sum, rnd)
            if len(tag_index[sample_cls]):
                sample_index.append(tag_index[sample_cls].pop()) 
            elif len(tag_index[sample_cls])==0:
                # print('cls:{} is None'.format(sample_cls))
                tag_index[sample_cls] = deepcopy(sort_index[sample_cls])
                # tag_index[sample_cls] = list(sort_index[sample_cls])
                sample_index.append(tag_index[sample_cls].pop()) 
        # print('sample_index',sample_index[:10])
        data_idx.append(sample_index)
    cnt = 0
    std = [pd.Series(Counter([id2targets[j] for j in data])).describe().std() for data in data_idx]
    print('std:',std)
    print('label std:',np.mean(std))
    for data in data_idx:
        if cnt%20==0:
            a = [id2targets[j] for j in data]
            print(Counter(a))
            print('\n')
        cnt+=1

    from mpl_toolkits.mplot3d import Axes3D
    # 生成随机数
    alpha = [0.5] * 3  # 三维平狄利克雷分布
    N = 1000;
    L = len(alpha)  # 样本数N=1000
    gamma_rnd = np.zeros([L, N]);
    dirichlet_rnd = np.zeros([L, N])
    for n in range(N):
        for i in range(L):
            gamma_rnd[i, n] = np.random.gamma(alpha[i], 1)
        # 逐样本归一化（对维度归一化）
        Z_d = np.sum(gamma_rnd[:, n])
        dirichlet_rnd[:, n] = gamma_rnd[:, n] / Z_d
    # 绘制散点图
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(dirichlet_rnd[0, :], dirichlet_rnd[1, :], dirichlet_rnd[2, :])
    ax.view_init(30, 60)
    # print(data_idx[0])
    return data_idx, std 
  
# # data_name = 'EMNIST'
# # data_name = 'CIFAR10'
# if data_name =='EMNIST':
#     alpha_value = 0.1
#     nums_cls = 62 #62 10
#     nums_wk = 100
#     nums_sample=6979 #6979 500
# else:
#     alpha_value = 0.1
#     nums_cls =  10 #62 10
#     nums_wk =   100
#     nums_sample=500 #6979 500    
# data_from_dirichlet(data_name, alpha_value, nums_cls, nums_wk, nums_sample)

# 导入模块
#'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 生成随机数
alpha = [0.5]*3 # 三维平狄利克雷分布
N = 1000; L = len(alpha) # 样本数N=1000
gamma_rnd = np.zeros([L, N]); dirichlet_rnd = np.zeros([L, N])
for n in range(N):
    for i in range(L):
        gamma_rnd[i, n]=np.random.gamma(alpha[i], 1)
    # 逐样本归一化（对维度归一化）
    Z_d = np.sum(gamma_rnd[:, n])
    dirichlet_rnd[:, n] = gamma_rnd[:, n]/Z_d
# 绘制散点图
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(dirichlet_rnd[0, :], dirichlet_rnd[1, :], dirichlet_rnd[2, :])
ax.view_init(30, 60)
#'''