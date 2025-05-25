import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock
import numpy as np
from torch.utils.data import SubsetRandomSampler
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import time
import random
from math import exp
from copy import deepcopy
import ray
import argparse
from torchsummary import summary
from tensorboardX import SummaryWriter
from dirichlet_data import data_from_dirichlet
from torch.autograd import Variable
from sam import SAM
#from torchvision.models import vgg11
from torchvision.models import vgg11
os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"


# 加入存档，log

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=1, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=16, type=int, help='# batch_size')
parser.add_argument('--E', default=1, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedAvg', type=str, help='alg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.998', type=float, help='lr_decay')
parser.add_argument('--data_name', default='CIFAR100', type=str, help='imagenet,CIFAR100')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
#parser.add_argument('--lr_ps', default='0.15', type=float, help='only for FedAdam ')

parser.add_argument('--lr_ps', default='1', type=float, help='only for FedAdam ')

parser.add_argument('--alpha_value', default='0.1', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.1', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=0.01, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='lenet5', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.85, type=float, help=' for mom_step')
parser.add_argument('--p', default=10, type=float, help=' for mom_step')
parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--freeze-layers', type=bool, default=False)

parser.add_argument('--datapath', type=str, default="./data")
parser.add_argument('--num_gpus_per', default=1, type=float, help=' for mom_step')
parser.add_argument('--normalization', default='BN', type=str, help=' for mom_step')
parser.add_argument('--pre', default=1, type=int, help=' for mom_step')
parser.add_argument('--print', default=1, type=int, help=' for mom_step')

args = parser.parse_args()
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
print(torch.cuda.is_available())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)
num_gpus_per = args.num_gpus_per# num_gpus_per = 0.16

num_gpus = len(gpu_idx.split(','))
#num_gpus_per = 1
data_name = args.data_name
CNN=args.CNN
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
'''
if args.data_name == 'CIFAR10':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
if args.data_name == 'CIFAR100':

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                     std=[0.2675, 0.2565, 0.2761])])
'''
if data_name == 'imagenet':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
import datasets as local_datasets

if data_name == 'imagenet':
    train_dataset = local_datasets.TinyImageNetDataset(
        root=os.path.join(args.datapath, 'tiny-imagenet-200'),
        split='train',
        transform=transform_train
    )

if data_name == 'CIFAR10':

    train_dataset = datasets.CIFAR10(
        "./data",
        train=True,
        download=False,
        transform=transform_train)
elif data_name == 'EMNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        split='byclass',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )

elif data_name == 'CIFAR100':
    train_dataset = datasets.cifar.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform_train
         )
elif data_name == 'MNIST':
    train_dataset = datasets.EMNIST(
        "./data",
        #split='mnist',
        split='balanced',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,)),
        ])
    )


def get_data_loader(pid, data_idx, batch_size, data_name):
    """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
    sample_chosed = data_idx[pid]
    train_sampler = SubsetRandomSampler(sample_chosed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,num_workers=0)
    return train_loader


def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
    ])


    if data_name  == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='test',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_test)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)
    elif data_name == 'EMNIST':
        test_dataset = datasets.EMNIST("./data", split='byclass', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test 
        #transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    elif data_name == 'MNIST':
        #test_dataset = datasets.EMNIST("./data",split='mnist', train=False, transform=transforms.Compose([
        test_dataset = datasets.EMNIST("./data", split='balanced', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader
    
def get_data_loader_train(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ])
    if data_name  == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=True, transform=transform_test)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)
    elif data_name == 'EMNIST':
        test_dataset = datasets.EMNIST("./data", split='byclass', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=True, transform=transform_test 
        #transforms.Compose([
        #    transforms.ToTensor(),
        #    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        )
    elif data_name == 'MNIST':
        #test_dataset = datasets.EMNIST("./data",split='mnist', train=False, transform=transforms.Compose([
        test_dataset = datasets.EMNIST("./data", split='balanced', train=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3248,))]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader

if data_name  == 'imagenet':
    def evaluate(model, test_loader,train_loader):
        """Evaluates the accuracy of the model on a validation dataset."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target =target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        return 100. * correct / total,torch.tensor(0),torch.tensor(0)
else:
    def evaluate(model, test_loader,train_loader):
        """Evaluates the accuracy of the model on a validation dataset."""
        criterion = nn.CrossEntropyLoss()
        model.eval()
        correct = 0
        total = 0
        test_loss=0
        train_loss=0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                test_loss+= criterion(outputs, target)

            for batch_idx, (data, target) in enumerate(train_loader):
                data_train = data.to(device)
                target_train = target.to(device)
                outputs_train = model(data_train)
                train_loss+= criterion(outputs_train, target_train)
        return 100. * correct / total,test_loss/ len(test_loader),train_loss/ len(train_loader)


class ConvNet_EMNIST(nn.Module):
    """TF Tutorial for EMNIST."""
    def __init__(self):
        super(ConvNet_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size=3)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3)
        self.fc1 = nn.Linear(9216,128)
        self.fc2 = nn.Linear(128,62)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class SCAFNET(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(SCAFNET, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 62)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

#resnet18
class shakes_LSTM(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(shakes_LSTM, self).__init__()
        embedding_dim = 8
        hidden_size = 100
        num_LSTM = 2
        input_length = 80
        self.n_cls = 80

        self.embedding = nn.Embedding(input_length, embedding_dim)
        self.stacked_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_LSTM)
        self.fc = nn.Linear(hidden_size, self.n_cls)
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # lstm accepts in this style
        output, (h_, c_) = self.stacked_LSTM(x)
        # Choose last hidden layer
        last_hidden = output[-1, :, :]
        x = self.fc(last_hidden)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


import torch.nn as nn
import torchvision.models as models

#'''
#resnet18
class ResNetpre(nn.Module):
  def __init__(self, num_classes=10, l2_norm=False):
    super(ResNetpre, self).__init__()
    self.l2_norm = l2_norm
    self.in_planes = 64

    if args.pre==1:

      #resnet18=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
      resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
      resnet18 = models.resnet18()
    resnet18.fc = nn.Linear(512,  num_classes)
    #self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
    #                           stride=2, padding=1, bias=False)
    #self.bn1 = nn.GroupNorm(2, 64)
    #resnet18.conv1 = self.conv1
    #resnet18.bn1 = self.bn1

    # Change BN to GN
    if args.normalization=='GN':
      resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

      resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
      resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
      resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
      resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

      resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

      resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

      resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

    #assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
    self.model = resnet18

  def forward(self, x):
    x = self.model(x)
    return x
  def get_weights(self):
    return {k: v.cpu() for k, v in self.state_dict().items()}

  def set_weights(self, weights):
    self.load_state_dict(weights)

  def get_gradients(self):
    grads = []
    for p in self.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads

  def set_gradients(self, gradients):
      for g, p in zip(gradients, self.parameters()):
          if g is not None:
              p.grad = torch.from_numpy(g)
#'''


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
#'''

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False):
        super(ResNet, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2, 64)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        if l2_norm:
            self.linear = nn.Linear(512 * block.expansion, num_classes, bias=False)
        else:
            self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = F.avg_pool2d(out, 4)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        if self.l2_norm:
            #with torch.no_grad():
                #w = self.linear.weight.data.clone()
                #w = F.normalize(w, dim=1, p=2)
                #self.linear.weight.copy_(w)
            #self.linear = F.normalize(self.linear)
            self.linear.weight.data = F.normalize(self.linear.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
            logit = self.linear(out)
        else:
            logit = self.linear(out)
            
        if return_feature==True:
            return out, logit
        else:
            return logit
        
    
    def forward_classifier(self,x):
        logit = self.linear(x)
        return logit        


    def sync_online_and_global(self):
        state_dict=self.state_dict()
        for key in state_dict:
            if 'global' in key:
                x=(key.split("_global"))
                online=(x[0]+x[1])
                state_dict[key]=state_dict[online]
        self.load_state_dict(state_dict)
        

    def get_weights(self):
      return {k: v.cpu() for k, v in self.state_dict().items()}
  
    def set_weights(self, weights):
      self.load_state_dict(weights)
  
    def get_gradients(self):
      grads = []
      for p in self.parameters():
          grad = None if p.grad is None else p.grad.data.cpu().numpy()
          grads.append(grad)
      return grads
  
    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
#'''

import torch.nn as nn
import torchvision.models as models
'''
#resnet18
class ResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10, l2_norm=False):
    super(ResNet, self).__init__()
    self.l2_norm = l2_norm
    self.in_planes = 64
    
    if args.pre==1:
      resnet18=models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
      resnet18 = models.resnet18()
    resnet18.fc = nn.Linear(512,  num_classes)
    #self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
    #                           stride=2, padding=1, bias=False)
    #self.bn1 = nn.GroupNorm(2, 64)
    #resnet18.conv1 = self.conv1
    #resnet18.bn1 = self.bn1
  
    # Change BN to GN 
    if args.normalization=='GN':
      resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
  
      resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
      resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
      resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
      resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
  
      resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
      resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
  
      resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
      resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
  
      resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
      resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
    
    #assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'            
    self.model = resnet18

  def forward(self, x):
    x = self.model(x)
    return x
  def get_weights(self):
    return {k: v.cpu() for k, v in self.state_dict().items()}

  def set_weights(self, weights):
    self.load_state_dict(weights)

  def get_gradients(self):
    grads = []
    for p in self.parameters():
        grad = None if p.grad is None else p.grad.data.cpu().numpy()
        grads.append(grad)
    return grads

  def set_gradients(self, gradients):
      for g, p in zip(gradients, self.parameters()):
          if g is not None:
              p.grad = torch.from_numpy(g)
'''

def blockVGG(covLayerNum,inputChannel, outputChannel, kernelSize, withFinalCov1:bool):
    layer = nn.Sequential()
    layer.add_module('conv2D1',nn.Conv2d(inputChannel, outputChannel,kernelSize,padding=1))
    layer.add_module('relu-1',nn.ReLU())
    for i in range(covLayerNum - 1):
        layer.add_module('conv2D{}'.format(i),nn.Conv2d(outputChannel, outputChannel,kernelSize,padding=1))
        layer.add_module('relu{}'.format(i),nn.ReLU())
    if withFinalCov1:
        layer.add_module('Conv2dOne',nn.Conv2d(outputChannel,outputChannel, 1))
    layer.add_module('max-pool',nn.MaxPool2d(2,2))
    return layer

# 定义网络结构
#VGG11
#'''
class VGG11_10(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = blockVGG(1,3,64,3,False)
        
        self.layer2 = blockVGG(1,64,128,3,False)
        
        self.layer3 = blockVGG(2,128,256,3,False)
        
        self.layer4 = blockVGG(2,256,512,3,False)
        
        self.layer5 = blockVGG(2,512,512,3,False)
        self.layer6 = nn.Sequential(
            nn.Linear(512,100),
            nn.ReLU(),
            nn.Linear(100,10),
            # nn.ReLU(),
            # nn.Softmax(1)
        )
    def forward(self,x:torch.Tensor):
        x = self.layer1(x) # 执行卷积神经网络部分
        x = self.layer2(x) # 执行全连接部分
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.layer6(x)
        return F.log_softmax(x, dim=1)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class VGG11_100(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = blockVGG(1,3,64,3,False)
        
        self.layer2 = blockVGG(1,64,128,3,False)
        
        self.layer3 = blockVGG(2,128,256,3,False)
        
        self.layer4 = blockVGG(2,256,512,3,False)
        
        self.layer5 = blockVGG(2,512,512,3,False)
        self.layer6 = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,100),
            # nn.ReLU(),
            # nn.Softmax(1)
        )
    def forward(self,x:torch.Tensor):
        x = self.layer1(x) # 执行卷积神经网络部分
        x = self.layer2(x) # 执行全连接部分
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.shape[0], -1)
        x = self.layer6(x)
        return F.log_softmax(x, dim=1)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
#'''

'''
class VGG11_10(nn.Module):
    def __init__(self):
        super().__init__()
        vgg11 = models.vgg11()
        vgg11.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        vgg11.classifier = nn.Sequential(
            nn.Linear(512 , 512),  # 减少神经元数量
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),  # 减少神经元数量
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 10),  # CIFAR-10有10个类
        )
        self.model = vgg11


    def forward(self, x):
        x = self.model(x)
        return x
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)



class VGG11_100(nn.Module):

    def __init__(self):
        super().__init__()

        vgg11 = models.vgg11()
        vgg11.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        vgg11.classifier = nn.Sequential(
            nn.Linear(512 , 512),  # 减少神经元数量
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),  # 减少神经元数量
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 100),  # CIFAR-10有10个类
        )
        self.model = vgg11

    def forward(self, x):
        x = self.model(x)
        return x
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
                
'''
#Lenet5
class Lenet5_10(nn.Module):
    """TF Tutorial for CIFAR."""
    
    def __init__(self):
        super(Lenet5_10, self).__init__()   
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384) 
        self.fc2 = nn.Linear(384, 192) 
        self.fc3 = nn.Linear(192, self.n_cls)
    def forward(self, x):      
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

class Lenet5_100(nn.Module):
    
    def __init__(self):
        super(Lenet5_100, self).__init__()    
        self.n_cls = 100
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384) 
        self.fc2 = nn.Linear(384, 192) 
        self.fc3 = nn.Linear(192, self.n_cls)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
                
                
from torch import nn
import math
class VGG(nn.Module):
    def __init__(self, features, num_classes=100, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Linear(512 * 2 * 2, 4096),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes)
            nn.Linear(512, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
            else:
                layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)
cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

if CNN == 'vgg19':
  def ConvNet100():
      model = VGG(make_layers(cfg['E']))
      return model
  def ConvNet():
      model = VGG(make_layers(cfg['E']),num_classes=10)
      return model  

#'''               
if CNN == 'resnet18':
    def ConvNet(num_classes=10, l2_norm=False):
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10, l2_norm=l2_norm)
    def ConvNet100(num_classes=100, l2_norm=False):
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=100, l2_norm=l2_norm)
    def ConvNet200(num_classes=200, l2_norm=False):
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200, l2_norm=l2_norm)
#''' 

#'''
if CNN == 'resnet18pre':
    def ConvNet(num_classes=10):
        return ResNetpre( num_classes=10)
    def ConvNet100(num_classes=100):
        return ResNetpre( num_classes=100)

    def ConvNet200(num_classes=200):
        return ResNetpre( num_classes=200)
#'''
if CNN == 'vgg11':
    def ConvNet():
        return VGG11_10()
    def ConvNet100():
        return VGG11_100() 
   
if CNN == 'lenet5':  
    def ConvNet():
        return Lenet5_10()
    def ConvNet100():
        return Lenet5_100()

class ConvNet_EMNIST(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(ConvNet_EMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.dropout1(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNet_MNIST(nn.Module):
    """TF Tutorial for EMNIST."""

    def __init__(self):
        super(ConvNet_MNIST, self).__init__()
        self.fc1 = nn.Linear(784, 47)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
            

@ray.remote
class ParameterServer(object):
    def __init__(self, lr, alg, tau, selection, data_name,num_workers):
        if data_name == 'CIFAR10':
            self.model = ConvNet()
        elif data_name == 'EMNIST':
            self.model = SCAFNET()
            #self.model = ConvNet_EMNIST()
        elif data_name == 'CIFAR100':
            self.model = ConvNet100()
        elif data_name == 'MNIST':
            self.model = ConvNet_MNIST()
        if data_name == 'imagenet':
            self.model = ConvNet200()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.momen_v = None
        #self.gamma = 0.9
        self.gamma=args.gamma
        #self.gamma = 0.9
        self.beta = 0.99  # 论文是0.99
        self.alg = alg
        self.num_workers = num_workers
  
        self.lr_ps = lr
        self.lg = 1.0
        self.ps_c = None
        self.c_all = None
        # 上一代的c
        self.c_all_pre = None
        self.tau = tau
        self.selection = selection
        self.cnt = 0
        self.alpha = args.alpha
        self.h = {}
        self.momen_m={}

    def attention(self, c_cur, c_pre):
        '''
        手动softmax
        Q: 当前机子，当前轮的信息
        K: 当前机子，上一轮的信息
        '''
        produces = []
        for cur, pre in zip(c_cur, c_pre):
            # 对于每个机子
            total_produce = 0
            for k, v in cur.items():
                # 对于该机子的每一层
                Q = F.normalize(v.view(1, -1))
                K = F.normalize(pre[k].view(1, -1))
                # 出问题，重来
                layer_produce = exp(Q.mm(K.t()).item())
                total_produce += layer_produce
            produces.append(total_produce)
        sum_all = sum(produces)
        alpha = [p / sum_all for p in produces]
        # print('~~~~~',alpha)
        return alpha

    def attention_api(self, c_cur, c_pre):
        '''
        api softmax
        Q: 当前机子，当前轮的信息
        K: 当前机子，上一轮的信息
        '''
        produces = []
        for cur, pre in zip(c_cur, c_pre):
            # 对于每个机子
            total_produce = 0
            for k, v in cur.items():
                # 对于该机子的每一层
                Q = F.normalize(v.view(1, -1))
                K = F.normalize(pre[k].view(1, -1))
                layer_produce = Q.mm(K.t()).item()
                total_produce += layer_produce
            produces.append(total_produce)
        alpha = torch.tensor(produces)
        alpha = F.softmax(alpha, dim=0).tolist()
        # print('~~~~~',alpha)
        return alpha

    def attention_global(self, c_cur):
        '''
        手动softmax
        Q: 全局的（平均的），当前轮的信息
        K: 当前机子，当前轮的信息
        '''
        produces = []
        # 获取全局平均信息
        global_c = self.ps_c
        for cur in c_cur:
            # 对于每个机子
            total_produce = 0
            for k, v in cur.items():
                # 对于该机子的每一层
                Q = F.normalize(v.view(1, -1))
                K = F.normalize(global_c[k].view(1, -1))
                layer_produce = exp(Q.mm(K.t()).item())
                total_produce += layer_produce
            produces.append(total_produce)
        sum_all = sum(produces)
        alpha = [p / sum_all for p in produces]
        # print('~~~~~',alpha)
        return alpha

    def set_pre_c(self, c):
        self.c_all_pre = c


    def apply_weights_avg(self, num_workers, *weights):
        '''
        weights: delta_w
        '''
        clint_weight = {}
        div=0
        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        global_weights={}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] +=  v/(num_workers*self.selection)
                else:
                    sum_weights[k] = v/(num_workers*self.selection)
        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] = ps_w[k] + args.lr_ps*sum_weights[k]
        self.model.set_weights(global_weights)
        if args.print==1:
            for weight in weights:
                denom=0
                divergence=0
                for k, v in weight.items():
                    clint_weight[k] = ps_w[k] + v
                    divergence += ((clint_weight[k] - global_weights[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                div = div + divergence / (denom * (num_workers * self.selection))
            cosinesimilarity=0
            for weight in weights:
                a_dot_b=0
                a_norm=0
                b_norm=0
                for k, v in weight.items():
                        #clint_weight[k] = ps_w[k] + v
                        a = sum_weights[k]
                        b = v
                        a_dot_b += (a * b).sum()
                        a_norm += (a * a).sum()
                        b_norm += (b * b).sum()
                cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
            return self.model.get_weights(),div.detach().numpy(),cosinesimilarity.detach().numpy()
        else:
            return self.model.get_weights()

    def calculate_divergence(self, num_workers, *weights):
        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        global_weights={}
        divergence = 0
        denom = 0
        div=0
        clint_weight = {}  # delta_w : sum_weights
        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        global_weights={}
        for weight in weights:
            for k, v in weight.items():
                    sum_weights[k] = v/(num_workers*self.selection)
        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] = ps_w[k] + sum_weights[k]
        for weight in weights:
            for k, v in weight.items():
                    clint_weight[k] = ps_w[k] + v
                    divergence += ((clint_weight[k] - global_weights[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                    div = div + divergence / (denom * (num_workers * self.selection))

        return div.detach().numpy()

    def calculate_cosinesimilarity(self, num_workers, *weights):
        sum_weights = {}  # delta_w : sum_weights
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] +=  v/(num_workers*self.selection)
                else:
                    sum_weights[k] = v/(num_workers*self.selection)
        a_dot_b = 0
        a_norm = 0
        b_norm = 0
        cosinesimilarity=0
        for weight in weights:
            for k, v in weight.items():
                    #clint_weight[k] = ps_w[k] + v
                    a = sum_weights[k]
                    b = v
                    a_dot_b += (a * b).sum()
                    a_norm += (a * a).sum()
                    b_norm += (b * b).sum()
            cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
        return cosinesimilarity.detach().numpy()

        #return self.model.get_weights(),div.detach().numpy()



    def apply_weights_moment(self, num_workers, *weights):
        #self.gamma=0.9
        sum_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += v / (num_workers*self.selection)
                else:
                    sum_weights[k] = v / (num_workers*self.selection)
        weight_ps = self.model.get_weights()
        if not self.momen_v:
            self.momen_v = deepcopy(sum_weights)
        else:
            for k, v in self.momen_v.items():
                self.momen_v[k] = self.gamma * v + sum_weights[k]
        seted_weight = {}
        for k, v in weight_ps.items():
            seted_weight[k] = v + args.lr_ps*self.momen_v[k]
        self.model.set_weights(seted_weight)

        if args.print==1:
            div=0
            clint_weight={}
            for weight in weights:
                denom=0
                divergence=0
                for k, v in weight.items():
                    clint_weight[k] = weight_ps[k] + v
                    divergence += ((clint_weight[k] - seted_weight[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                div = div + divergence / (denom * (num_workers * self.selection))
            cosinesimilarity=0
            for weight in weights:
                a_dot_b=0
                a_norm=0
                b_norm=0
                for k, v in weight.items():
                        a = self.momen_v[k]
                        b = v
                        a_dot_b += (a * b).sum()
                        a_norm += (a * a).sum()
                        b_norm += (b * b).sum()
                cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
            return self.model.get_weights(),div.detach().numpy(),cosinesimilarity.detach().numpy()
        else:
            return self.model.get_weights()

    def apply_weights_adam(self, num_workers, *weights):
        '''
        收到的是Δweight，
        '''
        delta_t = {}
        for weight in weights:
            for k, v in weight.items():
                if k in delta_t.keys():
                    delta_t[k] += v / (num_workers * selection)
                else:
                    delta_t[k] = v / (num_workers * selection)
        weight_ps = self.model.get_weights()
        if not self.momen_v:
            for k, v in delta_t.items():
                #self.momen_m[k]=0.1*delta_t[k]
                self.momen_m[k]=delta_t[k]
                
        else:
            for k, v in delta_t.items():
                self.momen_m[k]=0.9*self.momen_m[k]+0.1*delta_t[k]
                #self.momen_m[k]=0.9*self.momen_m[k]+delta_t[k]
          
        if not self.momen_v:
            self.momen_v = deepcopy(delta_t)
            for k, v in delta_t.items():
                # adam
                #self.momen_v[k] = self.beta * 0.1 + (1 - self.beta) * v.mul(v)
                # adagrad
                 self.momen_v[k] = v.mul(v)
        else:
            for k, v in self.momen_v.items():
                # adam
                self.momen_v[k] = self.beta * v + (1 - self.beta) * delta_t[k].mul(delta_t[k])
                # adagrad
                # self.momen_v[k]=v +  delta_t[k].mul(delta_t[k])

        seted_weight = {}
        for k, v in weight_ps.items():
            # if k=='fc1.weight':
            # print('delat', delta_t[k] / (self.momen_v[k].sqrt()+ tau))
            seted_weight[k] = v +0.15* self.momen_m[k] / (self.momen_v[k].sqrt() + self.tau)

        self.model.set_weights(seted_weight)
        return self.model.get_weights()

    def apply_weights_atte(self, num_workers, *weights):
        '''

        '''
        ps_w = self.model.get_weights()  # w : ps_w
        global_weights={}
        if not self.c_all_pre:
            alpha = [1 / num_workers] * num_workers
        else:
            alpha = self.attention(self.c_all, self.c_all_pre)
        sum_weights = {}
        for weight, cur_alpha in zip(weights, alpha):
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += v * cur_alpha
                else:
                    sum_weights[k] = v * cur_alpha
        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] = ps_w[k] + sum_weights[k]
        self.model.set_weights(global_weights)
        if args.print==1:
            clint_weight={}
            global_weights={}
            for weight in weights:
                denom=0
                divergence=0
                for k, v in weight.items():
                    clint_weight[k] = ps_w[k] + v
                    divergence += ((clint_weight[k] - global_weights[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                div = div + divergence / (denom * (num_workers * self.selection))
            cosinesimilarity=0
            for weight in weights:
                a_dot_b=0
                a_norm=0
                b_norm=0
                for k, v in weight.items():
                        #clint_weight[k] = ps_w[k] + v
                        a = sum_weights[k]
                        b = v
                        a_dot_b += (a * b).sum()
                        a_norm += (a * a).sum()
                        b_norm += (b * b).sum()
                cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
            return self.model.get_weights(),div.detach().numpy(),cosinesimilarity.detach().numpy()
        return self.model.get_weights()

    def apply_weights_global_atte(self, num_workers, *weights):
        '''
        weights: delta_wi
        '''
        ps_w = self.model.get_weights()  # w : ps_w
        global_weights={}
        if not self.c_all:
            alpha = [1 / num_workers] * num_workers
        else:
            alpha = self.attention_global(self.c_all)
            self.cnt += 1
            if self.cnt % 50 == 0:
                self.alpha = alpha
        sum_weights = {}
        for weight, cur_alpha in zip(weights, alpha):
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += v * cur_alpha
                else:
                    sum_weights[k] = v * cur_alpha

        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] = ps_w[k] + sum_weights[k]
        self.model.set_weights(global_weights)

        if args.print==1:
            clint_weight={}
            global_weights={}
            for weight in weights:
                denom=0
                divergence=0
                for k, v in weight.items():
                    clint_weight[k] = ps_w[k] + v
                    divergence += ((clint_weight[k] - global_weights[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                div = div + divergence / (denom * (num_workers * self.selection))
            cosinesimilarity=0
            for weight in weights:
                a_dot_b=0
                a_norm=0
                b_norm=0
                for k, v in weight.items():
                        #clint_weight[k] = ps_w[k] + v
                        a = sum_weights[k]
                        b = v
                        a_dot_b += (a * b).sum()
                        a_norm += (a * a).sum()
                        b_norm += (b * b).sum()
                cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
            return self.model.get_weights(),div.detach().numpy(),cosinesimilarity.detach().numpy()
        return self.model.get_weights()

    def produce(self, cur, pres):
        '''
        作用：   对于机子j，输出它由所有机子的状态生成自己的表示的权重
        输入    cur: 当前要处理的某一个机子
                pres: 上一代所有机子的更新量(cur所依赖的信息)
        '''
        e_list = []
        for pre in pres:
            sum_e = 0
            for k, v in cur.items():
                cur_i, pre_i = v.view(1, -1), pre[k].view(1, -1)
                cur_i, pre_i = F.normalize(cur_i), F.normalize(pre_i)
                e = exp(cur_i.mm(pre_i.t()).item())
                sum_e += e
            e_list.append(sum_e)
        sum_all = sum(e_list)
        prod = [e / sum_all for e in e_list]
        return prod

    def attention_compex(self):
        '''

        输入     c_cur:所有机子发来的更新量
                c_pre:上一代所有机子的更新量
        输出    每个机子的表示组成的列表
        '''
        c_cur = self.c_all
        c_pre = self.c_all_pre
        attention_list = [self.produce(cur, c_pre) for cur in c_cur]
        # print('(attention_list):',attention_list[0])
        return attention_list

    def self_attention(self):
        c_cur = self.c_all
        if not self.c_all:
            return torch.eye(num_workers).tolist()
        attention = [self.produce(cur, c_cur) for cur in c_cur]
        # print('(attention):',attention[0])
        return attention

    def apply_weights_mutilayer(self, num_workers, *weights):
        '''
        双层的注意力
        '''
        if not self.c_all_pre:
            alpha = torch.eye(num_workers).tolist()
        else:
            alpha = self.attention_compex()

        sum_weights_list = []
        for cur_alpha in alpha:
            sum_weights = {}
            for weight, cur_alpha_j in zip(weights, cur_alpha):
                for k, v in weight.items():
                    if k in sum_weights.keys():
                        sum_weights[k] += v * cur_alpha_j
                    else:
                        sum_weights[k] = v * cur_alpha_j
            sum_weights_list.append(sum_weights)
        new_weights = {}
        for weight in sum_weights_list:
            for k, v in weight.items():
                if k in new_weights.keys():
                    new_weights[k] += v / num_workers
                else:
                    new_weights[k] = v / num_workers
        self.model.set_weights(new_weights)
        return self.model.get_weights()

    def apply_weights_selfatte(self, num_workers, *weights):
        ps_w = self.model.get_weights()  # w : ps_w
        alpha = self.self_attention()
        sum_weights_list = []
        for cur_alpha in alpha:
            sum_weights = {}
            for weight, cur_alpha_j in zip(weights, cur_alpha):
                for k, v in weight.items():
                    if k in sum_weights.keys():
                        sum_weights[k] += v * cur_alpha_j
                    else:
                        sum_weights[k] = v * cur_alpha_j
            sum_weights_list.append(sum_weights)
        new_weights = {}
        for weight in sum_weights_list:
            for k, v in weight.items():
                if k in new_weights.keys():
                    new_weights[k] += v / num_workers
                else:
                    new_weights[k] = v / num_workers
        for k, v in new_weights.items():  # w = w + delta_w
            new_weights[k] = ps_w[k] + new_weights[k]

        self.model.set_weights(new_weights)
        if args.print==1:
            clint_weight={}
            global_weights={}
            global_weights =new_weights
            for weight in weights:
                denom=0
                divergence=0
                for k, v in weight.items():
                    clint_weight[k] = ps_w[k] + v
                    divergence += ((clint_weight[k] - global_weights[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                div = div + divergence / (denom * (num_workers * self.selection))
            cosinesimilarity=0
            for weight in weights:
                a_dot_b=0
                a_norm=0
                b_norm=0
                for k, v in weight.items():
                        #clint_weight[k] = ps_w[k] + v
                        a = sum_weights[k]
                        b = v
                        a_dot_b += (a * b).sum()
                        a_norm += (a * a).sum()
                        b_norm += (b * b).sum()
                cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
            return self.model.get_weights(),div.detach().numpy(),cosinesimilarity.detach().numpy()
        return self.model.get_weights()

    def apply_weights_avgSWA(self, num_workers, *weights):
        '''
        weights: delta_w
        '''
        #self.alpha=self.alpha
        global_weights={}
        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] += 1 / (self.num_workers * self.selection) * v
                    # sum_weights[k]+=v / num_workers
                else:
                    sum_weights[k] = 1 / (self.num_workers * self.selection) * v
        
        for k, v in sum_weights.items():  # w = w + delta_w
            global_weights[k] =ps_w[k]+sum_weights[k]*self.alpha
            #sum_weights[k] =self.alpha*ps_w[k]+ (ps_w[k] + sum_weights[k])*(1-self.alpha)
        self.model.set_weights(global_weights)
        if args.print==1:
            div=0
            clint_weight={}
            for weight in weights:
                denom=0
                divergence=0
                for k, v in weight.items():
                    clint_weight[k] = ps_w[k] + v
                    divergence += ((clint_weight[k] - global_weights[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                div = div + divergence / (denom * (num_workers * self.selection))
            cosinesimilarity=0
            for weight in weights:
                a_dot_b=0
                a_norm=0
                b_norm=0
                for k, v in weight.items():
                        #clint_weight[k] = ps_w[k] + v
                        a = sum_weights[k]
                        b = v
                        a_dot_b += (a * b).sum()
                        a_norm += (a * a).sum()
                        b_norm += (b * b).sum()
                cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
            return self.model.get_weights(),div.detach().numpy(),cosinesimilarity.detach().numpy()
        else:
            return self.model.get_weights()
        
        
    def apply_weights_avgAGM(self, num_workers, weights, ps_c):
        '''
        weights: delta_w
        '''
        tao=0.2
        lamda=0.85
        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] += 1 / (self.num_workers * self.selection) * v
                else:
                    sum_weights[k] = 1 / (self.num_workers * self.selection) * v
        for k, v in sum_weights.items():  # w = w + delta_w
            sum_weights[k] = tao*(ps_w[k] + sum_weights[k])+(1-tao)*(ps_w[k]-lamda*ps_c[k])
        for k, v in sum_weights.items():  # w = w + delta_w
            ps_c[k] = -(sum_weights[k]-ps_w[k])
        for k, v in sum_weights.items():  # w = w + delta_w
            sum_weights[k] =ps_w[k]-lamda*ps_c[k]
        self.model.set_weights(sum_weights)
        return self.model.get_weights(),ps_c
        
    def apply_weights_avgACG(self, num_workers, *weights):
        self.gamma=0.85
        ps_w = self.model.get_weights()
        sum_weights = {}
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():
                    sum_weights[k] += 1 / (self.num_workers * self.selection) * v
                else:
                    sum_weights[k] = 1 / (self.num_workers * self.selection) * v
        if not self.momen_v:
            self.momen_v = deepcopy(sum_weights)
        else:
            for k, v in self.momen_v.items():
                self.momen_v[k] = self.gamma * v + sum_weights[k]
        seted_weight = {}
        for k, v in ps_w.items():
            seted_weight[k] = v + self.momen_v[k]
        self.model.set_weights(seted_weight)
        if args.print==1:
            div=0
            clint_weight={}
            global_weights =seted_weight
            for weight in weights:
                denom=0
                divergence=0
                for k, v in weight.items():
                    clint_weight[k] = ps_w[k] + v
                    divergence += ((clint_weight[k] - global_weights[k]) ** 2).sum()
                    denom += ((clint_weight[k]) ** 2).sum()
                div = div + divergence / (denom * (num_workers * self.selection))
            cosinesimilarity=0
            for weight in weights:
                a_dot_b=0
                a_norm=0
                b_norm=0
                for k, v in weight.items():
                        a = self.momen_v[k]
                        b = v
                        a_dot_b += (a * b).sum()
                        a_norm += (a * a).sum()
                        b_norm += (b * b).sum()
                cosinesimilarity =cosinesimilarity+ a_dot_b / (((a_norm) ** 0.5) * ((b_norm) ** 0.5)*num_workers*self.selection)
            return self.model.get_weights(), self.momen_v,div.detach().numpy(),cosinesimilarity.detach().numpy()

        return self.model.get_weights(), self.momen_v




    def apply_weights_FedDyn(self, num_workers, weights,h):
        '''
        weights: delta_w
        '''
        args.alpha=0.01
        alpha=args.alpha
        ps_w = self.model.get_weights()  # w : ps_w
        sum_weights = {}  # delta_w : sum_weights
        for weight in weights:
            for k, v in weight.items():
                if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_weights[k] += 1 / (self.num_workers * self.selection) * v
                else:
                    sum_weights[k] = 1 / (self.num_workers * self.selection) * v

        sum_h= {}  # delta_w : sum_weights
        for hi in h:
            for k, v in hi.items():
                if k in sum_h.keys():  # delta_w = \sum (delta_wi/#wk)
                    sum_h[k] += 1 / (self.num_workers * self.selection) * v
                else:
                    sum_h[k] = 1 / (self.num_workers * self.selection) * v
        for k, v in sum_weights.items():
            if k in self.h.keys():  # delta_w = \sum (delta_wi/#wk)
                self.h[k] = self.h[k] - alpha * sum_weights[k]
            else:
                self.h[k] = - alpha * sum_weights[k]

        for k, v in sum_weights.items():  # w = w + delta_w

            sum_weights[k] = ps_w[k] + sum_weights[k]-self.h[k]
        self.model.set_weights(sum_weights)
        return self.model.get_weights()



    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.apply_weights_avg,
            'FedMoment': self.apply_weights_moment,
            #'cddplus': self.apply_weights_avg,
            #'cdd': self.apply_weights_avg,
            'SCAFFOLD': self.apply_weights_avg,
            'SCAFFOLD+': self.apply_weights_avg,
            #'SCAFFOLDM': self.apply_weights_moment,
            'IGFL_atte': self.apply_weights_atte,
            'IGFL': self.apply_weights_avg,
            #'cdd_ci_plus': self.apply_weights_moment,
            'mutilayer-atte': self.apply_weights_mutilayer,
            'self-atte': self.apply_weights_selfatte,
            'global-atte': self.apply_weights_global_atte,
            'FedAdam': self.apply_weights_adam,
            'FedAvg_atte': self.apply_weights_global_atte,  # Option2
            'FedDyn': self.apply_weights_FedDyn,
            #'fedDyn': self.apply_weights_fedDyn,
            'only-atte-self': self.apply_weights_selfatte,
            'momentum-step': self.apply_weights_avg,
            'FedCM':self.apply_weights_avg,
            #'IGFL_prox':self.apply_weights_avg,
            'FedDC': self.apply_weights_avg,
            #'cddplus_ci':self.apply_weights_moment,
            'IGFL+':self.apply_weights_avg,
            'FedAGM':self.apply_weights_avgAGM,
            'FedSAM':self.apply_weights_avg,
            'FedSAM+':self.apply_weights_avg,
            'MoFedSAM':self.apply_weights_avg,
            'FedSAMS':self.apply_weights_avg,
            #'FedSWA':self.apply_weights_avg,
            'FedSWA':self.apply_weights_avgSWA,
            'FedSWAS':self.apply_weights_avgSWA,
            'Fedprox':self.apply_weights_avg,
            'stem':self.apply_weights_avg,
            'FedACG': self.apply_weights_avgACG,
            'SCAFM':self.apply_weights_avg,
            'Fedspeed':self.apply_weights_avg,
            'Moon':self.apply_weights_avg,
            'FedSTORM': self.apply_weights_avg,
            'FedNesterov':self.apply_weights_avg,

            

        }

    def apply_weights_func2(self, alg, num_workers, weights,ps_c):
        self.load_dict()

        return self.func_dict.get(alg, None)(num_workers, weights,ps_c)

    def apply_weights_func(self, alg, num_workers, *weights):
        self.load_dict()
        return self.func_dict.get(alg, None)(num_workers, *weights)

    def apply_ci(self, alg, num_workers, *cis):
        '''
        平均所有的ds发来的sned_ci: delta_ci
        '''
        if 'atte' in alg:
            # 先将当前状态传给self.c_all_pre, 再平均所有的ds发来的ci，更新self.ps_c
            self.set_pre_c(self.c_all)
            self.c_all = cis
        args.gamma=0.1
        sum_c = {}  # delta_c :sum_c
        for ci in cis:
            for k, v in ci.items():
                if k in sum_c.keys():
                    sum_c[k] += v / (num_workers * selection)
                else:
                    sum_c[k] = v / (num_workers * selection)
        if self.ps_c == None:
            self.ps_c = sum_c
            return self.ps_c
        for k, v in self.ps_c.items():
            if alg in {'FedSTORM','FedNesterov'}:
                self.ps_c[k] = sum_c[k]
            #if alg in {'FedCM','FedDyn','IGFL_prox','FedAGM','FedDC','IGFL'}:
            if alg in {'IGFL_prox'}:
                self.ps_c[k] = v * args.gamma +  sum_c[k]
            if alg in {'FedCM','IGFL_prox','FedAGM','IGFL','MoFedSAM','stem'}:
                self.ps_c[k] = v +  sum_c[k]
            else:
                #self.ps_c[k] = v +  sum_c[k] * selection # c = c + selection * delta_c
                #self.ps_c[k] = v +  sum_c[k] * selection
                self.ps_c[k] = v + sum_c[k] * args.gamma
        return self.ps_c

    def get_weights(self):
        return self.model.get_weights()

    def get_ps_c(self):
        return self.ps_c

    def get_state(self):
        return self.ps_c, self.c_all

    def set_state(self, c_tuple):
        self.ps_c = c_tuple[0]
        self.c_all = c_tuple[1]

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_attention(self):
        return self.alpha

@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'EMNIST':
            # self.model = SCAFNET().to(device)
            self.model = ConvNet_EMNIST().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        elif data_name == 'MNIST':
            self.model = ConvNet_MNIST().to(device)
        if data_name == 'imagenet':
            self.model = ConvNet200().to(device)



        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.pid = pid
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        #if self.lr_decay:
            #self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: pow(self.lr_decay, epoch))
        self.pre_ps_weight = None
        self.pre_loc_weight = None
        self.flag = False
        self.ci = None
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.alpha = args.alpha
        self.gamma = args.gamma

    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_id_loader(self, index):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        if not c_dict.get(index):
            return
        self.ci = c_dict[index]


        
    def state_hi_loader(self, index):
        if not hi_dict.get(index):
            return
        self.hi = hi_dict[index]

    def state_Li_loader(self, index):
        if not Li_dict.get(index):
            return
        self.Li = Li_dict[index]

    def get_train_loss(self):
        return self.loss

    def update_fedavg(self, weights, E, index,lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def update_Moon(self, weights, E, index,lr):
        self.model.set_weights(weights)
        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )

        cos = torch.nn.CosineSimilarity(dim=-1)
        mu = 0.001
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                data.requires_grad = False
                target.requires_grad = False
                target = target.long()
                w = deepcopy(self.model.get_weights())
                #_, pro1, out  = self.model(data)
                with torch.no_grad():
                    self.model.set_weights(weights)
                    pro2 = self.model(data)
                    self.model.set_weights(self.ci)
                    pro3 = self.model(data)
                self.model.set_weights(w)
                pro1 = self.model(data)
                self.model.to('cuda')
                #_, pro2, _ = self.model(data)
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                #previous_net.to('cpu')
                temperature=0.5
                logits /= temperature
                labels = torch.zeros(data.size(0)).cuda().long()
                #self.model.cuda()
                #pro1 = self.model(data)
                loss2 = mu * self.criterion(logits, labels)
                #loss1 = self.criterion(out, target)
                loss1 = self.criterion(pro1, target)
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.step()
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            # origin
            self.ci[k] = v
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def update_stem(self, weights, E, index, ps_c, lr):
        num_workers = int(self.num_workers * selection)
        self.model.set_weights(weights)
        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        # momen_v=deepcopy(zero_weight)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        w2=deepcopy(self.model.get_weights())
        old_w = deepcopy(self.model.get_weights())
        sum=deepcopy(zero_weight)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                w2=deepcopy(old_w)
                old_w = deepcopy(self.model.get_weights())
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                g1 = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():
                    g1[k] =(old_w[k]-v)/lr
                    sum[k]=sum[k]+g1[k]

                self.model.set_weights(w2)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                g2 = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():
                    g2[k] =(w2[k]-v)/lr
                #self.model.set_weights(w2)
                new_weights = deepcopy(self.model.get_weights())
                
                for k, v in new_weights.items():
                    new_weights[k] = old_w[k] - g1[k]* lr+(1 - self.alpha) * (ps_c[k] - g2[k])* lr
                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送
        for k, v in self.model.get_weights().items():
            # origin
            self.ci[k] = sum[k] / (E * len(self.data_iterator))
        self.loss = loss.item()
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            send_ci[k] = - ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        return delta_w, send_ci    

    def update_Fedprox(self, weights, E, index,lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
                #lr=lr*0.95
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                ## Weight L2 loss
                reg_loss = 0
                loss_cg = 0
                alpha = args.alpha=0.01
                for n, p in model.named_parameters():
                    weights[n]=weights[n].to(device)
                    L1=alpha / 2 * torch.sum((p -weights[n]) * (p - weights[n]))
                    #reg_loss += L1.item()
                    #reg_loss=0
                    #L2=
                    loss_cg +=L1.item()
                loss = ce_loss + loss_cg
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w
        
    def update_FedSWA(self, weights, E, index,lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        a1=lr
        a2=args.rho*lr
        i=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                i=i+1
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
                #lr=lr*0.98
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                lr=(1-i/(len(self.data_iterator)*E))*a1+(i/(len(self.data_iterator)*E))*a2
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w
        # return self.model.get_weights()
        
    def update_FedSWAS(self, weights, E, index, ps_c,lr):
        '''
        返回：send_ci: tilda_ci
        原版
        '''
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        #zero_weight = deepcopy(self.model.get_weights())
        #for k, v in zero_weight.items():
        #    zero_weight[k] = zero_weight[k] - zero_weight[k]
        #sum=deepcopy(zero_weight)
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        a1=lr
        a2=args.rho*lr
        i=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
                i=i+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g                           
                new_weights = deepcopy(self.model.get_weights())
                # w = w  + tilda_w + c - ps_c
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] +ps_c[k])*lr  # y_i = y_i -lr*(-ci + c)
                    #new_weights[k] = v - (-self.ci[k])
                    
                self.model.set_weights(new_weights)
                #lr=(1-i/50)*a1+i/50*a2
                lr=(1-i/(len(self.data_iterator)*E))*a1+(i/(len(self.data_iterator)*E))*a2
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        lr=(a1+a2)/2
        send_ci = deepcopy(self.model.get_weights())
        #ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +self.ci[k]-ps_c[k]
            #self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator) * lr)
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            #send_ci[k] = -ci[k] + self.ci[k]
            send_ci[k] = -ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        # return self.model.get_weights(), send_ci                                #返回
        return delta_w, send_ci

    def update_fedDyn(self, weights, E, index, lr):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)
        #num_workers = int(self.num_workers * selection)
        self.model.set_weights(weights)
        #fixed_params = {n: p for n, p in self.model.named_parameters()}
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]

        if self.hi == None:
            self.hi = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.state_hi_loader(index)
        alpha=args.alpha=0.01
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                ## Weight L2 loss
                reg_loss = 0
                lg_loss=0
                for n, p in model.named_parameters():
                    self.hi[n] = self.hi[n].to(device)
                    weights[n]=weights[n].to(device)
                    lossh=torch.sum((p*(self.hi[n]-weights[n])))
                    l1=torch.sum((p-weights[n])**2)
                    lg_loss +=l1.item()
                    reg_loss += lossh.item()
                loss = ce_loss+0.5*alpha*reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        for k, v in self.model.get_weights().items():
            self.hi[k] = self.hi[k].to('cpu')
            weights[k]=weights[k].to('cpu')
            #self.ci[k] = 1 / (E * len(self.data_iterator)) * (v - weights[k])
        for k, v in self.model.get_weights().items():
            #self.hi[k] = self.hi[k] +alpha*(v - weights[k])
            self.hi[k] = self.hi[k] +  (v - weights[k])
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        hi_copy = deepcopy(self.hi)
        hi_dict[index] = hi_copy
        return delta_w,hi_copy

    def update_SAM(self, weights, E, index,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0,rho=0.05)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                #loss_function(output,model(input)).backward() 
                self.criterion(self.model(data), target).backward()
                self.optimizer.second_step(zero_grad=True)

                
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def update_Fedspeed(self, weights, E, index, lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        # 进入循环体之前，先装载数据集，以及状态
        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        self.data_id_loader(index)
        self.state_id_loader(index)
        rho = 0.1
        lamda=10
        alpha=1
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0, rho= rho)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                old_weights = deepcopy(self.model.get_weights())
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # loss_function(output,model(input)).backward()
                self.criterion(self.model(data), target).backward()
                self.optimizer.second_step(zero_grad=True)
                new_weights = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v +self.ci[k]*lr-(old_weights[k]-weights[k])*lr/lamda
                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        for k, v in self.model.get_weights().items():
            self.ci[k] = self.ci[k]- (v - weights[k])/lamda
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] =(v-lamda*self.ci[k])- weights[k]
        return delta_w

    def update_MoFedSAM(self, weights, E, index, ps_c,lr):
        num_workers = int(self.num_workers * selection)
        self.model.set_weights(weights)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        if self.ci == None:

            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        base_optimizer = torch.optim.SGD
        self.gamma=0.9
        #momen_v=deepcopy(zero_weight)
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr*(1-self.gamma), momentum=0,rho=0.1)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                #old_w = deepcopy(self.model.get_weights())
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                self.optimizer.second_step(zero_grad=True)
                new_weights = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():                
                    new_weights[k] =new_weights[k]- self.gamma *ps_c[k]*lr
                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送
        for k, v in self.model.get_weights().items():        
            self.ci[k] = -1 / (E * len(self.data_iterator)*lr) * (v - weights[k])
        self.loss = loss.item()
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            send_ci[k] = - ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        return delta_w, send_ci
    def update_SAMS(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        #sum=deepcopy(zero_weight)
        
        for e in range(int(E)-1):
            #if e<int(E/2+1):
                #print(e)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g                           
                new_weights = deepcopy(self.model.get_weights())
                # w = w  + tilda_w + c - ps_c
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] +ps_c[k])  # y_i = y_i -lr*(-ci + c)
                    #new_weights[k] = v - (-self.ci[k])
                    
                self.model.set_weights(new_weights)
        send_ci = deepcopy(self.model.get_weights())
        ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            self.ci[k] =( weights[k]-v) / ((int(E)-1) * len(self.data_iterator)) +self.ci[k]-ps_c[k] 
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ps_c[k] + self.ci[k]
            
        for e in range(int(E)-1,E):            
            base_optimizer = torch.optim.SGD
            self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0,rho=0.1)                        
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.first_step(zero_grad=True)
             
                self.criterion(self.model(data), target).backward()
                self.optimizer.second_step(zero_grad=True)
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        # return self.model.get_weights(), send_ci                                #返回
        return delta_w, send_ci

    def update_SAMplus(self, weights, E, index, ps_c,lr):
        '''
        返回：send_ci: tilda_ci
        原版
        '''
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0,rho=0.05)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                self.optimizer.second_step(zero_grad=True)
                new_weights = deepcopy(self.model.get_weights())
                # w = w  + tilda_w + c - ps_c
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] +ps_c[k])*lr  # y_i = y_i -lr*(-ci + c)
                    #new_weights[k] = v - (-self.ci[k])
                self.model.set_weights(new_weights)
        send_ci = deepcopy(self.model.get_weights())
        ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +self.ci[k]-ps_c[k] 
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            #send_ci[k] = -ci[k] + self.ci[k]
            send_ci[k] = -ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        # return self.model.get_weights(), send_ci                                #返回
        return delta_w, send_ci

    def update_scafplus(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g                           
                new_weights = deepcopy(self.model.get_weights())
                # w = w  + tilda_w + c - ps_c
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (-self.ci[k] +ps_c[k])  # y_i = y_i -lr*(-ci + c)
                    #new_weights[k] = v - (-self.ci[k])
                    
                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        send_ci = deepcopy(self.model.get_weights())
        #ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)) +self.ci[k]-ps_c[k]
            #self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator))
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            #send_ci[k] = -ci[k] + self.ci[k]
            send_ci[k] = -ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        c_dict[index] = deepcopy(self.ci)                             #返回
        return delta_w, send_ci
    def update_scafM(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g
                new_weights = deepcopy(self.model.get_weights())
                # w = w  + tilda_w + c - ps_c
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - 0.9*( ps_c[k])  # y_i = y_i -lr*(-ci + c)
                    # new_weights[k] = v - (-self.ci[k])

                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        send_ci = deepcopy(self.model.get_weights())
        ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator)) + self.ci[k] - ps_c[k]
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            # send_ci[k] = -ci[k] + self.ci[k]
            send_ci[k] = -ci[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        # ci_copy = deepcopy(self.ci)
        c_dict[index] = deepcopy(self.ci)
        # return self.model.get_weights(), send_ci                                #返回
        return delta_w, send_ci

    def update_scaf(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        if self.ci == None:  # ci_0 = 0 , c = 0
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        #zero_weight = deepcopy(self.model.get_weights())
        #for k, v in zero_weight.items():
        #   zero_weight[k] = zero_weight[k] - zero_weight[k]
        #sum=deepcopy(zero_weight)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                lg_loss = 0
                loss_c = self.criterion(output, target)
                for n, p in model.named_parameters():
                    ps_c[n] = ps_c[n].to(device)
                    self.ci[n] = self.ci[n].to(device)
                    weights[n]=weights[n].to(device)
                    ps_c1=torch.flatten(ps_c[n])
                    ci1=torch.flatten(self.ci[n])
                    p = torch.flatten(p)
                    lossh=(p *(-ci1.detach() +ps_c1.detach())).sum()
                    lg_loss +=lossh.item()
                loss = loss_c +lg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        del ci1,ps_c1
        send_ci = deepcopy(self.model.get_weights())
        ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k]=weights[k].to('cpu')
            ci[k]=ci[k].to('cpu')
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +ci[k]-ps_c[k]
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ci[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        #ci_copy = deepcopy(self.ci)
        c_dict[index] = deepcopy(self.ci)
        return delta_w, send_ci

    def update_FedDC(self, weights, E, index, ps_c, lr):
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sch_step, gamma=sch_gamma)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        # print(self.optimizer.state_dict()['param_groups'][0]['lr'])
        num_workers = int(self.num_workers * selection)
        self.model.set_weights(weights)
        # fixed_params = {n: p for n, p in self.model.named_parameters()}

        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = zero_weight
        if ps_c == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            ps_c = zero_weight
        if self.hi == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.hi = zero_weight
        # 进入循环体之前，先装载数据集，以及状态
        #del zero_weight

        self.data_id_loader(index)
        self.state_id_loader(index)
        self.state_hi_loader(index)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                ce_loss = self.criterion(output, target)
                ## Weight L2 loss
                reg_loss = 0
                loss_cg = 0
                alpha = args.alpha=0.01
                for n, p in model.named_parameters():
                    ps_c[n] = ps_c[n].to(device)
                    self.ci[n] = self.ci[n].to(device)
                    self.hi[n] = self.hi[n].to(device)
                    weights[n] = weights[n].to(device)
                    L1 = alpha / 2 * torch.sum(
                        (p - (weights[n] - self.hi[n])) * (p - (weights[n] - self.hi[n]))) + torch.sum(
                        p * (-self.ci[n] + ps_c[n]))
                    loss_cg += L1.item()
                loss = ce_loss + loss_cg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()

        # 迭代了所有的E轮后，更新本地的ci 并发送

        send_ci = deepcopy(self.model.get_weights())
        ci=deepcopy(self.ci)
        for k, v in self.model.get_weights().items():
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k]=weights[k].to('cpu')
            ci[k]=ci[k].to('cpu')
            self.ci[k] =( weights[k]-v) / (E * len(self.data_iterator)*lr) +ci[k]-ps_c[k]
        self.loss = loss.item()

        for k, v in self.model.get_weights().items():
            send_ci[k] = -ci[k] + self.ci[k]

        for k, v in self.model.get_weights().items():
            self.hi[k]=self.hi[k].to('cpu')
            self.hi[k] = self.hi[k] + (v - weights[k])
        self.loss = loss.item()
        del ci
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        # ci_copy = deepcopy(self.ci)
        c_dict[index] = deepcopy(self.ci)
        # hi_copy = deepcopy(self.hi)
        hi_dict[index] = deepcopy(self.hi)
        # return delta_w, delta_g_cur, hi_copy
        return delta_w,send_ci

    def update_FedCM(self, weights, E, index, ps_c,lr):
        self.model.set_weights(weights)
        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.gamma=0.9
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr*(1-self.gamma), weight_decay=1e-3)

        #momen_v=deepcopy(zero_weight)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                #if batch_idx==2:
                #x    break
                data = data.to(device)
                target = target.to(device)
                #old_w = deepcopy(self.model.get_weights())
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                new_weights = deepcopy(self.model.get_weights())
                for k, v in new_weights.items():
                    new_weights[k] =new_weights[k] -self.gamma *ps_c[k]*lr
                self.model.set_weights(new_weights)
        # 迭代了所有的E轮后，更新本地的ci 并发送
        for k, v in self.model.get_weights().items():
            # origin
            self.ci[k] = -1 / (E * len(self.data_iterator)*lr) * (v - weights[k])
        self.loss = loss.item()
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            send_ci[k] = - ps_c[k] + self.ci[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        # 维护state字典
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        return delta_w, send_ci
    def update_IGFL(self, weights, E, index, ps_c, lr):
        num_workers = int(self.num_workers * selection)
        lr=lr

        self.model.set_weights(weights)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.alpha=0.95
        lr=lr-self.alpha/num_workers*lr

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                #forward_weight = deepcopy(self.model.get_weights())
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()                
                #backward_weight = deepcopy(self.model.get_weights())
                #for k, v in backward_weight.items():
                #    backward_weight[k] =  forward_weight[k]-backward_weight[k]
                new_weights = deepcopy(self.model.get_weights())
                    
                for k, v in self.model.get_weights().items():
                    #ps_c[k] =ps_c[k].to('cpu')
                    #self.ci[k]=self.ci[k].to('cpu')
                    #for compare scaf
                    #new_weights[k] =  v - self.ci[k] + ps_c[k]
                    #new_weights[k] =  v - backward_weight[k] * self.alpha-(1/num_workers * backward_weight[k] - 1/num_workers * self.ci[k] + ps_c[k])*(1-self.alpha) #是否这样 no!
                    #new_weights[k] =  v+ backward_weight[k] * self.alpha- (1/num_workers * backward_weight[k] - 1/num_workers * self.ci[k] + ps_c[k])*(self.alpha)
                    #new_weights[k] = v  - (1 / num_workers * backward_weight[k] - 1 / num_workers * self.ci[k] + ps_c[k]) * self.alpha
                    new_weights[k] = v - ( - 1 / num_workers * self.ci[k] + ps_c[k]) * self.alpha
                    #new_weights[k] =  v+backward_weight[k]* self.alpha + (- 1/num_workers * self.ci[k] + ps_c[k])*(1-self.alpha)
                    #new_weights[k] = v - backward_weight[k] + (backward_weight[k]/num_workers+(- 1 / self.num_workers *self.ci[k] + ps_c[k])*1.05)
                    #new_weights[k] = v + (- 1 / num_workers *self.ci[k] + ps_c[k])*0.9                    
                    #new_weights[k] = v- backward_weight[k] + backward_weight[k] * self.alpha + (1 / num_workers * backward_weight[k] - 1 / num_workers *self.ci[k] + ps_c[k])*(2-self.alpha)
                    #new_weights[k] = v - ( ps_c[k]- 1 / num_workers *self.ci[k]+1/num_workers * backward_weight[k])*(self.alpha)
                    #new_weights[k] = v + ( ps_c[k])*(self.alpha)                                                
                self.model.set_weights(new_weights)
        for k, v in self.model.get_weights().items():
            self.ci[k] = -1 / (E * len(self.data_iterator)) * (v - weights[k])
            #-1 / (E * len(self.data_iterator) * lr) * (v - weights[k])
        self.loss = loss.item()
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            send_ci[k] =self.ci[k]-ps_c[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w, send_ci
    def update_IGFL_plus(self, weights, E, index, ps_c, lr):
        num_workers = int(self.num_workers * selection)
        lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        self.model.set_weights(weights)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                forward_weight = deepcopy(self.model.get_weights())
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                backward_weight = deepcopy(self.model.get_weights())
                for k, v in backward_weight.items():
                    backward_weight[k] = forward_weight[k] - backward_weight[k]
                new_weights = deepcopy(self.model.get_weights())

                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (1 / num_workers * backward_weight[k] - 1 / num_workers * self.ci[k] + ps_c[
                        k]) * self.alpha

                self.model.set_weights(new_weights)
        for k, v in self.model.get_weights().items():
            self.ci[k] = -1 / (E * len(self.data_iterator)) * (v - weights[k])
            # -1 / (E * len(self.data_iterator) * lr) * (v - weights[k])
        self.loss = loss.item()
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            send_ci[k] = self.ci[k] - ps_c[k]
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w, send_ci

    def update_FedSTORM(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        if self.ci == None:
            zero_weight = deepcopy(self.model.get_weights())
            for k, v in zero_weight.items():
                zero_weight[k] = zero_weight[k] - zero_weight[k]
            self.ci = deepcopy(zero_weight)
        if ps_c == None:
            ps_c = deepcopy(zero_weight)
        self.data_id_loader(index)
        self.state_id_loader(index)
        #args.alpha=(1-lr)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()  # y_i = y_i-lr*g
                new_weights = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v - (args.alpha)*(-self.ci[k] + ps_c[k]) *lr # y_i = y_i -lr*(-ci + c)
                self.model.set_weights(new_weights)
        for k, v in self.model.get_weights().items():
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator)*lr) - args.alpha*(-self.ci[k] + ps_c[k])
            #(weights[k] - v) / (E * len(self.data_iterator) * lr) + self.ci[k] - ps_c[k]
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        dt= deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
            dt[k]=(weights[k]-v)/ (E * len(self.data_iterator)*lr)
        c_dict[index] = deepcopy(self.ci)  # 返回
        return delta_w,dt

    def update_FedNesterov(self, weights, E, index, ps_c, lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        args.alpha=args.alpha
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                new_weights = deepcopy(self.model.get_weights())
                for k, v in self.model.get_weights().items():
                    new_weights[k] = v +args.alpha*(ps_c[k]/ (E * len(self.data_iterator)) )
                self.model.set_weights(new_weights)
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
        for k, v in self.model.get_weights().items():
            weights[k] = weights[k].to('cpu')
        delta_w = deepcopy(self.model.get_weights())
        #dt= deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
            #dt[k]=(weights[k]-v)/ (E * len(self.data_iterator)*lr)
        return delta_w,delta_w


    def update_FedACG(self, weights, E, index, ps_c, lr):
        for k, v in weights.items():
            weights[k] = weights[k] + ps_c[k] * 0.85
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                reg_loss = 0
                for n, p in model.named_parameters():
                    weights[n] = weights[n].to(device)
                    L1=((p - weights[n].detach()) ** 2).sum()
                    reg_loss += L1.item()
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
        for k, v in self.model.get_weights().items():
            weights[k] = weights[k].to('cpu')
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.update_fedavg,  # base FedAvg
            'FedMoment': self.update_fedavg,  # add moment
            #'cddplus': self.update_cddplus,  # 老式实现glfl
            #'cdd': self.update_cdd,  # 待清除
            'IGFL_atte': self.update_IGFL,  # self attetion
            'IGFL': self.update_IGFL,  # 新式实现glfl
            'SCAFFOLD': self.update_scaf,  # scaf
            'SCAFFOLD+': self.update_scafplus,  # scaf
            'mutilayer-atte': self.update_IGFL,  # 双层attetion
            'self-atte': self.update_IGFL,  # self attetion
            'global-atte': self.update_IGFL,  # Q=全局信息，
            'FedAdam': self.update_fedavg,  # FedAdam
            #'FedAvg_atte': self.update_only_atte,  # only attetion
            #'collection': self.update_collection,
            #'only-atte-self': self.update_only_atte,
            #'momentum-step': self.update_momentum_step,
            'FedDyn':self.update_fedDyn,
            'FedCM':self.update_FedCM,
            #'IGFL_prox':self.update_cddplus_c_prox,
            'FedDC':self.update_FedDC,
            #'cddplus_ci':self.update_cddplus_ci,
            'FedSTORM':self.update_FedSTORM,
            #'SCAFFOLDM':self.update_SCAFFOLDM,
            'FedSAM':self.update_SAM,
            'FedSAM+':self.update_SAMplus,
            'MoFedSAM':self.update_MoFedSAM,
            'FedSAMS':self.update_SAMS,
            'FedSWA':self.update_FedSWA,
            'FedSWAS':self.update_FedSWAS,
            'Fedprox':self.update_Fedprox,
            'stem':self.update_stem,
            'FedACG':self.update_FedACG,
            'SCAFM':self.update_scafM,
            'IGFL+': self.update_IGFL_plus,
            'Fedspeed':self.update_Fedspeed,
            'Moon':self.update_Moon,
            'FedNesterov':self.update_FedNesterov,


        }

    def update_func(self, alg, weights, E, index,lr, ps_c=None):
        self.load_dict()
        if alg in {'SCAFFOLD', 'IGFL', 'IGFL_atte', 'mutilayer-atte', 'self-atte', 'global-atte', 'only-atte','FedAvg_atte','IGFL_atte','IGFL+','FedNesterov',
                   'collection', 'only-atte-self', 'momentum-step','FedCM','IGFL_prox','FedDC','cddplus_ci','FedSTORM','SCAFFOLDM','SCAFFOLD+','FedSAM+','MoFedSAM','FedSAMS','FedSWAS','Fedprox','stem','FedACG','SCAFM'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)



if __name__ == "__main__":
    # 获取args
    epoch = args.epoch
    num_workers = args.num_workers
    batch_size = args.batch_size
    lr = args.lr
    E = args.E
    lr_decay = args.lr_decay  # for CIFAR10
    # lr_decay = 1
    alg = args.alg
    data_name = args.data_name
    selection = args.selection
    tau = args.tau
    lr_ps = args.lr_ps
    alpha_value = args.alpha_value
    alpha = args.alpha
    extra_name = args.extname
    check = args.check
    T_part = args.T_part
    c_dict = {}
    lr_decay=args.lr_decay
    
    hi_dict = {}
    Li_dict = {}
    import time

    localtime = time.asctime( time.localtime(time.time()) )

    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value,extra_name,localtime)
    c_dict = {}  # state dict
    assert alg in {
        'FedAvg',
        'FedMoment',
        'cddplus',
        'cdd',
        'SCAFFOLD',
        'IGFL_atte',
        'IGFL',
        'mutilayer-atte',
        'self-atte',
        'global-atte',
        'FedAdam',
        'FedAvg_atte',
        'collection',
        'only-atte-self',
        'momentum-step',
        'FedDyn',
        'FedCM',
        'IGFL_prox',
        'FedDC',
        'cddplus_ci',
        'FedAGM',
        'SCAFFOLDM',
        'SCAFFOLD+',
        'FedSAM',
        'FedSAM+',
        'MoFedSAM',
        'FedSAMS',
        'FedSWA',
        'FedSWAS',
        'Fedprox',
        'stem',
        'FedACG',
        'SCAFM',
        'IGFL+',
        'Fedspeed',
        'Moon',
        'FedSTORM',
        'FedNesterov',

    }
    #  配置logger
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("./log/{}-{}-{}-{}-{}-{}-{}.txt"
                                 .format(alg, data_name, lr, num_workers, batch_size, E, lr_decay))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(comment=alg)

    nums_cls = 100
    if data_name == 'CIFAR10':
        nums_cls = 10
    if data_name == 'CIFAR100':
        nums_cls = 100
    if data_name == 'EMNIST':
        nums_cls = 62
    if data_name == 'MNIST':
        nums_cls = 47
    if data_name == 'imagenet':
        nums_cls = 200

    nums_sample = 500
    if data_name == 'CIFAR10':
        nums_sample = int(50000/(args.num_workers))
    if data_name == 'EMNIST':
        nums_sample = 6979
    if data_name == 'MNIST':
        nums_sample = 500
    if data_name == 'CIFAR100':
        nums_sample = int(50000/(args.num_workers))
    if data_name == 'imagenet':
        nums_sample = int(100000 / (args.num_workers))

    #data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)




    import pickle
    if args.data_name=='imagenet':
        # 存储变量的文件的名字
        if args.alpha_value==0.6:
          filename = 'data_idx.data'
        if args.alpha_value==0.1:
          filename = 'data_idx100000_0.1.data'
        #filename = 'data_idx100000_0.05.data'
        #f = open(filename, 'wb')
        # 将变量存储到目标文件中区
        #pickle.dump(data_idx, f)
        #关闭文件
        #f.close()
        f = open(filename, 'rb')
        # 将文件中的变量加载到当前工作区
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)
        logger.info('std:{}'.format(std))
    #
    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

    ps = ParameterServer.remote(lr_ps, alg, tau, selection, data_name,num_workers)
    if data_name == 'imagenet':
        model = ConvNet200().to(device)
    if data_name == 'CIFAR10':
        model = ConvNet().to(device)
    elif data_name == 'EMNIST':
        model = SCAFNET().to(device)
        #model = ConvNet_EMNIST().to(device)
    elif data_name == 'CIFAR100':
        model = ConvNet100().to(device)
    elif data_name == 'MNIST':
        # model = SCAFNET().to(device)
        model = ConvNet_MNIST().to(device)
    if check:
        model_CKPT = torch.load(checkpoint_path)
        print('loading checkpoint!')
        model.load_state_dict(model_CKPT['state_dict'])
        c_dict = model_CKPT['c_dict']
        Li_dict = model_CKPT['Li_dict']
        hi_dict = model_CKPT['hi_dict']
        ps_state = model_CKPT['ps_state']
        ray.get(ps.set_state.remote(ps_state))
        epoch_s = model_CKPT['epoch']
        data_idx = model_CKPT['data_idx']
    else:
        epoch_s = 0
        # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection/args.p))]
    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},rho:{}'
                .format(extra_name, alg, E, data_name, epoch, lr,alpha_value,alpha,args.CNN,args.rho))
    # logger.info('data_idx{}'.format(data_idx))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")
    
    '''
    if data_name == 'CIFAR10':
        ckpt_dict = torch.load("/data/backups/ljk/glfl/checkpoint/ckpt-FedDC-0.1-cifar10vgg19FedDC-0.6-cifar10vgg19FedDC-Wed Jun 14 21:31:50 2023")
        weights = ckpt_dict['state_dict']      
        model.load_state_dict(weights)
    if data_name == 'CIFAR100':
        ckpt_dict = torch.load("/home/ljk/glfl/checkpoint/ckpt-FedMoment-0.03-cifar100vgg11pre-0.6-cifar100vgg11pre-Tue May 23 16:52:03 2023")
        weights = ckpt_dict['state_dict']      
        model.load_state_dict(weights) 
    '''
    ps.set_weights.remote(model.get_weights())
    current_weights = ps.get_weights.remote()
    ps_c = ps.get_ps_c.remote()

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    zero=model.get_weights()
    #print(delta_g_sum)
    for k, v in model.get_weights().items():
        zero[k]=zero[k]-zero[k]
    ps_c=deepcopy(zero)
    if args.alg== 'FedDyn':
        h=deepcopy(zero)
    del zero
    div=[]
    sim=[]
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()
        index = np.arange(num_workers)  # 100
        lr=lr*lr_decay
        np.random.shuffle(index)
        index = index[:int(num_workers * selection)]  # 10id
        if alg in {'SCAFFOLD','SCAFFOLD+', 'IGFL_atte', 'mutilayer-atte', 'self-atte', 'global-atte', 'FedAvg_atte','IGFL+','FedNesterov',
                   'collection', 'only-atte-self', 'momentum-step','FedCM','IGFL_prox','cddplus_ci','IGFL','SCAFFOLDM','FedDC','FedSAM+','MoFedSAM','FedSWAS','stem','SCAFM','FedSTORM',}:
            weights_and_ci=[]
            n=int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci +[ worker.update_func.remote(alg, current_weights, E , idx,lr,ps_c) for worker, idx in
                    zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)

            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)

            weights = [w for w, ci in weights_and_ci]
            ci = [ci for w, ci in weights_and_ci]
            ps_c = ps.apply_ci.remote(alg, num_workers, *ci)
            if args.print==1:
                current = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current = ray.get(current)
                current_weights = current[0]
                divergence= current[1]
                similarity =current[2]
                print('similarity:',similarity)
                print('divergence:',divergence)
                sim.append(similarity)
                div.append(divergence)
                model.set_weights(current_weights)
            else:
                current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current_weights = ray.get(current_weights)
                model.set_weights(current_weights)
            del weights
            del ci

        elif alg in {'FedAvg', 'FedMoment', 'FedAdam', 'FedSAM', 'FedSWA', 'Fedprox', 'Fedspeed', 'Moon'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                # worker_sel = workers[i:i + int(n / 2)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                     zip(workers, index_sel)]

            #current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
            time3 = time.time()
            print(epochidx, '    ', time3 - start_time1)
            if args.print==1:
                current = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current = ray.get(current)
                current_weights = current[0]
                divergence= current[1]
                similarity =current[2]
                print('similarity:',similarity)
                print('divergence:',divergence)
                sim.append(similarity)
                div.append(divergence)
                model.set_weights(current_weights)
            else:
                current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current_weights = ray.get(current_weights)
                model.set_weights(current_weights)
            
            


        if alg in {'FedAGM', 'FedACG'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                           worker, idx in
                                           zip(workers, index_sel)]
                time3 = time.time()
                print(epochidx, '    ', time3 - start_time1)
            if args.print==1:
                current = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current = ray.get(current)
                current_weights = current[0]
                ps_c = current[1]
                divergence = current[2]
                similarity =current[3]
                print('similarity:',similarity)
                print('divergence:',divergence)
                sim.append(similarity)
                div.append(divergence)
                model.set_weights(current_weights)
            else:
                current = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current = ray.get(current)
                current_weights=current[0]
                ps_c = current[1]
                model.set_weights(current_weights)
            del weights
        if alg in {'FedNesterov1'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                           worker, idx in
                                           zip(workers, index_sel)]
                time3 = time.time()
                print(epochidx, '    ', time3 - start_time1)
            if args.print==1:
                current = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current = ray.get(current)
                current_weights = current[0]
                ps_c = current[1]
                divergence = current[2]
                similarity =current[3]
                print('similarity:',similarity)
                print('divergence:',divergence)
                sim.append(similarity)
                div.append(divergence)
                model.set_weights(current_weights)
            else:
                current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
                current_weights = ray.get(current_weights)
                #current_weights=current[0]
                #ps_c = current[1]
                model.set_weights(current_weights)
            del weights




        if alg in {'FedDyn'}:
            weights_and_hi=[]
            n=int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_hi = weights_and_hi + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]

            weights_and_hi = ray.get(weights_and_hi)
            weights = [w for w, hi in weights_and_hi]
            hi = [hi for w, hi in weights_and_hi]
            current_weights=ps.apply_weights_func2.remote(alg, num_workers, weights, hi)
            #current_weights = ps.apply_weights_func.remote(alg, num_workers, *weights)
            model.set_weights(ray.get(current_weights))

        end_time1 = time.time()
        print(epochidx, '    ', end_time1 - time3)
        print(epochidx, '    ', end_time1 - start_time1)

        if epochidx % 10 == 0:
            start_time1 = time.time()
            print('测试')
            test_loss=0
            train_loss=0
            if alg in {'FedAGM','FedACG'}:
                model.set_weights(current_weights)
            else:
                #model.set_weights(ray.get(current_weights))
                model.set_weights(current_weights)
            accuracy,test_loss,train_loss = evaluate(model, test_loader,train_loader)
            end_time1 = time.time()
            print('测试完毕','    ',end_time1-start_time1)         
            test_loss=test_loss.to('cpu')
            loss_train_median=train_loss.to('cpu')        
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                ps_state = ps.get_state.remote()
                #torch.save({'epoch': epochidx + 1, 'state_dict': model.state_dict(), 'c_dict': c_dict,
                #             'ps_state': ray.get(ps_state), 'data_idx': data_idx},
                #           checkpoint_path)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info("Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}".format(epochidx, accuracy,
                                                                                                     loss_train_median,test_loss,
                                                                                                     no_improve))
                                                                                                     
            print("Iter {}: \t accuracy is {:.1f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}".format(epochidx, accuracy,
                                                                                               loss_train_median,test_loss,
                                                                                               no_improve))
            # logger.info('attention:{}'.format(ray.get(ps.get_attention.remote())))
            if np.isnan(loss_train_median):
                logger.info('nan~~')
                break
            X_list.append(epochidx)
            result_list.append(accuracy)
            result_list_loss.append(loss_train_median)
            test_list_loss.append(test_loss)


    logger.info("Final accuracy is {:.2f}.".format(accuracy))
    endtime = time.time()
    logger.info('time is pass:{}'.format(endtime - start))
    x = np.array(X_list)
    result = np.array(result_list)

    # the last is time cost
    #result_list_loss.append(endtime - start)

    result_loss = np.array(result_list_loss)
    test_list_loss=np.array(test_list_loss)
    #x2 = np.array(X_list)
    div=np.array(div)

    save_name = './plot/alg_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}'.format(alg, E, num_workers, epoch,
                                                                                          lr, alpha_value, selection,alpha,
                                                                                          extra_name,args.gamma,args.rho,args.CNN,endtime)

    save_name2 = './model/model_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}'.format(alg, E, num_workers, epoch,
                                                                                          lr, alpha_value, selection,alpha,
                                                                                          extra_name,args.gamma,args.rho,args.CNN,endtime)

    save_name3 = './div/alg_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}'.format(alg, E, num_workers, epoch,
                                                                                          lr, alpha_value, selection,alpha,
                                                                                          extra_name,args.gamma,args.rho,args.CNN,endtime)
    torch.save(model.state_dict(), save_name2)
    save_name = save_name + '.npy'
    save_name2 = save_name2 + '.pth'
    save_name3 = save_name3 + '.npy'
    np.save(save_name, (x, result, result_loss,test_list_loss))
    if args.print==1:
        np.save(save_name3, (div,sim) )
    ray.shutdown()