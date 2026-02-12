import gc
import multiprocessing

# 加入存档，log
import argparse
from datetime import datetime

from sam import SAM

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=1001, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=50, type=int, help='# batch_size')
parser.add_argument('--E', default=5, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedLESAM', type=str, help='FedAvg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.998', type=float, help='lr_decay')
parser.add_argument('--data_name', default='CIFAR100', type=str, help='imagenet,CIFAR100')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
parser.add_argument('--lr_ps', default='1', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.1', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.1', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=0.01, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='lenet5', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.85, type=float, help=' for mom_step')
parser.add_argument('--p', default=10, type=float, help=' for mom_step')
parser.add_argument('--datapath', type=str, default="./data")
parser.add_argument('--num_gpus_per', default=1, type=float, help=' for mom_step')
parser.add_argument('--normalization', default='BN', type=str, help=' for mom_step')
parser.add_argument('--print', default=0, type=int, help=' for mom_step')
parser.add_argument("--rho", type=float, default=0.1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--optimizer', default='SGD', type=str, help='adam')
parser.add_argument("--preprint", type=int, default=10, help="")
parser.add_argument("--lora", type=int, default=0, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
parser.add_argument("--pix", type=float, default=224, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--K', default=50, type=int, help='#workers')
parser.add_argument('--freeze', default=1, type=int, help='# batch_size')


args = parser.parse_args()
seed = 42
print(args.lora)
import os

gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx


from torch.utils.data import SubsetRandomSampler, random_split, DataLoader
import math
import torch
print(torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import random
from copy import deepcopy
import ray
from tensorboardX import SummaryWriter
from dirichlet_data import data_from_dirichlet
from models.resnet import ResNet18, ResNet50, ResNet10
from models.resnet_bn import ResNet18BN, ResNet50BN, ResNet10BN, ResNet34BN
from model import swin_tiny_patch4_window7_224 as swin_tiny
from model import swin_small_patch4_window7_224 as swin_small
from model import swin_large_patch4_window7_224_in22k as swin_large
from model import swin_base_patch4_window7_224_in22k as swin_base
from vit_model import vit_base_patch16_224_in21k as vit_B
from vit_model import vit_large_patch16_224_in21k as vit_L
from peft import LoraConfig, get_peft_model, TaskType
from models.DeiTTiny import deit_tiny

print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16

num_gpus = len(gpu_idx.split(','))
# num_gpus_per = 1
data_name = args.data_name
CNN = args.CNN
if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base']:
    lora_config = LoraConfig(
        r=args.r,  # 低秩矩阵的秩，通常在 4 到 64 之间[^18^]
        lora_alpha=args.r*2,  # 缩放参数，通常为 r 的 2 到 32 倍[^18^]
        lora_dropout=0.05,  # Dropout 比率，防止过拟合[^18^]
        bias="none",  # 不训练偏置项[^18^]
        task_type="IMAGE_CLASSIFICATION",  # 任务类型，根据具体任务选择[^18^]
        target_modules=['attn.qkv', 'attn.proj']  # 目标模块，根据模型结构指定[^18^]
    )


if CNN in ['VIT-B', 'swin_tiny', 'swin_large', 'VIT-L', 'swin_small', 'swin_base', 'resnet18pre', 'resnet50pre',
           'resnet101pre'] and args.pix == 224:
    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为 ResNet-18 输入的大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),  # 将图像大小调整为 ResNet-18 输入的大小
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])

    if data_name == 'CIFAR100' or data_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
else:
    if data_name == 'CIFAR10' or data_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
        )
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    if data_name == 'imagenet':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
        ])
        transform_test = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

import dataset as local_datasets

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
        download=True,
        transform=transform_train)


elif data_name == 'CIFAR100':
    train_dataset = datasets.cifar.CIFAR100(
        "./data",
        train=True,
        download=True,
        transform=transform_train
    )

if args.alpha_value==1:
    generator = torch.Generator().manual_seed(42)
    total_size = len(train_dataset)
    print(total_size)
    subset_size = total_size // args.num_workers
    remainder = total_size % args.num_workers  # 计算剩余的样本数
    # 创建分割大小列表
    split_sizes = [subset_size] * (args.num_workers-1)+ [subset_size + remainder]
    subsets = random_split(train_dataset, split_sizes, generator=generator)

    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = DataLoader(subsets[pid], batch_size=args.batch_size, shuffle=True)
        return train_loader

if args.alpha_value!=1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        sample_chosed = data_idx[pid]
        train_sampler = SubsetRandomSampler(sample_chosed)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler, num_workers=0, generator=torch.Generator().manual_seed(seed))
        return train_loader


def get_data_loader_test(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""

    if data_name == 'imagenet':
        test_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='test',
            transform=transform_test
        )
    if data_name == 'CIFAR10':
        test_dataset = datasets.CIFAR10("./data", train=False, transform=transform_test)

    elif data_name == 'CIFAR100':
        test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test
                                               )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return test_loader


def get_data_loader_train(data_name):
    """Safely downloads data. Returns training/validation set dataloader."""
    if data_name == 'imagenet':
        train_dataset = local_datasets.TinyImageNetDataset(
            root=os.path.join(args.datapath, 'tiny-imagenet-200'),
            split='train',
            transform=transform_train
        )
    if data_name == 'CIFAR10':
        train_dataset = datasets.CIFAR10("./data", train=True, transform=transform_train)
        # test_dataset = datasets.cifar.CIFAR100("./data", train=False, transform=transform_test)

    elif data_name == 'CIFAR100':
        train_dataset = datasets.cifar.CIFAR100("./data", train=True, transform=transform_train
                                                )

    train_dataset = torch.utils.data.Subset(train_dataset, range(1000))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=200,
        shuffle=False,
        num_workers=4)
    return train_loader




def evaluate(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss += criterion(outputs, target)

        for batch_idx, (data, target) in enumerate(train_loader):
            data_train = data.to(device)
            target_train = target.to(device)
            outputs_train = model(data_train)
            train_loss += criterion(outputs_train, target_train)
    torch.cuda.empty_cache()
    return 100. * correct / total, test_loss / len(test_loader), train_loss / len(train_loader)






from torch import nn
from torchvision import datasets, transforms, models


class Lenet5(nn.Module):
    """TF Tutorial for CIFAR."""

    def __init__(self, num_classes=10):
        super(Lenet5, self).__init__()
        self.n_cls = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



if CNN == 'lenet5':
    def ConvNet():
        return Lenet5(num_classes=10)
    def ConvNet100():
        return Lenet5(num_classes=100)

if CNN == 'swin_tiny':
    def ConvNet():
        return swin_tiny(num_classes=10)
    def ConvNet100():
        return swin_tiny(num_classes=100)
    def ConvNet200():
        return swin_tiny(num_classes=200)

if CNN == 'swin_large':
    def ConvNet():
        return swin_large(num_classes=10)
    def ConvNet100():
        return swin_large(num_classes=100)
    def ConvNet200():
        return swin_large(num_classes=200)
if CNN == 'swin_small':
    def ConvNet():
        return swin_small(num_classes=10)
    def ConvNet100():
        return swin_small(num_classes=100)
    def ConvNet200():
        return swin_small(num_classes=200)

if CNN == 'swin_base':
    def ConvNet():
        return swin_base(num_classes=10)
    def ConvNet100():
        return swin_base(num_classes=100)
    def ConvNet200():
        return swin_base(num_classes=200)

if CNN == 'VIT-B':
    def ConvNet():
        return vit_B(num_classes=10)
    def ConvNet100():
        return vit_B(num_classes=100)
    def ConvNet200():
        return vit_B(num_classes=200)

if CNN == 'VIT-L':
    def ConvNet():
        return vit_L(num_classes=10)
    def ConvNet100():
        return vit_L(num_classes=100)
    def ConvNet200():
        return vit_L(num_classes=200)

if CNN == 'resnet34':
    def ConvNet(num_classes=10):
        return ResNet34BN(num_classes=10)
    def ConvNet100(num_classes=100):
        return ResNet34BN(num_classes=100)
    def ConvNet200(num_classes=200):
        return ResNet34BN(num_classes=200)

if CNN == 'resnet10':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10):
            return ResNet10BN(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10BN(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10BN(num_classes=200)
    if args.normalization == 'GN':
        def ConvNet(num_classes=10):
            return ResNet10(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet10(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet10(num_classes=200)

if CNN == 'resnet50':
    def ConvNet(num_classes=10):
        return ResNet50BN(num_classes=10)
    def ConvNet100(num_classes=100):
        return ResNet50BN(num_classes=100)
    def ConvNet200(num_classes=200):
        return ResNet50BN(num_classes=200)

if CNN == 'resnet18':
    if args.normalization == 'BN':
        def ConvNet(num_classes=10, l2_norm=False):
            return ResNet18BN(num_classes=10)
        def ConvNet100(num_classes=100, l2_norm=False):
            return ResNet18BN(num_classes=100)
        def ConvNet200(num_classes=200, l2_norm=False):
            return ResNet18BN(num_classes=200)
    if args.normalization == 'GN':
        def ConvNet(num_classes=10):
            return ResNet18(num_classes=10)
        def ConvNet100(num_classes=100):
            return ResNet18(num_classes=100)
        def ConvNet200(num_classes=200):
            return ResNet18(num_classes=200)



if CNN == 'deit_tiny':
    def ConvNet(num_classes=10):
        return deit_tiny(num_classes=10, img_size=32)
    def ConvNet100(num_classes=100):
        return deit_tiny(num_classes=100, img_size=32)
    def ConvNet200(num_classes=200):
        return deit_tiny(num_classes=200, img_size=64)

import torch


@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        self.pid = pid
        if data_name == 'CIFAR10':
            self.model = ConvNet().to(device)
        elif data_name == 'CIFAR100':
            self.model = ConvNet100().to(device)
        if data_name == 'imagenet':
            self.model = ConvNet200().to(device)
        if args.lora == 1:
            self.model = get_peft_model(self.model, lora_config)
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        self.flag = False
        self.ci = {}
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.momen_v = {}
        self.momen_m = {}
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.old = {}
        self.index = 0
        self.R = 0
        self.loss = 0
        self.t = torch.tensor([0], dtype=torch.float32, device='cpu')

    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)

    def state_ci_loader(self, index):
        if not ci_dict.get(index):
            return
        self.ci = ci_dict[index]

    def get_train_loss(self):
        return self.loss

    def get_param_name(self, param):
        # 获取参数的名称
        for name, p in self.model.named_parameters():
            if p is param:
                return name
        return None

    def state_id_loader(self, index):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        if not c_dict.get(index):
            return
        self.ci = c_dict[index]



    def update_fedavg_adamw(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                   param.requires_grad = True
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2), eps=1e-8)
        self.loss = 0
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss+=loss.item()/args.K
                loss.backward()

                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        #'''
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        #'''
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 10 == 0:
            print(index,'norm:', norm,'loss:',self.loss)
        return delta_w

    def update_fedavg(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)

        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                   param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-3)
        self.loss = 0
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_FedAdamW(self, weights, E, index, momen_m, momen_v, lr, step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                   param.requires_grad = True
        if momen_m=={}:
            momen_m = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}
        for k, v in self.model.named_parameters():
            momen_m[k] = momen_m[k].to(device)
        #self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
        #                                   weight_decay=args.alpha,amsgrad=False)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                           weight_decay=args.alpha,amsgrad=False)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step'] = step.to(device)
                self.optimizer.state[p]['exp_avg'] = torch.zeros_like(p.data).to(device)
                #self.optimizer.state[p]['exp_avg_sq'] = torch.full_like(p.data, momen_v[param_name]).to(device)
                self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].clone().detach().to(device)
        step = 0  # 新增步数计数
        self.loss = 0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                #with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma*lr/(args.K)))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                state = self.optimizer.state.get(p, None)
                momen_v[param_name] = state['exp_avg_sq'].clone().detach().to('cpu')
        delta_w = {}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 10 == 0:
            print(index,'norm:', norm,'loss:',self.loss)
        return delta_w, momen_v

    def update_scaf(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        if self.ci == {}:
            self.ci = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_ci_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001)
        with torch.no_grad():
            for k in ps_c:
                ps_c[k] = ps_c[k].to(device)
                self.ci[k] = self.ci[k].to(device)
                weights[k] = weights[k].to(device)

        step = 0  # 新增步数计数
        self.loss=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step +1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                lg_loss = 0
                loss_c = self.criterion(output, target)
                for n, p in self.model.named_parameters():
                    lossh = (p * (-self.ci[n] + ps_c[n])).sum()
                    lg_loss += lossh.item()
                loss = loss_c + lg_loss
                loss.backward()
                self.optimizer.step()

        send_ci = {}
        ci = {}
        with torch.no_grad():
            for k, v in self.model.named_parameters():
                v_cpu = v.detach().to('cpu')
                ps_c[k] = ps_c[k].to('cpu')
                self.ci[k] = self.ci[k].to('cpu')
                weights[k] = weights[k].to('cpu')
                ci[k] = self.ci[k]
                self.ci[k] = (weights[k] - v_cpu) / (args.K * lr) + ci[k] - ps_c[k]

        for k, v in self.model.named_parameters():
            send_ci[k] = -ci[k] + self.ci[k]
        delta_w = {}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        del data, target, output, loss, ci
        gc.collect()
        torch.cuda.empty_cache()
        return delta_w, send_ci

    def update_FedCM(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}

        for name, param in self.model.named_parameters():
            # 默认LoRA中其他层已冻结，此处确保分类头参与训练
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.data_id_loader(index)
        self.gamma = args.gamma
        #self.optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.001)
        self.optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr,
                                         weight_decay=0.001)
        for k, v in self.model.named_parameters():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to(device)
        step = 0  # 新增步数计数
        self.loss = 0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                for k, v in self.model.named_parameters():
                    if v.grad is None:
                        continue
                    v.grad.data = (1 - args.gamma) * v.grad.data + args.gamma * ps_c[k]
                self.optimizer.step()
        send_ci = {}
        for k, v in self.model.named_parameters():
            ps_c[k] = ps_c[k].to('cpu')

        for k, v in self.model.named_parameters():
            if v.requires_grad:
                send_ci[k] =  - 1 / (args.K * lr) * (v.cpu()  - weights[k])
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w, send_ci

    def update_Moon(self, weights, E, index,lr):
        self.model.load_state_dict(weights)
        if self.ci == {}:
            self.ci = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.named_parameters()}
        self.data_id_loader(index)
        self.state_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )

        cos = torch.nn.CosineSimilarity(dim=-1)
        mu = 0.001
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                data.requires_grad = False
                target.requires_grad = False
                target = target.long()
                w = deepcopy(self.model.get_weights())
                with torch.no_grad():
                    self.model.load_state_dict(weights)
                    pro2 = self.model(data)
                    self.model.load_state_dict(self.ci)
                    pro3 = self.model(data)
                self.model.load_state_dict(w)
                pro1 = self.model(data)
                self.model.to('cuda')
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                temperature=0.5
                logits /= temperature
                labels = torch.zeros(data.size(0)).cuda().long()
                loss2 = mu * self.criterion(logits, labels)
                loss1 = self.criterion(pro1, target)
                loss = loss1 + loss2
                loss.backward()
                self.optimizer.step()
        self.loss = loss.item()
        for k, v in self.model.state_dict().items():
            self.ci[k] = v
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_Fedprox(self, weights, E, index,lr):
        self.model.set_weights(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3 )
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
                alpha = 0.01
                for n, p in model.named_parameters():
                    weights[n]=weights[n].to(device)
                    L1=alpha / 2 * torch.sum((p -weights[n]) * (p - weights[n]))
                    loss_cg +=L1.item()
                loss = self.criterion(output, target)+ loss_cg
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def update_FedSWA(self, weights, E, index,lr):
        self.model.load_state_dict(weights)
        self.data_id_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr*2, weight_decay=1e-3 )
        lr1=lr
        rho=0
        lr2=rho*lr
        i=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                i=i+1
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr*2, weight_decay=1e-3 )
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                lr=(1-i/(len(self.data_iterator)*E))*lr1+(i/(len(self.data_iterator)*E))*lr2
        self.loss = loss.item()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_FedMoSWA(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)  # y_i = x, x:weights
        if self.ci == {}:
            self.ci = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        self.data_id_loader(index)
        self.state_id_loader(index)
        a1 = lr
        rho=0
        a2 = rho * lr
        i = 0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
                i = i + 1
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
                    new_weights[k] = v - (-self.ci[k] + ps_c[k]) * lr  # y_i = y_i -lr*(-ci + c)
                self.model.set_weights(new_weights)
                lr = (1 - i / (len(self.data_iterator) * E)) * a1 + (i / (len(self.data_iterator) * E)) * a2
        lr = (a1 + a2) / 2
        send_ci = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            self.ci[k] = (weights[k] - v) / (E * len(self.data_iterator) * lr) + self.ci[k] - ps_c[k]
        self.loss = loss.item()
        for k, v in self.model.get_weights().items():
            send_ci[k] = -ps_c[k] + self.ci[k]
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        return delta_w, send_ci


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

    def update_scafplus(self, weights, E, index, ps_c,lr):
        self.model.load_state_dict(weights)  # y_i = x, x:weights
        if self.ci == {}:  # ci_0 = 0 , c = 0
              self.ci = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.named_parameters()}
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.named_parameters()}

        self.data_id_loader(index)
        self.state_ci_loader(index)
        self.model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001)
        for k, v in self.model.named_parameters():
            ps_c[k]=ps_c[k].to(device)
            self.ci[k]=self.ci[k].to(device)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                for k, v in self.model.named_parameters():
                    v.grad.data = v.grad.data + ps_c[k]-self.ci[k]
                self.optimizer.step()  # y_i = y_i-lr*g
        with torch.no_grad():
            send_ci = {}
            # 先把控制变量/全局控制变到CPU（避免反复to产生临时大张量）
            for k, p in self.model.named_parameters():
                pc = ps_c[k].detach().cpu()
                ci = self.ci[k].detach().cpu()
                w = weights[k].detach().cpu()  # 全局参数
                p_cpu = p.detach().cpu()  # 本地训练后参数

                new_ci = (w - p_cpu) / (args.K * lr) + ci - pc
                self.ci[k] = new_ci  # 常驻变量，必须是“无图”的纯tensor
                send_ci[k] = -pc + new_ci
            # delta_w 也 detach
            delta_w = {}
            for k, t in self.model.state_dict().items():
                delta_w[k] = t.detach().cpu() - weights[k].detach().cpu()
        ci_copy = deepcopy(self.ci)
        c_dict[index] = ci_copy
        return delta_w, send_ci

    def update_IGFL(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}
        if self.ci == {}:
            self.ci = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        for k, v in self.model.named_parameters():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to(device)
                self.ci[k] = self.ci[k].to(device)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        args.gamma=0.9
        #self.alpha=0.95
        #lr =lr + self.alpha / num_workers * lr
        #lr = lr *(1-args.gamma)
        #self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr*(1-args.gamma), weight_decay=0.001)
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.001)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                for k, v in self.model.named_parameters():
                    if v.grad is None:
                        continue
                    v.grad.data = (1 - args.gamma) * v.grad.data + args.gamma * (ps_c[k]-1 / num_workers *(self.ci[k]-v.grad.data))
                self.optimizer.step()
                #for n, p in self.model.named_parameters():
                #    if not p.requires_grad:
                #        continue
                #    p.data.add_((1 / num_workers * self.ci[n]-ps_c[n])*args.gamma)
        send_ci = {}
        for k, v in self.model.named_parameters():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to('cpu')
        for k, v in self.model.named_parameters():
                send_ci[k] =  - 1 / (args.K*lr) * (v.cpu()  - weights[k])
        for k, v in self.model.named_parameters():
            self.ci[k] = -1 / (args.K*lr) * (v.cpu() - weights[k])
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k or "classifier" in k or "head" in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k or "classifier" in k or "head" in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w,send_ci

    def update_SAM(self, weights, E, index, lr):
        self.model.set_weights(weights)  # y_i = x, x:weights
        num_workers = self.num_workers
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        self.state_id_loader(index)
        zero_weight = deepcopy(self.model.get_weights())
        for k, v in zero_weight.items():
            zero_weight[k] = zero_weight[k] - zero_weight[k]
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, momentum=0, rho=0.05)
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                # loss_function(output,model(input)).backward()
                self.criterion(self.model(data), target).backward()
                self.optimizer.second_step(zero_grad=True)

        # 迭代了所有的E轮后，更新本地的ci 并发送ci,delta_w
        self.loss = loss.item()
        delta_w = deepcopy(self.model.get_weights())
        for k, v in self.model.get_weights().items():
            delta_w[k] = v - weights[k]
        return delta_w

    def update_NSAM(self, weights, E, index, ps_c, lr):
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.state_dict().items()}
        args.gamma=0.85
        args.rho=0.05
        for k, v in weights.items():
            weights[k] = weights[k] + ps_c[k] * args.gamma
        self.model.load_state_dict(weights)
        self.data_id_loader(index)
        base_optimizer = torch.optim.SGD
        self.optimizer = SAM(self.model.parameters(), base_optimizer, lr=lr, weight_decay=0.001, rho=args.rho)
        step = 0  # 新增步数计数
        self.loss = 0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step = step + 1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss += loss.item() / args.K
                loss.backward()
                self.optimizer.first_step(zero_grad=True)
                self.criterion(self.model(data), target).backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.second_step(zero_grad=True)
        for k, v in self.model.state_dict().items():
            weights[k] = weights[k].to('cpu')
        delta_w = {}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu()  - weights[k]
        return delta_w

    def update_FedACG(self, weights, E, index, ps_c, lr):
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.state_dict().items()}
        args.gamma=0.85
        for k, v in weights.items():
            weights[k] = weights[k] + ps_c[k] * args.gamma
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        step = 0  # 新增步数计数
        self.loss=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step +1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                reg_loss = 0
                for n, p in model.named_parameters():
                    weights[n] = weights[n].to(device)
                    L1 = ((p - weights[n].detach()) ** 2).sum()
                    reg_loss += L1.item()
                loss = self.criterion(output, target)+0.01*reg_loss
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_FedNesterov(self, weights, E, index, ps_c, lr):
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v, device='cpu') for k, v in self.model.state_dict().items()}
        args.gamma=0.85
        for k, v in weights.items():
            weights[k] = weights[k] + ps_c[k] * args.gamma
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        for name, param in self.model.named_parameters():
            # 默认LoRA中其他层已冻结，此处确保分类头参与训练
            if "classifier" in name or "head" in name:
                param.requires_grad = True
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-3)
        step = 0  # 新增步数计数
        self.loss=0
        for e in range(E):
            for batch_idx, (data, target) in enumerate(self.data_iterator):
                if step >= args.K:
                    break
                step=step +1
                data = data.to(device)
                target = target.to(device)
                self.model.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
        for k, v in self.model.state_dict().items():
            delta_w[k] = v.cpu() - weights[k]
        return delta_w


    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.update_fedavg,  # base FedAvg
            'SCAFFOLD': self.update_scaf,  # scaf
            'FedAdam': self.update_fedavg,  # FedAdam
            'FedCM': self.update_FedCM,
            'FedAvg_adamw': self.update_fedavg_adamw,
            'FedAdamW': self.update_FedAdamW,
            'FedSAM': self.update_SAM,
            'FedMoSAM': self.update_MoFedSAM,
            'FedSWA': self.update_FedSWA,
            'Fedprox': self.update_Fedprox,
            'Moon': self.update_Moon,
            'MoFedSWA': self.update_scafplus,
            'IGFL': self.update_IGFL,
            'FedNSAM': self.update_NSAM,
            'FedACG': self.update_FedACG,
            'FedMoment': self.update_fedavg,  # add moment
            'FedNesterov': self.update_FedNesterov,
        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None, step=None, shared_state=None,ci=None):
        self.load_dict()
        if alg in {'FedCM','FedACG','IGFL','FedNSAM','FedACG'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in {'FedLADA','FedAdamW','FedMuon'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, v, lr, step)
        if alg in {'SCAFFOLD','MoFedSWA','MoFedSAM'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        else:
            return self.func_dict.get(alg, None)(weights, E, index, lr)


def set_random_seed(seed=42):
    """
    设置随机种子以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def apply_weights_moment(num_workers, weights,model,momen_m):
    model.to('cpu')
    gamma = 0.9
    ps_w = model.state_dict()
    sum_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += 1 / (num_workers * selection) * v
            else:
                sum_weights[k] = 1 / (num_workers * selection) * v
    if momen_m == {}:
        momen_m = deepcopy(sum_weights)
    else:
        for k, v in momen_m.items():
            #momen_m[k] = gamma * v + sum_weights[k]*(1-gamma)
            momen_m[k] = gamma * v + sum_weights[k]
    seted_weight = {}
    for k, v in ps_w.items():
        ps_w[k] = v + momen_m[k]
    model.load_state_dict(ps_w)
    return model.state_dict(), momen_m


@torch.no_grad()
def apply_weights_adam(num_workers, weights,model,momen_m,momen_v):
    model.to('cpu')
    tau = 0.01
    beta = 0.98
    args.lr_ps=0.01
    train_name=[]
    for name, param in model.named_parameters():
        train_name.append(name)
    delta_t = {}
    for weight in weights:
        for k, v in weight.items():
            if k in delta_t.keys():
                delta_t[k] += v / (num_workers * selection)
            else:
                delta_t[k] = v / (num_workers * selection)
    weight_ps = model.state_dict()
    if momen_m == {}:
        for k, v in delta_t.items():
            momen_m[k] = delta_t[k]*0.1
    else:
        for k, v in delta_t.items():
            momen_m[k] = 0.9 * momen_m[k] + 0.1 * delta_t[k]
    if momen_v == {}:
        momen_v = deepcopy(delta_t)
        for k, v in delta_t.items():
            momen_v[k] = (1 - beta) * v.mul(v)
    else:
        for k, v in momen_v.items():
            momen_v[k] = beta * v + (1 - beta) * delta_t[k].mul(delta_t[k])
    seted_weight = {}
    for k, v in weight_ps.items():
        if k in train_name:
            seted_weight[k] = (v + args.lr_ps * momen_m[k] / (momen_v[k].sqrt() + tau))
        else:
            seted_weight[k]=v + delta_t[k]
    #self.t = self.t + 1
    model.load_state_dict(seted_weight)
    return model.state_dict(),momen_m,momen_v

@torch.no_grad()
def apply_weights_avgACG(num_workers, weights,model,momen_m):
    model.to('cpu')
    gamma = 0.85
    ps_w = model.state_dict()
    sum_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += 1 / (num_workers * selection) * v
            else:
                sum_weights[k] = 1 / (num_workers * selection) * v
    if momen_m == {}:
        momen_m = deepcopy(sum_weights)
    else:
        for k, v in momen_m.items():
            #momen_m[k] = gamma * v + sum_weights[k]*(1-gamma)
            momen_m[k] = gamma * v + sum_weights[k]
    seted_weight = {}
    for k, v in ps_w.items():
        ps_w[k] = v + momen_m[k]
    model.load_state_dict(ps_w)
    return model.state_dict(), momen_m



def apply_weights_FedCM(num_workers, weights,model):
    model.to('cpu')
    weights = [w for w in weights]
    sum_c = {}
    # 首先以第一个客户端为基础初始化 sum_c（避免判断逻辑）

    ps_w = model.state_dict()  # w : ps_w
    sum_weights = {}  # delta_w : sum_weights
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),sum_weights




@torch.no_grad()
def apply_weights_avg(num_workers, weights,model):
    model.to('cpu')
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if not torch.is_floating_point(ps_w[k]):
                continue
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        ps_w[k] = ps_w[k] + sum_weights[k]*args.lg
        #ps_w[k]=ps_w[k]*(1-args.alpha)
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()}





def apply_weights_SCAF(num_workers, weights,model,ps_c):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weightss = [w for w,_ in weights]
    sum_c = {}
    # 首先以第一个客户端为基础初始化 sum_c（避免判断逻辑）
    for k, v in m[0].items():
        sum_c[k] =v / (num_workers * selection)
    # 之后叠加剩余客户端的梯度
    for ci in m[1:]:
        for k, v in ci.items():
            sum_c[k]+= v / (num_workers * selection)
    if ps_c == {}:
        ps_c = {k: torch.zeros_like(v.cpu()) for k, v in model.named_parameters()}
        for k, v in m[0].items():
            ps_c[k]=sum_c[k]
    else:
        for k, v in m[0].items():
            if alg in {'SCAFFOLD'}:
                ps_c[k]=ps_c[k]+sum_c[k]*selection
            else:
                ps_c[k] = ps_c[k] + sum_c[k] * 0.4
    ps_w = model.state_dict()  # w : ps_w

    sum_weights = {}  # delta_w : sum_weights
    for weight in weightss:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),ps_c








def apply_weights_avg2(num_workers, weights,model):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weights = [w for w, _ in weights]
    scale = 1.0 / (num_workers * selection)
    sum_c = {}
    # 首先以第一个客户端为基础初始化 sum_c（避免判断逻辑）
    for k, v in m[0].items():
        #sum_c[k] = v.clone().mul_(scale)
        sum_c[k] =v / (num_workers * selection)
    # 之后叠加剩余客户端的梯度
    for ci in m[1:]:
        for k, v in ci.items():
            sum_c[k]+= v / (num_workers * selection)
            #sum_c[k].add_(v, alpha=scale)
    ps_w = model.state_dict()  # w : ps_w
    sum_weights = {}  # delta_w : sum_weights
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),sum_c




def apply_weights_FedLADA(num_workers, weights,model,momen_m):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weights = [w for w, _ in weights]
    scale = 1.0 / (num_workers * selection)
    momen_v = {}
    # 首先以第一个客户端为基础初始化 sum_c（避免判断逻辑）
    for k, v in m[0].items():
        momen_v[k] =v / (num_workers * selection)
    # 之后叠加剩余客户端的梯度
    for ci in m[1:]:
        for k, v in ci.items():
            momen_v[k]+= v / (num_workers * selection)


    ps_w = model.state_dict()  # w : ps_w
    sum_weights = {}  # delta_w : sum_weights
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)

    momen_m=momen_m
    for k, v in sum_weights.items():
        if k not in momen_m.keys():
            momen_m[k]=sum_weights[k]/lr
        else:
            momen_m[k] = sum_weights[k] / lr
            #momen_m[k] =args.alpha*momen_m[k]+(1-args.alpha)*sum_weights[k]/lr

    for k, v in sum_weights.items():  # w = w + delta_w
        ps_w[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return model.state_dict(),momen_m,momen_v





if __name__ == "__main__":
    # 获取args
    step = torch.tensor([0], dtype=torch.float32, device='cpu')
    seed=42
    set_random_seed(seed=seed)
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
    lr_decay = args.lr_decay

    hi_dict = {}
    Li_dict = {}
    mi_dict = {}
    vi_dict = {}
    ti_dict = {}


    import time

    localtime = time.asctime(time.localtime(time.time()))
    checkpoint_path = './checkpoint/ckpt-{}-{}-{}-{}-{}-{}'.format(alg, lr, extra_name, alpha_value, extra_name,
                                                                   localtime)
    c_dict = {}  # state dict
    assert alg in {
        'FedAvg',
        'SCAFFOLD',
        'FedAdam',
        'FedCM',
        'FedAvg_adamw',
        'FedLADA',
        'Local_Muon',
        'FedAdamW',
        'FedMuon',
        'FedSAM',
        'FedMoSAM',
        'MoFedSWA',
        'FedSWA',
        'Fedprox',
        'Moon',
        'IGFL',
        'FedNSAM',
        'FedACG',
        'FedMoment',
        'FedNesterov'

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
        nums_sample = int(50000 / (args.num_workers))
        # nums_sample = 500
    if data_name == 'CIFAR100':
        nums_sample = int(50000 / (args.num_workers))
        # nums_sample=500
    if data_name == 'imagenet':
        nums_sample = int(100000 / (args.num_workers))
        # nums_sample = 500

    import pickle

    filename = 'num_workers_{}-alpha_value_{}-data_{}'.format(num_workers, args.alpha_value, data_name)

    if args.alpha_value == 1:
        filename = 'data_idx.data'
        f = open(filename, 'rb')
        data_idx = pickle.load(f)

    import os
    import pickle

    filename = f'num_workers_{num_workers}-alpha_value_{alpha_value}-data_{data_name}'

    if os.path.exists(filename):
        # 文件存在，直接加载
        with open(filename, 'rb') as f:
            data_idx = pickle.load(f)
        print(f"加载已有数据索引文件: {filename}")
        std = None  # 若你需要 std，则存成 tuple 后一起加载
    else:
        # 文件不存在，生成并保存
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)
        with open(filename, 'wb') as f:
            pickle.dump(data_idx, f)
        print(f"生成并保存新数据索引文件: {filename}")

    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)

    hi_dict = {}
    Li_dict = {}
    mi_dict = {}
    vi_dict = {}
    ti_dict = {}
    ci_dict = {}

    if data_name == 'imagenet':
        model = ConvNet200().to('cpu')
    if data_name == 'CIFAR10':
        model = ConvNet().to('cpu')
    elif data_name == 'CIFAR100':
        model = ConvNet100().to('cpu')

    epoch_s = 0
    # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection / args.p))]
    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},rho:{}'
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.rho))
    #logger.info('data_idx{}'.format(data_idx))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")


    if args.CNN == 'VIT-B':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('vit_base_patch16_224_in21k.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'VIT-L':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('jx_vit_large_patch16_224_in21k-606da67d.pth', map_location=device)
            # 删除不需要的权重
            del_keys = ['head.weight', 'head.bias'] if model.has_logits \
                else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            for k in del_keys:
                del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head, pre_logits外，其他权重全部冻结
                if "head" not in name and "pre_logits" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
    #if args.pre == 1:
    if args.CNN == 'swin_tiny' and args.pre == 1:
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_tiny_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))


    if args.CNN == 'swin_small':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_small_patch4_window7_224.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.CNN == 'swin_base':

        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_base_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))
        # '''

    if args.CNN == 'swin_large':
        if args.weights != "":
            assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
            weights_dict = torch.load('swin_large_patch4_window7_224_22k.pth', map_location=device)["model"]
            # 删除有关分类类别的权重
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]
            print(model.load_state_dict(weights_dict, strict=False))

        if args.freeze_layers:
            for name, para in model.named_parameters():
                # 除head外，其他权重全部冻结
                if "head" not in name:
                    para.requires_grad_(False)
                else:
                    print("training {}".format(name))

    if args.lora == 1 and args.alg != 'FLORA':
        model = get_peft_model(model, lora_config)

    current_weights=model.state_dict()

    ps_c=None
    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    best_acc = 0
    no_improve = 0
    ps_c = {}
    model=model.to(device)
    momen_m={}
    momen_v = {}
    ps_c = {}
    m={}
    v = {}
    div = []
    sim = []
    for epochidx in range(epoch_s, epoch):
        index = np.arange(num_workers)  # 100
        np.random.shuffle(index)
        index = index[:int(num_workers * selection)]  # 10id
        start_time1 = time.time()
        eta_max=args.lr
        eta_min =0
        t=epochidx
        T=args.epoch
        lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))

        if alg in { 'FedCM','FedMoSAM','IGFL'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            time3 = time.time()
            current_weights, ps_c = apply_weights_avg2(num_workers, weights_and_ci, model)
            model.load_state_dict(current_weights)
            del weights_and_ci



        if alg in {'FedLADA','FedAdamW','FedMuon'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=m, v=v,step=step) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            current_weights, m, v = apply_weights_FedLADA(num_workers, weights_and_ci, model,m)

            model.load_state_dict(current_weights)
            step.add_(nums_sample / args.batch_size * args.E)




        if alg in {'SCAFFOLD','MoFedSWA'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [
                    worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=ps_c)
                    for worker, idx in zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            current_weights, ps_c = apply_weights_SCAF(num_workers, weights_and_ci, model, ps_c)
            model.load_state_dict(current_weights)
            print(len(ci_dict))
            del weights_and_ci




        elif alg in {'FedAvg', 'FedSWA','Fedprox', 'FedAvg_adamw','FedSAM','Local_Soap','Local_Muon','Local_Shampoo','Local_AdEMAMix'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_ref = ray.put(current_weights)
                # 然后传进去：
                weights = [worker.update_func.remote(alg, weights_ref, E, idx, lr) for worker, idx in
                           zip(workers, index_sel)]
            weights=ray.get(weights)
            current_weights = apply_weights_avg(num_workers, weights,model)
            model.load_state_dict(current_weights)
            del weights

        elif alg in {'FedAdam'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                           zip(workers, index_sel)]
            weights=ray.get(weights)
            current_weights,momen_m,momen_v = apply_weights_adam(num_workers, weights,model,momen_m,momen_v)
            model.load_state_dict(current_weights)
            del weights



        if alg in { 'FedACG','FedNSAM','FedNesterov'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = weights + [worker.update_func.remote(alg, current_weights, E, idx, lr, momen_m) for
                                     worker, idx in
                                     zip(workers, index_sel)]
                time3 = time.time()
            weights = ray.get(weights)
            current_weights,momen_m = apply_weights_avgACG(num_workers,weights,model,momen_m)
            model.load_state_dict(current_weights)
            del weights

        elif alg in {'FedMoment'}:
            weights = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights = [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                           zip(workers, index_sel)]
            weights=ray.get(weights)
            current_weights,momen_m = apply_weights_moment(num_workers, weights,model,momen_m)
            model.load_state_dict(current_weights)
            del weights

        end_time1 = time.time()
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1
        args.R = args.R + 1

        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            if accuracy > best_acc:
                best_acc = accuracy
                #ps_state = ps.get_state.remote()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},gamma:{},alpha:{},alpha_value:{},data:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.gamma, args.alpha, args.alpha_value,
                    args.data_name))

            print(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},data:{},gamma:{},alpha:{},alpha_value:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.data_name, args.gamma,
                    args.alpha, args.alpha_value))
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
    result_loss = np.array(result_list_loss)
    test_list_loss = np.array(test_list_loss)
    # div = np.array(div)
    now = datetime.now()
    save_name = './plot/alg_{}-data_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-optimizer{}-time{}'.format(
        alg,args.data_name, E, num_workers, epoch,
        args.lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.rho, args.CNN, args.optimizer, now)
    save_name2 = './model/model_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-time{}'.format(
        alg, E, num_workers, epoch,
        args.lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.rho, args.CNN, endtime)
    # torch.save(model.state_dict(), save_name2)
    save_name = save_name + '.npy'
    save_name2 = save_name2 + '.pth'
    np.save(save_name, (x, result, result_loss, test_list_loss))

    ray.shutdown()
