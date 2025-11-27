import math
import os

from sam import SAM

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--lg', default=1.0, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs to train')
parser.add_argument('--num_workers', default=100, type=int, help='#workers')
parser.add_argument('--batch_size', default=16, type=int, help='# batch_size')
parser.add_argument('--E', default=1, type=int, help='# batch_size')
parser.add_argument('--alg', default='FedMoment', type=str, help='alg')  # FedMoment cddplus cdd SCAF atte
parser.add_argument('--extname', default='EM', type=str, help='extra_name')
parser.add_argument('--gpu', default='0,1', type=str, help='use which gpus')
parser.add_argument('--lr_decay', default='0.99', type=float, help='lr_decay')
parser.add_argument('--data_name', default='imagenet', type=str, help='lr_decay')
parser.add_argument('--tau', default='0.01', type=float, help='only for FedAdam ')
parser.add_argument('--lr_ps', default='0.15', type=float, help='only for FedAdam ')
parser.add_argument('--alpha_value', default='0.6', type=float, help='for dirichlet')
parser.add_argument('--selection', default='0.06', type=float, help=' C')
parser.add_argument('--check', default=0, type=int, help=' if check')
parser.add_argument('--T_part', default=10, type=int, help=' for mom_step')
parser.add_argument('--alpha', default=1, type=float, help=' for mom_step')
parser.add_argument('--CNN', default='VIT-L', type=str, help=' for mom_step')
parser.add_argument('--gamma', default=0.9, type=float, help=' for mom_step')
parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                    help='initial weights path')
# 是否冻结权重
parser.add_argument('--p', default=1, type=int, help=' for mom_step')
parser.add_argument('--freeze-layers', type=bool, default=False)
parser.add_argument('--datapath', type=str,
                    default="./data")
parser.add_argument('--num_gpus_per', default=0.5, type=float, help=' for mom_step')
parser.add_argument('--rho', default=0.1, type=float, help='rho')
parser.add_argument('--optimizer', default='SGD', type=str, help='SGD,AdamW')
parser.add_argument("--preprint", type=int, default=5, help="")
parser.add_argument("--R", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--lora", type=int, default=0, help="")
parser.add_argument("--AdaLora", type=int, default=0, help="")
parser.add_argument("--r", type=int, default=16, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--beta1", type=float, default=0.9, help="the perturbation radio for the SAM optimizer.")
parser.add_argument("--beta2", type=float, default=0.999, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--K', default=20, type=int, help='#workers')
parser.add_argument('--freeze', default=1, type=int, help='# batch_size')
parser.add_argument("--pre", type=int, default=1, help="the perturbation radio for the SAM optimizer.")
parser.add_argument('--print', default=0, type=int, help=' for mom_step')

args = parser.parse_args()
print(args.lora)
gpu_idx = args.gpu
print('gpu_idx', gpu_idx)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx


#from muon import MuonWithAuxAdam
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import SubsetRandomSampler, random_split
import random
from math import exp
from copy import deepcopy
import ray
from tensorboardX import SummaryWriter
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, \
    RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from dirichlet_data import data_from_dirichlet

from peft import AdaLoraConfig, get_peft_model
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Adafactor, Trainer, TrainingArguments

from datasets import load_dataset, tqdm
from peft import LoraConfig, get_peft_model, TaskType


print(torch.cuda.is_available())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)
num_gpus_per = args.num_gpus_per  # num_gpus_per = 0.16
# num_gpus_per = 0.5
num_gpus = len(gpu_idx.split(','))

data_name = args.data_name
CNN = args.CNN


if args.CNN=='roberta_base':
    model_path='./roberta_base'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    lora_config = LoraConfig(
        r=args.r,  # LoRA attention dimension
        lora_alpha=args.r*2,  # Alpha scaling
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        #target_modules=['query', 'value', 'key','intermediate.dense','output.dense'] , # Target modules to apply LoRA
        #target_modules=['query', 'value', 'key'],
        target_modules=['query', 'value', 'key', 'dense'],
        #modules_to_save = ["classifier"],
    )
    #'''

    if args.AdaLora == 1:
        lora_config = AdaLoraConfig(
            target_r=args.r,
            init_r=args.r * 2,
            beta1=0.85,
            beta2=0.85,
            tinit=0,
            tfinal=500,
            deltaT=10,
            lora_alpha=args.r * 2,
            lora_dropout=0.05,
            #target_modules=['query', 'value', 'key', 'intermediate.dense', 'output.dense'],
            target_modules=['query', 'value', 'key'],
            task_type=TaskType.SEQ_CLS,
        )
    #'''

if args.data_name=='QQP':
    dataset_path = './data/QQP'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]



if args.data_name=='MNLI':
    model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)
    dataset_path = './data/MNLI'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    #model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).to(device)


if args.data_name=='STS-B':
    dataset_path = './data/sts-b'
    # 加载数据集
    dataset = load_dataset(dataset_path)

    dataset = dataset.rename_column("score", "label")
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    #tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])


    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='WNLI':
    dataset_path = './data/WNLI'
    # 加载数据集
    dataset = load_dataset(dataset_path)
    # 数据预处理
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]




if args.data_name=='RTE':
    dataset_path = './data/RTE'
    dataset = load_dataset(dataset_path)
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",
                         max_length=128)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name=='MRPC':
    def preprocess_function(example):
        return tokenizer(example["text1"], example["text2"], truncation=True, padding="max_length",max_length=128)
    dataset_path = './data/MRPC'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

if args.data_name=='qnli':
    def preprocess_function(examples):
        # 拼接问题和句子
        inputs = tokenizer(
            examples["text1"],
            examples["text2"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        labels = [1 if label == "entailment" else 0 for label in examples["label"]]
        inputs["labels"] = labels
        return inputs
    dataset_path = './data/qnli'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


if args.data_name=='sst2':
    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", return_tensors="pt",max_length=64)
    dataset_path = './data/sst2'
    dataset = load_dataset(dataset_path)
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]

if args.data_name=='cola':
    dataset_path = './data/cola'
    def preprocess_function(examples):
        return tokenizer(examples["Sentence"], padding="max_length", truncation=True,max_length=64)

    dataset = load_dataset(dataset_path)
    dataset = dataset.rename_column("Acceptability", "label")
    # 应用预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    #tokenized_dataset = tokenized_dataset.rename_column("Acceptability", "label")
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]


seed = 42
if args.alpha_value==1:
    def get_data_loader(pid, data_idx, batch_size, data_name):
        """Safely downloads data. Returns training/validation set dataloader. 使用到了外部的数据"""
        generator = torch.Generator().manual_seed(42)
        train_dataset = tokenized_dataset["train"]
        total_size = len(train_dataset)
        #print(total_size)
        subset_size = total_size // args.num_workers
        remainder = total_size % args.num_workers  # 计算剩余的样本数

        # 创建分割大小列表
        split_sizes = [subset_size] * (args.num_workers - 1) + [subset_size + remainder]
        subsets = random_split(train_dataset, split_sizes, generator=generator)
        sample_chosed = data_idx[pid]
        #train_sampler = SubsetRandomSampler(sample_chosed)
        #train_dataset = tokenized_dataset["train"]
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
    #train_dataset = tokenized_dataset["train"]
    test_dataset = tokenized_dataset["validation"]
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)
    return test_loader

def get_data_loader_train(data_name):
    #train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
    train_dataset = tokenized_dataset["train"].select(range(1000))
    #test_dataset = tokenized_dataset["validation"]
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)
    return test_loader

def evaluate2(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    start_time1 = time.time()
    print('evaluate')
    with torch.no_grad():
        for batch in tqdm(test_loader,disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss+= criterion(logits, target)
    accuracy = 100. * correct / total
    end_time1 = time.time()
    print('evaluate完毕', '    ', end_time1 - start_time1)
    return  accuracy , test_loss / len(test_loader), torch.tensor(0)

def evaluate(model, test_loader, train_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    test_loss = 0
    train_loss = 0
    start_time1 = time.time()
    print('evaluate')
    with torch.no_grad():
        for batch in tqdm(test_loader,disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            test_loss+= criterion(logits, target)
        for batch in tqdm(train_loader, disable=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target = batch["label"].to(device)
            model.zero_grad()
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            _, predicted = torch.max(logits.data, 1)
            train_loss += criterion(logits, target)
    accuracy = 100. * correct / total
    end_time1 = time.time()
    print('evaluate完毕', '    ', end_time1 - start_time1)
    return  accuracy , test_loss / len(test_loader), train_loss / len(train_loader)


#@ray.remote(num_cpus=1,num_gpus=num_gpus_per)
@ray.remote(num_gpus=num_gpus_per)
class DataWorker(object):

    def __init__(self, pid, data_idx, num_workers, lr, batch_size, alg, data_name, selection, T_part):
        self.alg = alg
        if args.CNN == 'roberta_base':
            model_path = './roberta_base'
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            if args.data_name == 'MNLI':
                self.model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

        if args.lora == 1 and args.alg!="FLORA":
            self.model = get_peft_model(self.model, lora_config)
            print(args.lora)
        self.pid = pid
        self.num_workers = num_workers
        self.data_iterator = None
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0
        self.lr_decay = lr_decay
        self.alg = alg
        self.data_idx = data_idx
        self.pre_ps_weight = None
        self.pre_loc_weight = None
        self.flag = False
        self.ci = {}
        self.selection = selection
        self.T_part = T_part
        self.Li = None
        self.hi = None
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.momen_v = {}
        self.momen_m = {}
        self.R =1
        self.t ={k:  torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in self.model.named_parameters()}
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2), amsgrad=False)


    def data_id_loader(self, index):
        '''
        在每轮的开始，该工人装载数据集，以充当被激活的第index个客户端
        '''
        self.data_iterator = get_data_loader(index, self.data_idx, batch_size, data_name)



    def state_id_loader(self, index, shared_state):
        '''
        在每轮的开始，该工人装载状态，以充当被激活的第index个客户端，使用外部的状态字典
        '''
        # c_dict = ray.get(c_dict_id)
        self.ci = ray.get(shared_state.get_ci_dict.remote(index))




    def get_train_loss(self):
        return self.loss

    def get_param_name(self, param):
        # 获取参数的名称
        for name, p in self.model.named_parameters():
            if p is param:
                return name
        return None


    def update_fedavg(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-3)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1  # 步数+1

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w

    def update_fedavg_adamw(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2))
        step = 0  # 新增步数计数
        self.loss =0
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss += loss.item() / args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1  # 步数+1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w

    def update_FedLADA(self, weights, E, index, momen_m, momen_v, lr, step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if momen_m=={}:
            momen_m = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

        momen_m2 = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        for k, v in self.model.named_parameters():
            if k not in momen_m.keys():
                continue
            momen_m[k] = momen_m[k].to(device)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=lr, weight_decay=0.01)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step']=step.to(device)
                self.optimizer.state[p]['exp_avg'] = momen_m2[param_name].clone().detach().to(device)
                self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].to(device)
        self.loss = 0
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1  # 步数+1
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if n not in momen_m.keys():
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma)/(args.K))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                #avg_val = self.optimizer.state[p]['exp_avg_sq'].mean()
                momen_v[param_name] = self.optimizer.state[p]['exp_avg_sq'].to('cpu')

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w, momen_v

    def update_FedAdamW(self, weights, E, index, momen_m, momen_v, lr, step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if momen_m=={}:
            momen_m = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

        momen_m2 = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        for k, v in self.model.named_parameters():
            if k not in momen_m.keys():
                continue
            momen_m[k] = momen_m[k].to(device)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                           lr=lr*(1-args.gamma), weight_decay=0.01/(1-args.gamma))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step']=step.to(device)
                self.optimizer.state[p]['exp_avg'] = momen_m2[param_name].clone().detach().to(device)
                self.optimizer.state[p]['exp_avg_sq'] =torch.full_like(p.data, momen_v[param_name]).to(device)
        self.loss = 0
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1  # 步数+1
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if n not in momen_m.keys():
                        continue
                    p.data.add_(momen_m[n].mul(args.gamma)/(args.K))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                avg_val = self.optimizer.state[p]['exp_avg_sq'].mean()
                momen_v[param_name] = avg_val.to('cpu')

        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w, momen_v

    def update_fedavg_adam(self, weights, E, index, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=0.01,
                                           betas=(args.beta1, args.beta2))
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1  # 步数+1
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w


    def update_FedCM(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.state_dict().items()}

        self.data_id_loader(index)
        self.gamma = 0.9
        args.gamma=0.9
        self.optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr*(1-args.gamma),
                                         weight_decay=0.001)
        for k, v in self.model.state_dict().items():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to(device)
        self.loss = 0
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step=step+1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                for n, p in self.model.named_parameters():
                    if not p.requires_grad or n not in ps_c.keys():
                        continue
                    p.data.add_(ps_c[n]*args.gamma/args.K)


        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        return delta_w

    def update_scaf(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        if self.ci == {}:
            self.ci = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v,device='cpu') for k, v in self.model.named_parameters()}
        # 进入循环体之前，先装载数据集，以及状态
        self.data_id_loader(index)
        #self.state_ci_loader(index)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.001)
        #with torch.no_grad():
        for k in ps_c:
            ps_c[k] = ps_c[k].to(device)
            self.ci[k] = self.ci[k].to(device)
            weights[k] = weights[k].to(device)

        self.loss = 0
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step = step + 1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                self.loss+=loss.item()/args.K

                lg_loss = 0
                loss_c =  self.criterion(logits, target.long())
                for n, p in self.model.named_parameters():
                    if 'lora' not in n:
                        continue
                    lossh = (p * (-self.ci[n] + ps_c[n])).sum()
                    lg_loss += lossh.item()
                loss = loss_c + lg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        send_ci = {}
        ci = {}
        #with torch.no_grad():
        for k, v in self.model.named_parameters():
            v_cpu = v.detach().to('cpu')
            ps_c[k] = ps_c[k].to('cpu')
            self.ci[k] = self.ci[k].to('cpu')
            weights[k] = weights[k].to('cpu')
            ci[k] = self.ci[k]
            self.ci[k] = (weights[k] - v_cpu) / (args.K * lr) + ci[k] - ps_c[k]

        for k, v in self.model.named_parameters():
            if 'lora' not in k:
                continue
            send_ci[k] = -ci[k] + self.ci[k]
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]

        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 4 == 0:
            print('norm:', norm,'loss:',self.loss)
        #del  target, output, loss, ci
        #torch.cuda.empty_cache()
        return delta_w, send_ci





    def update_fedavg_adamw_V(self, weights,momen_v, E, index, lr,step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        self.momen_m = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        if momen_v=={}:
            momen_v= {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}

        self.optimizer = LayerWiseAdamW(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=1e-2,
                                        betas=(args.beta1, args.beta2))
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                self.optimizer.state[p]['step_1'] = int(0)
                self.optimizer.state[p]['step_2'] = int(step)
                self.optimizer.state[p]['exp_avg'] = self.momen_m[param_name].clone().detach().to(device)
                self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].clone().detach().to(device)
                #self.optimizer.state[p]['exp_avg_sq_1'] = momen_v[param_name].clone().detach().to(device)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1  # 步数+1

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
        #print(momen_v)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                #print(param_name)
                momen_v[param_name] = self.optimizer.state[p]['exp_avg_sq'].clone().detach().to('cpu')
                #momen_v[param_name] = self.optimizer.state[p]['exp_avg_sq_1'].clone().detach().to('cpu')
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 10 == 0:
            print(norm)
        return delta_w, momen_v



    def update_FedadamW_CM(self, weights, E, index, ps_c, lr):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if ps_c == {}:
            ps_c = {k: torch.zeros_like(v) for k, v in self.model.named_parameters() if v.requires_grad}

        for k, v in self.model.named_parameters():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to(device)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr*(1-args.gamma), weight_decay=0.01,
                                           betas=(args.beta1, args.beta2))

        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                step += 1  # 步数+1
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                for n, p in self.model.named_parameters():
                    if not p.requires_grad:
                        continue
                    p.data.add_(-ps_c[n]*args.gamma*lr)

        send_ci = {}
        for k, v in self.model.named_parameters():
            if k in ps_c.keys():
                ps_c[k] = ps_c[k].to('cpu')

        for k, v in self.model.named_parameters():
            if v.requires_grad:
                send_ci[k] = - 1 / (args.K*lr) * (v.cpu() - weights[k])
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        norm=0
        for k, v in self.model.named_parameters():
            if k in delta_w.keys():
                norm += torch.norm(delta_w[k], p=2)
        if index % 10 == 0:
            print(norm)
        return delta_w,send_ci



    def update_fedavg_adamw_A(self, weights, E, index,momen_v,  lr,step):
        self.model.load_state_dict(weights)
        self.model.to(device)
        self.data_id_loader(index)
        if args.freeze==0:
            for name, param in self.model.named_parameters():
                if "classifier" in name or "head" in name:
                    param.requires_grad = True
        #self.optimizer = AdamW_A(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-2)
        self.momen_m = {k: torch.zeros_like(v) for k, v in self.model.named_parameters()}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                if param_name in self.momen_v.keys():
                    #self.optimizer.state[p]['step_2'] = step.to(device)
                    #self.optimizer.state[p]['step_1'] =  torch.tensor([0], dtype=torch.float32, device='cpu').to(device)
                    self.optimizer.state[p]['step'] = step.to(device)
                    self.optimizer.state[p]['exp_avg'] =self.momen_m[param_name].clone().detach().to(device)
                    self.optimizer.state[p]['exp_avg_sq'] = momen_v[param_name].clone().detach().to(device)
        step = 0  # 新增步数计数
        for e in range(E):
            for batch in tqdm(self.data_iterator, disable=True):
                if step >= args.K:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["label"].to(device)
                self.model.zero_grad()
                output = self.model(input_ids, attention_mask=attention_mask)
                logits = output.logits
                loss = self.criterion(logits, target.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                self.optimizer.step()
                step += 1  # 步数+1
        momen_v={}
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_name = self.get_param_name(p)
                momen_v[param_name] = self.optimizer.state[p]['exp_avg_sq'].clone().detach().to('cpu')
        if args.lora == 1:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items() if 'lora' in k}
            for k, v in self.model.state_dict().items():
                if 'lora' in k:
                    delta_w[k] = v.cpu() - weights[k]
        else:
            delta_w = {k: v.cpu() for k, v in self.model.state_dict().items()}
            for k, v in self.model.state_dict().items():
                delta_w[k] = v.cpu() - weights[k]
        return delta_w, momen_v

    def load_dict(self):
        self.func_dict = {
            'FedAvg': self.update_fedavg,  # base FedAvg
            'FedAvg_adamw': self.update_fedavg_adamw,
            'FedAvg_adamw_A': self.update_fedavg_adamw_A,
            'FedAvg_adamw_V': self.update_fedavg_adamw_V,
            'FedAvg_adam': self.update_fedavg_adam,
            'FedadamW_CM': self.update_FedadamW_CM,
            'SCAFFOLD': self.update_scaf,  # scaf
            'FedCM': self.update_FedCM,
            'FedAdamW': self.update_FedAdamW,
            'FedLADA': self.update_FedLADA,
            'FedAdam': self.update_fedavg,  # FedAdam


        }

    def update_func(self, alg, weights, E, index, lr, ps_c=None, v=None,step=None,shared_state=None):
        self.load_dict()
        if alg in { 'FedCM', 'MoFedSAM', 'FedSWAS',
                   'FedACG','FedLionM','FedadamW_CM','FedCM'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, lr)
        if alg in {'FedAvg_adamw_A', 'FedAvg_adamw_M','FedAvg_adam_A','FedLADA','FedAdamW'}:
            return self.func_dict.get(alg, None)(weights, E, index, ps_c, v, lr, step)
        if alg in { 'FedAvg_adamw_A','FedAdam_mini_A'}:
            return self.func_dict.get(alg, None)(weights, E, index,  v, lr,step)
        if alg in {'FedAvg_adamw_V','FedAvg_adamw_A','FedAvg_adamw_V2'}:
            return self.func_dict.get(alg, None)(weights, E=E, index=index,  lr=lr, step=step,momen_v=v)
        if alg in { 'FedAvg_adamw_P'}:
            return self.func_dict.get(alg, None)(weights, E, index, lr,shared_state,step)
        if alg in {'SCAFFOLD','SCAFFOLD+'}:
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
        if k in momen_m.keys():
            seted_weight[k] = (v + args.lr_ps * momen_m[k] / (momen_v[k].sqrt() + tau))
            #                   *math.sqrt(1 - self.beta ** self.t)/(1 - 0.9 ** self.t))
            #print(k, args.lr_ps / (self.momen_v[k].mean().sqrt()*math.sqrt(1 - self.beta ** self.t) + self.tau)/(1 - 0.9 ** self.t))
        else:
            seted_weight[k]=v
    #self.t = self.t + 1
    model.load_state_dict(seted_weight)
    return model.state_dict(),momen_m,momen_v

def apply_weights_avgACG(num_workers, weights,model,momen_v):
    gamma = args.gamma
    ps_w = model.state_dict()
    sum_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():
                sum_weights[k] += 1 / (num_workers * selection) * v
            else:
                sum_weights[k] = 1 / (num_workers * selection) * v
    if momen_v == {}:
        momen_v = deepcopy(sum_weights)
    else:
        for k, v in momen_v.items():
            # self.momen_v[k] = self.gamma * v +(1-self.gamma ) *sum_weights[k]
            momen_v[k] = gamma * v + sum_weights[k]
    seted_weight = {}
    for k, v in ps_w.items():
        seted_weight[k] = v + momen_v[k]
    model.load_state_dict(seted_weight)
    return model.state_dict(), momen_v


def apply_weights_FedLADA(num_workers, weights,model):
    model.to('cpu')
    m = [mi for _, mi in weights]
    weights = [w for w, _ in weights]
    scale = 1.0 / (num_workers * selection)
    sum_c = {}
    # 首先以第一个客户端为基础初始化 sum_c（避免判断逻辑）
    for k, v in m[0].items():
        sum_c[k] =v / (num_workers * selection)
    # 之后叠加剩余客户端的梯度
    for ci in m[1:]:
        for k, v in ci.items():
            sum_c[k]+= v / (num_workers * selection)
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
    return model.state_dict(),sum_weights,sum_c


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
            if alg in {'SCAFFOLD+'}:
                ps_c[k] = ps_c[k] + sum_c[k] * 0.2
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


@torch.no_grad()
def apply_weights_avg(num_workers, weights,model):
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items()}
    scale = 1.0 / (num_workers * selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if 'lora' in k and args.lora==1:
                sum_weights[k].add_(v, alpha=scale)  # inplace 加法
            else:
                sum_weights[k].add_(v, alpha=scale)
    # 将 server 模型加上 delta_w
    for k in ps_w.keys():
        ps_w[k].add_(sum_weights[k])  # inplace 加法
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()}




@torch.no_grad()
def apply_weights_avg2( num_workers,weights,model):
    m = [mi for _, mi in weights]
    weights = [w for w, _ in weights]
    sum_c = {}  # delta_c :sum_c
    for ci in m:
        for k, v in ci.items():
            if k in sum_c.keys():
                sum_c[k] += v / (num_workers * selection)
            else:
                sum_c[k] = v / (num_workers * selection)
    # 当前 server 模型参数（保持在当前设备上，不 .cpu()）
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {k: torch.zeros_like(v) for k, v in ps_w.items()}
    scale = 1.0 / (num_workers * selection)
    # 聚合 delta_wi
    for weight in weights:
        for k, v in weight.items():
            if 'lora' in k and args.lora==1:
                sum_weights[k].add_(v, alpha=scale)  # inplace 加法
            else:
                sum_weights[k].add_(v, alpha=scale)
    # 将 server 模型加上 delta_w
    for k in ps_w.keys():
        ps_w[k].add_(sum_weights[k])  # inplace 加法
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()},sum_c


def apply_weights_avg3(num_workers, weights,model):
    m = [mi for _, mi,_ in weights]
    m2 = [vi for _,_, vi in weights]
    weights = [w for w, _,_ in weights]
    scale = 1.0 / (num_workers * selection)
    sum_m = {}
    for k, v in m[0].items():
        sum_m[k] = v / (num_workers * selection)
    for ci in m[1:]:
        for k, v in ci.items():
            sum_m[k]+=v / (num_workers * selection)
    sum_m2 = {}
    for k, v in m2[0].items():
        sum_m2[k] = v.clone().mul_(scale)
    for ci in m2[1:]:
        for k, v in ci.items():
            sum_m2[k].add_(v, alpha=scale)
    ps_w = {k: v.cpu() for k, v in model.state_dict().items()}
    sum_weights = {}  # delta_w : sum_weights
    global_weights = {}
    for weight in weights:
        for k, v in weight.items():
            if k in sum_weights.keys():  # delta_w = \sum (delta_wi/#wk)
                sum_weights[k] += v / (num_workers * selection)
            else:
                sum_weights[k] = v / (num_workers * selection)
    for k, v in sum_weights.items():  # w = w + delta_w
        global_weights[k] = ps_w[k] + sum_weights[k]
    model.load_state_dict(ps_w)
    return {k: v.cpu() for k, v in model.state_dict().items()},sum_m,sum_m2




if __name__ == "__main__":
    # 获取args
    gpu_idx = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
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
        'FedCM',
        'FedAvg_adamw',
        'FedAdam',
        'FedAvg_adamw_A',
        'FedAvg_adam',
        'FedadamW_CM',
        'FedAvg_adamw_V',
        'FedLADA',
        'SCAFFOLD',
        'FedCM',
        'FedAdamW',

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

    nums_cls = 2
    if data_name == 'sst2':
        nums_cls = 2
    if data_name == 'MNLI':
        nums_cls = 3

    nums_sample = 500
    if data_name == 'sst2':
        nums_sample = int(67349/ (args.num_workers))
    if data_name == 'cola':
        nums_sample = int(8551/ (args.num_workers))
    if data_name == 'qnli':
        nums_sample = int(104743/ (args.num_workers))
    if data_name == 'MRPC':
        nums_sample = int(5801/ (args.num_workers))
    if data_name == 'RTE':
        nums_sample = int(2490/ (args.num_workers))
    if data_name == 'MNLI':
        nums_sample = int(392702/ (args.num_workers))
    if data_name == 'QQP':
        nums_sample = int(363846 / (args.num_workers))




    import pickle
    if args.alpha_value == 0.6:
        filename = 'data_idx.data'
    if args.alpha_value == 0.1:
        filename = 'data_idx100000_0.1.data'
    filename = 'data_idx100000_0.1.data'
    if args.alpha_value==1:
        f = open(filename, 'rb')
        data_idx = pickle.load(f)
    else:
        data_idx, std = data_from_dirichlet(data_name, alpha_value, nums_cls, num_workers, nums_sample)

    ray.init(ignore_reinit_error=True, num_gpus=num_gpus)



    if args.CNN == 'bert':
        model_path = '../glfl/BERT'
        model = BertForSequenceClassification.from_pretrained(model_path)

    if args.CNN == 'roberta_base':
        model_path = './roberta_base'
        model = RobertaForSequenceClassification.from_pretrained(model_path)

        if args.data_name == 'MNLI':
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)



    model=model.to(device)
    epoch_s = 0
    # c_dict = None,None
    workers = [DataWorker.remote(i, data_idx, num_workers,
                                 lr, batch_size=batch_size, alg=alg, data_name=data_name, selection=selection,
                                 T_part=T_part) for i in range(int(num_workers * selection))]


    logger.info('extra_name:{},alg:{},E:{},data_name:{}, epoch:{}, lr:{},alpha_value:{},alpha:{},CNN:{},gamma:{}'
                .format(extra_name, alg, E, data_name, epoch, lr, alpha_value, alpha, args.CNN, args.gamma))
    # logger.info('data_idx{}'.format(data_idx))

    test_loader = get_data_loader_test(data_name)
    train_loader = get_data_loader_train(data_name)
    print("@@@@@ Running synchronous parameter server training @@@@@@")

    if args.CNN == 'bert':
        model_path = '../glfl/BERT'
        model = BertForSequenceClassification.from_pretrained(model_path)

    if args.CNN == 'roberta_base':
        model_path = './roberta_base'
        model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=2)
        if args.data_name == 'MNLI':
            model = RobertaForSequenceClassification.from_pretrained(model_path, num_labels=3)

        if args.pre==0:
            config = RobertaConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=514,
                num_labels=2,
                type_vocab_size=1,
                pad_token_id=1,
                bos_token_id=0,
                eos_token_id=2
            )
            # 从头初始化模型
            model = RobertaForSequenceClassification(config).to('cpu')
            print('从头训练')


    if args.lora == 1 and args.alg!='FLORA':
        model = get_peft_model(model, lora_config)


    current_weights=model.state_dict()
    ps_c=None

    result_list, X_list = [], []
    result_list_loss = []
    test_list_loss = []
    start = time.time()
    # for early stop
    best_acc = 0
    no_improve = 0
    m = {k: torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in
                    model.named_parameters()}
    v = {k: torch.tensor([0], dtype=torch.float32, device='cpu') for k, v in
                    model.named_parameters()}
    if alg in ['FedAvg_adamw_A','FedAdam_mini_A']:
        m = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        v = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
        v =None
    momen_m={}
    momen_v = {}
    ps_c={}

    div = []
    sim = []
    step = torch.tensor([0], dtype=torch.float32, device='cpu')
    for epochidx in range(epoch_s, epoch):
        start_time1 = time.time()

        lr = lr * lr_decay

        if args.lr_decay==2:
            eta_max=args.lr
            eta_min=0
            t=epochidx
            T=args.epoch
            lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))

        index = np.arange(num_workers)  # 100
        np.random.shuffle(index)
        index = index[:int(num_workers * selection)]  # 10id
        index = np.sort(index)

        if alg in {'FedadamW_CM'}:
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




        elif alg in {

            'FedAvg', 'FedAdam', 'FedSAM', 'Fedprox', 'FedIT',
            'FFA_LoRA', 'LoRA_FAIR',
            'FedAvg_adam',
        }:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            #print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_avg(num_workers, weights,model)
            time4 = time.time()
            #print(epochidx, '    ', time4 - time3)
            model.load_state_dict(current_weights)

        elif alg in { 'FedAdam' }:
            weights = []
            index_sel = index
            weights = [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                       zip(workers, index_sel)]
            weights = ray.get(weights)
            time3 = time.time()
            # print(epochidx, '    ', time3 - start_time1)
            current_weights,moment_m = apply_weights_avg(num_workers, weights, model,momen_m)
            time4 = time.time()
            # print(epochidx, '    ', time4 - time3)
            model.load_state_dict(current_weights)



        if alg in {  'FedCM'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            time3 = time.time()
            current_weights, ps_c = apply_weights_FedCM(num_workers, weights_and_ci, model)
            #current_weights, ps_c = apply_weights_CM(num_workers, weights_and_ci, model,ps_c)
            model.load_state_dict(current_weights)
            del weights_and_ci


        if alg in { 'FedadamW_CM'}:
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
            #current_weights, ps_c = apply_weights_CM(num_workers, weights_and_ci, model,ps_c)
            model.load_state_dict(current_weights)
            del weights_and_ci

        if alg in {'SCAFFOLD','SCAFFOLD+'}:
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
            del weights_and_ci

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



        elif alg in { 'FedAvg_adamw'}:
            weights = []
            index_sel = index
            weights =  [worker.update_func.remote(alg, current_weights, E, idx, lr) for worker, idx in
                                 zip(workers, index_sel)]
            weights=ray.get(weights)
            time3 = time.time()
            #print(epochidx, '    ', time3 - start_time1)
            current_weights = apply_weights_avg(num_workers, weights,model)
            #current_weights = apply_weights_avg_fast(num_workers, weights,model)
            #time4 = time.time()
            #print(epochidx, '    ', time4 - time3)
            model.load_state_dict(current_weights)



        if alg in {'FedAvg_adamw_A','FedAvg_adamw_V'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [
                    worker.update_func.remote(alg,current_weights, E, idx, lr,v=v, step=step)
                    for
                    worker, idx in
                    zip(workers, index_sel)]
            weights_and_ci=ray.get(weights_and_ci)
            current_weights,v=apply_weights_avg2(num_workers, weights_and_ci,model)
            model.load_state_dict(current_weights)

        if alg in {'FedLADA','FedAdamW'}:
            weights_and_ci = []
            n = int(num_workers * selection)
            for i in range(0, n, int(n / args.p)):
                index_sel = index[i:i + int(n / args.p)]
                weights_and_ci = weights_and_ci + [worker.update_func.remote(alg, current_weights, E, idx, lr, ps_c=m, v=v,step=step) for
                                                   worker, idx in
                                                   zip(workers, index_sel)]
            weights_and_ci = ray.get(weights_and_ci)
            #current_weights,m,v = apply_weights_avg3(num_workers, weights_and_ci,model)
            current_weights, m, v = apply_weights_FedLADA(num_workers, weights_and_ci, model)
            model.load_state_dict(current_weights)
            step.add_(args.K)

        end_time1 = time.time()
        print(epochidx, '    ', end_time1 - start_time1)
        args.i = 1

        if epochidx % args.preprint == 0:
            start_time1 = time.time()
            print('测试')
            test_loss = 0
            train_loss = 0
            accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            if epochidx % (args.epoch-1) == 0 and epochidx != 0:
                accuracy, test_loss, train_loss = evaluate(model, test_loader, train_loader)
            end_time1 = time.time()
            print('测试完毕', '    ', end_time1 - start_time1)
            test_loss = test_loss.to('cpu')
            loss_train_median = train_loss.to('cpu')
            # early stop
            if accuracy > best_acc:
                best_acc = accuracy
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == 1000:
                    break

            writer.add_scalar('accuracy', accuracy, epochidx * E)
            writer.add_scalar('loss median', loss_train_median, epochidx * E)
            logger.info(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},gamma:{},rho:{},alpha_value:{},data:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.gamma, args.rho, args.alpha_value,
                    args.data_name))

            print(
                "Iter {}: \t accuracy is {:.2f}, train loss is {:.5f}, test loss is {:.5f}, no improve:{}, name:{},lr:{:.7f},CNN:{},GPU:{},data:{},gamma:{},rho:{},alpha_value:{}".format(
                    epochidx, accuracy,
                    loss_train_median, test_loss,
                    no_improve, args.alg, lr, args.CNN, args.gpu, args.data_name, args.gamma,
                    args.rho, args.alpha_value))

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
    save_name = './plot/alg_{}-data_{}-E_{}-#wk_{}-ep_{}-lr_{}-alpha_value_{}-selec_{}-alpha{}-{}-gamma{}-rho{}-CNN{}-optimizer{}-time{}'.format(
        alg,args.data_name, E, num_workers, epoch,
        lr, alpha_value, selection, alpha,
        extra_name, args.gamma, args.rho, args.CNN, args.optimizer, endtime)
    save_name = save_name + '.npy'
    np.save(save_name, (x, result, result_loss, test_list_loss))
    ray.shutdown()