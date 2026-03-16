

# README

## Improving generalization in federated learning with highly heterogeneous data via momentum-based stochastic controlled weight averaging 被ICML 2025录用！！

* 一张4090或者两张2080ti即可训练！！发顶会！！代码问题或者讨论+vx 15653218567

* 我的其他论文也都是这一套代码配置，均可复现！

* 个人主页：https://junkangliu0.github.io/


This repository contains the implementation of **FedSWA** and **FedMoSWA**, federated learning algorithms designed to improve generalization in the presence of highly heterogeneous client data. The work is based on the paper:

> *Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging (FedSWA & FedMoSWA)*.

Both algorithms extend upon **SCAFFOLD** and **SAM-type methods**, aiming to find **global flat minima** that lead to better test performance compared to FedAvg, FedSAM, and related approaches.

---
一张4090或者两张2080ti即可训练！！

## 🛠 Environment Setup
创建环境，要python=3.8，ray==1.0.0  ！！！
```
conda create -n fedswa python=3.8 -y
conda activate fedswa
```

### Requirements
* Python 3.8
* torch==2.4.1
* torchvision==0.19.1
* numpy
* ray==1.0.0
* tensorboardX==2.6.2.2
* peft==0.13.2
* transformers==4.46.3

You can install the dependencies with:

```bash
pip install -r requirements.txt
```


下载速度慢改一下镜像源
```
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
EOF
```

---

## Usage

To run the training with **SCAFFOLD** on **CIFAR100** using a **ResNet-18** backbone and Group Normalization:

```bash
python  main_FedSWA.py --alg FedSWA --lr 2e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50

python  main_FedSWA.py --alg MoFedSWA --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50

python  main_FedSWA.py --alg FedAvg --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 301  --extname FedMuon --lr_decay 2 --gamma 0.5  --CNN   resnet18 --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 100 --preprint 10  --rho 0.01 --pix 32 --lora 0 --K 50
```

* 这里解释一下 --num_gpus_per 0.1的意思是如果你用的是4090显卡24g显存，那么你每个客户端将分配0.1张显卡，即2.4g显存。
* --lr_decay 2 解释一下，这个是余弦学习率下降
* --gpu 0 是指使用的是第0块gpu（gpu序号）
* --alpha_value 0.1 是迪利克雷非立同分布常数
* --alpha_value 1 这个时候是iid情况
* --lora 0 是否使用lora微调，从头训练的情况下，不用lora微调 选0就行
* --normalization BN resnet的归一化层，我选的是BN层，这个效果更好，选择GN也行，收敛的慢
* --data_name timy imagenet数据集需要自己下载，网址在下面


另外注意！！FedSWA的学习率一般是FedAvg的两倍！lr=0.1*2
---


---
## 联邦大模型微调 vit


```bash
python  main_FedSWA.py --alg FedSWA --lr 2e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha  10  --epoch 101  --extname FedMuon --lr_decay 2 --gamma 0.85  --CNN   VIT-B --E 5 --batch_size 50   --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --pre 1 --num_workers 50 --preprint 10  --rho 0.01 --pix 224 --lora 1 --K 50
```

* --lora 1 使用lora微调
* --batch_size 16 显存限制原因，16效果还可以
* --num_gpus_per 0.2 五个客户端，每个客户端使用0.2张卡
* --lr 1e-3 这个学习率微调lora最好

  ---

下载模型权重网址：

vit-base：
https://huggingface.co/Junkang2/vit/tree/main

swin_transformer 
https://huggingface.co/Junkang2/swin_transformer/tree/main

## Dataset

数据集下载网址

Tiny-ImageNet：
https://huggingface.co/datasets/Junkang2/Tiny-ImageNet/upload/main

The code supports multiple datasets:

* **CIFAR-10 / CIFAR-100**
* **Tiny-ImageNet**
* **EMNIST**
* **MNIST**

Data will be automatically downloaded to the `./data` directory unless specified via `--datapath`.



## Key Arguments

* `--alg` : Algorithm to use (e.g., `FedAvg`, `FedSWA`, `FedMoSWA`, `SCAFFOLD+`).
* `--lr` : Initial learning rate.
* `--lr_decay` : Decay rate for learning rate scheduling.
* `--epoch` : Total number of training epochs.
* `--E` : Local epochs per communication round.
* `--batch_size` : Training batch size.
* `--alpha_value` : Dirichlet distribution parameter (controls data heterogeneity).
* `--alpha` : Momentum/variance reduction hyperparameter.
* `--gamma` : Server momentum coefficient.
* `--selection` : Fraction of clients selected each round.
* `--CNN` : Model architecture (`resnet18`, `vgg11`, `lenet5`, etc.).
* `--normalization` : Normalization layer (`BN` for BatchNorm, `GN` for GroupNorm).
* `--gpu` : GPU index(es) to use.
* `--num_gpus_per` : Fraction of GPU resources allocated per client.

---

## Algorithm Overview

* **FedSWA**: Incorporates **stochastic weight averaging** and cyclical learning rate schedules to find flatter global minima, outperforming FedAvg and FedSAM on heterogeneous data.
* **FedMoSWA**: Builds on FedSWA with **momentum-based variance reduction**, aligning local and global updates more effectively than SCAFFOLD.

---

## Results

Experiments on **CIFAR-10/100** and **Tiny-ImageNet** show that:

* **FedSWA** achieves better generalization than FedAvg and FedSAM.
* **FedMoSWA** further reduces client drift and improves optimization stability, especially under high heterogeneity.

---

## Citation

If you use this code, please cite the paper:

```
@inproceedings{liu2025fedswa,
  title={Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging},
  author={Liu, Junkang and Liu, Yuanyuan and Shang, Fanhua and Liu, Hongying and Liu, Jin and Feng, Wei},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

---











