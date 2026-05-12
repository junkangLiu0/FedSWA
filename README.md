

# FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data

<p align="center">
  <b>Momentum-Based Stochastic Controlled Weight Averaging for Robust Federated Generalization</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ICML-2025-blue">
  <img src="https://img.shields.io/badge/Federated%20Learning-Non--IID-orange">
  <img src="https://img.shields.io/badge/Backbone-ResNet%20%7C%20ViT%20%7C%20Swin-green">
  <img src="https://img.shields.io/badge/Training-Ray-lightgrey">
</p>

This repository contains the official implementation of **FedSWA** and **FedMoSWA**, two federated optimization algorithms designed to improve generalization under highly heterogeneous client data.

The central observation is simple yet important: in highly non-IID federated learning, local sharpness-aware methods such as FedSAM may find locally flat minima that are not aligned with the global objective. FedSWA instead promotes flatter global solutions through stochastic weight averaging, while FedMoSWA further introduces momentum-based stochastic control to align local and global update directions.

> **Paper**: *Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging*  
> **Venue**: ICML 2025  
> **Methods**: FedSWA, FedMoSWA  
> **Tasks**: CIFAR-10, CIFAR-100, Tiny-ImageNet  
> **Backbones**: LeNet-5, ResNet, ViT, Swin Transformer, DeiT  
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






---

## Highlights

- **Global flatness for federated generalization**  
  FedSWA revisits stochastic weight averaging in FL and targets globally flat minima instead of purely local sharpness reduction.

- **Momentum-based stochastic control**  
  FedMoSWA introduces a server-client control mechanism to reduce client drift and better align local updates with the global trajectory.

- **Strong performance under high heterogeneity**  
  The method is designed for challenging Dirichlet non-IID splits, especially `alpha_value = 0.1`.

- **Broad algorithm zoo**  
  The codebase includes FedAvg, SCAFFOLD, FedSAM, MoFedSAM, FedProx, MOON, FedCM, FedACG, FedNSAM, FedMoment, FedNesterov, FedSWA, and MoFedSWA.

- **Transformer fine-tuning support**  
  ViT and Swin backbones support LoRA-based parameter-efficient federated fine-tuning.

---

## Installation

### 1. Create environment

```bash
conda create -n fedswa python=3.8 -y
conda activate fedswa
````

### 2. Install dependencies

```bash
pip install torch==2.4.1 torchvision==0.19.1 numpy ray==1.0.0 tensorboardX==2.6.2.2 peft==0.13.2 transformers==4.46.3
```

Or, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

### 3. Optional: use a faster pip mirror

```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
EOF
```

### 4. Create runtime folders

```bash
mkdir -p data log checkpoint
```

---

## Datasets

The current training script directly supports:

| Dataset flag | Dataset       | Notes                                  |
| ------------ | ------------- | -------------------------------------- |
| `CIFAR10`    | CIFAR-10      | Automatically downloaded to `./data`   |
| `CIFAR100`   | CIFAR-100     | Automatically downloaded to `./data`   |
| `imagenet`   | Tiny-ImageNet | Expected at `./data/tiny-imagenet-200` |

Tiny-ImageNet should be organized as:

```text
data/
└── tiny-imagenet-200/
    ├── train/
    ├── val/
    └── test/
```

---

## Non-IID Data Partition

The script uses Dirichlet partitioning to simulate statistical heterogeneity across clients.

| Argument            | Meaning                                                                        |
| ------------------- | ------------------------------------------------------------------------------ |
| `--num_workers`     | Total number of federated clients                                              |
| `--selection`       | Fraction of clients sampled per round                                          |
| `--alpha_value`     | Dirichlet concentration parameter                                              |
| `--alpha_value 0.1` | Highly heterogeneous setting                                                   |
| `--alpha_value 0.3` | Moderately heterogeneous setting                                               |
| `--alpha_value 0.6` | Mildly heterogeneous setting                                                   |
| `--alpha_value 1`   | IID-like setting, but this version may require a prepared `data_idx.data` file |

Example:

```bash
--num_workers 100 --selection 0.1 --alpha_value 0.1
```

This means 100 total clients, 10 participating clients per round, and a highly non-IID split.

---

## Model Zoo

| Backbone flag | Model      |
| ------------- | ---------- |
| `lenet5`      | LeNet-5    |
| `resnet10`    | ResNet-10  |
| `resnet18`    | ResNet-18  |
| `resnet34`    | ResNet-34  |
| `resnet50`    | ResNet-50  |
| `deit_tiny`   | DeiT-Tiny  |
| `VIT-B`       | ViT-Base   |
| `VIT-L`       | ViT-Large  |
| `swin_tiny`   | Swin-Tiny  |
| `swin_small`  | Swin-Small |
| `swin_base`   | Swin-Base  |
| `swin_large`  | Swin-Large |

For ResNet backbones, the normalization layer can be selected by:

```bash
--normalization BN
```

or

```bash
--normalization GN
```

In most reproduced ResNet-18 experiments, `BN` is recommended.

---

## Algorithm Zoo

| Paper / Method name                |      Code flag | Status                                             |
| ---------------------------------- | -------------: | -------------------------------------------------- |
| FedAvg                             |       `FedAvg` | Stable                                             |
| SCAFFOLD                           |     `SCAFFOLD` | Stable                                             |
| FedAdam                            |      `FedAdam` | Stable                                             |
| FedCM                              |        `FedCM` | Stable                                             |
| FedAvg with AdamW local optimizer  | `FedAvg_adamw` | Requires `beta1/beta2` parser check                |
| FedAdamW-style federated optimizer |     `FedAdamW` | Experimental                                       |
| FedSAM                             |       `FedSAM` | Stable if model supports `get_weights/set_weights` |
| MoFedSAM                           |     `FedMoSAM` | Stable if model supports `get_weights/set_weights` |
| FedSWA                             |       `FedSWA` | Proposed method                                    |
| FedMoSWA / MoFedSWA                |     `MoFedSWA` | Proposed method                                    |
| FedProx                            |      `Fedprox` | Stable                                             |
| MOON                               |         `Moon` | Stable                                             |
| IGFL                               |         `IGFL` | Stable                                             |
| FedNSAM                            |      `FedNSAM` | Stable                                             |
| FedACG                             |       `FedACG` | Stable                                             |
| FedMoment                          |    `FedMoment` | Stable                                             |
| FedNesterov                        |  `FedNesterov` | Stable                                             |
| FedLADA                            |      `FedLADA` | Declared but mapping incomplete                    |
| FedMuon                            |      `FedMuon` | Declared but mapping incomplete                    |
| Local Muon                         |   `Local_Muon` | Declared but mapping incomplete                    |

> Naming note: the paper uses **FedMoSWA**, while the current code uses the command-line flag **`MoFedSWA`**.

---

## Main Arguments

| Argument          | Description                                       |
| ----------------- | ------------------------------------------------- |
| `--alg`           | Federated algorithm                               |
| `--lr`            | Initial learning rate                             |
| `--epoch`         | Number of communication rounds                    |
| `--E`             | Local epochs per round                            |
| `--K`             | Maximum local iterations per client per round     |
| `--batch_size`    | Local batch size                                  |
| `--data_name`     | Dataset name                                      |
| `--CNN`           | Backbone architecture                             |
| `--alpha_value`   | Dirichlet non-IID coefficient                     |
| `--selection`     | Client participation ratio                        |
| `--num_workers`   | Total number of clients                           |
| `--gpu`           | GPU index                                         |
| `--num_gpus_per`  | Ray GPU fraction per client worker                |
| `--normalization` | `BN` or `GN` for ResNet                           |
| `--rho`           | SAM/SWA perturbation or decay-related coefficient |
| `--gamma`         | Server momentum/control coefficient               |
| `--lora`          | Whether to use LoRA fine-tuning                   |
| `--r`             | LoRA rank                                         |
| `--pix`           | Input image resolution                            |
| `--preprint`      | Evaluation interval                               |
| `--extname`       | Extra experiment name                             |

---

# Quick Start

## FedSWA on CIFAR-100 with ResNet-18

```bash
python main_FedSWA.py --alg FedSWA --lr 2e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedSWA_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## FedMoSWA on CIFAR-100 with ResNet-18

```bash
python main_FedSWA.py --alg MoFedSWA --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname MoFedSWA_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## FedAvg baseline

```bash
python main_FedSWA.py --alg FedAvg --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedAvg_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

> In the paper-style ResNet-18 setting, FedSWA often uses a larger learning rate than FedAvg, e.g., `2e-1` for FedSWA and `1e-1` for FedAvg.

---

# Full Command Zoo

The following commands use a unified setting:

```text
Dataset: CIFAR-100
Backbone: ResNet-18
Clients: 100
Participation: 10%
Non-IID: Dirichlet alpha = 0.1
Local epochs: 5
Local steps: 50
Batch size: 50
GPU: 0
```

## 1. FedAvg

```bash
python main_FedSWA.py --alg FedAvg --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedAvg_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 2. SCAFFOLD

```bash
python main_FedSWA.py --alg SCAFFOLD --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname SCAFFOLD_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 3. FedAdam

```bash
python main_FedSWA.py --alg FedAdam --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedAdam_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 4. FedCM

```bash
python main_FedSWA.py --alg FedCM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedCM_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 5. FedAvg-AdamW

```bash
python main_FedSWA.py --alg FedAvg_adamw --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.01 --epoch 301 --extname FedAvgAdamW_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

> Code note: this branch references `args.beta1` and `args.beta2`. If your current parser does not define them, add `--beta1` and `--beta2` to the parser or set them directly in the code.

## 6. FedAdamW

```bash
python main_FedSWA.py --alg FedAdamW --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.01 --epoch 301 --extname FedAdamW_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 7. FedSAM

```bash
python main_FedSWA.py --alg FedSAM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedSAM_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.05 --pix 32 --lora 0 --K 50 --freeze 1
```

## 8. MoFedSAM

```bash
python main_FedSWA.py --alg FedMoSAM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname MoFedSAM_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.05 --pix 32 --lora 0 --K 50 --freeze 1
```

## 9. FedSWA

```bash
python main_FedSWA.py --alg FedSWA --lr 2e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedSWA_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 10. FedMoSWA / MoFedSWA

```bash
python main_FedSWA.py --alg MoFedSWA --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname MoFedSWA_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 11. FedProx

```bash
python main_FedSWA.py --alg Fedprox --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedProx_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 12. MOON

```bash
python main_FedSWA.py --alg Moon --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname MOON_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 13. IGFL

```bash
python main_FedSWA.py --alg IGFL --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname IGFL_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.9 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 14. FedNSAM

```bash
python main_FedSWA.py --alg FedNSAM --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedNSAM_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.05 --pix 32 --lora 0 --K 50 --freeze 1
```

## 15. FedACG

```bash
python main_FedSWA.py --alg FedACG --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedACG_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 16. FedMoment

```bash
python main_FedSWA.py --alg FedMoment --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedMoment_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.9 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## 17. FedNesterov

```bash
python main_FedSWA.py --alg FedNesterov --lr 1e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedNesterov_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

---

# Declared but Experimental Commands

The following algorithms are listed in the code-level `assert alg` set, but the current pasted script does not provide complete client-side mappings for them. Keep them as experimental entries until the corresponding `func_dict` mapping and update functions are completed.

## FedLADA

```bash
python main_FedSWA.py --alg FedLADA --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.01 --epoch 301 --extname FedLADA_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## FedMuon

```bash
python main_FedSWA.py --alg FedMuon --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.01 --epoch 301 --extname FedMuon_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## Local Muon

```bash
python main_FedSWA.py --alg Local_Muon --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 0.01 --epoch 301 --extname LocalMuon_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.5 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

---

# Transformer / LoRA Fine-Tuning

The code supports LoRA fine-tuning for ViT and Swin backbones.

## FedSWA with ViT-Base LoRA

```bash
python main_FedSWA.py --alg FedSWA --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 101 --extname FedSWA_CIFAR100_ViTB_LoRA_a01 --lr_decay 2 --gamma 0.85 --CNN VIT-B --E 5 --batch_size 16 --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --num_workers 50 --preprint 10 --rho 0.01 --pix 224 --lora 1 --r 16 --K 50 --freeze 1
```

## MoFedSWA with ViT-Base LoRA

```bash
python main_FedSWA.py --alg MoFedSWA --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 101 --extname MoFedSWA_CIFAR100_ViTB_LoRA_a01 --lr_decay 2 --gamma 0.5 --CNN VIT-B --E 5 --batch_size 16 --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --num_workers 50 --preprint 10 --rho 0.01 --pix 224 --lora 1 --r 16 --K 50 --freeze 1
```

## FedAvg with ViT-Base LoRA

```bash
python main_FedSWA.py --alg FedAvg --lr 1e-3 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 101 --extname FedAvg_CIFAR100_ViTB_LoRA_a01 --lr_decay 2 --gamma 0.5 --CNN VIT-B --E 5 --batch_size 16 --gpu 0 --p 1 --num_gpus_per 0.2 --normalization BN --selection 0.1 --print 0 --num_workers 50 --preprint 10 --rho 0.01 --pix 224 --lora 1 --r 16 --K 50 --freeze 1
```

> Implementation note: if the script reports `AttributeError: 'Namespace' object has no attribute 'freeze_layers'`, replace `args.freeze_layers` with `args.freeze` in the ViT/Swin loading blocks, or add `parser.add_argument("--freeze_layers", default=1, type=int)`.

---

# Recommended Reproduction Settings

## CIFAR-100, ResNet-18, highly heterogeneous

```bash
python main_FedSWA.py --alg FedSWA --lr 2e-1 --data_name CIFAR100 --alpha_value 0.1 --alpha 10 --epoch 301 --extname FedSWA_CIFAR100_R18_a01 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## CIFAR-100, ResNet-18, moderately heterogeneous

```bash
python main_FedSWA.py --alg FedSWA --lr 2e-1 --data_name CIFAR100 --alpha_value 0.3 --alpha 10 --epoch 301 --extname FedSWA_CIFAR100_R18_a03 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

## CIFAR-100, ResNet-18, mildly heterogeneous

```bash
python main_FedSWA.py --alg FedSWA --lr 2e-1 --data_name CIFAR100 --alpha_value 0.6 --alpha 10 --epoch 301 --extname FedSWA_CIFAR100_R18_a06 --lr_decay 2 --gamma 0.85 --CNN resnet18 --E 5 --batch_size 50 --gpu 0 --p 1 --num_gpus_per 0.1 --normalization BN --selection 0.1 --print 0 --num_workers 100 --preprint 10 --rho 0.01 --pix 32 --lora 0 --K 50 --freeze 1
```

---

# Logging and Outputs

The script writes logs to:

```text
./log/
```

and uses TensorBoardX with the algorithm name as the run comment.

To monitor training:

```bash
tensorboard --logdir runs
```

The test accuracy is evaluated every `--preprint` rounds.

Example:

```bash
--preprint 10
```

means the global model is evaluated every 10 communication rounds.

---

# Practical Notes

## GPU allocation

The argument:

```bash
--num_gpus_per 0.1
```

means each Ray worker is allocated 0.1 GPU. With 10 selected clients per round, this approximately uses one full GPU.

For one 24GB GPU, a common ResNet-18 setting is:

```bash
--selection 0.1 --num_workers 100 --num_gpus_per 0.1 --batch_size 50
```

For ViT/Swin LoRA fine-tuning, reduce the batch size and increase the GPU fraction if needed:

```bash
--batch_size 16 --num_gpus_per 0.2 --num_workers 50
```

## Learning rate

Recommended starting points:

| Method        | Suggested LR |
| ------------- | -----------: |
| FedAvg        |       `1e-1` |
| FedSWA        |       `2e-1` |
| MoFedSWA      |       `1e-1` |
| FedSAM        |       `1e-1` |
| ViT/Swin LoRA |       `1e-3` |

## Resolution

For ResNet on CIFAR:

```bash
--pix 32
```

For ViT/Swin:

```bash
--pix 224
```

---

# Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{liu2025fedswa,
  title={Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging},
  author={Liu, Junkang and Liu, Yuanyuan and Shang, Fanhua and Liu, Hongying and Liu, Jin and Feng, Wei},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

---

# Acknowledgement

This project builds on the broader literature of federated optimization, sharpness-aware minimization, stochastic weight averaging, and control-variate methods for non-IID federated learning.

FedSWA and FedMoSWA are designed to make federated training not only converge, but generalize.

```
```










