

# README

## Improving Generalization in Federated Learning with Highly Heterogeneous Data

This repository contains the implementation of **FedSWA** and **FedMoSWA**, federated learning algorithms designed to improve generalization in the presence of highly heterogeneous client data. The work is based on the paper:

> *Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging (FedSWA & FedMoSWA)*.

Both algorithms extend upon **SCAFFOLD** and **SAM-type methods**, aiming to find **global flat minima** that lead to better test performance compared to FedAvg, FedSAM, and related approaches.

---

## Requirements

* Python 3.8+
* PyTorch
* torchvision
* numpy
* matplotlib
* tensorboardX
* ray
* filelock

You can install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## Dataset

The code supports multiple datasets:

* **CIFAR-10 / CIFAR-100**
* **Tiny-ImageNet**
* **EMNIST**
* **MNIST**

Data will be automatically downloaded to the `./data` directory unless specified via `--datapath`.

---

## Usage

To run the training with **SCAFFOLD+** on **CIFAR100** using a **ResNet-18** backbone and Group Normalization:

```bash
python main.py \
  --alg SCAFFOLD+ \
  --lr 0.1 \
  --data_name CIFAR100 \
  --alpha_value 0.6 \
  --alpha 0.1 \
  --epoch 1001 \
  --extname CIFAR100 \
  --lr_decay 0.998 \
  --gamma 0.2 \
  --CNN resnet18 \
  --E 5 \
  --batch_size 50 \
  --gpu 0 \
  --p 1 \
  --num_gpus_per 0.1 \
  --normalization GN \
  --selection 0.1 \
  --print 0
```

---

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


