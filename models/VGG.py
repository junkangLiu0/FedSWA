import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
def blockVGG(covLayerNum, inputChannel, outputChannel, kernelSize, withFinalCov1: bool):
    layer = nn.Sequential()
    layer.add_module('conv2D1', nn.Conv2d(inputChannel, outputChannel, kernelSize, padding=1))
    layer.add_module('relu-1', nn.ReLU())
    for i in range(covLayerNum - 1):
        layer.add_module('conv2D{}'.format(i), nn.Conv2d(outputChannel, outputChannel, kernelSize, padding=1))
        layer.add_module('relu{}'.format(i), nn.ReLU())
    if withFinalCov1:
        layer.add_module('Conv2dOne', nn.Conv2d(outputChannel, outputChannel, 1))
    layer.add_module('max-pool', nn.MaxPool2d(2, 2))
    return layer

class VGG11_10(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = blockVGG(1, 3, 64, 3, False)

        self.layer2 = blockVGG(1, 64, 128, 3, False)

        self.layer3 = blockVGG(2, 128, 256, 3, False)

        self.layer4 = blockVGG(2, 256, 512, 3, False)

        self.layer5 = blockVGG(2, 512, 512, 3, False)
        self.layer6 = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            # nn.ReLU(),
            # nn.Softmax(1)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)  # 执行卷积神经网络部分
        x = self.layer2(x)  # 执行全连接部分
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
        self.layer1 = blockVGG(1, 3, 64, 3, False)

        self.layer2 = blockVGG(1, 64, 128, 3, False)

        self.layer3 = blockVGG(2, 128, 256, 3, False)

        self.layer4 = blockVGG(2, 256, 512, 3, False)

        self.layer5 = blockVGG(2, 512, 512, 3, False)
        self.layer6 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            # nn.ReLU(),
            # nn.Softmax(1)
        )

    def forward(self, x: torch.Tensor):
        x = self.layer1(x)  # 执行卷积神经网络部分
        x = self.layer2(x)  # 执行全连接部分
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