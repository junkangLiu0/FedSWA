import math
import torch
from torch.optim.optimizer import Optimizer


# from megatron.optimizer.l2_norm import l2_norm

def exists(val):
    return val is not None



class LayerWiseAdamW(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(LayerWiseAdamW, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        #total_lr = 0
        #l = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr']
            weight_decay = group['weight_decay']
            for p in group['params']:
                grad = p.grad
                if p.grad is None:
                    continue
                state = self.state[p]
                # Initialize state
                if 'step_1' not in state:
                    state['step_1'] = 0
                    state['step_2'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.tensor(0., device=p.device)
                    #state['exp_avg_sq_1'] = torch.tensor(0., device=p.device)
                state['step_1'] += 1
                state['step_2'] += 1
                mean_sq_grad = grad.pow(2).mean()
                state['exp_avg_sq'].mul_(beta2).add_(mean_sq_grad * (1 - beta2))
                #state['exp_avg_sq_1'] = torch.max(state['exp_avg_sq_1'], state['exp_avg_sq'])
                # 更新一阶动量
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)

                # 权重衰减
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                # 偏差修正
                bias_correction1 = 1 - beta1 ** state['step_1']
                bias_correction2 = 1 - beta2 ** state['step_2']
                #denom = (state['exp_avg_sq_1'] / bias_correction2).sqrt().add_(eps)
                denom = (state['exp_avg_sq'] / bias_correction2).sqrt().add_(eps)
                step_size = lr / bias_correction1
                # Parameter update
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)
                #l=l+1
                #total_lr=total_lr+lr/denom
        #if state['step_1']==1 or state['step_1']==40:
        #    print(f"Average learning rate for all layers: {total_lr/l}")
        return loss



