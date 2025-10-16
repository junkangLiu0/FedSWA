import torch
import torch.nn.functional as F


class Nesterov(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, gamma , adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10
        self.gamma=gamma

        defaults = dict(rho=rho, adaptive=adaptive,gamma=gamma ,**kwargs)
        super(Nesterov, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

        # self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            # group["adaptive"] = adaptive
        self.paras = None

    @torch.no_grad()
    def first_step(self, g_update):
        #inputs, labels, loss_func, model = self.paras
        alpha=0.1
        grad_norm = 0
        keys_list = list(g_update.keys())
        #print(keys_list)
        for group in self.param_groups:
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None or p.grad == None:
                    continue
                else:
                    key = keys_list[idx]
                    g_update[key] = g_update[key].to('cuda')
                    #g_update[key]=alpha*p.grad+(1-alpha)*g_update[key]
                    grad_norm += g_update[key].norm(p=2)**2
        grad_norm=grad_norm**0.5



        for group in self.param_groups:
            # if g_update !=None:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx, p in enumerate(group["params"]):
                p.requires_grad = True
                if g_update == None or p.grad == None:
                    continue
                else:
                    key = keys_list[idx]
                    g_update[key] = g_update[key].to('cuda')
                    #e_w = g_update[key] * scale.to(p)
                    key = keys_list[idx]
                    #e_w = g_update[key]*self.gamma-g_update[key] * scale.to(p)
                    e_w = g_update[key] * self.gamma
                p.add_(e_w )
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not self.state[p]:
                    continue
                # go back to "w" from "w + e(w)"
                p.sub_(self.state[p]["e_w"])
                self.state[p]["e_w"] = 0

    def step(self, g_update=None):
        inputs, labels, loss_func, model = self.paras
        self.first_step(g_update)
        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        self.second_step()
