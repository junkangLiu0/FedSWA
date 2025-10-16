import torch
import torch.nn.functional as F

from utils import *
class GAMASAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(GAMASAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            #group["adaptive"] = adaptive
        self.paras = None
        

    @torch.no_grad()
    def first_step(self):
        #first order sum 
        grad_norm = 0
        for group in self.param_groups:
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                grad_norm+=p.grad.norm(p=2)

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx,p in enumerate(group["params"]):
                p.requires_grad = True 
                if p.grad is None: 
                    continue
                # original SAM 
                # e_w = p.grad * scale.to(p)
                # ASAM 
                
                e_w=p.grad * scale.to(p)
                # climb to the local maximum "w + e(w)"
                p.add_(e_w * 1)  
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
    def step(self,):

        inputs, labels, loss_func, model,delta_list = self.paras
        param_list = param_to_vector(model)

        predictions = model(inputs)
        loss = torch.nn.functional.cross_entropy(predictions,labels,reduction='mean')
        self.zero_grad()
        loss.backward()

        self.first_step()

        param_list = param_to_vector(model)
        predictions = model(inputs)
        loss = loss_func(predictions, labels,param_list,delta_list)
        self.zero_grad()
        loss.backward()

        self.second_step()
