import torch
import torch.nn.functional as F


class LESAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(LESAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        #self.g_update=None
        for group in self.param_groups:
            group["rho"] = rho
            #group["adaptive"] = adaptive
        self.paras = None
        

    @torch.no_grad()
    def first_step(self,g_update,model):
        keys_list = list(g_update.keys())
        #first order sum 
        grad_norm = 0
        for group in self.param_groups:
            for idx,p in enumerate(group["params"]):
                #print(idx)
                key = keys_list[idx]
                if p.requires_grad == False or g_update[key].dtype == torch.long:
                    continue
                if g_update ==None: 
                    continue
                else:
                    key = keys_list[idx]
                    #grad_norm += g_update[key].float().norm(p=2)
                    #grad_norm += g_update[key].norm(p=2)
                    grad_norm+=g_update[key].norm(p=2)**2
        grad_norm = grad_norm ** 0.5
        #my_dict = {'a': 1, 'b': 2, 'c': 3}
        #index = 1  # 假设我们想要第二个元素
        #key = keys_list[index]
        #value = my_dict[key]
        for group in self.param_groups:
            #if g_update !=None: 
            scale = group["rho"] / (grad_norm + 1e-7)
            for idx,p in enumerate(group["params"]):
            #for idx,p in group["params"].items():
                if p.requires_grad == False or g_update[key].dtype == torch.long:
                    continue
                if g_update ==None: 
                    continue
                # original SAM 
                # e_w = p.grad * scale.to(p)
                # ASAM
                #e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                else:
                    key=keys_list[idx]
                    g_update[key]=g_update[key].to('cuda')
                    e_w=-g_update[key] * scale.to(p)
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


    def step(self,g_update=None):
        inputs, labels, loss_func, model = self.paras


        self.first_step(g_update,model)

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.second_step()
