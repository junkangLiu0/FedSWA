import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数式API

class SAM(torch.optim.Optimizer):  # 定义一个继承自PyTorch优化器的类SAM
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):  # 初始化方法
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"  # 确保rho是非负的
        self.max_norm = 10  # 设置最大范数为10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)  # 创建默认参数字典
        super(SAM, self).__init__(params, defaults)  # 调用父类的初始化方法

        self.base_optimizer = base_optimizer  # 保存基础优化器
        self.param_groups = self.base_optimizer.param_groups  # 获取参数组
        #self.g_update=None  # 注释掉的代码，可能用于更新梯度
        for group in self.param_groups:  # 遍历每个参数组
            group["rho"] = rho  # 设置每个参数组的rho值
            #group["adaptive"] = adaptive  # 注释掉的代码，可能用于设置自适应参数
        self.paras = None  # 初始化paras为None

    @torch.no_grad()  # 装饰器，表示该方法不需要计算梯度
    def first_step(self):  # 定义first_step方法
        grad_norm = 0  # 初始化梯度范数为0
        for group in self.param_groups:  # 遍历每个参数组
            for idx, p in enumerate(group["params"]):  # 遍历参数组中的每个参数
                p.requires_grad = True  # 设置参数需要梯度
                if p.grad is None:  # 如果参数的梯度为None
                    continue  # 跳过该参数
                    grad_norm += p.grad.norm(p=2)  # 计算梯度的L2范数并累加

        for group in self.param_groups:  # 遍历每个参数组
            scale = group["rho"] / (grad_norm + 1e-7)  # 计算缩放因子
            for idx, p in enumerate(group["params"]):  # 遍历参数组中的每个参数
                p.requires_grad = True  # 设置参数需要梯度
                if p.grad is None:  # 如果参数的梯度为None
                    continue  # 跳过该参数
                e_w = p.grad * scale.to(p)  # 计算扰动
                p.add_(e_w * 1)  # 更新参数，爬升到局部最大值
                self.state[p]["e_w"] = e_w  # 保存扰动到状态字典

    @torch.no_grad()  # 装饰器，表示该方法不需要计算梯度
    def second_step(self):  # 定义second_step方法
        for group in self.param_groups:  # 遍历每个参数组
            for p in group["params"]:  # 遍历参数组中的每个参数
                if p.grad is None or not self.state[p]:  # 如果参数的梯度为None或状态字典为空
                    continue  # 跳过该参数
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"
        self.max_norm = 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        #self.g_update=None
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


    def step(self,g_update=None):
        inputs, labels, loss_func, model = self.paras
        
        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()

        self.first_step(g_update)

        predictions = model(inputs)
        loss = loss_func(predictions, labels)
        self.zero_grad()
        loss.backward()
        self.second_step()