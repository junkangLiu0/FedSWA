import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数式API

class LESAM(torch.optim.Optimizer):  # 定义LESAM优化器类，继承自PyTorch的Optimizer
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):  # 初始化方法
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"  # 确保rho是非负的
        self.max_norm = 10  # 设置最大范数为10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)  # 创建默认参数字典
        super(LESAM, self).__init__(params, defaults)  # 调用父类的初始化方法

        self.base_optimizer = base_optimizer  # 保存基础优化器
        self.param_groups = self.base_optimizer.param_groups  # 获取参数组
        #self.g_update=None  # 注释掉的代码，可能用于更新梯度
        for group in self.param_groups:  # 遍历每个参数组
            group["rho"] = rho  # 设置每个参数组的rho值
            #group["adaptive"] = adaptive  # 注释掉的代码，可能用于设置自适应参数
        self.paras = None  # 初始化paras为None

    @torch.no_grad()  # 装饰器，表示该方法不需要计算梯度
    def first_step(self, g_update, model):  # 定义first_step方法
        keys_list = list(g_update.keys())  # 获取g_update的键列表
        grad_norm = 0  # 初始化梯度范数为0
        for group in self.param_groups:  # 遍历每个参数组
            for idx, p in enumerate(group["params"]):  # 遍历参数组中的每个参数
                key = keys_list[idx]  # 获取当前参数对应的键
                if p.requires_grad == False or g_update[key].dtype == torch.long:  # 如果参数不需要梯度或g_update的类型为long
                    continue  # 跳过该参数
                if g_update == None:  # 如果g_update为None
                    continue  # 跳过
                else:
                    key = keys_list[idx]  # 获取当前参数对应的键
                    grad_norm += g_update[key].norm(p=2)**2  # 计算梯度的L2范数的平方并累加
        grad_norm = grad_norm ** 0.5  # 计算梯度范数的平方根
        for group in self.param_groups:  # 遍历每个参数组
            scale = group["rho"] / (grad_norm + 1e-7)  # 计算缩放因子
            for idx, p in enumerate(group["params"]):  # 遍历参数组中的每个参数
                if p.requires_grad == False or g_update[key].dtype == torch.long:  # 如果参数不需要梯度或g_update的类型为long
                    continue  # 跳过该参数
                if g_update == None:  # 如果g_update为None
                    continue  # 跳过
                else:
                    key = keys_list[idx]  # 获取当前参数对应的键
                    g_update[key] = g_update[key].to('cuda')  # 将g_update移动到CUDA设备
                    e_w = -g_update[key] * scale.to(p)  # 计算扰动
                p.add_(e_w * 1)  # 更新参数，爬升到局部最大值
                self.state[p]["e_w"] = e_w  # 保存扰动到状态字典

    @torch.no_grad()  # 装饰器，表示该方法不需要计算梯度
    def second_step(self):  # 定义second_step方法
        for group in self.param_groups:  # 遍历每个参数组
            for p in group["params"]:  # 遍历参数组中的每个参数
                if p.grad is None or not self.state[p]:  # 如果参数的梯度为None或状态字典为空
                    continue  # 跳过该参数
                p.sub_(self.state[p]["e_w"])  # 从“w + e(w)”返回到“w”
                self.state[p]["e_w"] = 0  # 重置扰动为0

    def step(self, g_update=None):  # 定义step方法
        inputs, labels, loss_func, model = self.paras  # 解包参数

        self.first_step(g_update, model)  # 执行第一步

        predictions = model(inputs)  # 获取模型预测
        loss = loss_func(predictions, labels)  # 计算损失
        self.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播计算梯度

        self.second_step()  # 执行第二步
