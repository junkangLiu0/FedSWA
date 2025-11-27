import torch  # 导入 PyTorch 库
import torch.nn.functional as F  # 导入 PyTorch 的函数式 API

class SAMC(torch.optim.Optimizer):  # 定义一个继承自 PyTorch 优化器的类 SAMC
    def __init__(self, params, base_optimizer, rho, adaptive=False, **kwargs):  # 初始化方法
        assert rho >= 0.0, f"Invalid perturbation rate, should be non-negative: {rho}"  # 确保 rho 是非负的
        self.max_norm = 10  # 设置最大范数为 10

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)  # 创建默认参数字典
        super(SAMC, self).__init__(params, defaults)  # 调用父类的初始化方法

        self.base_optimizer = base_optimizer  # 设置基础优化器
        self.param_groups = self.base_optimizer.param_groups  # 获取参数组
        # self.g_update=None  # 注释掉的代码，可能用于存储梯度更新
        for group in self.param_groups:  # 遍历每个参数组
            group["rho"] = rho  # 设置每个参数组的 rho 值
            # group["adaptive"] = adaptive  # 注释掉的代码，可能用于设置自适应参数
        self.paras = None  # 初始化 paras 为 None

    @torch.no_grad()  # 装饰器，表示该方法不需要计算梯度
    def first_step(self, g_update):  # 定义 first_step 方法
        keys_list = list(g_update.keys())  # 获取 g_update 的键列表
        # first order sum  # 注释，表示一阶和
        grad_norm = 0  # 初始化梯度范数为 0
        for group in self.param_groups:  # 遍历每个参数组
            for idx, p in enumerate(group["params"]):  # 遍历每个参数
                # print(idx)  # 注释掉的代码，可能用于调试
                p.requires_grad = True  # 设置参数需要计算梯度
                if g_update == None:  # 如果 g_update 为 None
                    continue  # 跳过当前循环
                else:  # 否则
                    key = keys_list[idx]  # 获取当前索引对应的键
                    #grad_norm += g_update[key].norm(p=2)  # 注释掉的代码，可能用于计算梯度范数
                    grad_norm += g_update[key].norm(p=2) ** 2  # 计算梯度的 L2 范数的平方并累加
        grad_norm = grad_norm ** 0.5  # 计算梯度范数的平方根

        # my_dict = {'a': 1, 'b': 2, 'c': 3}  # 注释掉的示例字典
        # index = 1  # 假设我们想要第二个元素  # 注释掉的示例索引
        # key = keys_list[index]  # 注释掉的示例获取键
        # value = my_dict[key]  # 注释掉的示例获取值
        for group in self.param_groups:  # 遍历每个参数组
            # if g_update !=None:  # 注释掉的代码，可能用于检查 g_update
            scale = group["rho"] / (grad_norm + 1e-7)  # 计算缩放因子
            for idx, p in enumerate(group["params"]):  # 遍历每个参数
                # for idx,p in group["params"].items():  # 注释掉的代码，可能用于遍历参数
                p.requires_grad = True  # 设置参数需要计算梯度
                if g_update == None:  # 如果 g_update 为 None
                    continue  # 跳过当前循环
                # original SAM  # 注释，表示原始 SAM
                # e_w = p.grad * scale.to(p)  # 注释掉的代码，可能用于计算扰动
                # ASAM  # 注释，表示自适应 SAM
                # e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)  # 注释掉的代码，可能用于计算自适应扰动
                else:  # 否则
                    key = keys_list[idx]  # 获取当前索引对应的键
                    g_update[key] = g_update[key].to('cuda')  # 将 g_update 的值移动到 CUDA 设备
                    e_w = -g_update[key] * scale.to(p)  # 计算扰动
                # climb to the local maximum "w + e(w)"  # 注释，表示爬升到局部最大值
                p.add_(e_w * 1)  # 更新参数
                self.state[p]["e_w"] = e_w  # 存储扰动

    @torch.no_grad()  # 装饰器，表示该方法不需要计算梯度
    def second_step(self):  # 定义 second_step 方法
        for group in self.param_groups:  # 遍历每个参数组
            for p in group["params"]:  # 遍历每个参数
                if p.grad is None or not self.state[p]:  # 如果参数的梯度为 None 或状态为空
                    continue  # 跳过当前循环
                # go back to "w" from "w + e(w)"  # 注释，表示从 "w + e(w)" 返回到 "w"
                p.sub_(self.state[p]["e_w"])  # 恢复参数
                self.state[p]["e_w"] = 0  # 重置扰动

    def step(self, g_update=None):  # 定义 step 方法
        #inputs, labels, loss_func, model = self.paras  # 注释掉的代码，可能用于存储输入、标签、损失函数和模型

        self.first_step(g_update)  # 调用 first_step 方法

        #predictions = model(inputs)  # 注释掉的代码，可能用于计算预测
        #loss = loss_func(predictions, labels)  # 注释掉的代码，可能用于计算损失
        #self.zero_grad()  # 注释掉的代码，可能用于清零梯度
        #loss.backward()  # 注释掉的代码，可能用于反向传播

        self.second_step()  # 调用 second_step 方法
