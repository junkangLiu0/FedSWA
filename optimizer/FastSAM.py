import torch


class FastSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.params = list(params)
        self.rho = rho
        self.defaults = kwargs
        super().__init__(self.params, self.defaults)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "FastSAM requires closure"

        # 1. Forward and backward (standard)
        loss = closure()

        # 2. Compute perturbation
        grad_norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for p in self.params if p.grad is not None
            ])
        ) + 1e-12

        scale = self.rho / grad_norm
        for p in self.params:
            if p.grad is None:
                continue
            e = p.grad * scale
            p.add_(e)  # Apply perturbation temporarily

        # 3. Reuse the same gradient as approximation (no second backward)
        self.base_optimizer.step()

        # 4. Remove perturbation
        for p in self.params:
            if p.grad is None:
                continue
            e = p.grad * scale
            p.sub_(e)

        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()
