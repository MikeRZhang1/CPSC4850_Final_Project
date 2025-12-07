import torch
from torch.optim import Optimizer

class MomentumSGD_Strong_Wolfe(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, c1=1e-4, c2=0.9, 
                 alpha_max=0.1, max_ls_iter=100):
        defaults = dict(lr=lr, momentum=momentum, c1=c1, c2=c2,
                        alpha_max=alpha_max, max_ls_iter=max_ls_iter)
        super().__init__(params, defaults)
    
    def _gather_flat_params(self):
        return torch.cat([p.data.view(-1) for group in self.param_groups for p in group['params']]) 
    
    def _gather_flat_grad(self):
        return torch.cat([p.grad.view(-1) for group in self.param_groups for p in group['params']])
    
    def _set_params_from_flat(self, flat):
        idx = 0
        for group in self.param_groups:
            for p in group['params']:
                numel = p.numel()
                p.data.copy_(flat[idx:idx+numel].view_as(p))
                idx += numel

    def _phi(self, closure, x0, d, alpha):
        self._set_params_from_flat(x0 + alpha * d)
        loss = closure()
        g = self._gather_flat_grad()
        return loss.item(), g
    
    def _phi_derivative(self, grad, d):
        return torch.dot(grad, d).item()

    def _zoom(self, closure, x0, d, alo, ahi, phi0, der0, c1, c2, max_ls_iter):
        for _ in range(max_ls_iter):
            aj = 0.5 * (alo + ahi)
            phi_j, g_j = self._phi(closure, x0, d, aj)
            if phi_j > phi0 + c1 * aj * der0:
                ahi = aj
            else:
                der_j = self._phi_derivative(g_j, d)
                if abs(der_j) <= -c2 * der0:
                    return aj
                if der_j * (ahi - alo) >= 0:
                    ahi = alo
                alo = aj
        return aj

    # Strong Wolfe line search
    def _strong_wolfe(self, closure, x0, d, phi0, der0, 
                      c1, c2, alpha1, alpha_max, max_ls_iter):
        alpha_prev = 0.0
        phi_prev = phi0

        alpha = alpha1
        for _ in range(max_ls_iter):
            phi_a, g_a = self._phi(closure, x0, d, alpha)

            if (phi_a > phi0 + c1 * alpha * der0) or \
               (phi_a >= phi_prev and _ > 0):
                return self._zoom(closure, x0, d,
                                  alpha_prev, alpha,
                                  phi0, der0, c1, c2, max_ls_iter)

            der_a = self._phi_derivative(g_a, d)

            if abs(der_a) <= -c2 * der0:
                return alpha

            if der_a >= 0:
                return self._zoom(closure, x0, d,
                                  alpha, alpha_prev,
                                  phi0, der0, c1, c2, max_ls_iter)

            alpha_prev = alpha
            phi_prev = phi_a
            alpha = 0.5 * (alpha + alpha_max)

        return alpha

    def step(self, closure):
        loss = closure()
        g = self._gather_flat_grad()

        for group in self.param_groups:
            momentum = group['momentum']
            c1, c2 = group['c1'], group['c2']
            alpha_max = group['alpha_max']
            alpha1 = group['lr']
            max_ls_iter = group['max_ls_iter']

        g = self._gather_flat_grad()

        if 'momentum_buffer' not in self.state:
            self.state['momentum_buffer'] = torch.zeros_like(g)
        self.state['momentum_buffer'].mul_(momentum).add_(g, alpha=-1.0)

        d = self.state['momentum_buffer']

        x0 = self._gather_flat_params()
        phi0 = loss.item()
        der0 = torch.dot(g, d).item()
        alpha = self._strong_wolfe(
            closure, x0, d, phi0, der0,
            c1, c2, alpha1, alpha_max, max_ls_iter
        )
        new_x = x0 + alpha * d
        self._set_params_from_flat(new_x)

        return loss

# class Momentum_Trust_Region():

# class Adam_Strong_Wolfe():

# class Adam_Trust_Region():