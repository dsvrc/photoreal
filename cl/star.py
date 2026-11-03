"""
STAR (Eskandar et al., ICLR 2025) on the replay batch: penalise how much the predictions on
buffered clips change under a small worst-case weight perturbation.

    1. reference = the unperturbed prediction (no gradient)
    2. delta0 = a random perturbation of every weight matrix W, with norm eps * ||W||
    3. one ascent step at theta + delta0: delta = delta0 + gamma * ||W|| * g / ||g||
    4. loss = lam * D(theta + delta); the gradient reaches theta through functional_call

D is the mean squared difference to the reference; it replaces the paper's classifier KL, which
equals it for fixed-variance Gaussian outputs. Every replayed clip counts. All passes share the
time steps, noise, condition-dropout and dropout masks. Defaults are the paper's Table 8 values.
"""

import torch
from torch.func import functional_call

from cl.common import fixed_rng


class STAR:
    def __init__(self, model, lam=0.1, gamma=0.01, eps=0.001):
        self.model, self.lam, self.gamma, self.eps = model, lam, gamma, eps
        self.weights = {n: p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2}

    def loss(self, diffusion, x_start, cond, seed):
        t = torch.randint(0, diffusion.num_timesteps, (len(x_start),), device=x_start.device)
        x_t = diffusion.q_sample(x_start, t)
        steps = torch.as_tensor(diffusion.timestep_map, device=t.device)[t]  # what SpacedDiffusion feeds the model
        kwargs = {"y": cond["y"], "cond_drop_prob": 0.2}  # the training forward's condition dropout

        def predict(params):
            with fixed_rng(seed):
                return functional_call(self.model, params, (x_t, steps), kwargs) if params else self.model(x_t, steps, **kwargs)

        with torch.no_grad():
            reference = predict({})
        delta0 = {}
        for n, W in self.weights.items():
            r = torch.randn_like(W)
            delta0[n] = r * (self.eps * W.detach().norm() / r.norm().clamp_min(1e-12))
        probe = {n: (W.detach() + delta0[n]).requires_grad_(True) for n, W in self.weights.items()}
        divergence = (predict(probe) - reference).pow(2).mean()
        grads = torch.autograd.grad(divergence, list(probe.values()), allow_unused=True)
        perturbed = {}
        for (n, W), g in zip(self.weights.items(), grads):
            delta = delta0[n]
            if g is not None and g.norm() > 0:
                delta = delta + self.gamma * W.detach().norm() * g / g.norm()
            perturbed[n] = W + delta
        return self.lam * (predict(perturbed) - reference).pow(2).mean()
