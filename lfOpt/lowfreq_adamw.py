import math

import torch


class LowFreqAdamW(torch.optim.Optimizer):
    """
    SGD-style optimizer with batch-wise random-walk smoothing of dL/dlogits.

    Usage in training:
        opt = LowFreqAdamW(model.parameters(), lr=1e-3, ...)
        logits = model(x)
        logits.register_hook(lambda g: opt.shape_grad(g, x_embed))
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        sigma=1.0,
        lam=0.5,
        eta=1e-2,
        center=True,
        spectral_cap=True,
        scale_match=True,
        adapt_mode="global",  # "none" | "global" | "local"
        tgt_offdiag=0.1,
        q_percentile=50.0,
        sigma_beta=0.05,
        sigma_min=1e-6,
        sigma_max=1e3,
        k_nn=5,
        base_impl=torch.optim.SGD,
    ):
        # Base optimizer: by default SGD; can be swapped for AdamW if desired.
        self.base = base_impl(params, lr=lr, weight_decay=weight_decay)

        self.sigma0 = float(sigma)
        self.lam = float(lam)
        self.eta = float(eta)
        self.center = bool(center)
        self.spectral_cap = bool(spectral_cap)
        self.scale_match = bool(scale_match)

        self.adapt_mode = str(adapt_mode)
        self.tgt_offdiag = float(tgt_offdiag)
        self.q_percentile = float(q_percentile)
        self.sigma_beta = float(sigma_beta)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.k_nn = int(k_nn)

        self._sigma_ema = None

        super().__init__(self.base.param_groups, self.base.defaults)
        self.state = self.base.state

    @torch.no_grad()
    def pairwise_dist2(self, x):
        """Squared Euclidean distances in flattened input space."""
        b = x.shape[0]
        xf = x.view(b, -1).to(torch.float64)
        g = xf @ xf.t()
        n = g.diag().view(b, 1)
        return (n + n.t() - 2.0 * g).clamp_min(0.0)

    @torch.no_grad()
    def sigma_from_dist2_global(self, dist2):
        """Single sigma adapted from batch pairwise distances."""
        b = dist2.size(0)
        mask = ~torch.eye(b, dtype=torch.bool, device=dist2.device)
        d2 = dist2[mask]
        if d2.numel() > 0:
            q = torch.quantile(d2, self.q_percentile / 100.0).item()
        else:
            q = float(self.sigma0**2)
        q = max(q, 1e-18)

        denom = -2.0 * math.log(max(min(self.tgt_offdiag, 0.999), 1e-6))
        sigma_batch = math.sqrt(q / denom)
        sigma_batch = min(max(sigma_batch, self.sigma_min), self.sigma_max)

        if self._sigma_ema is None:
            self._sigma_ema = sigma_batch
        else:
            self._sigma_ema = (1.0 - self.sigma_beta) * self._sigma_ema + self.sigma_beta * sigma_batch

        return float(self._sigma_ema)

    @torch.no_grad()
    def sigma_from_dist2_local(self, dist2):
        """Per-sample sigma_i from k-NN distances."""
        b = dist2.size(0)
        d = dist2.clamp_min(0.0).sqrt()
        eye = torch.eye(b, device=d.device, dtype=d.dtype)
        topk = torch.topk(d + eye * 1e9, k=self.k_nn, largest=False).values
        s_i = topk.mean(1)
        sigma_i = s_i.clamp_min(self.sigma_min).clamp_max(self.sigma_max)
        return sigma_i

    @torch.no_grad()
    def kernel_from_inputs(self, x):
        """
        Compute random-walk kernel K and the sigma used.
        """
        dist2 = self.pairwise_dist2(x)

        if self.adapt_mode == "none":
            sigma = self.sigma0
            k = torch.exp(-dist2 / (2.0 * (sigma**2)))
        elif self.adapt_mode == "global":
            sigma = self.sigma_from_dist2_global(dist2)
            k = torch.exp(-dist2 / (2.0 * (sigma**2)))
        elif self.adapt_mode == "local":
            sigma_i = self.sigma_from_dist2_local(dist2)
            denom = 2.0 * (sigma_i.view(-1, 1) * sigma_i.view(1, -1))
            k = torch.exp(-(dist2 / denom.clamp_min(1e-18)))
            sigma = sigma_i
        else:
            raise ValueError(f"Unknown adapt_mode={self.adapt_mode}")

        k = 0.5 * (k + k.t())
        return k, sigma

    @torch.no_grad()
    def make_transition_matrix(self, x):
        k, _ = self.kernel_from_inputs(x)
        dcol = k.sum(0).clamp_min(1e-12)
        p = k / dcol.view(1, -1)

        if self.eta > 0.0:
            b = k.size(0)
            p = (1.0 - self.eta) * p + self.eta * (torch.ones_like(p) / b)

        if self.spectral_cap:
            v = torch.randn(p.size(0), 1, device=p.device, dtype=p.dtype)
            v /= v.norm() + 1e-12
            v = p.t() @ (p @ v)
            s = torch.sqrt((v.t() @ v).clamp_min(0)).item()
            if s > 1.0:
                p = p / s

        return p

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def shape_grad(self, grad, x):
        """
        Smooth per-sample gradient `grad` using the random-walk kernel built from `x`.
        """
        b = grad.shape[0]
        r_flat = grad.view(b, -1).to(torch.float64)
        p = self.make_transition_matrix(x)

        if self.center:
            r_bar = r_flat.mean(0, keepdim=True)
            rc = r_flat - r_bar
        else:
            r_bar = None
            rc = r_flat

        eye = torch.eye(b, device=p.device, dtype=p.dtype)
        a = (1.0 - self.lam) * eye + self.lam * p
        r_tilde = a @ rc

        if r_bar is not None:
            r_tilde = r_tilde + r_bar

        if self.scale_match:
            rn = r_flat.norm()
            rtn = r_tilde.norm()
            r_tilde = (rn / (rtn + 1e-12)) * r_tilde

        out = r_tilde.to(dtype=grad.dtype, device=grad.device).view_as(grad)
        return out

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.base.step()
        return loss
