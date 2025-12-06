# low_freq_adam_rw.py
import math
import torch


class LowFreqAdam(torch.optim.Optimizer):
    """
    Adam/AdamW with batch-wise random-walk smoothing of dL/dlogits.
    Mean-preserving, column-stochastic P with teleportation ensures unbiasedness;
    centering + spectral cap give strict variance contraction on non-constant modes.
    Use via logits.register_hook(lambda g: opt.shape_grad(g, x_embed)).
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        m=32,
        sigma=1.0,
        lam=0.5,
        scale_match=True,
        eta=1e-2,                 # teleportation strength (PageRank)
        center=True,              # subtract/add back batch mean before/after smoothing
        spectral_cap=True,        # cap ||P||_2 to <= 1
        adam_impl=torch.optim.AdamW,
    ):
        self.base = adam_impl(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.m       = int(m)
        self.sigma   = float(sigma)
        self.lam     = float(lam)
        self.scale_match  = bool(scale_match)
        self.eta     = float(eta)
        self.center  = bool(center)
        self.spectral_cap = bool(spectral_cap)

        self.W = None            # [m, D_flat] projection for building batch kernel features
        self.clip_grad_norm = None
        self.ddp_model = None
        self.debug = False
        super().__init__(self.base.param_groups, self.base.defaults)
        self.state = self.base.state

    @torch.no_grad()
    def _init_W(self, xf):
        # xf: [B, D]
        B, D = xf.shape
        Xc = xf - xf.mean(0, keepdim=True)
        # SVD in float32 for speed, promote later
        U, S, Vh = torch.linalg.svd(Xc.to(torch.float32), full_matrices=False)
        k = min(self.m, Vh.size(0))
        Wp = Vh[:k].to(dtype=torch.float64, device=xf.device)  # [k, D]
        if k < self.m:
            extra = torch.randn(self.m - k, D, device=xf.device, dtype=torch.float64) / math.sqrt(D)
            Wp = torch.cat([Wp, extra], dim=0)
        # unit-variance rows w.r.t. current data
        std = (Xc.to(torch.float64) @ Wp.t()).std(0).clamp_min(1e-3)
        self.W = (Wp / std[:, None]).contiguous()  # [m, D], float64

    @torch.no_grad()
    def _make_P_rw(self, x):
        """
        Build column-stochastic, mean-preserving random-walk matrix P in float64.
        x: arbitrary tensor with batch dimension first -> flattened to [B, D].
        """
        B = x.shape[0]
        D = x[0].numel()
        xf = x.view(B, D).to(torch.float64)

        if self.W is None or self.W.size(1) != D or self.W.size(0) != self.m:
            self._init_W(xf)

        # periodic features theta = 2π X W^T, wrapped to [-π,π)
        theta = 2 * math.pi * (xf @ self.W.t())                 # [B, m]
        d = theta[:, None, :] - theta[None, :, :]               # [B, B, m]
        d = (d + math.pi) % (2 * math.pi) - math.pi

        # symmetric positive kernel
        K = torch.exp(-(d.pow(2).sum(-1)) / (2.0 * self.sigma ** 2))  # [B, B]
        K = 0.5 * (K + K.t())

        # column-stochastic P = K D^{-1}
        dcol = K.sum(0).clamp_min(1e-12)
        P = K / dcol.view(1, -1)

        # teleportation to make primitive (strict spectral gap) and preserve 1^T P = 1^T
        if self.eta > 0.0:
            P = (1.0 - self.eta) * P + self.eta * (torch.ones_like(P) / B)

        # optional spectral cap: scale so ||P||_2 <= 1 (2 power iterations)
        if self.spectral_cap:
            v = torch.randn(B, 1, device=P.device, dtype=P.dtype)
            v /= v.norm() + 1e-12
            v = P.t() @ (P @ v)
            v /= v.norm() + 1e-12
            s = torch.sqrt((v.t() @ (P.t() @ (P @ v))).clamp_min(0)).item()
            if s > 1.0:
                P = P / s

        return P  # float64

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def shape_grad(self, grad, x):
        """
        Transform dL/dlogits -> batch-smoothed gradient.
        grad: [B, ...] tensor of same shape as logits.
        x   : any embedding with leading dim B used to build the batch kernel.
        """
        B = grad.shape[0]
        r_flat = grad.view(B, -1)                                  # [B, C_eff]
        dtype_out = grad.dtype
        P = self._make_P_rw(x).to(dtype=r_flat.dtype, device=r_flat.device)  # [B,B]

        # mean-preserving affine smoothing on centered component
        if self.center:
            r_bar = r_flat.mean(0, keepdim=True)
            rc = r_flat - r_bar
        else:
            r_bar = None
            rc = r_flat

        I = torch.eye(B, device=r_flat.device, dtype=r_flat.dtype)
        A = (1.0 - self.lam) * I + self.lam * P                    # [B,B]
        r_tilde_flat = A @ rc
        if r_bar is not None:
            r_tilde_flat = r_tilde_flat + r_bar

        if self.scale_match:
            rn = r_flat.norm()
            rtn = r_tilde_flat.norm()
            r_tilde_flat = (rn / (rtn + 1e-12)) * r_tilde_flat

        r_tilde = r_tilde_flat.view_as(grad).to(dtype_out)

        if self.debug:
            def _chk(name, t):
                if t is None: return
                if not torch.isfinite(t).all():
                    print(f"[LowFreqAdam DEBUG] {name} has non-finite values; max|.|={t.detach().abs().max().item():.3e}")
            _chk("grad_in", grad)
            _chk("P", P)
            _chk("grad_out", r_tilde)

        return r_tilde

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0:
            params = [p for group in self.base.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)

        self.base.step()
        return loss

    def state_dict(self):
        state = self.base.state_dict()
        state["_lowfreq_W"] = self.W
        # store scalar knobs
        state["_lowfreq_cfg"] = {
            "m": self.m, "sigma": self.sigma, "lam": self.lam,
            "eta": self.eta, "center": self.center, "spectral_cap": self.spectral_cap,
            "scale_match": self.scale_match,
        }
        return state

    def load_state_dict(self, state_dict):
        self.W = state_dict.pop("_lowfreq_W", None)
        _cfg = state_dict.pop("_lowfreq_cfg", None)
        if _cfg is not None:
            self.m = int(_cfg.get("m", self.m))
            self.sigma = float(_cfg.get("sigma", self.sigma))
            self.lam = float(_cfg.get("lam", self.lam))
            self.eta = float(_cfg.get("eta", self.eta))
            self.center = bool(_cfg.get("center", self.center))
            self.spectral_cap = bool(_cfg.get("spectral_cap", self.spectral_cap))
            self.scale_match = bool(_cfg.get("scale_match", self.scale_match))
        self.base.load_state_dict(state_dict)
