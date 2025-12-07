# import math
# import torch


# class LowFreqAdamW(torch.optim.Optimizer):
#     """
#     Adam optimizer with low-frequency kernel shaping of the logits gradient before the update.
#     """

#     def __init__(
#         self,
#         params,
#         lr=1e-3,
#         betas=(0.9, 0.999),
#         eps=1e-8,
#         weight_decay=0.0,
#         m=4,
#         sigma=1.0,
#         lam=0.5,
#         scale_match=True,
#         base_impl=torch.optim.Adam,
#     ):
#         self.base = base_impl(
#             params,
#             lr=lr,
#             # betas=betas,
#             # eps=eps,
#             weight_decay=weight_decay,
#         )
#         self.m, self.sigma, self.lam, self.scale_match = int(m), float(sigma), float(lam), bool(scale_match)
#         self.W = None
#         self.clip_grad_norm = None
#         self.ddp_model = None
#         self.debug = False  # set to True to enable debug checks
#         super().__init__(self.base.param_groups, self.base.defaults)
#         self.state = self.base.state

#     @torch.no_grad()
#     def _khat(self, x):
#         B = x.shape[0]
#         D = x[0].numel()
#         xf = x.view(B, D).to(torch.float64)

#         if self.W is None:
#             Xc = xf - xf.mean(0, keepdim=True)
#             _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
#             k = min(self.m, Vh.size(0))
#             Wp = Vh[:k]
#             if k < self.m:
#                 extra = torch.randn(self.m - k, D, device=x.device, dtype=torch.float64) / math.sqrt(D)
#                 Wp = torch.cat([Wp, extra], dim=0)
#             self.W = Wp.to(x.device)

#         theta = 2 * math.pi * (xf @ self.W.t())
#         d = theta[:, None, :] - theta[None, :, :]
#         d = (d + math.pi) % (2 * math.pi) - math.pi
#         K = torch.exp(-(d.pow(2).sum(-1)) / (2.0 * self.sigma**2))
#         K = 0.5 * (K + K.t())
#         dv = K.diag().clamp_min(1e-12).sqrt()
#         Khat = (K / dv).t() / dv
#         # keep Khat in floating-point; caller will cast to the gradient dtype
#         return Khat, K

#     def zero_grad(self, set_to_none=True):
#         self.base.zero_grad(set_to_none=set_to_none)

#     @torch.no_grad()
#     def shape_grad(self, grad, x):
#         """
#         Transform dL/dlogits -> low-frequency-shaped gradient.
#         Intended to be used from a logits.register_hook.
#         """
#         B = grad.shape[0]
#         r_flat = grad.view(B, -1)
#         # ensure Khat matches the gradient dtype/device
#         Khat, K = self._khat(x)
#         Khat = Khat.to(dtype=grad.dtype, device=grad.device)
#         K = K.to(dtype=grad.dtype, device=grad.device)
#         r_tilde_flat = (1.0 - self.lam) * r_flat + self.lam * (Khat @ r_flat)
#         if self.scale_match:
#             rn = r_flat.norm()
#             rtn = r_tilde_flat.norm()
#             r_tilde_flat = (rn / (rtn + 1e-12)) * r_tilde_flat
#         r_tilde = r_tilde_flat.view_as(grad)

#         # Logging diagnostics for SAM runs
#         eye = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
#         k_norm = K.norm().item()
#         k_off_norm = (K - eye * K).norm().item()
#         k_diag_sqrt_sum = torch.sqrt(K.diag().sum().clamp_min(0)).item()
#         self.last_kernel_stats = {
#             "k_norm": k_norm,
#             "k_off_norm": k_off_norm,
#             "k_diag_sqrt_sum": k_diag_sqrt_sum,
#         }

#         if self.debug:
#             def _check(name, t):
#                 if t is None:
#                     return
#                 if not torch.isfinite(t).all():
#                     print(f"[LowFreqAdam DEBUG] non-finite values in {name}: "
#                           f"max_abs={t.detach().abs().max().item():.3e}")
#                 else:
#                     vmax = t.detach().abs().max().item()
#                     if vmax > 1e6:
#                         print(f"[LowFreqAdam DEBUG] large values in {name}: max_abs={vmax:.3e}")

#             _check("grad_in", grad)
#             _check("grad_out", r_tilde)
#             _check("Khat", Khat)

#         return r_tilde

#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             with torch.enable_grad():
#                 loss = closure()

#         if self.clip_grad_norm is not None and self.clip_grad_norm > 0.0:
#             params = [p for group in self.base.param_groups for p in group["params"]]
#             torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)

#         self.base.step()
#         return loss

#     def state_dict(self):
#         state = self.base.state_dict()
#         state["_lowfreq_W"] = self.W
#         return state

#     def load_state_dict(self, state_dict):
#         self.W = state_dict.pop("_lowfreq_W", None)
#         self.base.load_state_dict(state_dict)

import math
import torch


class LowFreqAdamW(torch.optim.Optimizer):
    """
    Adam optimizer with low-frequency kernel shaping of the logits gradient.
    No sigma EMA, no mixer, lambda=1, no column-stochastic normalization.
    Sigma is chosen per-batch by matching a target energy ratio tied to the batch variance.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.0,
        m=4,
        sigma=1.0,     # only used as starting bracket center
        scale_match=True,
        kappa=0.5,          # target fraction of mean-zero energy to remove (0..1)
        bisect_iters=8,     # log-sigma bisection steps
        sigma_min=1e-3,     # bracket for sigma (angular units)
        sigma_max=None,     # default: pi*sqrt(m)
        base_impl=torch.optim.Adam,
    ):
        self.base = base_impl(params, lr=lr, weight_decay=weight_decay)
        self.m = int(m)
        self.sigma_init = float(sigma)
        self.scale_match = bool(scale_match)
        self.kappa = float(kappa)
        self.bisect_iters = int(bisect_iters)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max) if sigma_max is not None else (math.pi * math.sqrt(self.m))

        self.W = None
        self.clip_grad_norm = None
        self.debug = False
        super().__init__(self.base.param_groups, self.base.defaults)
        self.state = self.base.state

    @torch.no_grad()
    def _ensure_W(self, x):
        B = x.shape[0]
        D = x[0].numel()
        xf = x.view(B, D).to(torch.float64)
        if self.W is None:
            Xc = xf - xf.mean(0, keepdim=True)
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            k = min(self.m, Vh.size(0))
            Wp = Vh[:k]
            if k < self.m:
                extra = torch.randn(self.m - k, D, device=x.device, dtype=torch.float64) / math.sqrt(D)
                Wp = torch.cat([Wp, extra], dim=0)
            self.W = Wp.to(x.device)

    @torch.no_grad()
    def _Khat_sigma(self, x, sigma):
        # degree-normalized symmetric kernel \hat K = D^{-1/2} K D^{-1/2}
        B = x.shape[0]
        D = x[0].numel()
        xf = x.view(B, D).to(torch.float64)
        self._ensure_W(x)
        theta = 2 * math.pi * (xf @ self.W.t())                    # [B,m]
        d = theta[:, None, :] - theta[None, :, :]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        K = torch.exp(-(d.pow(2).sum(-1)) / (2.0 * float(sigma) ** 2))  # [B,B]
        K = 0.5 * (K + K.t())
        dv = K.diag().clamp_min(1e-12).sqrt()
        Khat = (K / dv).t() / dv
        return Khat, K

    @torch.no_grad()
    def _energy_ratio(self, r_flat, Khat):
        num = (Khat @ r_flat).pow(2).sum()
        den = (r_flat.pow(2).sum()).clamp_min(1e-24)
        return (num / den).item()

    @torch.no_grad()
    def _pick_sigma(self, r_flat, x):
        # target = 1 - kappa * v, where v = ||r - mean(r)||^2 / ||r||^2
        B = r_flat.size(0)
        r_bar = r_flat.mean(0, keepdim=True)
        rc = r_flat - r_bar
        v = (rc.pow(2).sum() / r_flat.pow(2).sum().clamp_min(1e-24)).item()
        c_min = 1.0 - v                      # achievable as sigma -> +inf
        c_tgt = 1.0 - self.kappa * v
        c_tgt = float(min(max(c_tgt, c_min), 1.0))

        lo = max(self.sigma_min, 1e-6)
        hi = max(self.sigma_max, lo * 10.0)
        # bisection in log-sigma
        for _ in range(self.bisect_iters):
            mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
            Khat_mid, _ = self._Khat_sigma(x, mid)
            phi = self._energy_ratio(r_flat, Khat_mid)  # monotone ↓ in sigma
            if phi > c_tgt:     # too much energy left -> increase sigma
                lo = mid
            else:
                hi = mid
        sigma_star = hi
        Khat_star, K_star = self._Khat_sigma(x, sigma_star)
        return sigma_star, Khat_star, K_star, v, c_tgt

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def shape_grad(self, grad, x):
        # r_tilde = \hat K_{sigma*} r, sigma* chosen to hit energy ratio 1 - kappa*v
        B = grad.shape[0]
        r_flat = grad.view(B, -1).to(torch.float64)

        sigma_star, Khat, K, v, c_tgt = self._pick_sigma(r_flat, x)
        r_tilde_flat = (Khat @ r_flat)

        if self.scale_match:
            rn = r_flat.norm()
            rtn = r_tilde_flat.norm()
            r_tilde_flat = (rn / (rtn + 1e-12)) * r_tilde_flat

        out = r_tilde_flat.to(dtype=grad.dtype, device=grad.device).view_as(grad)

        # diagnostics
        eye = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        self.last_kernel_stats = {
            "sigma_star": float(sigma_star),
            "var_frac_v": float(v),
            "target_ratio": float(c_tgt),
            "achieved_ratio": float(self._energy_ratio(r_flat, Khat)),
            "K_norm": K.norm().item(),
            "K_off_norm": (K - eye * K).norm().item(),
        }
        if self.debug:
            for k, v_ in self.last_kernel_stats.items():
                if isinstance(v_, float):
                    if not math.isfinite(v_): print(f"[LFAdamW] {k} non-finite")
        return out

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.clip_grad_norm:
            params = [p for g in self.base.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad_norm)
        self.base.step()
        return loss

    def state_dict(self):
        sd = self.base.state_dict()
        sd["_lowfreq_W"] = self.W
        return sd

    def load_state_dict(self, state_dict):
        self.W = state_dict.pop("_lowfreq_W", None)
        self.base.load_state_dict(state_dict)
