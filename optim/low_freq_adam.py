import math
import torch


class LowFreqAdam(torch.optim.Optimizer):
    """
    Adam optimizer with low-frequency kernel shaping of the logits gradient before the update.
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
        adam_impl=torch.optim.Adam,
    ):
        self.base = adam_impl(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.m, self.sigma, self.lam, self.scale_match = int(m), float(sigma), float(lam), bool(scale_match)
        self.W = None
        self.clip_grad_norm = None
        self.ddp_model = None
        self.debug = False  # set to True to enable debug checks
        super().__init__(self.base.param_groups, self.base.defaults)
        self.state = self.base.state

    @torch.no_grad()
    def _khat(self, x):
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

        theta = 2 * math.pi * (xf @ self.W.t())
        d = theta[:, None, :] - theta[None, :, :]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        K = torch.exp(-(d.pow(2).sum(-1)) / (2.0 * self.sigma**2))
        K = 0.5 * (K + K.t())
        dv = K.diag().clamp_min(1e-12).sqrt()
        Khat = (K / dv).t() / dv
        return Khat.to(dtype=x.dtype)

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def shape_grad(self, grad, x):
        """
        Transform dL/dlogits -> low-frequency-shaped gradient.
        Intended to be used from a logits.register_hook.
        """
        B = grad.shape[0]
        r_flat = grad.view(B, -1)
        Khat = self._khat(x)
        r_tilde_flat = (1.0 - self.lam) * r_flat + self.lam * (Khat @ r_flat)
        if self.scale_match:
            rn = r_flat.norm()
            rtn = r_tilde_flat.norm()
            r_tilde_flat = (rn / (rtn + 1e-12)) * r_tilde_flat
        r_tilde = r_tilde_flat.view_as(grad)

        if self.debug:
            def _check(name, t):
                if t is None:
                    return
                if not torch.isfinite(t).all():
                    print(f"[LowFreqAdam DEBUG] non-finite values in {name}: "
                          f"max_abs={t.detach().abs().max().item():.3e}")
                else:
                    vmax = t.detach().abs().max().item()
                    if vmax > 1e6:
                        print(f"[LowFreqAdam DEBUG] large values in {name}: max_abs={vmax:.3e}")

            _check("grad_in", grad)
            _check("grad_out", r_tilde)
            _check("Khat", Khat)

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
        return state

    def load_state_dict(self, state_dict):
        self.W = state_dict.pop("_lowfreq_W", None)
        self.base.load_state_dict(state_dict)
