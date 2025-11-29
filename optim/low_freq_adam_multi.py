import math
import torch

from torch.optim import Adam


def wrap(ang):
    return (ang + math.pi) % (2 * math.pi) - math.pi  # [-pi, pi)


@torch.no_grad()
def torus_heat_gram(theta, sigma, N=0):
    # theta: [B, m] in [-pi,pi). Returns degree-normalized Gram \hat K.
    d = wrap(theta[:, None, :] - theta[None, :, :])  # [B,B,m]
    shifts = torch.arange(-N, N + 1, device=theta.device, dtype=theta.dtype)
    dd = d.unsqueeze(-1) - (2 * math.pi) * shifts.view(1, 1, 1, -1)  # [B,B,m,2N+1]
    K1d = torch.exp(-(dd**2) / (2 * sigma**2)).sum(-1)  # [B,B,m]
    K = K1d.prod(-1)
    K = 0.5 * (K + K.t())
    dv = K.diag().clamp_min(1e-12).sqrt()
    return ((K / dv).t() / dv)


@torch.no_grad()
def pca_init_rows(Z, m):
    # Z: [B,D], row-orth init W in R^{m×D} with unit-variance rows (w.r.t. Z)
    Zc = Z - Z.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Zc, full_matrices=False)
    W = Vh[:m, :]
    std = (Zc @ W.t()).std(0).clamp_min(1e-3)
    return (W / std[:, None]).contiguous()


class LowFreqAdamMulti(torch.optim.Optimizer):
    """
    Low-frequency Adam that can aggregate multiple embedding sources (per-layer).
    specs: dict name -> {"m": int, "sigma": float, "alpha": float}
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.0,
                 specs=None, lam=0.5, scale_match=True, N_images=0):
        self.base = Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        self.specs = specs or {}
        self.W = {}  # name -> [m,D]
        self.lam, self.scale_match, self.N_images = float(lam), bool(scale_match), int(N_images)
        super().__init__(self.base.param_groups, self.base.defaults)

    @torch.no_grad()
    def _Kbar(self, embeds):
        Ksum, asum = None, 0.0
        for name, Z in embeds.items():
            if name not in self.specs:
                continue
            m = int(self.specs[name]["m"])
            sigma = float(self.specs[name]["sigma"])
            a = float(self.specs[name]["alpha"])
            if name not in self.W:
                self.W[name] = pca_init_rows(Z, m).to(Z.device, Z.dtype)
            theta = wrap(2 * math.pi * (Z @ self.W[name].t()))
            Khat = torus_heat_gram(theta, sigma=sigma, N=self.N_images).to(Z.dtype)
            Ksum = Khat if Ksum is None else (Ksum + a * Khat)
            asum += a
        return (Ksum / max(asum, 1e-12)) if Ksum is not None else None

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure):
        # closure returns: logits, y, embeds dict name->tensor, loss
        logits, y, embeds, loss = closure()
        # flatten logits/targets if needed
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits_flat = logits.view(-1, C)
            y_flat = y.view(-1)
        else:
            logits_flat = logits
            y_flat = y

        with torch.no_grad():
            Bflat = logits_flat.size(0)
            p = logits_flat.softmax(1)
            r = p
            r[torch.arange(Bflat, device=logits_flat.device), y_flat] -= 1.0
            r /= Bflat
            Kbar = self._Kbar({k: v.detach() for k, v in embeds.items() if k in self.specs})
            if Kbar is not None:
                r_tilde = (1.0 - self.lam) * r + self.lam * (Kbar @ r)
                if self.scale_match:
                    r_tilde *= (r.norm() / (r_tilde.norm() + 1e-12)).detach()
            else:
                r_tilde = r

        logits_flat.backward(gradient=r_tilde)
        self.base.step()
        return loss
