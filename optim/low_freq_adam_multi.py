import math, torch
from torch.optim import Adam

def wrap(ang):
    return (ang + math.pi) % (2*math.pi) - math.pi  # [-pi, pi)

@torch.no_grad()
def torus_rbf(theta, sigma, degree_norm=True, eps=1e-12):
    """
    Gaussian kernel on the torus using wrapped differences (no image sum).
    theta: [B, m] angles in [-pi, pi). Returns K or degree-normalized \hat K.
    K_ij = exp(-||wrap(theta_i - theta_j)||^2 / (2 sigma^2))
    """
    dtype0 = theta.dtype
    if dtype0 != torch.float64:
        theta = theta.to(torch.float64)
    d = wrap(theta[:, None, :] - theta[None, :, :])              # [B,B,m]
    K = torch.exp(-(d.pow(2).sum(-1)) / (2.0 * (float(sigma) ** 2)))  # [B,B]
    K = 0.5 * (K + K.t())                                        # symmetrize
    if degree_norm:
        dv = K.diag().clamp_min(eps).sqrt()
        K = (K / dv).t() / dv
    return K.to(dtype0)

@torch.no_grad()
def pca_init_rows(Z, m):
    # Z: [B,D], row-orth init W in R^{m×D} with unit-variance rows (w.r.t. Z)
    Zc = Z - Z.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(Zc.to(torch.float32), full_matrices=False)
    W = Vh[:m, :].to(Z.dtype, Z.device)
    std = (Zc @ W.t()).std(0).clamp_min(1e-3)
    return (W / std[:, None]).contiguous()

class LowFreqAdamLM(torch.optim.Optimizer):
    """
    Low-frequency Adam using a torus Gaussian kernel.
    specs: dict name -> {"m": int, "sigma": float, "alpha": float}
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, wd=0.0,
                 specs=None, lam=0.5, scale_match=True, degree_norm=True):
        self.base = Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        self.specs = specs or {}
        self.W = {}                  # name -> [m,D]
        self.lam = float(lam)
        self.scale_match = bool(scale_match)
        self.degree_norm = bool(degree_norm)
        super().__init__(self.base.param_groups, self.base.defaults)

    @torch.no_grad()
    def _Kbar(self, embeds):
        Ksum, asum = None, 0.0
        for name, Z in embeds.items():
            if name not in self.specs:
                continue
            m = int(self.specs[name]["m"])
            sigma = float(self.specs[name]["sigma"])
            alpha = float(self.specs[name]["alpha"])
            if alpha == 0:
                continue
            if name not in self.W:
                self.W[name] = pca_init_rows(Z, m).to(Z.device, Z.dtype)
            theta = wrap(2 * math.pi * (Z @ self.W[name].t()))
            K = torus_rbf(theta, sigma=sigma, degree_norm=self.degree_norm).to(Z.dtype)
            Ksum = (alpha * K) if (Ksum is None) else (Ksum + alpha * K)  # fixed weighting
            asum += alpha
        return None if Ksum is None else (Ksum / max(asum, 1e-12))

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure):
        """
        closure must return: logits [B,C], y [B],
                             embeds dict name -> [B,D] (detached),
                             loss (scalar)
        """
        logits, y, embeds, loss = closure()
        with torch.no_grad():
            B = logits.size(0)
            p = logits.softmax(1)
            r = p
            r[torch.arange(B, device=logits.device), y] -= 1.0
            r /= B
            Kbar = self._Kbar({k: v.detach() for k, v in embeds.items()})
            if Kbar is not None:
                r_tilde = (1.0 - self.lam) * r + self.lam * (Kbar @ r)
            else:
                r_tilde = r
            if self.scale_match:
                r_tilde *= (r.norm() / (r_tilde.norm() + 1e-12)).detach()
        logits.backward(gradient=r_tilde)
        self.base.step()
        return loss
