import math, torch
from torch.optim import Adam

def wrap(ang):
    return (ang + math.pi) % (2*math.pi) - math.pi  # [-pi, pi)

@torch.no_grad()
def torus_rbf(theta, sigma, degree_norm=True, eps=1e-12):
    # theta: [N, m] in [-pi,pi)
    theta64 = theta if theta.dtype == torch.float64 else theta.to(torch.float64)
    d = wrap(theta64[:, None, :] - theta64[None, :, :])               # [N,N,m]
    K = torch.exp(-(d.pow(2).sum(-1)) / (2.0 * (float(sigma) ** 2)))  # [N,N]
    K = 0.5 * (K + K.t())
    if degree_norm:
        dv = K.diag().clamp_min(eps).sqrt()
        K = (K / dv).t() / dv
    return K.to(theta.dtype)

@torch.no_grad()
def pca_init_rows(Z, m):
    # Z: [N,D] (float32 recommended for SVD speed)
    Zf = Z.to(torch.float32)
    Zc = Zf - Zf.mean(0, keepdim=True)
    _,_,Vh = torch.linalg.svd(Zc, full_matrices=False)
    W = Vh[:m,:].to(Z.dtype, Z.device)
    std = (Zc @ W.t()).std(0).clamp_min(1e-3)
    return (W / std[:,None]).contiguous()

class LowFreqAdamMulti(torch.optim.Optimizer):
    """
    Torus-Gaussian LF-Adam with per-layer kernels.
    Works for:
      classification: logits [B,C], embeds[name] [B,D]
      language model: logits [B,T,C], embeds[name] [B,T,D]
    No giant [BT,BT] kernels; applies K per sequence when T is present.
    specs: dict name -> {"m": int, "sigma": float, "alpha": float}
    """
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8, wd=0.0,
                 specs=None, lam=0.5, scale_match=True, degree_norm=True):
        self.base = Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        self.specs = specs or {}
        self.W = {}  # name -> [m,D]
        self.lam = float(lam)
        self.scale_match = bool(scale_match)
        self.degree_norm = bool(degree_norm)
        super().__init__(self.base.param_groups, self.base.defaults)

    @torch.no_grad()
    def _ensure_W(self, name, Z2d):
        if name not in self.W:
            self.W[name] = pca_init_rows(Z2d, int(self.specs[name]["m"])).to(Z2d.device, Z2d.dtype)

    @torch.no_grad()
    def _apply_Kbar(self, embeds, r, B=None, T=None):
        # r: [N,C] where N=B or N=B*T
        if B is None:  # classification case
            Ksum, asum = None, 0.0
            for name, Z in embeds.items():
                if name not in self.specs: continue
                spec = self.specs[name]
                alpha = float(spec["alpha"])
                if alpha == 0: continue
                Z2d = Z  # [B,D]
                self._ensure_W(name, Z2d)
                theta = wrap(2*math.pi * (Z2d @ self.W[name].t()))
                K = torus_rbf(theta, sigma=float(spec["sigma"]), degree_norm=self.degree_norm).to(Z2d.dtype)
                Ksum = alpha * K if Ksum is None else (Ksum + alpha * K)
                asum += alpha
            if Ksum is None or asum == 0: return r
            return (Ksum / asum) @ r

        # language model: apply per sequence to avoid [BT,BT]
        C = r.size(1)
        out = torch.zeros_like(r)
        for b in range(B):
            Ksum_b, asum_b = None, 0.0
            for name, Z in embeds.items():
                if name not in self.specs: continue
                spec = self.specs[name]; alpha = float(spec["alpha"])
                if alpha == 0: continue
                Zb = Z[b]              # [T,D]
                self._ensure_W(name, Zb)
                theta_b = wrap(2*math.pi * (Zb @ self.W[name].t()))  # [T,m]
                Kb = torus_rbf(theta_b, sigma=float(spec["sigma"]), degree_norm=self.degree_norm).to(Zb.dtype)  # [T,T]
                Ksum_b = alpha * Kb if Ksum_b is None else (Ksum_b + alpha * Kb)
                asum_b += alpha
            if Ksum_b is None or asum_b == 0:
                out[b*T:(b+1)*T] = r[b*T:(b+1)*T]
            else:
                Kbar_b = (Ksum_b / asum_b)                             # [T,T]
                out[b*T:(b+1)*T] = Kbar_b @ r[b*T:(b+1)*T]              # [T,C]
        return out

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure):
        # closure must return: logits, y, embeds(dict name->tensor), loss
        logits, y, embeds, loss = closure()

        # figure out shapes and flatten if LM
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits_f = logits.reshape(B*T, C)
            y_f = y.reshape(B*T)
            # make sure all embeds[name] are 3D and we keep them that way; _apply_Kbar will index per b
            embeds_use = {k: v.detach() for k, v in embeds.items()}
            Binfo, Tinfo = B, T
        else:
            B, C = logits.shape
            logits_f = logits
            y_f = y
            # ensure 2D embeds
            embeds_use = {k: v.detach() for k, v in embeds.items()}
            Binfo, Tinfo = None, None

        with torch.no_grad():
            N = logits_f.size(0)
            p = logits_f.softmax(1)
            r = p
            r[torch.arange(N, device=logits.device), y_f] -= 1.0
            r /= N
            Kr = self._apply_Kbar(embeds_use, r, B=Binfo, T=Tinfo)
            r_tilde = (1.0 - self.lam) * r + self.lam * Kr
            if self.scale_match:
                r_tilde *= (r.norm() / (r_tilde.norm() + 1e-12)).detach()

        logits_f.backward(gradient=r_tilde.to(logits.dtype))
        self.base.step()
        return loss
