import math, torch
from torch.optim import Adam

def wrap(a): return (a + math.pi) % (2*math.pi) - math.pi

@torch.no_grad()
def _power_iter_rho(K, iters=2):
    v = torch.randn(K.size(0), 1, device=K.device, dtype=K.dtype)
    v /= (v.norm() + 1e-12)
    for _ in range(iters):
        v = K @ v; v /= (v.norm() + 1e-12)
    return (v.t() @ (K @ v)).squeeze().clamp_min(1e-12)

@torch.no_grad()
def torus_heat_gram(theta, sigma, N=0, lipschitz=True):
    # theta: [T, m] (per-sequence). Returns degree-normalized, optionally spectral-capped.
    dtype0 = theta.dtype
    if dtype0 != torch.float64: theta = theta.to(torch.float64)

    d  = wrap(theta[:,None,:] - theta[None,:,:])                         # [T,T,m]
    sh = torch.arange(-N, N+1, device=theta.device, dtype=theta.dtype)
    dd = d.unsqueeze(-1) - (2*math.pi) * sh.view(1,1,1,-1)               # [T,T,m,2N+1]
    K1 = torch.exp(-(dd**2) / (2.0 * (float(sigma)**2))).sum(-1)         # [T,T,m]
    K  = K1.prod(-1)                                                     # [T,T]
    K  = 0.5*(K + K.t())
    Dv = K.diag().clamp_min(1e-12).sqrt()
    K  = (K / Dv).t() / Dv                                               # degree-normalize
    if lipschitz:
        rho = _power_iter_rho(K, iters=2)
        if rho > 1: K = K / rho
    return K.to(dtype0)

@torch.no_grad()
def build_block_kernel(Z, W, m, sigma, N=0, window=None):
    """
    Z: [B,T,D] per-sequence embeddings (detach). W: [m,D] or None per batch item.
    window: int or None. If set, zero out entries with |i-j|>window (temporal locality).
    Return block-diagonal Khat of shape [B*T, B*T].
    """
    B, T, D = Z.shape
    Kblocks = []
    if W is None: W = [None]*B
    outW = []
    for b in range(B):
        Zb = Z[b]                                         # [T,D]
        if W[b] is None:
            # PCA init rows for this sequence
            Zc = Zb - Zb.mean(0, keepdim=True)
            _,_,Vh = torch.linalg.svd(Zc.to(torch.float32), full_matrices=False)
            Wb = (Vh[:m,:]).to(Zb.dtype, Zb.device)
            std = (Zc @ Wb.t()).std(0).clamp_min(1e-3)
            Wb = (Wb / std[:,None]).contiguous()
        else:
            Wb = W[b]
        theta = wrap(2*math.pi * (Zb @ Wb.t()))           # [T,m]
        Kb = torus_heat_gram(theta, sigma=sigma, N=N, lipschitz=True)  # [T,T]
        if window is not None and window >= 0:
            idx = torch.arange(T, device=Z.device)
            mask = (idx[:,None]-idx[None,:]).abs() <= int(window)
            Kb = Kb * mask.to(Kb.dtype)
        Kblocks.append(Kb)
        outW.append(Wb)
    Khat = torch.block_diag(*Kblocks)                     # [B*T, B*T]
    return Khat.contiguous(), outW

class LowFreqAdamLM(torch.optim.Optimizer):
    """
    For language models: logits [B,T,C], embeds 'h' = hidden states [B,T,D].
    Block-diagonal K per sequence, optional temporal window. Adaptive lambda.
    """
    def __init__(self, params, lr=3e-4, betas=(0.9,0.999), eps=1e-8, wd=0.0,
                 m=8, sigma=0.8, lam=0.3, N_images=0, window=None,
                 scale_match=False, lam_warmup=200):
        self.base = Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd)
        self.m, self.sigma, self.N_images = int(m), float(sigma), int(N_images)
        self.window = None if window is None else int(window)
        self.scale_match = bool(scale_match)
        self.lam, self.lam_warmup, self.step_id = float(lam), int(lam_warmup), 0
        self.W_per_seq = None  # list of [m,D] per batch item; re-init when batch size changes
        super().__init__(self.base.param_groups, self.base.defaults)

    def zero_grad(self, set_to_none=True): self.base.zero_grad(set_to_none=set_to_none)

    def step(self, closure):
        # closure must return: logits [B,T,C], y [B,T], embeds {'h':[B,T,D]}, loss
        logits, y, embeds, loss = closure()
        B, T, C = logits.shape
        H = embeds['h'].detach()                           # [B,T,D]
        if self.W_per_seq is None or len(self.W_per_seq) != B:
            self.W_per_seq = [None]*B

        with torch.no_grad():
            # residuals r over flattened tokens
            p = logits.float().softmax(-1)
            r = p.view(B*T, C)
            r[torch.arange(B*T, device=logits.device), y.view(-1)] -= 1.0
            r /= (B*T)

            # block-diagonal K across sequences; optional time window
            Khat, self.W_per_seq = build_block_kernel(H, self.W_per_seq,
                                                      m=self.m, sigma=self.sigma,
                                                      N=self.N_images, window=self.window)
            # adaptive lambda based on how different (K - I)r is
            Kr = Khat @ r
            diff = (Kr - r)
            ratio = (diff.norm() / (r.norm() + 1e-12)).clamp_min(0.)
            lam_curr = self.lam * min(1.0, float(self.step_id + 1) / max(1, self.lam_warmup))
            lam_eff  = lam_curr / (1.0 + ratio.item())     # shrink when preconditioner is aggressive
            r_tilde  = (1.0 - lam_eff) * r + lam_eff * Kr

            if self.scale_match:
                r_tilde *= (r.norm() / (r_tilde.norm() + 1e-12)).detach()

        logits.view(B*T, C).backward(gradient=r_tilde.to(logits.dtype))
        self.base.step()
        self.step_id += 1
        return loss
