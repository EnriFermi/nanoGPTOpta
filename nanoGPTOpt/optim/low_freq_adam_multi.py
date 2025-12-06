# low_freq_adam_perlayer.py
import math, torch
from torch.optim import AdamW

def wrap(a):  # map to [-pi, pi)
    return (a + math.pi) % (2*math.pi) - math.pi

@torch.no_grad()
def pca_rows(Z, m):
    Zf = Z.to(torch.float32)
    Zc = Zf - Zf.mean(0, keepdim=True)
    _,_,Vh = torch.linalg.svd(Zc, full_matrices=False)
    W = Vh[:m,:].to(dtype=Z.dtype, device=Z.device)
    std = (Zc @ W.t()).std(0).clamp_min(1e-3)
    return (W / std[:,None]).contiguous()

@torch.no_grad()
def torus_rbf_Khat(theta, sigma, degree_norm=True):
    # theta: [N,m]; returns \hat K (degree-normalized) in float64 for stability
    th = theta if theta.dtype==torch.float64 else theta.to(torch.float64)
    d  = wrap(th[:,None,:] - th[None,:,:])                  # [N,N,m]
    K  = torch.exp(-(d.pow(2).sum(-1)) / (2.0*(sigma**2)))  # [N,N]
    K  = 0.5*(K + K.t())
    if degree_norm:
        dv = K.diag().clamp_min(1e-12).sqrt()
        K  = (K / dv).t() / dv
    return K

@torch.no_grad()
def _kernel_block(theta_I, theta_J, sigma):
    I = theta_I.size(0); J = theta_J.size(0)
    thI = theta_I.to(torch.float64); thJ = theta_J.to(torch.float64)
    d   = wrap(thI[:,None,:] - thJ[None,:,:])                     # [I,J,m]
    return torch.exp(-(d.pow(2).sum(-1)) / (2.0*(sigma**2)))      # [I,J]

@torch.no_grad()
def rbf_Khat_matvec(theta, v, sigma, degree_norm=True, row_chunk=512, col_chunk=2048):
    # y = \hat K v without building K; theta:[N,m], v:[N,C] (float32/16 ok)
    N, m = theta.shape; C = v.size(1)
    th = theta.to(torch.float64); vv = v.to(torch.float64)
    # degree vector d_i = sum_j K_ij
    d = torch.zeros(N, dtype=torch.float64, device=th.device)
    for i0 in range(0, N, row_chunk):
        i1 = min(i0+row_chunk, N)
        acc = torch.zeros(i1-i0, dtype=torch.float64, device=th.device)
        for j0 in range(0, N, col_chunk):
            j1 = min(j0+col_chunk, N)
            acc += _kernel_block(th[i0:i1], th[j0:j1], sigma).sum(1)
        d[i0:i1] = acc
    d.clamp_min_(1e-12)
    invsqrt_d = d.rsqrt()
    u = vv * invsqrt_d.view(N,1)
    y = torch.zeros(N, C, dtype=torch.float64, device=th.device)
    for i0 in range(0, N, row_chunk):
        i1 = min(i0+row_chunk, N)
        acc = torch.zeros(i1-i0, C, dtype=torch.float64, device=th.device)
        for j0 in range(0, N, col_chunk):
            j1 = min(j0+col_chunk, N)
            Kblk = _kernel_block(th[i0:i1], th[j0:j1], sigma)
            acc += Kblk @ u[j0:j1]
        y[i0:i1] = acc * invsqrt_d[i0:i1].view(-1,1)
    return y.to(v.dtype)

@torch.no_grad()
def safe_mix(r, Kr, lam):
    # ensure ||(1-λ)r + λKr|| <= ||r||, arbitrary r,Kr with same shape
    rr = (r*r).sum().clamp_min(1e-12)
    a  = (r*Kr).sum() / rr
    b  = (Kr*Kr).sum() / rr
    den = (b + 1.0 - 2.0*a).clamp_min(1e-12)
    lam_max = (2.0*(1.0 - a) / den).clamp(min=0.0, max=1.0)
    lam_eff = torch.minimum(torch.as_tensor(lam, device=r.device, dtype=r.dtype), lam_max)
    return (1.0 - lam_eff) * r + lam_eff * Kr

class LowFreqAdamPerLayer(torch.optim.Optimizer):
    """
    Per-layer LF Adam with arbitrary loss and per-layer kernels.
    You pass a 'layer_specs' dict:
      layer_specs[name] = {
         'params': list_of_parameters_for_this_layer,
         'embed_key': name_in_embeds_dict,           # embeds[embed_key] -> [B,D] or [B,T,D]
         'm': int, 'sigma': float, 'alpha': float    # alpha used only if you aggregate, else ignored
      }
    For each layer ℓ, we compute r_base = d(loss)/d(logits). Then r̃_ℓ = (1-λ)r_base + λ ( \hat K_ℓ r_base ),
    and call autograd.grad(logits, params_ℓ, grad_outputs=r̃_ℓ), accumulating only on that layer’s params.
    """
    def __init__(self, params, layer_specs, lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0,
                 lam=0.3, degree_norm=True, chunked=False, row_chunk=512, col_chunk=2048,
                 scale_match=False, lam_warmup=0):
        self.base = AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.specs = layer_specs
        self.W = {name: None for name in layer_specs}    # per-layer W
        self.lam = float(lam)
        self.degree_norm = bool(degree_norm)
        self.scale_match = bool(scale_match)
        self.lam_warmup = int(lam_warmup)
        self.step_id = 0
        self.chunked = bool(chunked)
        self.row_chunk, self.col_chunk = int(row_chunk), int(col_chunk)
        super().__init__(self.base.param_groups, self.base.defaults)

    @staticmethod
    def _flatten_logits_and_targets(logits, y):
        if logits.dim() == 3:
            B,T,C = logits.shape
            return logits.reshape(B*T, C), y.reshape(B*T), (B,T,C)
        else:
            B,C = logits.shape
            return logits, y, (B,1,C)

    @torch.no_grad()
    def _prepare_theta(self, name, Z2d, m, sigma):
        if self.W[name] is None:
            self.W[name] = pca_rows(Z2d, m).to(dtype=Z2d.dtype, device=Z2d.device)
        theta = wrap(2*math.pi * (Z2d @ self.W[name].t()))  # [N,m]
        return theta

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def accumulate(self, logits, y, embeds, loss, scale=1.0):
        # Works with arbitrary loss (scalar or mean of per-sample). We use autograd.grad to get dL/dlogits.
        logits_f, y_f, shape_info = self._flatten_logits_and_targets(logits, y)  # [N,C], [N]
        N, C = logits_f.shape

        # derivative of the actual loss wrt logits (no CE assumptions)
        r_base = torch.autograd.grad(
            outputs=loss, inputs=logits, grad_outputs=torch.ones_like(loss),
            retain_graph=True, create_graph=False, allow_unused=False
        )[0].detach()                   # same shape as logits
        r_base = r_base.reshape(N, C)

        lam_eff = self.lam * (min(1.0, (self.step_id + 1)/max(1,self.lam_warmup)) if self.lam_warmup>0 else 1.0)

        # per-layer pass: build Khat_ℓ from layer's embedding and apply only to that layer's params
        for name, spec in self.specs.items():
            if spec.get('params', None) is None: 
                continue
            Z = embeds[spec['embed_key']]
            Z2d = Z.reshape(N, -1).detach()
            theta = self._prepare_theta(name, Z2d, int(spec['m']), float(spec['sigma']))

            if self.chunked:
                Kr = rbf_Khat_matvec(theta, r_base, sigma=float(spec['sigma']),
                                      degree_norm=self.degree_norm,
                                      row_chunk=self.row_chunk, col_chunk=self.col_chunk)
            else:
                Khat = torus_rbf_Khat(theta, sigma=float(spec['sigma']), degree_norm=self.degree_norm)
                Kr = (Khat @ r_base.to(torch.float64)).to(r_base.dtype)

            r_tilde = safe_mix(r_base, Kr, lam_eff)
            if self.scale_match:
                r_tilde *= (r_base.norm() / (r_tilde.norm() + 1e-12)).detach()
            r_tilde_full = r_tilde.reshape(*shape_info)

            grads = torch.autograd.grad(
                outputs=logits, inputs=spec['params'],
                grad_outputs=r_tilde_full, retain_graph=True, allow_unused=True
            )
            for p, g in zip(spec['params'], grads):
                if g is None: 
                    continue
                if p.grad is None:
                    p.grad = (g * float(scale)).detach()
                else:
                    p.grad.add_((g * float(scale)).detach())

    def step_base(self):
        self.base.step()
        self.step_id += 1


# ---------------- Dirichlet (simplex) variant ----------------

@torch.no_grad()
def stick_break(p: torch.Tensor) -> torch.Tensor:
    # p: [B,C], sum=1. returns z in [0,1]^{C-1} (stick-breaking coordinates)
    B, C = p.shape
    z = torch.empty(B, C-1, dtype=p.dtype, device=p.device)
    s = torch.zeros(B, 1, dtype=p.dtype, device=p.device)
    eps = 1e-12
    for i in range(C-1):
        denom = (1 - s).clamp_min(eps)
        z[:, i] = (p[:, i:i+1] / denom).squeeze(1)
        s = s + p[:, i:i+1]
    return z.clamp(0, 1)


@torch.no_grad()
def jacobi_orthonormal_on_01(x: torch.Tensor, a: float, b: float, M: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Orthonormal shifted Jacobi basis on [0,1] w.r.t. weight x^a (1-x)^b, a,b > -1.
    Returns Phi: [B, M+1] and eigenvalues lam: [M+1] with lam_n = n(n + a + b + 1).
    """
    t = (2.0 * x - 1.0).to(torch.float64)
    alpha = float(b)
    beta = float(a)
    B = t.shape[0]
    M = int(M)
    Phi = torch.empty(B, M + 1, dtype=torch.float64, device=t.device)

    def h_n(n):
        a0, b0 = alpha, beta
        num = torch.exp(torch.lgamma(torch.tensor(n + a0 + 1.0)) + torch.lgamma(torch.tensor(n + b0 + 1.0)))
        den = torch.exp(torch.lgamma(torch.tensor(n + 1.0)) + torch.lgamma(torch.tensor(n + a0 + b0 + 1.0)))
        return (2.0 ** (a0 + b0 + 1.0)) * num / ((2 * n + a0 + b0 + 1.0) * den)

    P0 = torch.ones_like(t)
    norm0 = (2.0 ** (-(a + b + 1.0))) * h_n(0)
    Phi[:, 0] = (P0 / math.sqrt(norm0))
    if M == 0:
        lam0 = torch.tensor([0.0], dtype=torch.float64, device=t.device)
        return Phi.to(x.dtype), lam0

    P1 = 0.5 * ((2.0 + alpha + beta) * t + (alpha - beta))
    norm1 = (2.0 ** (-(a + b + 1.0))) * h_n(1)
    Phi[:, 1] = (P1 / math.sqrt(norm1))

    Pnm1, Pn = P0, P1
    for n in range(1, M):
        A = 2.0 * (n + 1.0) * (n + alpha + beta + 1.0)
        Bc = (2.0 * n + alpha + beta + 1.0)
        C = 2.0 * (n + alpha) * (n + beta)
        denom1 = (2.0 * n + alpha + beta + 1.0) * (2.0 * n + alpha + beta + 2.0)
        denom2 = (2.0 * n + alpha + beta) * (2.0 * n + alpha + beta + 2.0)
        Acoef = A / denom1
        Bcoef = (alpha**2 - beta**2) / denom2
        Ccoef = C / ((2.0 * n + alpha + beta) * (2.0 * n + alpha + beta + 1.0))
        Pnp1 = ((Acoef * Bc) * t + Bcoef) * Pn - Ccoef * Pnm1
        normn1 = (2.0 ** (-(a + b + 1.0))) * h_n(n + 1)
        Phi[:, n + 1] = (Pnp1 / torch.sqrt(normn1))
        Pnm1, Pn = Pn, Pnp1

    n = torch.arange(0, M + 1, device=t.device, dtype=torch.float64)
    lam = n * (n + (a + b + 1.0))
    return Phi.to(x.dtype), lam


@torch.no_grad()
def dirichlet_batch_kernel(probs: torch.Tensor, alpha: torch.Tensor, M: int, t_heat: float) -> torch.Tensor:
    """
    probs: [B,C] on simplex; alpha: [C] Dirichlet parameters (>0).
    Build BxB heat kernel via separable stick-breaking Jacobi features with degree M and time t_heat.
    """
    B, C = probs.shape
    assert alpha.numel() == C
    z = stick_break(probs)  # [B, C-1]
    a = alpha[:-1]
    b = torch.flip(alpha, dims=[0]).cumsum(0)
    b = torch.flip(b, dims=[0])[1:]
    K = None
    for i in range(C - 1):
        Phi_i, lam_i = jacobi_orthonormal_on_01(z[:, i], a[i].item() - 1.0, b[i].item() - 1.0, M)
        w = torch.exp(-t_heat * lam_i).to(Phi_i.dtype)
        Ki = (Phi_i * w.view(1, -1)) @ Phi_i.t()
        K = Ki if K is None else (K * Ki)
    K = 0.5 * (K + K.t())
    return K


class LowFreqAdamDirichlet(torch.optim.Optimizer):
    """
    Adam with batch-wise Dirichlet–Jacobi heat kernel on the probability simplex.
    For logits [B,C]: Kr = K_BB(probs; alpha, M, t_heat) @ r, r = dL/dlogits.
    For logits [B,T,C]: applies the same K_BB to each time slice.
    """
    def __init__(self, params, lr=3e-4, betas=(0.9,0.999), eps=1e-8, weight_decay=0.0,
                 alpha=None, M=6, t_heat=0.1, lam=0.3, scale_match=False):
        self.base = Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        self.M = int(M)
        self.t_heat = float(t_heat)
        self.lam = float(lam)
        self.scale_match = bool(scale_match)
        super().__init__(self.base.param_groups, self.base.defaults)

    def zero_grad(self, set_to_none=True):
        self.base.zero_grad(set_to_none=set_to_none)

    def _apply_K(self, probs, r):
        if self.alpha is None:
            alpha = torch.ones(probs.size(1), dtype=probs.dtype, device=probs.device)
        else:
            alpha = self.alpha.to(probs.device, probs.dtype)
        K = dirichlet_batch_kernel(probs, alpha, self.M, self.t_heat).to(r.dtype)
        return K @ r

    def step(self, closure):
        logits, _, embeds_unused, loss = closure()
        if logits.dim() == 3:
            B, T, C = logits.shape
            probs = logits.detach().float().softmax(-1).mean(1)  # [B,C]
            N = B * T
            r_full = torch.autograd.grad(
                outputs=loss,
                inputs=logits,
                grad_outputs=torch.ones_like(loss),
                retain_graph=False,
                create_graph=False,
            )[0].detach()
            r = r_full.reshape(N, C)
            Kr = self._apply_K(probs, r.view(B, T, C).reshape(B, T * C)).view(N, C)
        else:
            B, C = logits.shape
            probs = logits.detach().float().softmax(-1)
            r = torch.autograd.grad(
                outputs=loss,
                inputs=logits,
                grad_outputs=torch.ones_like(loss),
                retain_graph=False,
                create_graph=False,
            )[0].detach()
            Kr = self._apply_K(probs, r)

        r_tilde = safe_mix(r, Kr, self.lam)
        if self.scale_match:
            r_tilde *= (r.norm() / (r_tilde.norm() + 1e-12)).detach()
        logits.reshape(-1, logits.shape[-1]).backward(gradient=r_tilde.to(logits.dtype))
        self.base.step()
        return loss
