"""Gated local-Fourier adapter applied to the raw strain features.

The adapter refines the per-strain feature matrix before graph encoding by
mixing a local (depthwise convolution) branch with a spectral (learnable
frequency multiplier) branch:

    u   = Linear(LayerNorm(e))                      # [B, r, L]
    A   = Conv1d(u, groups=r)                       # local branch
    B   = irfft(M * rfft(u))                        # spectral branch
    V   = 2 g A + 2 (1 - g) B                       # g in (0, 1)
    out = e + alpha * Up(gelu(V).transpose(1, 2))

With the default zero-initialised gate (g = 0.5) this reduces to V = A + B.
The gate value is a bottleneck-channel fusion weight, not a residue
importance or causal weight.

Shipped checkpoint: ``frozen/<dataset>/s<seed>/adapter_ckpt.pt``
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFAdapter(nn.Module):
    def __init__(self, C=100, r=16, m=16, alpha_init=0.1,
                 gated="dynamic", spectral="learnable"):
        super().__init__()
        assert gated in ("dynamic", "static", "fixed")
        assert spectral in ("learnable", "identity")
        self.gated = gated
        self.spectral = spectral
        self.r = r
        self.C = C
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.norm = nn.LayerNorm(C)
        self.down = nn.Linear(C, r)
        self.local = nn.Conv1d(r, r, 3, padding=1, groups=r, bias=False)
        self.m = m
        self.mult_r = None
        self.mult_i = None
        if spectral == "learnable":
            k = min(m, 566 // 2 + 1)
            self.mult_r = nn.Parameter(torch.ones(1, r, k))
            self.mult_i = nn.Parameter(torch.zeros(1, r, k))
        if gated == "dynamic":
            self.W_g = nn.Linear(2 * r, r)
            nn.init.zeros_(self.W_g.weight)
            nn.init.zeros_(self.W_g.bias)
        elif gated == "static":
            self.beta = nn.Parameter(torch.zeros(r))
        self.up = nn.Linear(r, C)

    def _fourier(self, u):
        if self.spectral == "identity":
            return u
        spec = torch.fft.rfft(u, dim=-1)
        nf = spec.shape[-1]
        k = min(self.m, nf)
        re_orig, im_orig = spec.real, spec.imag
        re = re_orig.clone()
        im = im_orig.clone()
        re[:, :, :k] = (self.mult_r[:, :, :k] * re_orig[:, :, :k]
                        - self.mult_i[:, :, :k] * im_orig[:, :, :k])
        im[:, :, :k] = (self.mult_r[:, :, :k] * im_orig[:, :, :k]
                        + self.mult_i[:, :, :k] * re_orig[:, :, :k])
        return torch.fft.irfft(torch.complex(re, im), n=u.shape[-1], dim=-1)

    def forward(self, e):
        u = self.down(self.norm(e)).transpose(1, 2)
        A = self.local(u)
        B = self._fourier(u)
        if self.gated == "dynamic":
            q = torch.cat([A.mean(dim=-1), B.mean(dim=-1)], dim=-1)
            g = torch.sigmoid(self.W_g(q)).unsqueeze(-1)
        elif self.gated == "static":
            g = torch.sigmoid(self.beta).view(1, -1, 1)
        else:
            V = A + B
            return self.up(F.gelu(V).transpose(1, 2))
        V = 2.0 * g * A + 2.0 * (1.0 - g) * B
        return self.up(F.gelu(V).transpose(1, 2))

    def apply(self, e):
        return e + self.alpha * self.forward(e)