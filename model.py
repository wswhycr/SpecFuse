"""SpecFuse prediction heads.

Token layout
------------
A strain is represented by ``2 + B`` tokens: two graph tokens (source and
reference embeddings from the frozen encoder) followed by B spectral tokens.
Source tokens query reference tokens in directional cross-attention.

Classes
-------
SpecFuseHead      full model (2 graph + B spectral tokens, directional attention)
VariantHead       ablation variants sharing the same tokeniser
                  "graph"         2 graph tokens only
                  "spec"          B spectral tokens only
                  "noattn"        all tokens concatenated into an MLP
                  "noattn_graph"  2 graph tokens into an MLP
                  "noattn_spec"   B spectral tokens into an MLP
"""
import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.mha = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                         batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn, dim))


class SpecFuseHead(nn.Module):
    """Full model: two graph tokens + B spectral tokens, directional attention."""

    def __init__(self, n_bands=4, d_v=272, dim=128, heads=4, ffn=256,
                 dropout=0.4, blocks=2, z_dim=128):
        super().__init__()
        self.dim, self.n_bands, self.dropout = dim, n_bands, dropout
        self.ln_z = nn.LayerNorm(z_dim, eps=1e-5)
        self.ln_p = nn.LayerNorm(z_dim, eps=1e-5)
        self.zmap = nn.Identity() if dim == z_dim else nn.Linear(z_dim, dim)
        self.e_z = nn.Parameter(torch.zeros(dim))
        self.e_p = nn.Parameter(torch.zeros(dim))
        self.e_b = nn.Parameter(torch.zeros(n_bands, dim))
        for e in (self.e_z, self.e_p, self.e_b):
            nn.init.normal_(e, 0.0, 0.02)
        self.ws = nn.Linear(d_v, dim)
        self.bs = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([Block(dim, heads, ffn, dropout)
                                     for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(64, 1))

    def spec_tokens(self, v):
        return self.ws(v) + self.bs + self.e_b.unsqueeze(0)

    def tokens(self, z, p, v, with_graph=True):
        s = self.spec_tokens(v)
        if not with_graph:
            return s
        gz = self.zmap(self.ln_z(z)).unsqueeze(1) + self.e_z
        gp = self.zmap(self.ln_p(p)).unsqueeze(1) + self.e_p
        return torch.cat([gz, gp, s], dim=1)

    def forward(self, t_src, t_ref, mode="D", return_attn=False):
        B, S = t_src.shape[0], t_src.shape[1]
        dev = t_src.device
        ng = 2 if S > self.n_bands else 0
        kv_pad = torch.zeros(B, S, dtype=torch.bool, device=dev)
        pool = torch.ones(B, S, device=dev)
        if S != self.n_bands and mode == "C":
            # spectral-only pooling: graph tokens attend but do not pool
            kv_pad[:, :ng] = True
            pool[:, :ng] = 0.0
        o = t_src
        aw = None
        for blk in self.blocks:
            q = blk.ln1(o)
            kv = blk.ln1(t_ref)
            if return_attn:
                a, aw = blk.mha(q, kv, kv, key_padding_mask=kv_pad,
                                need_weights=True, average_attn_weights=False)
            else:
                a, _ = blk.mha(q, kv, kv, key_padding_mask=kv_pad,
                               need_weights=False)
            o = o + blk.drop(a)
            o = o + blk.drop(blk.ffn(blk.ln2(o)))
        pooled = (o * pool.unsqueeze(-1)).sum(1) / pool.sum(1).unsqueeze(-1)
        out = self.head(pooled).squeeze(-1)
        return (out, aw) if return_attn else out

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VariantHead(nn.Module):
    """Ablation heads (single source, or attention-free fusion)."""

    VARIANTS = ("full", "graph", "spec", "noattn", "noattn_graph", "noattn_spec")

    def __init__(self, variant="full", n_bands=4, d_v=272, dim=128, heads=4,
                 ffn=256, dropout=0.4, blocks=2, z_dim=128, mlp_width=256):
        super().__init__()
        assert variant in self.VARIANTS, variant
        self.variant, self.dim, self.n_bands = variant, dim, n_bands
        self._flatten = variant.startswith("noattn")
        self.dropout = dropout
        self.ln_z = nn.LayerNorm(z_dim, eps=1e-5)
        self.ln_p = nn.LayerNorm(z_dim, eps=1e-5)
        self.zmap = nn.Identity() if dim == z_dim else nn.Linear(z_dim, dim)
        self.e_z = nn.Parameter(torch.zeros(dim))
        self.e_p = nn.Parameter(torch.zeros(dim))
        if variant not in ("graph", "noattn_graph"):
            self.e_b = nn.Parameter(torch.zeros(n_bands, dim))
        for p in (self.e_z, self.e_p):
            nn.init.normal_(p, 0.0, 0.02)
        self.ws = nn.Linear(d_v, dim)
        self.bs = nn.Parameter(torch.zeros(dim))
        self.blocks = nn.ModuleList([] if self._flatten else
                                    [Block(dim, heads, ffn, dropout)
                                     for _ in range(blocks)])
        if self._flatten:
            n_tok = {"noattn_graph": 2, "noattn_spec": n_bands}.get(
                variant, 2 + n_bands)
            in_dim = 2 * n_tok * dim
            self.head = nn.Sequential(nn.Linear(in_dim, mlp_width),
                                      nn.GELU(), nn.Dropout(dropout),
                                      nn.Linear(mlp_width, 64), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))
        else:
            self.head = nn.Sequential(nn.Linear(dim, 64), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))

    def spec_tokens(self, v):
        return self.ws(v) + self.bs + self.e_b.unsqueeze(0)

    def tokens(self, z, p, v, with_graph=True):
        if self.variant in ("graph", "noattn_graph"):
            gz = self.zmap(self.ln_z(z)).unsqueeze(1) + self.e_z
            gp = self.zmap(self.ln_p(p)).unsqueeze(1) + self.e_p
            return torch.cat([gz, gp], dim=1)
        s = self.spec_tokens(v)
        if self.variant in ("spec", "noattn_spec") or not with_graph:
            return s
        gz = self.zmap(self.ln_z(z)).unsqueeze(1) + self.e_z
        gp = self.zmap(self.ln_p(p)).unsqueeze(1) + self.e_p
        return torch.cat([gz, gp, s], dim=1)

    def forward(self, t_src, t_ref, mode="D", return_attn=False):
        if self._flatten:
            x = torch.cat([t_src.flatten(1), t_ref.flatten(1)], dim=-1)
            return self.head(x).squeeze(-1)
        B, S = t_src.shape[0], t_src.shape[1]
        dev = t_src.device
        kv_pad = torch.zeros(B, S, dtype=torch.bool, device=dev)
        pool = torch.ones(B, S, device=dev)
        o = t_src
        aw = None
        for blk in self.blocks:
            q = blk.ln1(o)
            kv = blk.ln1(t_ref)
            if return_attn:
                a, aw = blk.mha(q, kv, kv, key_padding_mask=kv_pad,
                                need_weights=True, average_attn_weights=False)
            else:
                a, _ = blk.mha(q, kv, kv, key_padding_mask=kv_pad,
                               need_weights=False)
            o = o + blk.drop(a)
            o = o + blk.drop(blk.ffn(blk.ln2(o)))
        pooled = (o * pool.unsqueeze(-1)).sum(1) / pool.sum(1).unsqueeze(-1)
        out = self.head(pooled).squeeze(-1)
        return (out, aw) if return_attn else out

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(variant="full", n_bands=4, d_v=272, dim=128, heads=4, ffn=256,
                dropout=0.4, blocks=2, mlp_width=256):
    """Factory: full head for variant='full', ablation head otherwise."""
    if variant == "full":
        return SpecFuseHead(n_bands=n_bands, d_v=d_v, dim=dim, heads=heads,
                            ffn=ffn, dropout=dropout, blocks=blocks)
    return VariantHead(variant=variant, n_bands=n_bands, d_v=d_v, dim=dim,
                       heads=heads, ffn=ffn, dropout=dropout, blocks=blocks,
                       mlp_width=mlp_width)


def mlp_width_for(target_params, n_bands=4, dim=128, fixed_other=0):
    """Integer MLP width whose attention-free head matches ``target_params``."""
    in_dim = 2 * (2 + n_bands) * dim
    best = None
    for w in range(64, 4096, 8):
        m = nn.Sequential(nn.Linear(in_dim, w), nn.GELU(), nn.Dropout(0.0),
                          nn.Linear(w, 64), nn.GELU(), nn.Dropout(0.0),
                          nn.Linear(64, 1))
        n = sum(p.numel() for p in m.parameters()) + fixed_other
        d = abs(n - target_params) / target_params
        if best is None or d < best[0]:
            best = (d, w, n)
        if n > target_params * 1.2:
            break
    return best


if __name__ == "__main__":
    # Parameter-count self-check against the numbers reported in the paper.
    shapes = dict(n_bands=4, d_v=272, dim=128, heads=4, ffn=256, dropout=0.4)
    checks = [
        ("full", 2, 256, 309633),
        ("noattn", 2, 171, 310252),
        ("noattn_graph", 2, 474, 309467),
        ("noattn_spec", 2, 251, 309820),
    ]
    for variant, blocks, width, expect in checks:
        m = build_model(variant=variant, blocks=blocks, mlp_width=width, **shapes)
        got = m.n_params()
        flag = "OK " if got == expect else "DIFF"
        print(f"{flag} {variant:14s} blocks={blocks} width={width} "
              f"params={got} (paper {expect})")