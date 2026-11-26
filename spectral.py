"""Spectral token construction and dataset assembly.

Each strain is described by B frequency bands of the shared F-dynamic adapter
output, summarised by Q DCT coefficients of the real and imaginary parts plus a
log-power statistic per band.  The resulting descriptor has shape
[B, r * (2Q + 1)] and is normalised with statistics of the inner-training set.

Frozen inputs (shipped)
-----------------------
frozen/<dataset>/s<seed>/adapter_ckpt.pt   F-dynamic adapter weights
frozen/<dataset>/s<seed>/z1.pt             graph-encoder output for each strain
frozen/<dataset>/s<seed>/proj_z1.pt        projected graph embedding

Splits (shipped)
----------------
splits/<dataset>/s<seed>/{inner_train,inner_val,test}.npz
    rows, src, dst, y  (train/val/test are disjoint in rows)

The spectral cache is rebuilt on first use and written to cache/.
"""
import os

import numpy as np
import torch

from adapter import GatedFAdapter
from graphData import graphDataset

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(ROOT, "splits")
FROZEN = os.path.join(ROOT, "frozen")
CACHE = os.path.join(ROOT, "cache")

SPLIT_NAMES = ("inner_train", "inner_val", "test")


def dct_basis(n_b, q, device="cpu"):
    k = torch.arange(n_b, dtype=torch.float64, device=device)
    i = torch.arange(q, dtype=torch.float64, device=device).unsqueeze(1)
    R = torch.cos(np.pi * (k + 0.5) * i / n_b) * np.sqrt(2.0 / n_b)
    R[0] = R[0] / np.sqrt(2.0)
    return R.float()


def band_indices(L, device, nb):
    K = L // 2 + 1
    idx = torch.full((K,), -1, dtype=torch.long)
    edges = np.linspace(1, K, nb + 1).astype(int)
    for b in range(nb):
        idx[edges[b]:edges[b + 1]] = b
    return idx.to(device)


def cache_path(ds, seed, nb, q):
    return os.path.join(CACHE, f"{ds}_s{seed}_B{nb}Q{q}.npz")


def _load_splits(ds, seed):
    out = {}
    for s in SPLIT_NAMES:
        p = os.path.join(SPLITS, ds, f"s{seed}", f"{s}.npz")
        d = np.load(p)
        out[s] = {k: torch.as_tensor(d[k]) for k in ("rows", "src", "dst", "y")}
    return out


def build(ds, seed, nb, q, device="cpu", force=False):
    """Return {'z','p','v_norm','mu','sd','n_bins','L'} for one dataset/seed."""
    out = cache_path(ds, seed, nb, q)
    if os.path.exists(out) and not force:
        d = np.load(out)
        return {k: (torch.as_tensor(d[k]) if k not in ("n_bins", "L")
                    else d[k])
                for k in ("z", "p", "v_norm", "mu", "sd", "n_bins", "L")}

    # 1. raw strain features -> k-mer matrix
    g = graphDataset(ds, strip=True)
    X = g.data.x
    L = X.shape[1] // 100

    # 2. refine with the frozen adapter, then take the real FFT
    st = os.path.join(FROZEN, ds, f"s{seed}")
    ad = GatedFAdapter(100, r=16, m=16, alpha_init=0.03, gated="dynamic")
    ad.load_state_dict(torch.load(os.path.join(st, "adapter_ckpt.pt"),
                                  map_location="cpu", weights_only=False))
    ad = ad.to(device).eval()
    with torch.no_grad():
        U = ad.down(ad.norm(X.to(device).view(-1, L, 100))).transpose(1, 2)
        Xs = torch.fft.rfft(U, dim=-1, norm="ortho")

    # 3. band-wise DCT summary
    bidx = band_indices(L, torch.device(device), nb)
    nbins = [int((bidx == b).sum()) for b in range(nb)]
    assert min(nbins) >= q, f"band bins {nbins} < Q={q}"
    parts = []
    for b in range(nb):
        sel = (bidx == b).nonzero(as_tuple=True)[0]
        Xb = Xs[:, :, sel]
        R = dct_basis(len(sel), q, device).to(Xb.device)
        CR = torch.einsum("qk,nck->ncq", R, Xb.real)
        CI = torch.einsum("qk,nck->ncq", R, Xb.imag)
        P = torch.log1p((Xb.real ** 2 + Xb.imag ** 2).mean(dim=2))
        parts.append(torch.cat([CR.reshape(CR.shape[0], -1),
                                CI.reshape(CI.shape[0], -1), P], dim=1))
    V = torch.stack(parts, dim=1)                       # [N, nb, r*(2q+1)]

    # 4. frozen graph embeddings
    z = torch.load(os.path.join(st, "z1.pt"), map_location="cpu",
                   weights_only=False).float()
    p = torch.load(os.path.join(st, "proj_z1.pt"), map_location="cpu",
                   weights_only=False).float()

    # 5. normalise with inner-training statistics
    sp = _load_splits(ds, seed)
    uniq = np.unique(np.concatenate([sp["inner_train"]["src"].numpy(),
                                     sp["inner_train"]["dst"].numpy()]))
    uniq = torch.as_tensor(uniq.astype(np.int64))
    D = V.shape[-1]
    Vu = V[uniq].reshape(-1, nb * D)
    mu = Vu.mean(dim=0)
    sd = Vu.std(dim=0, correction=0).clamp_min(1e-6)
    Vn = (V.reshape(-1, nb * D) - mu) / sd

    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, z=z.numpy(), p=p.numpy(),
                        v_raw=V.reshape(-1, nb * D).cpu().numpy().astype(np.float32),
                        v_norm=Vn.cpu().numpy().astype(np.float32),
                        mu=mu.cpu().numpy().astype(np.float32),
                        sd=sd.cpu().numpy().astype(np.float32),
                        n_bins=np.array(nbins, dtype=np.int32),
                        L=np.array([L], dtype=np.int32))
    return {"z": z, "p": p, "v_norm": Vn, "mu": mu, "sd": sd,
            "n_bins": np.array(nbins), "L": np.array([L])}


def dataset(ds, seed, nb=4, q=8, device="cpu", force=False):
    """Model-ready tensors plus the three split index sets.

    The spectral cache is always built on CPU (numpy round-trip) and only then
    moved to ``device``; this keeps the cache device-independent.
    """
    D = build(ds, seed, nb, q, force=force)
    vn = D["v_norm"]
    vn = vn if torch.is_tensor(vn) else torch.as_tensor(vn)
    V = vn.reshape(vn.shape[0], nb, vn.shape[-1] // nb).to(device)
    F = {"z": D["z"].to(device), "p": D["p"].to(device), "v": V,
         "d_v": int(V.shape[-1])}
    for s, v in _load_splits(ds, seed).items():
        F[s] = {k: v[k].to(device) for k in v}
    return F