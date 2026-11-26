"""PLM baselines: frozen whole-strain embeddings into a two-layer MLP.

This is the reference implementation behind the protein-language-model
comparison (Table 4).  Each pretrained model contributes one embedding per
strain; the pair representation is the concatenation of the source embedding,
the reference embedding and their absolute difference, mapped to a distance by
a two-layer MLP.

Protocol (identical to the SpecFuse experiments):
    512 hidden units, 256 in the second layer, dropout 0.2, MSE loss,
    AdamW lr 1e-3 / wd 1e-5, batch 1024, 300 epochs, gradient clipping 1.0,
    checkpoint selected on inner-validation MAE.

Examples
--------
    python plm_baseline.py --dataset h1n1_566 --seed 42 --model ProtT5
    python plm_baseline.py --dataset h3n2_566_asym --seed 42 --model ESM2
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(ROOT, "embeddings", "plm_strains")
SPLITS = os.path.join(ROOT, "splits_asym")

HID, BS, EPOCHS, LR, WD, DO = 512, 1024, 300, 1e-3, 1e-5, 0.2
MODELS = ["BioGPT", "ESM2", "ProGen2", "ProtBERT", "ProtT5"]


def metrics(y, p):
    e = p - y
    mse = float((e ** 2).mean())
    return {"mae": float(e.abs().mean()),
            "mse": mse,
            "rmse": float(mse ** 0.5),
            "r2": float(1 - mse / float(((y - y.mean()) ** 2).mean())),
            "n": int(len(y))}


class PairMLP(nn.Module):
    """[src, dst, |src - dst|] -> distance."""

    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d * 3, HID), nn.GELU(), nn.Dropout(DO),
            nn.Linear(HID, HID // 2), nn.GELU(), nn.Dropout(DO),
            nn.Linear(HID // 2, 1))

    def forward(self, src, dst):
        return self.net(torch.cat([src, dst, (src - dst).abs()], dim=1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="h1n1_566 | h3n2_566_asym | bvic_585_asym")
    ap.add_argument("--seed", type=int, default=42, help="42 | 123 | 2024")
    ap.add_argument("--model", default="ProtT5", choices=MODELS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs_plm"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed * 1000 + 11)

    emb_path = os.path.join(EMB, f"{args.dataset}__{args.model}.npz")
    X = np.load(emb_path)["emb"].astype(np.float32)
    sp = np.load(os.path.join(SPLITS, f"{args.dataset}_s{args.seed}.npz"))
    assert int(sp["n_nodes"]) == X.shape[0], (int(sp["n_nodes"]), X.shape)

    t = {k: {kk: torch.as_tensor(sp[f"{k}_{kk}"], device=dev)
             for kk in ("src", "dst", "y")}
         for k in ("inner_train", "inner_val", "test")}

    # normalise with inner-training strains only
    trn = np.unique(np.r_[t["inner_train"]["src"].cpu().numpy(),
                          t["inner_train"]["dst"].cpu().numpy()])
    mu = X[trn].mean(0, keepdims=True)
    sd = X[trn].std(0, keepdims=True) + 1e-6
    Xs = torch.as_tensor(((X - mu) / sd).astype(np.float32), device=dev)

    out = os.path.join(args.out, f"{args.dataset}__{args.model}_s{args.seed}")
    os.makedirs(out, exist_ok=True)

    model = PairMLP(Xs.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.MSELoss()
    best, best_ep = 1e9, 0
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        g = torch.Generator(device="cpu").manual_seed(
            (args.seed * 1000 + 77) * 1000 + ep)
        perm = torch.randperm(len(t["inner_train"]["src"]), generator=g)
        model.train()
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS].to(dev)
            loss = lossf(model(Xs[t["inner_train"]["src"][idx]],
                               Xs[t["inner_train"]["dst"][idx]]),
                         t["inner_train"]["y"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vm = float((model(Xs[t["inner_val"]["src"]],
                              Xs[t["inner_val"]["dst"]])
                        - t["inner_val"]["y"]).abs().mean())
        if vm < best - 1e-6:
            best, best_ep = vm, ep
            torch.save({"state": model.state_dict(), "epoch": ep,
                        "val_mae": vm}, os.path.join(out, "best.pt"))
        if ep % 50 == 0 or ep == 1:
            print(f"  ep {ep} val {vm:.4f} best {best:.4f}@{best_ep} "
                  f"{(time.perf_counter() - t0) / ep:.2f}s/ep", flush=True)

    ck = torch.load(os.path.join(out, "best.pt"), map_location=dev,
                    weights_only=False)
    model.load_state_dict(ck["state"])
    model.eval()
    res = {"dataset": args.dataset, "model": args.model, "seed": args.seed,
           "dim": int(Xs.shape[1]), "best_epoch": ck["epoch"],
           "params": int(sum(q.numel() for q in model.parameters())),
           "wall_min": round((time.perf_counter() - t0) / 60, 2)}
    with torch.no_grad():
        for split in ("inner_val", "test"):
            m = metrics(t[split]["y"],
                        model(Xs[t[split]["src"]], Xs[t[split]["dst"]]))
            res[split] = m
            print(f"  {split}: MAE {m['mae']:.5f} R2 {m['r2']:.5f} n={m['n']}")

    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print("DONE", json.dumps({k: res[k] for k in
                              ("dataset", "model", "seed", "params")}
                             | {"test_mae": res["test"]["mae"],
                                "test_r2": res["test"]["r2"]}))


if __name__ == "__main__":
    main()