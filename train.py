"""Train a SpecFuse prediction head.

Examples
--------
Full model, one dataset and seed:
    python train.py --dataset h1n1_566 --seed 42

Parameter-matched single-source upper bound (Section 4.4.3):
    python train.py --dataset h1n1_566 --seed 42 \
        --variant noattn_graph --blocks 2 --width 474

The head is selected on inner-validation MAE; test MAE is reported only once.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import spectral
from model import build_model

ROOT = os.path.dirname(os.path.abspath(__file__))

PAPER_CFG = dict(n_bands=4, q=8, dim=128, heads=4, ffn=256, dropout=0.4,
                 blocks=2, lr=1e-3, wd=1e-5, batch=512, epochs=1200)

# MLP widths that match the full head's parameter count (Section 4.4.3).
MATCHED_WIDTH = {"noattn": 171, "noattn_graph": 474, "noattn_spec": 251}


@torch.no_grad()
def predict(model, F, split, batch=1024):
    model.eval()
    sp = F[split]
    n = len(sp["src"])
    out = torch.empty(n, device=sp["src"].device)
    for i in range(0, n, batch):
        j = slice(i, min(i + batch, n))
        ts = model.tokens(F["z"][sp["src"][j]], F["p"][sp["src"][j]],
                          F["v"][sp["src"][j]])
        tr = model.tokens(F["z"][sp["dst"][j]], F["p"][sp["dst"][j]],
                          F["v"][sp["dst"][j]])
        out[j] = model(ts, tr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="h1n1_566 | h3n2_566_asym | bvic_585_asym")
    ap.add_argument("--seed", type=int, default=42, help="42 | 123 | 2024")
    ap.add_argument("--variant", default="full",
                    choices=["full", "graph", "spec", "noattn",
                             "noattn_graph", "noattn_spec"])
    ap.add_argument("--epochs", type=int, default=PAPER_CFG["epochs"])
    ap.add_argument("--blocks", type=int, default=PAPER_CFG["blocks"])
    ap.add_argument("--width", type=int, default=None,
                    help="MLP width for attention-free variants; "
                         "defaults to the parameter-matched value")
    ap.add_argument("--batch", type=int, default=PAPER_CFG["batch"])
    ap.add_argument("--lr", type=float, default=PAPER_CFG["lr"])
    ap.add_argument("--wd", type=float, default=PAPER_CFG["wd"])
    ap.add_argument("--dropout", type=float, default=PAPER_CFG["dropout"])
    ap.add_argument("--out", default=os.path.join(ROOT, "runs"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(PAPER_CFG)
    cfg.update(blocks=args.blocks, batch=args.batch, lr=args.lr, wd=args.wd,
               dropout=args.dropout, variant=args.variant)
    width = args.width or MATCHED_WIDTH.get(args.variant, 256)

    tag = f"{args.variant}_bl{args.blocks}_w{width}"
    out_dir = os.path.join(args.out, tag, args.dataset, f"s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    F = spectral.dataset(args.dataset, args.seed, cfg["n_bands"], cfg["q"],
                         device)
    tr, va = F["inner_train"], F["inner_val"]

    torch.manual_seed(args.seed * 1000 + 42)
    model = build_model(variant=args.variant, n_bands=cfg["n_bands"],
                        d_v=F["d_v"], dim=cfg["dim"], heads=cfg["heads"],
                        ffn=cfg["ffn"], dropout=cfg["dropout"],
                        blocks=cfg["blocks"], mlp_width=width).to(device)
    n_params = model.n_params()
    print(f"[{tag}] {args.dataset} s{args.seed} params={n_params}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["wd"])
    lossf = torch.nn.MSELoss()
    best = {"epoch": 0, "val_mae": float((va["y"] - predict(model, F,
                                                            "inner_val")).abs().mean())}
    torch.save({"state": model.state_dict(), "epoch": 0,
                "val_mae": best["val_mae"], "cfg": cfg},
               os.path.join(out_dir, "best.pt"))

    hist = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        gen = torch.Generator(device=device).manual_seed(
            (args.seed * 1000 + 4242) * 1000 + ep)
        perm = torch.randperm(len(tr["src"]), device=device, generator=gen)
        tot, nb = 0.0, 0
        for i in range(0, len(perm), cfg["batch"]):
            j = perm[i:i + cfg["batch"]]
            ts = model.tokens(F["z"][tr["src"][j]], F["p"][tr["src"][j]],
                              F["v"][tr["src"][j]])
            trr = model.tokens(F["z"][tr["dst"][j]], F["p"][tr["dst"][j]],
                               F["v"][tr["dst"][j]])
            pred = model(ts, trr)
            loss = lossf(pred, tr["y"][j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(j)
            nb += len(j)

        mae = float((va["y"] - predict(model, F, "inner_val")).abs().mean())
        if mae < best["val_mae"] - 1e-6:
            best = {"epoch": ep, "val_mae": mae}
            torch.save({"state": model.state_dict(), "epoch": ep,
                        "val_mae": mae, "cfg": cfg},
                       os.path.join(out_dir, "best.pt"))
        hist.append({"epoch": ep, "val_mae": mae, "train_mse": tot / max(nb, 1)})
        if ep % 50 == 0 or ep == args.epochs:
            print(f"  ep{ep} val={mae:.6f} best={best['val_mae']:.6f}"
                  f"@{best['epoch']} {time.time() - t0:.0f}s", flush=True)

    ckpt = torch.load(os.path.join(out_dir, "best.pt"), map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["state"])
    test_mae = float((F["test"]["y"] - predict(model, F, "test")).abs().mean())
    test_mse = float(((F["test"]["y"] - predict(model, F, "test")) ** 2).mean())
    y_var = float(((F["test"]["y"] - F["test"]["y"].mean()) ** 2).mean())
    r2 = 1.0 - test_mse / y_var

    meta = dict(dataset=args.dataset, seed=args.seed, variant=args.variant,
                width=width, params=n_params, best_epoch=ckpt["epoch"],
                best_val_mae=ckpt["val_mae"], test_mae=test_mae,
                test_mse=test_mse, test_r2=r2, n_test=len(F["test"]["y"]),
                wall_min=round((time.time() - t0) / 60, 2), cfg=cfg)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print("DONE", json.dumps({k: meta[k] for k in
                              ("dataset", "seed", "variant", "params",
                               "best_epoch", "test_mae", "test_r2")}))


if __name__ == "__main__":
    main()