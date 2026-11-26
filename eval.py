"""Evaluate a trained SpecFuse head on the held-out test split.

Examples
--------
    python eval.py --run runs/full_bl2_w256/h1n1_566/s42

    # MC-dropout estimate used for the reported test MAE
    python eval.py --run runs/full_bl2_w256/h1n1_566/s42 --mc 100
"""
import argparse
import json
import os

import numpy as np
import torch

import spectral
from model import build_model

DS_NAME = {"h1n1_566": "H1N1", "h3n2_566_asym": "H3N2",
           "bvic_585_asym": "B/Victoria"}


def metrics(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - y
    mae = float(np.abs(err).mean())
    mse = float((err ** 2).mean())
    r2 = 1.0 - mse / float(((y - y.mean()) ** 2).mean())
    return dict(mae=mae, mse=mse, rmse=float(np.sqrt(mse)), r2=r2, n=int(len(y)))


@torch.no_grad()
def forward(model, F, split, batch=1024):
    sp = F[split]
    out = torch.empty(len(sp["src"]), device=sp["src"].device)
    for i in range(0, len(sp["src"]), batch):
        j = slice(i, min(i + batch, len(sp["src"])))
        ts = model.tokens(F["z"][sp["src"][j]], F["p"][sp["src"][j]],
                          F["v"][sp["src"][j]])
        tr = model.tokens(F["z"][sp["dst"][j]], F["p"][sp["dst"][j]],
                          F["v"][sp["dst"][j]])
        out[j] = model(ts, tr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="directory containing best.pt")
    ap.add_argument("--split", default="test",
                    choices=["inner_val", "test"])
    ap.add_argument("--mc", type=int, default=0,
                    help="number of Monte-Carlo dropout passes (0 = off)")
    args = ap.parse_args()

    ckpt = torch.load(os.path.join(args.run, "best.pt"), map_location="cpu",
                      weights_only=False)
    cfg = ckpt["cfg"]
    parts = args.run.replace("\\", "/").rstrip("/").split("/")
    dataset, seed = parts[-2], int(parts[-1].lstrip("s"))
    variant = cfg.get("variant", "full")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    F = spectral.dataset(dataset, seed, cfg["n_bands"], cfg["q"], device)
    model = build_model(variant=variant, n_bands=cfg["n_bands"], d_v=F["d_v"],
                        dim=cfg["dim"], heads=cfg["heads"], ffn=cfg["ffn"],
                        dropout=cfg["dropout"], blocks=cfg["blocks"],
                        mlp_width=cfg.get("mlp_width", 256)).to(device)
    model.load_state_dict(ckpt["state"])

    if args.mc <= 0:
        model.eval()
        res = metrics(F[args.split]["y"].cpu().numpy(),
                      forward(model, F, args.split).cpu().numpy())
        res["mode"] = "deterministic"
    else:
        model.train()                     # keep dropout active
        preds = torch.stack([forward(model, F, args.split)
                             for _ in range(args.mc)])
        pred = preds.mean(0)
        res = metrics(F[args.split]["y"].cpu().numpy(), pred.cpu().numpy())
        res["mode"] = f"mc{args.mc}"
        res["mc_std"] = float(preds.std(0).mean())

    res.update(dataset=dataset, dataset_name=DS_NAME.get(dataset, dataset),
               seed=seed, variant=variant, params=ckpt["state"] and None,
               best_epoch=ckpt["epoch"], best_val_mae=ckpt["val_mae"])
    res.pop("params")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()