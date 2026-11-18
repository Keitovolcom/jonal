"""Compare toy2d conditions (seed mean ± std): does EDD appear as the input dimension grows?

python3 toy2d/compare_toy.py --sweep-dir results/toy2d-XXX
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_toy import detect_phases  # noqa: E402


def load(sweep: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(sweep.glob("toy2d_*")):
        if not (run / "DONE").exists():
            continue
        cfg = json.loads((run / "config.json").read_text())
        m = pd.read_csv(run / "metrics.csv")
        ph = detect_phases(m)
        cond = f"d={cfg['dim']}" + (f", σ={cfg['sigma']:g}" if cfg["dim"] > 2 else "") + \
            f", γ={cfg['label_noise']:g}, lr={cfg['lr']:g}"
        rows.append(dict(cond=cond, dim=cfg["dim"], sigma=cfg["sigma"], noise=cfg["label_noise"],
                         lr=cfg["lr"], seed=cfg["seed"], run=run, metrics=m,
                         epoch_noisy_fit=_first(m, "train_error_noisy", 0.01),
                         min_test_any=m["test_error"].min(), **ph))
    return pd.DataFrame(rows)


def _first(m, col, thr):
    if m[col].isna().all():
        return np.nan
    hit = m.loc[m[col] <= thr, "epoch"]
    return int(hit.iloc[0]) if len(hit) else np.nan


# One-factor sweeps: (name, varied factor, fixed conditions, title). Only runs matching every
# fixed condition are compared; d=2 has no extra dims, so sigma is not fixed for it.
SWEEPS = [
    ("dim_g0p2_s0p3_lr1e-3", "dim", dict(noise=0.2, lr=1e-3, sigma=0.3), "γ=0.2, σ=0.3 (d>2), lr=1e-3"),
    ("dim_g0p2_s0p3_lr1e-4", "dim", dict(noise=0.2, lr=1e-4, sigma=0.3), "γ=0.2, σ=0.3 (d>2), lr=1e-4"),
    ("sigma_d50_g0p2_lr1e-3", "sigma", dict(dim=50, noise=0.2, lr=1e-3), "d=50, γ=0.2, lr=1e-3"),
    ("noise_d2_lr1e-3", "noise", dict(dim=2, lr=1e-3), "d=2, lr=1e-3"),
    ("noise_d50_s0p3_lr1e-3", "noise", dict(dim=50, sigma=0.3, lr=1e-3), "d=50, σ=0.3, lr=1e-3"),
    ("lr_d2_g0p2", "lr", dict(dim=2, noise=0.2), "d=2, γ=0.2"),
    ("lr_d10_s0p3_g0p2", "lr", dict(dim=10, sigma=0.3, noise=0.2), "d=10, σ=0.3, γ=0.2"),
    ("lr_d50_s0p3_g0p2", "lr", dict(dim=50, sigma=0.3, noise=0.2), "d=50, σ=0.3, γ=0.2"),
]
FACTOR_LABEL = {"dim": "d", "sigma": "σ", "noise": "γ", "lr": "lr"}
MAP_EPOCHS = [1, 10, 100, 1000, 20000]


def select(df, factor, fixed):
    keep = np.ones(len(df), bool)
    for k, v in fixed.items():
        col = df[k].to_numpy()
        match = np.isclose(col, v)
        if k == "sigma":
            match |= df["dim"].to_numpy() == 2
        keep &= match
    return df[keep].sort_values(factor)


def curve_panel(sub, factor, path, title):
    cols = [("test_error", "test error"), ("train_error_noisy", "$E_N$"),
            ("train_error_clean", "$E_C$"), ("test_loss", "test loss")]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 3.6))
    for k, (val, g) in enumerate(sub.groupby(factor, sort=True)):
        ep = g["metrics"].iloc[0]["epoch"].clip(lower=0.8).to_numpy()
        color = plt.cm.viridis(k / max(sub[factor].nunique() - 1, 1) * 0.85)
        for ax, (col, lab) in zip(axes, cols):
            ys = np.stack([m[col].to_numpy() for m in g["metrics"]])
            if np.isnan(ys).all():
                continue
            mu, sd = np.nanmean(ys, 0), np.nanstd(ys, 0)
            ax.plot(ep, mu, color=color, label=f"{FACTOR_LABEL[factor]}={val:g}")
            ax.fill_between(ep, mu - sd, mu + sd, color=color, alpha=0.2, lw=0)
            ax.set_title(lab)
    for ax in axes:
        ax.set_xscale("log"); ax.grid(alpha=0.3); ax.set_xlabel("epoch")
    axes[3].set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{FACTOR_LABEL[factor]} sweep  |  fixed: {title}  (mean ± std over {sub['seed'].nunique()} seeds)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def map_panel(sub, factor, path, title, seed=0):
    """Rows = factor values (one seed), columns = epochs; eps-averaged argmax map."""
    from plot_toy import CMAP, scatter_points
    runs = sub[sub["seed"] == seed]
    fig, axes = plt.subplots(len(runs), len(MAP_EPOCHS), figsize=(2.3 * len(MAP_EPOCHS), 2.4 * len(runs)),
                             squeeze=False)
    for r, (_, row) in enumerate(runs.iterrows()):
        run = row["run"]
        cfg = json.loads((run / "config.json").read_text())
        ev = np.load(run / "eval_epochs.npy")
        pred = np.load(run / "grid_pred.npy", mmap_mode="r")
        data = dict(np.load(run / "data.npz"))
        lim = cfg["grid_limit"]
        m = row["metrics"]
        for c, e in enumerate(MAP_EPOCHS):
            i = int(np.argmin(np.abs(ev - e)))
            ax = axes[r, c]
            ax.imshow(pred[i], cmap=CMAP, vmin=-0.5, vmax=2.5, extent=(-lim, lim, -lim, lim),
                      interpolation="nearest")
            scatter_points(ax, data, lim, small=True)
            ax.set_title(f"ep {ev[i]}  test {m['test_error'].iloc[i]:.3f}", fontsize=8)
        axes[r, 0].set_ylabel(f"{FACTOR_LABEL[factor]}={row[factor]:g}", fontsize=10)
    fig.suptitle(f"{FACTOR_LABEL[factor]} sweep  |  fixed: {title}  (seed {seed}; argmax, "
                 "eps-averaged for d>2)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    a = ap.parse_args()
    df = load(a.sweep_dir)
    out = a.sweep_dir / "comparison"
    out.mkdir(exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    keep = ["cond", "seed", "min_epoch", "min_test", "peak_epoch", "peak_test", "final_test",
            "rise", "second_descent", "edd", "epoch_noisy_fit", "min_test_any"]
    df[keep].sort_values(["cond", "seed"]).to_csv(out / "per_run_summary.csv", index=False)
    agg = df.groupby("cond")[keep[2:]].agg(["mean", "std"])
    agg.to_csv(out / "condition_summary.csv")
    print(df[keep].sort_values(["cond", "seed"]).to_string(index=False))

    for name, factor, fixed, title in SWEEPS:
        sub = select(df, factor, fixed)
        if sub[factor].nunique() < 2:
            continue
        curve_panel(sub, factor, out / f"curves_{name}.png", title)
        map_panel(sub, factor, out / f"maps_{name}.png", title)
        s = sub.groupby(factor)[["min_test", "peak_test", "final_test", "second_descent", "epoch_noisy_fit"]]
        s.mean().round(3).to_csv(out / f"summary_{name}.csv")
        print(f"\n[{name}] fixed {title}\n" + s.mean().round(3).to_string())


if __name__ == "__main__":
    main()
