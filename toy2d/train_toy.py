"""Epoch-wise double descent on 2-D, 3-class Gaussian blobs (optionally embedded in d dims).

Class information lives only in (x1, x2); dims 3..d are label-independent N(0, sigma^2)
noise drawn independently for every train/test sample. At every evaluation epoch this
script logs the four error series + losses, per-sample predictions, the prediction map
on a 300x300 grid of the (x1, x2) plane, and predictions along pair lines x(alpha).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

NUM_CLASSES = 3
GRID_LIMIT = 2.75
ALPHAS = np.round(np.arange(-0.5, 1.5 + 1e-9, 0.01), 2)
PAIR_MODES = ("clean-clean", "clean-noisy", "test-test")


def token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--sigma", type=float, default=0.3, help="std of extra dims (ignored when dim=2)")
    p.add_argument("--label-noise", type=float, default=0.2)
    p.add_argument("--blob-std", type=float, default=0.5)
    p.add_argument("--n-per-class", type=int, default=1000)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--num-log-evals", type=int, default=200, help="log-spaced evals after epoch 100")
    p.add_argument("--grid-size", type=int, default=300)
    p.add_argument("--grid-eps-samples", type=int, default=16)
    p.add_argument("--num-pairs", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results/toy2d"))
    p.add_argument("--name", default=None)
    return p.parse_args()


def run_name(a: argparse.Namespace) -> str:
    sigma = "none" if a.dim == 2 else token(a.sigma)
    return (f"toy2d_d-{a.dim}_sigma-{sigma}_noise-{token(a.label_noise)}"
            f"_width-{a.width}_lr-{token(a.lr)}_seed-{a.seed}")


def eval_epochs(total: int, num_log: int) -> list[int]:
    dense = list(range(0, min(100, total) + 1))
    if total <= 100:
        return dense
    logs = np.unique(np.round(np.logspace(2, math.log10(total), num_log)).astype(int))
    return sorted(set(dense) | set(logs.tolist()) | {total})


def make_data(a: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(a.seed)
    angles = np.pi / 2 + 2 * np.pi * np.arange(NUM_CLASSES) / NUM_CLASSES
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    def blobs() -> tuple[np.ndarray, np.ndarray]:
        y = np.repeat(np.arange(NUM_CLASSES), a.n_per_class)
        x2 = centers[y] + a.blob_std * rng.standard_normal((len(y), 2))
        extra = a.sigma * rng.standard_normal((len(y), a.dim - 2))
        return np.concatenate([x2, extra], axis=1).astype(np.float32), y

    x_train, y_orig = blobs()
    x_test, y_test = blobs()
    n = len(y_orig)
    noisy_idx = np.sort(rng.choice(n, int(n * a.label_noise), replace=False))
    y_train = y_orig.copy()
    y_train[noisy_idx] = (y_orig[noisy_idx] + rng.integers(1, NUM_CLASSES, len(noisy_idx))) % NUM_CLASSES
    is_noisy = np.zeros(n, dtype=bool)
    is_noisy[noisy_idx] = True
    return dict(x_train=x_train, y_train=y_train, y_orig=y_orig, is_noisy=is_noisy,
                noisy_idx=noisy_idx, x_test=x_test, y_test=y_test, centers=centers)


def make_pairs(data: dict[str, np.ndarray], num_pairs: int, seed: int) -> dict[str, np.ndarray]:
    """Nearest-neighbour pairs, same definitions as interpolation.py (L2 in full input space)."""
    rng = np.random.default_rng(seed + 10_000)
    out = {}
    clean = np.flatnonzero(~data["is_noisy"])
    noisy = data["noisy_idx"]

    def nearest(query_x, pool_idx, pool_x):
        d = ((query_x[:, None, :] - pool_x[None, :, :]) ** 2).sum(-1)
        return d

    for mode in PAIR_MODES:
        if mode == "clean-clean":
            q = rng.choice(clean, num_pairs, replace=False)
            xq, lq = data["x_train"][q], data["y_orig"][q]
            pool, xp, lp = clean, data["x_train"][clean], data["y_orig"][clean]
        elif mode == "clean-noisy":
            if len(noisy) == 0:
                continue
            q = rng.choice(noisy, min(num_pairs, len(noisy)), replace=False)
            xq, lq = data["x_train"][q], data["y_orig"][q]
            pool, xp, lp = clean, data["x_train"][clean], data["y_orig"][clean]
        else:
            q = rng.choice(len(data["y_test"]), num_pairs, replace=False)
            xq, lq = data["x_test"][q], data["y_test"][q]
            pool = np.arange(len(data["y_test"]))
            xp, lp = data["x_test"], data["y_test"]
        d = nearest(xq, pool, xp)
        d[lq[:, None] != lp[None, :]] = np.inf
        if mode != "clean-noisy":
            d[q[:, None] == pool[None, :]] = np.inf
        j = d.argmin(1)
        x0, x1 = xq, xp[j]
        if mode == "test-test":
            orig, noise = lq, lq
        else:
            orig, noise = lq, data["y_train"][q]
        out[mode] = dict(i0=q, i1=pool[j], x0=x0, x1=x1, original=orig, noisy=noise,
                         distance=np.sqrt(d[np.arange(len(q)), j]))
    return out


class MLP(nn.Module):
    def __init__(self, dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, width),
                                 nn.ReLU(), nn.Linear(width, NUM_CLASSES))

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def predict(model, x: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    return torch.cat([F.softmax(model(x[i:i + chunk]), 1) for i in range(0, len(x), chunk)])


@torch.no_grad()
def grid_probs(model, grid2: torch.Tensor, dim: int, sigma: float, m: int, seed: int,
               zero_eps: bool) -> torch.Tensor:
    """Mean softmax over m fixed eps draws per grid point (identical draws every epoch)."""
    if dim == 2:
        return predict(model, grid2)
    if zero_eps:
        return predict(model, torch.cat([grid2, grid2.new_zeros(len(grid2), dim - 2)], 1))
    gen = torch.Generator(device=grid2.device)
    out, chunk = [], 8192
    for c, i in enumerate(range(0, len(grid2), chunk)):
        g = grid2[i:i + chunk]
        gen.manual_seed(seed * 1_000_003 + c)
        eps = sigma * torch.randn(len(g), m, dim - 2, generator=gen, device=g.device)
        x = torch.cat([g[:, None, :].expand(-1, m, -1), eps], 2).reshape(-1, dim)
        out.append(F.softmax(model(x), 1).reshape(len(g), m, NUM_CLASSES).mean(1))
    return torch.cat(out)


def plane_inst_s(pred: np.ndarray) -> float:
    h = pred[:, 1:] != pred[:, :-1]
    v = pred[1:, :] != pred[:-1, :]
    return float((h.sum() + v.sum()) / (h.size + v.size))


def main() -> None:
    a = parse_args()
    if a.dim == 2:
        a.sigma = 0.0
    name = a.name or run_name(a)
    out = a.output_dir / name
    if (out / "metrics.csv").exists():
        raise SystemExit(f"{out} already exists")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(f"cuda:{a.gpu}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    data = make_data(a)
    pairs = make_pairs(data, a.num_pairs, a.seed)
    np.savez(out / "data.npz", **data)
    np.savez(out / "pairs.npz", **{f"{m}__{k}": v for m, d in pairs.items() for k, v in d.items()})
    epochs_to_eval = eval_epochs(a.epochs, a.num_log_evals)
    cfg = dict(vars(a), output_dir=str(a.output_dir), name=name, num_classes=NUM_CLASSES,
               grid_limit=GRID_LIMIT, eval_epochs=epochs_to_eval, alphas=ALPHAS.tolist(),
               pair_modes=list(pairs), optimizer="adam", weight_decay=0.0, scheduler=None,
               grid_eps="fixed across epochs; per-chunk generator seeded by seed",
               torch_version=torch.__version__)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(a.seed)
    model = MLP(a.dim, a.width).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    xtr = torch.tensor(data["x_train"], device=dev)
    ytr = torch.tensor(data["y_train"], device=dev)
    yor = torch.tensor(data["y_orig"], device=dev)
    noisy = torch.tensor(data["is_noisy"], device=dev)
    xte = torch.tensor(data["x_test"], device=dev)
    yte = torch.tensor(data["y_test"], device=dev)
    lin = torch.linspace(-GRID_LIMIT, GRID_LIMIT, a.grid_size, device=dev)
    gy, gx = torch.meshgrid(lin.flip(0), lin, indexing="ij")  # row 0 = top (x2 max)
    grid2 = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    al = torch.tensor(ALPHAS, device=dev, dtype=torch.float32)[None, :, None]
    lines = {m: (1 - al) * torch.tensor(p["x0"], device=dev)[:, None] + al * torch.tensor(p["x1"], device=dev)[:, None]
             for m, p in pairs.items()}

    T, G = len(epochs_to_eval), a.grid_size
    store = dict(
        grid_pred=np.zeros((T, G, G), np.uint8), grid_prob=np.zeros((T, G, G, 3), np.uint8),
        train_prob=np.zeros((T, len(ytr), 3), np.float16), test_prob=np.zeros((T, len(yte), 3), np.float16),
        **{f"line_pred__{m}": np.zeros((T, len(p["x0"]), len(ALPHAS)), np.uint8) for m, p in pairs.items()})
    if a.dim > 2:
        store["grid0_pred"] = np.zeros((T, G, G), np.uint8)
        store["grid0_prob"] = np.zeros((T, G, G, 3), np.uint8)

    fields = ["epoch", "test_error", "train_error", "train_error_clean", "train_error_noisy",
              "noisy_pred_original", "noisy_pred_other", "test_loss", "train_loss", "train_loss_clean",
              "train_loss_noisy", "plane_inst_s", "plane_inst_t", "plane0_inst_s", "plane0_inst_t",
              "elapsed_sec"]
    fcsv = open(out / "metrics.csv", "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    writer.writeheader()
    log = open(out / "train.log", "a")
    t0 = time.time()

    def evaluate(ti: int, epoch: int) -> None:
        model.eval()
        ptr, pte = predict(model, xtr), predict(model, xte)
        lptr = -torch.log(ptr.clamp_min(1e-12))
        nll_tr = lptr.gather(1, ytr[:, None]).squeeze(1)
        nll_te = -torch.log(pte.clamp_min(1e-12)).gather(1, yte[:, None]).squeeze(1)
        htr, hte = ptr.argmax(1), pte.argmax(1)
        wrong = (htr != ytr).float()
        row = dict(epoch=epoch, test_error=(hte != yte).float().mean().item(),
                   train_error=wrong.mean().item(), train_error_clean=wrong[~noisy].mean().item(),
                   test_loss=nll_te.mean().item(), train_loss=nll_tr.mean().item(),
                   train_loss_clean=nll_tr[~noisy].mean().item())
        if noisy.any():
            row.update(train_error_noisy=wrong[noisy].mean().item(),
                       noisy_pred_original=(htr[noisy] == yor[noisy]).float().mean().item(),
                       noisy_pred_other=((htr[noisy] != yor[noisy]) & (htr[noisy] != ytr[noisy])).float().mean().item(),
                       train_loss_noisy=nll_tr[noisy].mean().item())
        store["train_prob"][ti] = ptr.cpu().numpy()
        store["test_prob"][ti] = pte.cpu().numpy()
        for m, xl in lines.items():
            store[f"line_pred__{m}"][ti] = predict(model, xl.reshape(-1, a.dim)).argmax(1).reshape(xl.shape[:2]).cpu().numpy()
        for key, zero in (("grid", False), ("grid0", True)):
            if key == "grid0" and a.dim == 2:
                continue
            pg = grid_probs(model, grid2, a.dim, a.sigma, a.grid_eps_samples, a.seed, zero).reshape(G, G, 3)
            pred = pg.argmax(-1).cpu().numpy().astype(np.uint8)
            store[f"{key}_pred"][ti] = pred
            store[f"{key}_prob"][ti] = (pg * 255).round().clamp(0, 255).byte().cpu().numpy()
            pk = "plane" if key == "grid" else "plane0"
            row[f"{pk}_inst_s"] = plane_inst_s(pred)
            if ti > 0:
                row[f"{pk}_inst_t"] = float((pred != store[f"{key}_pred"][ti - 1]).mean())
        row["elapsed_sec"] = round(time.time() - t0, 1)
        writer.writerow(row)
        fcsv.flush()
        if ti % 10 == 0 or epoch == a.epochs:
            msg = (f"[{name}] ep {epoch:6d} test {row['test_error']:.4f} train {row['train_error']:.4f} "
                   f"clean {row['train_error_clean']:.4f} noisy {row.get('train_error_noisy', float('nan')):.4f} "
                   f"inst_s {row['plane_inst_s']:.4f} t={row['elapsed_sec']:.0f}s")
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        model.train()

    gen = torch.Generator(device=dev)
    gen.manual_seed(a.seed)
    n = len(ytr)
    ti = 0
    evaluate(ti, 0)
    ti += 1
    for epoch in range(1, a.epochs + 1):
        perm = torch.randperm(n, device=dev, generator=gen)
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if epoch == epochs_to_eval[ti]:
            evaluate(ti, epoch)
            ti += 1
            if ti == T:
                break
    fcsv.close()
    np.save(out / "eval_epochs.npy", np.array(epochs_to_eval))
    for k, v in store.items():
        np.save(out / f"{k}.npy", v)
    torch.save(model.state_dict(), out / "final_model.pt")
    (out / "DONE").write_text(f"{time.time() - t0:.1f}\n")
    print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
