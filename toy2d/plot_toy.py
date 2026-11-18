"""Figures, animations and extra analyses for one toy2d run (output of train_toy.py).

python3 toy2d/plot_toy.py --run-dir results/toy2d-XXX/<run> [--phase-epochs 30,600] [--no-anim]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

CLASS_COLORS = np.array([[0.85, 0.20, 0.20], [0.20, 0.65, 0.25], [0.20, 0.35, 0.85]])
CMAP = ListedColormap(CLASS_COLORS * 0.55 + 0.45)  # pale fills so points stay visible
SERIES = {"test": "#d62728", "train": "#1f77b4", "clean": "#2ca02c", "noisy": "#9467bd"}


def smooth(y: np.ndarray, k: int = 5) -> np.ndarray:
    s = pd.Series(y).rolling(k, center=True, min_periods=1).median()
    return s.to_numpy()


def detect_phases(m: pd.DataFrame) -> dict:
    """Phase I/II boundary = early test-error minimum, II/III boundary = later test-error peak."""
    ep = m["epoch"].to_numpy()
    s = smooth(m["test_error"].to_numpy())
    valid = ep >= 1
    if m["train_error_noisy"].notna().any():
        en = m["train_error_noisy"].to_numpy()
        # minimum before noisy samples are half memorised
        half = np.flatnonzero(valid & (en < 0.5 * np.nanmax(en[valid])))
        lim = half[0] if len(half) else len(ep) - 1
    else:
        lim = len(ep) - 1
    cand = np.flatnonzero(valid[: lim + 1])
    i1 = cand[np.argmin(s[cand])] if len(cand) else 1
    i2 = i1 + int(np.argmax(s[i1:]))
    out = dict(min_epoch=int(ep[i1]), min_test=float(s[i1]), peak_epoch=int(ep[i2]),
               peak_test=float(s[i2]), final_test=float(s[-1]),
               rise=float(s[i2] - s[i1]), second_descent=float(s[i2] - s[-1]))
    out["edd"] = bool(out["rise"] > 0.01 and out["second_descent"] > 0.5 * out["rise"])
    return out


def phase_lines(ax, phases):
    for e in phases:
        ax.axvline(max(e, 1), color="gray", ls="--", lw=0.8)


def plot_curves(m: pd.DataFrame, phases: list[int], title: str, path: Path) -> None:
    ep = m["epoch"].clip(lower=0.8)
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8), sharex=True)
    ax = axes[0]
    ax.plot(ep, m["test_error"], c=SERIES["test"], label="test")
    ax.plot(ep, m["train_error"], c=SERIES["train"], label="train (noisy labels)")
    ax.set_ylabel("error")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(ep, m["train_error_clean"], c=SERIES["clean"], label="$E_C$ (clean)")
    if m["train_error_noisy"].notna().any():
        ax.plot(ep, m["train_error_noisy"], c=SERIES["noisy"], label="$E_N$ (noisy)")
        ax.plot(ep, m["noisy_pred_original"], c=SERIES["noisy"], ls=":", label="$X_N$ predicts original")
    ax.set_ylabel("train error")
    ax.legend(fontsize=8)
    ax = axes[2]
    ax.plot(ep, m["test_loss"], c=SERIES["test"], label="test")
    ax.plot(ep, m["train_loss"], c=SERIES["train"], label="train")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("log")
        phase_lines(ax, phases)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("epoch")
    axes[0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter_points(ax, data, lim, small=False):
    x, yo, yt, nz = data["x_train"], data["y_orig"], data["y_train"], data["is_noisy"]
    c = ~nz
    ax.scatter(x[c, 0], x[c, 1], s=2 if small else 3, c=CLASS_COLORS[yo[c]], lw=0, alpha=0.6)
    # noisy: fill = noisy label, ring = original label
    ax.scatter(x[nz, 0], x[nz, 1], s=10 if small else 14, c=CLASS_COLORS[yt[nz]],
               edgecolors=CLASS_COLORS[yo[nz]], linewidths=1.0, marker="o")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def phase_of(epoch, phases):
    if not phases:
        return ""
    if epoch <= phases[0]:
        return "Phase I"
    if len(phases) < 2 or epoch <= phases[1]:
        return "Phase II"
    return "Phase III"


def map_figure(run, epochs, data, cfg, key, phases, path, overlay=True):
    lim = cfg["grid_limit"]
    ext = (-lim, lim, -lim, lim)
    pred = np.load(run / f"{key}_pred.npy", mmap_mode="r")
    prob = np.load(run / f"{key}_prob.npy", mmap_mode="r")
    ev = np.load(run / "eval_epochs.npy")
    idx = [int(np.argmin(np.abs(ev - e))) for e in epochs]
    fig, axes = plt.subplots(2, len(idx), figsize=(2.6 * len(idx), 5.4))
    for j, i in enumerate(idx):
        axes[0, j].imshow(pred[i], cmap=CMAP, vmin=-0.5, vmax=2.5, extent=ext, interpolation="nearest")
        axes[1, j].imshow(np.asarray(prob[i]) / 255.0 * 0.8 + 0.1, extent=ext, interpolation="bilinear")
        for r in range(2):
            if overlay:
                scatter_points(axes[r, j], data, lim, small=True)
            else:
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[0, j].set_title(f"ep {ev[i]}  {phase_of(ev[i], phases)}", fontsize=9)
    axes[0, 0].set_ylabel("argmax")
    axes[1, 0].set_ylabel("prob RGB")
    fig.suptitle(f"{cfg['name']}  [{key}]", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_animation(run, data, cfg, m, phases, keys, path, fps=12):
    lim = cfg["grid_limit"]
    ext = (-lim, lim, -lim, lim)
    ev = np.load(run / "eval_epochs.npy")
    arrs = [(np.load(run / f"{k}_pred.npy", mmap_mode="r"), np.load(run / f"{k}_prob.npy", mmap_mode="r")) for k in keys]
    ncol = 2 * len(keys) + 1
    fig = plt.figure(figsize=(3.3 * ncol, 3.9))
    gs = fig.add_gridspec(1, ncol)
    ims = []
    for c, (k, (pr, pb)) in enumerate(zip(keys, arrs)):
        a1 = fig.add_subplot(gs[0, 2 * c]); a2 = fig.add_subplot(gs[0, 2 * c + 1])
        i1 = a1.imshow(pr[0], cmap=CMAP, vmin=-0.5, vmax=2.5, extent=ext, interpolation="nearest")
        i2 = a2.imshow(np.asarray(pb[0]) / 255 * 0.8 + 0.1, extent=ext, interpolation="bilinear")
        scatter_points(a1, data, lim, small=True)
        scatter_points(a2, data, lim, small=True)
        sub = "" if k == "grid" else " (eps=0)"
        a1.set_title(f"argmax{sub}", fontsize=9); a2.set_title(f"prob RGB{sub}", fontsize=9)
        ims.append((i1, i2))
    ac = fig.add_subplot(gs[0, -1])
    e = m["epoch"].clip(lower=0.8)
    ac.plot(e, m["test_error"], c=SERIES["test"], label="test")
    ac.plot(e, m["train_error"], c=SERIES["train"], label="train")
    if m["train_error_noisy"].notna().any():
        ac.plot(e, m["train_error_noisy"], c=SERIES["noisy"], lw=0.8, label="$E_N$")
    ac.set_xscale("log"); ac.set_ylim(0, 1); ac.legend(fontsize=7); ac.grid(alpha=0.3)
    phase_lines(ac, phases)
    vline = ac.axvline(0.8, color="k")
    title = fig.suptitle("", fontsize=10)
    fig.tight_layout()

    def update(i):
        for (i1, i2), (pr, pb) in zip(ims, arrs):
            i1.set_data(pr[i]); i2.set_data(np.asarray(pb[i]) / 255 * 0.8 + 0.1)
        vline.set_xdata([max(ev[i], 0.8)] * 2)
        row = m.iloc[i]
        title.set_text(f"{cfg['name']}   epoch {ev[i]}   {phase_of(ev[i], phases)}   "
                       f"test err {row.test_error:.3f}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(ev), blit=False)
    anim.save(path.with_suffix(".mp4"), writer=animation.FFMpegWriter(fps=fps, bitrate=2400), dpi=100)
    anim.save(path.with_suffix(".gif"), writer=animation.PillowWriter(fps=fps), dpi=60)
    plt.close(fig)


def plane_instability(run, cfg, m, phases, key, outdir):
    pred = np.load(run / f"{key}_pred.npy")
    ev = np.load(run / "eval_epochs.npy")
    inst_s =((pred[:, :, 1:] != pred[:, :, :-1]).sum((1, 2)) + (pred[:, 1:, :] != pred[:, :-1, :]).sum((1, 2))) / \
        (pred[0, :, 1:].size + pred[0, 1:, :].size)
    flips = pred[1:] != pred[:-1]
    inst_t = np.concatenate([[np.nan], flips.mean((1, 2))])
    gaps = np.concatenate([[np.nan], np.diff(ev)])
    pd.DataFrame(dict(epoch=ev, inst_s=inst_s, inst_t=inst_t, inst_t_per_epoch=inst_t / gaps)).to_csv(
        outdir / f"plane_instability_{key}.csv", index=False)
    cum = flips.sum(0)
    np.save(outdir / f"flip_count_{key}.npy", cum.astype(np.uint16))
    # cumulative flip maps per phase
    # flips[i] is the change from eval i to i+1; skip the init -> epoch-1 jump
    bounds = [1] + [int(np.searchsorted(ev, p)) for p in phases] + [len(ev) - 1]
    return ev, inst_s, inst_t, gaps, flips, bounds


def plot_instability(ev, inst_s, inst_t, gaps, flips, bounds, data, cfg, phases, key, path):
    lim = cfg["grid_limit"]
    ext = (-lim, lim, -lim, lim)
    e = np.clip(ev, 0.8, None)
    nseg = len(bounds) - 1
    fig = plt.figure(figsize=(4 + 3 * (nseg + 1), 3.6))
    gs = fig.add_gridspec(2, nseg + 2, width_ratios=[2.2] + [1] * (nseg + 1))
    a = fig.add_subplot(gs[0, 0]); a.plot(e, inst_s, c="k"); a.set_ylabel("plane INST_S")
    b = fig.add_subplot(gs[1, 0], sharex=a); b.plot(e, inst_t, c="C1", label="per eval step")
    b2 = b.twinx(); b2.plot(e, inst_t / gaps, c="C0", lw=0.8, label="per epoch"); b2.set_yscale("log")
    b.set_ylabel("plane INST_T"); b2.set_ylabel("per epoch", color="C0")
    for ax in (a, b):
        ax.set_xscale("log"); phase_lines(ax, phases); ax.grid(alpha=0.3)
    b.set_xlabel("epoch")
    names = ["Phase I", "Phase II", "Phase III"] if len(phases) == 2 else [f"seg {i}" for i in range(nseg)]
    segs = [(bounds[i], bounds[i + 1], names[i] if i < len(names) else "") for i in range(nseg)] + [(1, len(ev) - 1, "all")]
    for j, (s0, s1, nm) in enumerate(segs):
        ax = fig.add_subplot(gs[:, j + 1])
        cnt = flips[s0:s1].sum(0)
        im = ax.imshow(np.log1p(cnt), cmap="magma", extent=ext)
        ax.scatter(data["x_train"][data["is_noisy"], 0], data["x_train"][data["is_noisy"], 1],
                   s=1.5, c="cyan", lw=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"log(1+flips) {nm}\nep {ev[s0]}–{ev[s1]}", fontsize=8)
    fig.suptitle(f"{cfg['name']} [{key}]  (cyan = noisy train points)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def occupancy(run, cfg, phases, outdir):
    pairs = np.load(run / "pairs.npz")
    ev = np.load(run / "eval_epochs.npy")
    al = np.array(cfg["alphas"])
    rows = []
    reps = {"init": 1}
    if len(phases) >= 1:
        reps["I/II boundary"] = phases[0]
    if len(phases) >= 2:
        reps["II/III boundary (peak)"] = phases[1]
    reps["final"] = int(ev[-1])
    modes = cfg["pair_modes"]
    fig, axes = plt.subplots(len(modes), len(reps), figsize=(3 * len(reps), 2.4 * len(modes)), squeeze=False)
    for r, mode in enumerate(modes):
        pr = np.load(run / f"line_pred__{mode}.npy")
        orig = pairs[f"{mode}__original"][None, :, None]
        noi = pairs[f"{mode}__noisy"][None, :, None]
        o = (pr == orig).mean(1)
        n = ((pr == noi) & (noi != orig)).mean(1)
        ot = 1 - o - n
        inst_s = (pr[:, :, 1:] != pr[:, :, :-1]).sum(2).mean(1)
        inst_t = np.concatenate([[np.nan], (pr[1:] != pr[:-1]).mean((1, 2))])
        for ti, e in enumerate(ev):
            rows.append(dict(mode=mode, epoch=e, line_inst_s=inst_s[ti], line_inst_t=inst_t[ti],
                             original_mean=o[ti].mean(), noisy_mean=n[ti].mean(), other_mean=ot[ti].mean(),
                             other_in_0_1=ot[ti][(al >= 0) & (al <= 1)].mean()))
        for c, (nm, e) in enumerate(reps.items()):
            ti = int(np.argmin(np.abs(ev - e)))
            ax = axes[r, c]
            ax.stackplot(al, o[ti], n[ti], ot[ti], colors=["#2ca02c", "#9467bd", "#ff7f0e"],
                         labels=["original", "noisy", "other"])
            ax.axvline(0, c="k", lw=0.6); ax.axvline(1, c="k", lw=0.6)
            ax.set_ylim(0, 1); ax.set_xlim(al[0], al[-1])
            ax.set_title(f"{mode} | {nm} (ep {ev[ti]})", fontsize=8)
    axes[0, -1].legend(fontsize=7, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("alpha (x0 → x1)")
    fig.tight_layout()
    fig.savefig(outdir / "occupancy.png", dpi=130)
    plt.close(fig)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "line_metrics.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    for mode in modes:
        d = df[df["mode"] == mode]
        e = d["epoch"].clip(lower=0.8)
        axes[0].plot(e, d["line_inst_s"], label=mode)
        axes[1].plot(e, d["line_inst_t"], label=mode)
        axes[2].plot(e, d["other_in_0_1"], label=mode)
    for ax, t in zip(axes, ["line INST_S (#label changes on segment)", "line INST_T", "'other' share on α∈[0,1]"]):
        ax.set_xscale("log"); ax.set_title(t, fontsize=9); phase_lines(ax, phases); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "line_metrics.png", dpi=130)
    plt.close(fig)


def per_point(run, data, cfg, phases, outdir):
    ev = np.load(run / "eval_epochs.npy")
    tr = np.load(run / "train_prob.npy").argmax(-1)
    te = np.load(run / "test_prob.npy").argmax(-1)
    bounds = [1] + [int(np.searchsorted(ev, p)) for p in phases] + [len(ev) - 1]
    names = ["I", "II", "III"][: len(bounds) - 1]
    rows = []
    groups = {"train clean": tr[:, ~data["is_noisy"]], "train noisy": tr[:, data["is_noisy"]], "test": te}
    for g, p in groups.items():
        if p.shape[1] == 0:
            continue
        for k, nm in enumerate(names):
            seg = p[bounds[k]: bounds[k + 1] + 1]
            rate = (seg[1:] != seg[:-1]).mean(0)
            rows.append(dict(group=g, phase=nm, mean_flip_rate=rate.mean(), frac_any_flip=(rate > 0).mean()))
    pd.DataFrame(rows).to_csv(outdir / "per_point_instability.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--phase-epochs", default=None, help="override, e.g. '30,600'")
    ap.add_argument("--no-anim", action="store_true")
    a = ap.parse_args()
    run = a.run_dir
    cfg = json.loads((run / "config.json").read_text())
    m = pd.read_csv(run / "metrics.csv")
    data = dict(np.load(run / "data.npz"))
    outdir = run / "figures"
    outdir.mkdir(exist_ok=True)
    ph = detect_phases(m)
    if a.phase_epochs:
        phases = [int(x) for x in a.phase_epochs.split(",")]
    else:
        # II/III boundary only when a second descent was actually detected
        phases = ([ph["min_epoch"], ph["peak_epoch"]] if ph["edd"]
                  else [ph["min_epoch"]] if ph["rise"] > 0.005 else [])
    ph["phases_used"] = phases
    (outdir / "phases.json").write_text(json.dumps(ph, indent=2))
    plot_curves(m, phases, cfg["name"], outdir / "learning_curves.png")
    ev = np.load(run / "eval_epochs.npy")
    snaps = sorted({1, 3, 10, *phases, int(np.sqrt(max(phases[-1], 1) * ev[-1])) if phases else 1000, int(ev[-1])})
    snaps = [s for s in snaps if s <= ev[-1]][:8]
    keys = ["grid"] + (["grid0"] if cfg["dim"] > 2 else [])
    for k in keys:
        map_figure(run, snaps, data, cfg, k, phases, outdir / f"maps_{k}.png")
        res = plane_instability(run, cfg, m, phases, k, outdir)
        plot_instability(*res, data, cfg, phases, k, outdir / f"plane_instability_{k}.png")
    occupancy(run, cfg, phases, outdir)
    per_point(run, data, cfg, phases, outdir)
    if not a.no_anim:
        make_animation(run, data, cfg, m, phases, keys, outdir / "maps_anim")
    print(run.name, json.dumps(ph))


if __name__ == "__main__":
    main()
