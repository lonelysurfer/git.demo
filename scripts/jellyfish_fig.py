# -*- coding: utf-8 -*-
"""密度水母图：四面板 hero 图（分布演变 / 分组时序 / 方差分解 / 响应面）。

用法示例见 SKILL.md；输入为长表 CSV。
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure

WIN_COLORS = [PALETTE["blue_main"], PALETTE["orange_main"], PALETTE["teal_main"], PALETTE["violet_main"]]


def blend_cmap(color):
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list("blend", [(1, 1, 1), (r, g, b)])


def draw_jellyfish(ax, xc, values, color, hw=0.36):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 8:
        return {}
    kde = gaussian_kde(values)
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = 0.12 * (hi - lo)
    grid = np.linspace(max(lo - pad, values.min() - 1e-9), hi + pad, 240)
    dens = kde(grid)
    dens = dens / dens.max()
    cmap = blend_cmap(color)
    poly = Polygon(np.concatenate([
        np.column_stack([xc - hw * dens, grid]),
        np.column_stack([xc + hw * dens[::-1], grid[::-1]])]), closed=True,
        facecolor="none", edgecolor="none")
    ax.add_patch(poly)
    rgba = np.repeat(cmap(dens)[:, None, :], 60, axis=1)
    rgba[:, :, 3] = np.repeat((0.25 + 0.75 * dens)[:, None], 60, axis=1)
    im = ax.imshow(rgba, extent=[xc - hw, xc + hw, grid[0], grid[-1]],
                   origin="lower", aspect="auto", zorder=2, interpolation="bilinear")
    im.set_clip_path(poly)
    kde_w = gaussian_kde(values, bw_method=kde.factor * 1.9)
    dw = kde_w(grid); dw = dw / dw.max()
    wide = np.concatenate([
        np.column_stack([xc - hw * dw * 1.28, grid]),
        np.column_stack([xc + hw * dw[::-1] * 1.28, grid[::-1]])])
    ax.add_patch(Polygon(wide, closed=True, facecolor=color, alpha=0.10, edgecolor=color, lw=0.5, zorder=1))
    mu, sd = float(np.mean(values)), float(np.std(values))
    ax.plot([xc, xc], [max(mu - 1.4 * sd, grid[0]), min(mu + 1.4 * sd, grid[-1])],
            color=PALETTE["black"], lw=1.4, zorder=4)
    ax.scatter([xc], [mu], s=13, facecolor="white", edgecolor=PALETTE["black"], lw=0.8, zorder=5)
    return {"mu": mu, "sigma": sd, "n": int(len(values))}


def main():
    ap = argparse.ArgumentParser(description="密度水母图四面板")
    ap.add_argument("--data", required=True)
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--entity-col", default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--windows", default="0,24,72,168,336", help="逗号分隔的时间窗边界")
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--time-max", type=float, default=None)
    ap.add_argument("--decomp-cols", default="", help="逗号分隔的协变量列（面板 C 顺序方差分解）")
    ap.add_argument("--surface-csv", default="", help="面板 D 响应面数据（含 x/y/z 三列）")
    ap.add_argument("--surface-x", default="x")
    ap.add_argument("--surface-y", default="y")
    ap.add_argument("--surface-z", default="z")
    ap.add_argument("--title", default="数值概率密度函数演变与水母图")
    ap.add_argument("--subtitle", default="—— 按时间窗的分布特征、组间不确定性与来源分解 ——")
    ap.add_argument("--out", default="fig_jellyfish")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    edges = [float(v) for v in args.windows.split(",")]
    df["t"] = pd.to_numeric(df[args.time_col], errors="coerce") * 1.0
    df["v"] = pd.to_numeric(df[args.value_col], errors="coerce") * args.value_scale
    tmax = args.time_max if args.time_max else edges[-1]
    df = df.dropna(subset=["t", "v"])
    df = df[df["t"].between(edges[0], tmax)]

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(126)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.18, 1.0], hspace=0.42, wspace=0.34,
                          left=0.055, right=0.905, top=0.865, bottom=0.10)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # 面板 A
    stats_all = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sub = df[(df["t"] >= lo) & (df["t"] <= hi)] if i == 0 else df[(df["t"] > lo) & (df["t"] <= hi)]
        st = draw_jellyfish(ax_a, i + 1.0, sub["v"].values, WIN_COLORS[i % len(WIN_COLORS)])
        stats_all.append((f"{lo:g}–{hi:g}", st))
    ax_a.set_xticks(range(1, len(edges)))
    ax_a.set_xticklabels([f"{name}\n(n={st['n'] if st else 0})" for name, st in stats_all])
    ax_a.set_xlim(0.35, len(edges) - 0.65)
    ax_a.set_ylim(-5, df["v"].max() * 1.06)
    ax_a.set_ylabel("数值")
    ax_a.set_title("各时间窗概率密度与水母图", fontsize=8, pad=5)
    ax_a.legend(handles=[
        Line2D([0], [0], color=PALETTE["black"], lw=1.4, label="均值线 (μ)"),
        Patch(facecolor=PALETTE["blue_secondary"], alpha=0.18, label="均值±1σ 轮廓"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=PALETTE["black"], markersize=4.5, label="均值点")],
        loc="upper right", fontsize=6.2)
    cax = fig.add_axes([0.925, 0.56, 0.012, 0.27])
    cax.imshow(np.linspace(1, 0, 256).reshape(-1, 1), aspect="auto", cmap=blend_cmap(PALETTE["blue_main"]))
    cax.set_xticks([]); cax.set_yticks([])
    cax.set_title("概率密度", fontsize=6, pad=3)

    # 面板 B：分组或总体
    if args.group_col and args.group_col in df.columns:
        for k, (gname, g) in enumerate(df.groupby(args.group_col)):
            gb = g.groupby(pd.cut(g["t"], bins=edges), observed=True)["v"]
            x = g.groupby(pd.cut(g["t"], bins=edges), observed=True)["t"].median()
            ax_b.plot(x.values, gb.median().values, color=WIN_COLORS[k % 4], lw=1.3,
                      marker="o", ms=2.2, label=str(gname))
            ax_b.fill_between(x.values, gb.quantile(0.25).values, gb.quantile(0.75).values,
                              color=WIN_COLORS[k % 4], alpha=0.16, lw=0)
        ax_b.legend(fontsize=5.8, loc="upper right")
    else:
        gb = df.groupby(pd.cut(df["t"], bins=edges), observed=True)["v"]
        x = df.groupby(pd.cut(df["t"], bins=edges), observed=True)["t"].median()
        ax_b.plot(x.values, gb.median().values, color=PALETTE["blue_main"], lw=1.4, marker="o", ms=2.4)
        ax_b.fill_between(x.values, gb.quantile(0.25).values, gb.quantile(0.75).values,
                          color=PALETTE["blue_main"], alpha=0.16, lw=0)
    ax_b.set_xlabel("时间"); ax_b.set_ylabel("数值")
    ax_b.set_title("时序分布与不确定性", fontsize=8)

    # 面板 C：来源分解或离散度演化
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    if args.decomp_cols:
        cols = [c for c in args.decomp_cols.split(",") if c in df.columns]
        tss_list, seq_all = [], [np.zeros(len(centers)) for _ in cols]
        unexpl = np.zeros(len(centers))
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            d = df[(df["t"] > lo) & (df["t"] <= hi)].dropna(subset=cols + ["v"])
            if len(d) < 8:
                for arr in seq_all:
                    arr[k] = np.nan
                unexpl[k] = np.nan
                continue
            X = np.column_stack([d[c].values for c in cols])
            yv = d["v"].values
            Xs = (X - X.mean(0)) / np.where(X.std(0) == 0, 1, X.std(0))
            tss = ((yv - yv.mean()) ** 2).sum()
            Xacc = np.ones((len(yv), 1))
            prev, seq = tss, []
            for kk in range(Xs.shape[1]):
                Xacc = np.column_stack([Xacc, Xs[:, kk]])
                b, *_ = np.linalg.lstsq(Xacc, yv, rcond=None)
                rss = ((yv - Xacc @ b) ** 2).sum()
                seq.append(max(prev - rss, 0)); prev = rss
            for j in range(len(cols)):
                seq_all[j][k] = seq[j] / max(tss, 1e-9)
            unexpl[k] = prev / max(tss, 1e-9)
        bases = np.zeros(len(centers))
        for j, c in enumerate(cols):
            ax_c.bar(centers, seq_all[j], bottom=bases, width=(edges[1] - edges[0]) * 0.62,
                     color=WIN_COLORS[j % 4], alpha=0.85, edgecolor="white", lw=0.4, label=c)
            bases = bases + seq_all[j]
        ax_c.bar(centers, unexpl, bottom=bases, width=(edges[1] - edges[0]) * 0.62,
                 color=PALETTE["neutral_light"], edgecolor="white", lw=0.4, label="未解释")
        ax_c.set_ylim(0, 1.0); ax_c.legend(fontsize=5.6, ncol=2, loc="upper right")
    else:
        stds, iqrw = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            d = df[(df["t"] > lo) & (df["t"] <= hi)] if lo else df[(df["t"] >= lo) & (df["t"] <= hi)]
            stds.append(d["v"].std()); iqrw.append(d["v"].quantile(0.75) - d["v"].quantile(0.25))
        ax_c.plot(centers, stds, color=PALETTE["red_strong"], lw=1.4, marker="o", ms=3, label="窗内标准差")
        ax_c.plot(centers, iqrw, color=PALETTE["blue_secondary"], lw=1.4, marker="s", ms=3, label="窗内 IQR")
        ax_c.legend(fontsize=5.8)
    ax_c.set_xlabel("时间窗中心"); ax_c.set_ylabel("方差贡献占比" if args.decomp_cols else "离散度")
    ax_c.set_title("不确定性来源分解" if args.decomp_cols else "离散度演化", fontsize=8)

    # 面板 D：响应面或 KDE 等高线
    if args.surface_csv and Path(args.surface_csv).exists():
        dsa = pd.read_csv(args.surface_csv)
        xb, yb = 8, 8
        piv = dsa.pivot_table(index=args.surface_y, columns=args.surface_x,
                              values=args.surface_z, aggfunc="mean")
        xs = piv.columns.values.astype(float); ys = piv.index.values.astype(float)
        M = piv.values.astype(float)
        Sm = gaussian_filter(np.nan_to_num(M, nan=np.nanmean(M)), sigma=1.2)
        cf = ax_d.contourf(xs, ys, Sm, levels=12, cmap="Blues")
        ax_d.contour(xs, ys, Sm, levels=5, colors="white", linewidths=0.5)
        fig.colorbar(cf, ax=ax_d, pad=0.02, fraction=0.055).set_label("均值", fontsize=6)
    else:
        kde = gaussian_kde(df[["t", "v"]].values.T)
        xs = np.linspace(df["t"].min(), df["t"].max(), 120)
        ys = np.linspace(df["v"].min(), df["v"].max(), 120)
        Xg, Yg = np.meshgrid(xs, ys)
        Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        cf = ax_d.contourf(Xg, Yg, Zg, levels=12, cmap="Blues")
        fig.colorbar(cf, ax=ax_d, pad=0.02, fraction=0.055).set_label("密度", fontsize=6)
    ax_d.set_xlabel(args.surface_x if args.surface_csv else "时间")
    ax_d.set_ylabel(args.surface_y if args.surface_csv else "数值")
    ax_d.set_title("二维响应面 / 密度", fontsize=8)

    for ax, tag, dx in [(ax_a, "A", -0.035), (ax_b, "B", -0.10), (ax_c, "C", -0.10), (ax_d, "D", -0.10)]:
        ax.text(dx, 1.05, tag, transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.suptitle(args.title, fontsize=11, y=0.99)
    fig.text(0.5, 0.945, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
