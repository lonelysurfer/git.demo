# -*- coding: utf-8 -*-
"""相空间螺旋轨迹：两个耦合状态变量的演化映射（仿参考图3）。

数据契约: --data CSV 含 --t-col、--x-col、--y-col、--series-col（多序列）。
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
from matplotlib.collections import LineCollection
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def shoelace_area(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def lag_estimate(t, x, y):
    xn = (x - x.mean()) / (x.std() + 1e-12)
    yn = (y - y.mean()) / (y.std() + 1e-12)
    best_lag, best_c = 0, -2
    dt = np.median(np.diff(np.sort(t))) if len(t) > 2 else 1.0
    for lag in np.arange(-len(xn) // 3, len(xn) // 3 + 1):
        if lag >= 0:
            c = np.corrcoef(xn[lag:], yn[:len(yn) - lag])[0, 1] if len(xn) - lag > 3 else -2
        else:
            c = np.corrcoef(xn[:lag], yn[-lag:])[0, 1] if len(yn) + lag > 3 else -2
        if np.isfinite(c) and c > best_c:
            best_c, best_lag = c, lag
    return abs(best_lag) * dt


def main():
    ap = argparse.ArgumentParser(description="相空间螺旋轨迹图")
    ap.add_argument("--data", required=True)
    ap.add_argument("--t-col", required=True)
    ap.add_argument("--x-col", required=True)
    ap.add_argument("--y-col", required=True)
    ap.add_argument("--series-col", required=True)
    ap.add_argument("--series-names", default="", help="逗号分隔，按系列出现顺序")
    ap.add_argument("--x-name", default="状态变量 X")
    ap.add_argument("--y-name", default="状态变量 Y")
    ap.add_argument("--couple-name", default="耦合强度（相关系数）")
    ap.add_argument("--lag-name", default="滞后时间 (s)")
    ap.add_argument("--annots", default="", help="分号分隔的每序列标注，如 '滞后最强;中等;较弱'")
    ap.add_argument("--title", default="状态-浓度相空间的演化轨迹映射")
    ap.add_argument("--subtitle", default="—— 耦合路径及其时间演化 ——")
    ap.add_argument("--conclusions", default="轨迹呈环状;滞后效应显著;耦合强度随层级变化")
    ap.add_argument("--out", default="fig_phase_spiral")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df.sort_values(args.t_col)
    names = ([s for s in args.series_names.split(",")] if args.series_names
             else sorted(df[args.series_col].astype(str).unique()))
    series_order = list(dict.fromkeys(df[args.series_col].astype(str)))
    colors = [PALETTE["blue_main"], PALETTE["orange_main"], PALETTE["teal_main"], PALETTE["violet_main"]]

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(120)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.5, wspace=0.38,
                          left=0.06, right=0.965, top=0.855, bottom=0.145)
    ax_a = fig.add_subplot(gs[0, :])
    ax_bs, ax_bar, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])

    # ---------- A: 螺旋轨迹 ----------
    all_xy = df[[args.x_col, args.y_col]].dropna().values
    xg = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), 100)
    yg = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), 100)
    for k, (sr, name) in enumerate(zip(series_order, names)):
        g = df[df[args.series_col].astype(str) == sr].sort_values(args.t_col)
        x, yv, tv = g[args.x_col].values, g[args.y_col].values, g[args.t_col].values
        pts = np.array([x, yv]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", array=tv[:-1], lw=1.8, alpha=0.85)
        ax_a.add_collection(lc)
        ax_a.scatter(x, yv, s=10, color=colors[k % 4], edgecolor="white", lw=0.4, zorder=4,
                     label=name)
        ann = args.annots.split(";") if args.annots else []
        if k < len(ann):
            ax_a.annotate(ann[k], xy=(x[len(x) // 2], yv[len(yv) // 2]),
                          xytext=(x[len(x) // 2] + (x.max() - x.min()) * 0.18,
                                  yv[len(yv) // 2] + (yv.max() - yv.min()) * 0.12),
                          fontsize=6.2, color=colors[k % 4],
                          arrowprops=dict(arrowstyle="->", color=colors[k % 4], lw=0.7),
                          bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=colors[k % 4], lw=0.6))
    ax_a.set_xlabel(args.x_name); ax_a.set_ylabel(args.y_name)
    ax_a.set_title("相空间演化轨迹（按时间着色）", fontsize=8, pad=5)
    ax_a.legend(fontsize=6.2, loc="upper right")
    ax_a.autoscale()
    cbar = fig.colorbar(lc, ax=ax_a, pad=0.012, fraction=0.03)
    cbar.set_label("时间 t (s)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    # ---------- B: 每序列双轴时序小图（纵轴按序列错开） ----------
    n = len(series_order)
    # 使用可嵌套方案：在 ax_bs 区域内画三条（共享图例），纵轴错开
    offsets = np.linspace(0, 1.0, n + 1)
    for k, (sr, name) in enumerate(zip(series_order, names)):
        g = df[df[args.series_col].astype(str) == sr].sort_values(args.t_col)
        xv = norm(g[args.x_col].values) + (n - 1 - k)
        yv = norm(g[args.y_col].values) + (n - 1 - k)
        ax_bs.plot(g[args.t_col].values, xv, color=PALETTE["blue_main"], lw=1.2, label=args.x_name if k == 0 else None)
        ax_bs.plot(g[args.t_col].values, yv, color=PALETTE["orange_main"], lw=1.2, ls="--",
                   label=args.y_name if k == 0 else None)
    ax_bs.set_yticks([n - 0.5 - k for k in range(n)])
    ax_bs.set_yticklabels(names, fontsize=6)
    ax_bs.set_xlabel("时间 (s)")
    ax_bs.set_title("典型序列的双指标时序", fontsize=8)
    ax_bs.legend(fontsize=5.6, ncol=2, loc="upper right")

    # ---------- C: 耦合强度与滞后 ----------
    areas, lags, corrs = [], [], []
    for sr in series_order:
        g = df[df[args.series_col].astype(str) == sr].sort_values(args.t_col)
        areas.append(shoelace_area(g[args.x_col].values, g[args.y_col].values))
        lags.append(lag_estimate(g[args.t_col].values, g[args.x_col].values, g[args.y_col].values))
        corrs.append(abs(np.corrcoef(g[args.x_col].values, g[args.y_col].values)[0, 1]))
    xpos = np.arange(n)
    ax_bar.bar(xpos - 0.18, corrs, width=0.32, color=PALETTE["orange_main"], alpha=0.85,
               label=args.couple_name)
    ax_bar.set_ylabel(args.couple_name, fontsize=6.5, color=PALETTE["orange_main"])
    ax_bar2 = ax_bar.twinx()
    ax_bar2.bar(xpos + 0.18, lags, width=0.32, color=PALETTE["green_strong"], alpha=0.85,
                label=args.lag_name)
    ax_bar2.set_ylabel(args.lag_name, fontsize=6.5, color=PALETTE["green_strong"])
    ax_bar.set_xticks(xpos); ax_bar.set_xticklabels(names, fontsize=6)
    ax_bar.set_title("耦合强度与滞后效应", fontsize=8)
    h1, l1 = ax_bar.get_legend_handles_labels()
    h2, l2 = ax_bar2.get_legend_handles_labels()
    ax_bar.legend(h1 + h2, l1 + l2, fontsize=5.4, loc="upper left")

    # ---------- D: 轨迹密度 ----------
    kde = gaussian_kde(all_xy.T)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
    cf = ax_d.contourf(Xg, Yg, Zg, levels=14, cmap="turbo")
    ax_d.set_xlabel(args.x_name); ax_d.set_ylabel(args.y_name)
    ax_d.set_title("轨迹密度分布", fontsize=8, pad=5)
    cbar2 = fig.colorbar(cf, ax=ax_d, pad=0.02, fraction=0.055)
    cbar2.set_label("轨迹密度", fontsize=6)
    cbar2.ax.tick_params(labelsize=5.5)

    for ax, tag, dx in [(ax_a, "A", -0.03), (ax_bs, "B", -0.07), (ax_bar, "C", -0.09), (ax_d, "D", -0.09)]:
        ax.text(dx, 1.05, tag, transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.935, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])
    kw = dict(boxstyle="round,pad=0.45", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    concl = "    ".join("%s %s" % (a1, b1) for a1, b1 in zip("①②③", args.conclusions.split(";")))
    fig.text(0.5, 0.028, "关键结论    " + concl, ha="center", va="center", fontsize=6.3,
             bbox=kw, wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
