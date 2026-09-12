# -*- coding: utf-8 -*-
"""时空场演化切片云图：多时刻 2D 场快照 + 剖面 + 演化示意（仿参考图6）。"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def main():
    ap = argparse.ArgumentParser(description="时空场演化切片云图")
    ap.add_argument("--field-data", required=True, help="长表 CSV: 时间, x, y, 场值1[, 场值2]")
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--x-col", required=True)
    ap.add_argument("--y-col", required=True)
    ap.add_argument("--v1-col", required=True)
    ap.add_argument("--v2-col", default="")
    ap.add_argument("--times", default="", help="逗号分隔的快照时刻（留空自动等距取 5 个）")
    ap.add_argument("--n-snap", type=int, default=5)
    ap.add_argument("--x-name", default="径向位置 x")
    ap.add_argument("--y-name", default="径向位置 y")
    ap.add_argument("--v1-name", default="场值 1")
    ap.add_argument("--v2-name", default="场值 2")
    ap.add_argument("--title", default="多尺度时空演化的场动态切片云图")
    ap.add_argument("--subtitle", default="—— 场快照、剖面与推进过程 ——")
    ap.add_argument("--out", default="fig_field_slices")
    args = ap.parse_args()

    df = pd.read_csv(args.field_data)
    df["t"] = pd.to_numeric(df[args.time_col], errors="coerce")
    for c in [args.x_col, args.y_col, args.v1_col] + ([args.v2_col] if args.v2_col else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t", args.x_col, args.y_col, args.v1_col])
    times_all = np.sort(df["t"].unique())
    if args.times:
        snaps = [float(v) for v in args.times.split(",")]
    else:
        snaps = list(times_all[np.linspace(0, len(times_all) - 1, args.n_snap).astype(int)])
    snaps_t = [min(times_all, key=lambda x: abs(x - s)) for s in snaps]

    apply_py_nature_style(font_size=7.0)
    v1_all = df[args.v1_col].values
    vmin, vmax = np.nanpercentile(v1_all, 1), np.nanpercentile(v1_all, 99)
    n = len(snaps_t)

    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(150)))
    gs_top = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.09], wspace=0.25,
                              left=0.05, right=0.96, top=0.80, bottom=0.63)
    axes_a = []
    for k, tv in enumerate(snaps_t):
        ax = fig.add_subplot(gs_top[0, k])
        g = df[np.isclose(df["t"], tv)]
        piv = g.pivot_table(index=args.y_col, columns=args.x_col, values=args.v1_col)
        cf = ax.contourf(piv.columns.values, piv.index.values, piv.values,
                         levels=14, cmap="turbo", vmin=vmin, vmax=vmax, extend="both")
        ax.set_title("t = %g" % tv, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 0:
            ax.set_ylabel(args.y_name, fontsize=6.5)
        axes_a.append(ax)
    cax = fig.add_subplot(gs_top[0, -1])
    fig.colorbar(cf, cax=cax).set_label(args.v1_name, fontsize=6)
    cax.tick_params(labelsize=5)
    fig.text(0.5, 0.615, "场峰值随时间由边界向中心推进", ha="center", fontsize=6.5,
             color=PALETTE["neutral_dark"])
    tag = fig.add_axes([0.02, 0.80, 0.05, 0.06]); tag.axis("off")
    tag.text(0, 0.5, "A", fontsize=9, fontweight="bold")

    # B: v1 剖面（沿 y 均值）
    ax_b = fig.add_subplot(2, 2, 3)
    cmap_p = plt.get_cmap("viridis")
    for k, tv in enumerate(snaps_t):
        g = df[np.isclose(df["t"], tv)]
        prof = g.groupby(args.x_col)[args.v1_col].mean()
        c = cmap_p(0.15 + 0.8 * k / max(n - 1, 1))
        ax_b.plot(prof.index.values, prof.values, color=c, lw=1.3, marker="o", ms=2,
                  label="t = %g" % tv)
    ax_b.set_xlabel(args.x_name); ax_b.set_ylabel(args.v1_name)
    ax_b.set_title("%s 随时间的变化曲线" % args.v1_name, fontsize=8)
    ax_b.legend(fontsize=5.4, ncol=2)
    tag2 = fig.add_axes([0.06, 0.42, 0.05, 0.05]); tag2.axis("off")
    tag2.text(0, 0.5, "B", fontsize=9, fontweight="bold")

    # C: v2 剖面（若有）
    if args.v2_col:
        ax_c = fig.add_subplot(2, 2, 4)
        for k, tv in enumerate(snaps_t):
            g = df[np.isclose(df["t"], tv)]
            prof = g.groupby(args.x_col)[args.v2_col].mean()
            c = matplotlib.cm.turbo(0.15 + 0.8 * k / max(n - 1, 1))
            ax_c.plot(prof.index.values, prof.values, color=c, lw=1.3, marker="s", ms=2,
                      label="t = %g" % tv)
        ax_c.set_xlabel(args.x_name); ax_c.set_ylabel(args.v2_name)
        ax_c.set_title("%s 径向分布剖面" % args.v2_name, fontsize=8)
        ax_c.legend(fontsize=5.4, ncol=2)
        tag3 = fig.add_axes([0.55, 0.42, 0.05, 0.05]); tag3.axis("off")
        tag3.text(0, 0.5, "C", fontsize=9, fontweight="bold")
    else:
        ax_c = None

    # D 行：演化示意（缩略快照 + 箭头 + 放大框）
    gs_bot = fig.add_gridspec(1, n + 1, width_ratios=[1] * n + [0.08], wspace=0.22,
                              left=0.05, right=0.96, top=0.335, bottom=0.13)
    for k, tv in enumerate(snaps_t):
        ax = fig.add_subplot(gs_bot[0, k])
        g = df[np.isclose(df["t"], tv)]
        piv = g.pivot_table(index=args.y_col, columns=args.x_col, values=args.v1_col)
        ax.contourf(piv.columns.values, piv.index.values, piv.values,
                    levels=10, cmap="turbo", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("t = %g" % tv, fontsize=6)
        if k == 0:
            ax.set_ylabel("表面", fontsize=6)
    axm = fig.add_subplot(gs_bot[0, -1]); axm.axis("off")
    axm.annotate("", xy=(0.5, 0.5), xytext=(0.0, 0.5),
                 arrowprops=dict(arrowstyle="-|>", color=PALETTE["blue_main"], lw=1.4))
    axm.text(0.5, 0.9, "中心", ha="center", fontsize=6)
    tag4 = fig.add_axes([0.02, 0.335, 0.05, 0.05]); tag4.axis("off")
    tag4.text(0, 0.5, "D", fontsize=9, fontweight="bold")

    fig.suptitle(args.title, fontsize=11, y=0.995)
    fig.text(0.5, 0.962, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
