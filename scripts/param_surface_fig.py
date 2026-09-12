# -*- coding: utf-8 -*-
"""参数敏感性拓扑景观：双 3D 曲面 + 剖面 + 梯度热图（仿参考图5）。"""
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
    ap = argparse.ArgumentParser(description="参数敏感性拓扑景观")
    ap.add_argument("--grid-data", required=True, help="规则网格 CSV: x, y, z1[, z2]")
    ap.add_argument("--x-col", required=True)
    ap.add_argument("--y-col", required=True)
    ap.add_argument("--z1-col", required=True)
    ap.add_argument("--z2-col", default="")
    ap.add_argument("--x-name", default=None)
    ap.add_argument("--y-name", default=None)
    ap.add_argument("--z1-name", default="参数 1")
    ap.add_argument("--z2-name", default="参数 2")
    ap.add_argument("--title", default="状态依赖型参数的敏感性拓扑景观")
    ap.add_argument("--subtitle", default="—— 双参数响应面与梯度分布 ——")
    ap.add_argument("--out", default="fig_param_surface")
    args = ap.parse_args()

    g = pd.read_csv(args.grid_data).sort_values([args.y_col, args.x_col])
    xs = np.sort(g[args.x_col].unique())
    ys = np.sort(g[args.y_col].unique())
    X, Y = np.meshgrid(xs, ys)
    Z1 = g[args.z1_col].values.reshape(len(ys), len(xs))
    has2 = bool(args.z2_col) and args.z2_col in g.columns
    Z2 = g[args.z2_col].values.reshape(len(ys), len(xs)) if has2 else None
    x_name = args.x_name or args.x_col
    y_name = args.y_name or args.y_col

    gy, gx = np.gradient(Z1, ys, xs)
    G1 = np.sqrt(gx ** 2 + gy ** 2)
    if has2:
        gy2, gx2 = np.gradient(Z2, ys, xs)
        G2 = np.sqrt(gx2 ** 2 + gy2 ** 2)
    i1, j1 = np.unravel_index(np.argmax(G1), G1.shape)

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(128)))
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax2 = fig.add_subplot(2, 2, 2, projection="3d") if has2 else None
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4) if has2 else fig.add_subplot(2, 2, 4)

    for ax, Z, name, tag, dx in ([(ax1, Z1, args.z1_name, "A", -0.02)] +
                                 ([(ax2, Z2, args.z2_name, "B", 0.02)] if has2 else [])):
        sf = ax.plot_surface(X, Y, Z, cmap="turbo", linewidth=0.1,
                             edgecolor=(0, 0, 0, 0.08), alpha=0.97)
        ax.contourf(X, Y, Z, levels=10, zdir="z", offset=Z.min() - 0.1 * Z.ptp(), cmap="turbo", alpha=0.5)
        ax.set_zlim(Z.min() - 0.1 * Z.ptp(), Z.max() * 1.02)
        gz = np.sqrt(np.gradient(Z, xs, ys)[0] ** 2 + np.gradient(Z, xs, ys)[1] ** 2)
        gi, gj = np.unravel_index(np.argmax(gz), gz.shape)
        ax.scatter(X[gi, gj], Y[gi, gj], Z[gi, gj], marker="*", s=150,
                   color=PALETTE["red_strong"], edgecolor="black", lw=0.5, zorder=10)
        ax.text(X[gi, gj], Y[gi, gj], Z[gi, gj] * 1.04, "高敏感区域", fontsize=6,
                ha="center", color=PALETTE["red_strong"])
        ax.set_xlabel(x_name, fontsize=6.5, labelpad=1)
        ax.set_ylabel(y_name, fontsize=6.5, labelpad=1)
        ax.set_zlabel(name, fontsize=6.5, labelpad=1)
        ax.set_title("%s 的敏感性拓扑" % name, fontsize=8, pad=2)
        ax.view_init(elev=28, azim=-55)
        fig.colorbar(sf, ax=ax, shrink=0.5, pad=0.06).ax.tick_params(labelsize=5)
        ax.text(dx, 1.02, tag, transform=ax.transAxes, fontsize=9, fontweight="bold")

    # C: 剖面曲线族
    slice_idx = np.linspace(0, len(ys) - 1, min(5, len(ys))).astype(int)
    cmap_p = plt.get_cmap("YlGnBu")
    for k, yi in enumerate(slice_idx):
        c = cmap_p(0.25 + 0.65 * k / max(len(slice_idx) - 1, 1))
        ax3.plot(xs, Z1[yi], color=c, lw=1.2, label="%s=%g" % (y_name.split("(")[0], ys[yi]))
    ax3.set_xlabel(x_name); ax3.set_ylabel(args.z1_name)
    ax3.set_title("%s 剖面分析" % args.z1_name, fontsize=8)
    ax3.legend(fontsize=5.2)
    tag_panel(ax3, "C", -0.09, 1.03)

    # D: 梯度热图
    cf = ax4.contourf(X, Y, G1, levels=14, cmap="Blues")
    gi, gj = np.unravel_index(np.argmax(G1), G1.shape)
    ax4.scatter(xs[gj], ys[gi], marker="*", s=60, color=PALETTE["red_strong"], zorder=5)
    ax4.set_xlabel(x_name); ax4.set_ylabel(y_name)
    ax4.set_title("|梯度| 剧烈区域", fontsize=8)
    fig.colorbar(cf, ax=ax4, pad=0.02, fraction=0.055).ax.tick_params(labelsize=5)
    tag_panel(ax4, "D", -0.09, 1.03)

    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.935, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])
    kw = dict(boxstyle="round,pad=0.45", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    fig.text(0.5, 0.028, "关键结论    参数随 %s 与 %s 呈显著非线性变化；高敏感区集中在 %s 高值与 %s 交叠处；"
             "使用变参数建模比常数假设更能刻画真实特性。" % (x_name, y_name, y_name, x_name),
             ha="center", va="center", fontsize=6.3, bbox=kw, wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
