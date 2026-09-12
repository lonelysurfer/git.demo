# -*- coding: utf-8 -*-
"""时空对齐热力图：多源指标按时间轴对齐的四面板 hero 图。"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure

CMAP = LinearSegmentedColormap.from_list("align", ["#F5F7FB", "#9DB8D9", "#3F6FB5", "#123C7A"])


def norm(v):
    v = np.asarray(v, float)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 0 else v * 0


def main():
    ap = argparse.ArgumentParser(description="时空对齐热力图四面板")
    ap.add_argument("--data", required=True)
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--indicator-cols", required=True, help="逗号分隔，2-3 个指标列")
    ap.add_argument("--indicator-names", required=True, help="逗号分隔，与列一一对应")
    ap.add_argument("--entity-col", default=None)
    ap.add_argument("--bins", type=int, default=14)
    ap.add_argument("--max-time", type=float, default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--group-a-value", type=str, default=None)
    ap.add_argument("--group-b-value", type=str, default=None)
    ap.add_argument("--group-a-name", default="组A")
    ap.add_argument("--group-b-name", default="组B")
    ap.add_argument("--zoom-lo", type=float, default=None)
    ap.add_argument("--zoom-hi", type=float, default=None)
    ap.add_argument("--finding", default="对齐清洗后各指标在时间轴上保持一致。")
    ap.add_argument("--title", default="多源异构时序数据的时空对齐热力图")
    ap.add_argument("--subtitle", default="—— 多指标时间轴一致性分析 ——")
    ap.add_argument("--out", default="fig_alignment")
    args = ap.parse_args()

    ind_cols = [c for c in args.indicator_cols.split(",")]
    ind_names = [c for c in args.indicator_names.split(",")]
    df = pd.read_csv(args.data)
    df["t"] = pd.to_numeric(df[args.time_col], errors="coerce")
    for c in ind_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    tmax = args.max_time if args.max_time else np.nanmax(df["t"])
    df = df.dropna(subset=["t"] + ind_cols)
    df = df[df["t"].between(0, tmax)]
    edges = np.linspace(0, tmax, args.bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    df["bin"] = pd.cut(df["t"], bins=edges, include_lowest=True)
    grp = df.groupby("bin", observed=True)
    mat_rows = [norm(grp[c].mean().fillna(0).values) for c in ind_cols]
    mat = np.vstack(mat_rows)

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(140)))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0], hspace=0.46, wspace=0.32,
                          left=0.085, right=0.93, top=0.885, bottom=0.155)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    im = ax_a.imshow(mat, aspect="auto", cmap=CMAP,
                     extent=[0, tmax, len(ind_cols) - 0.5, -0.5], vmin=0, vmax=1)
    for y in np.arange(len(ind_cols) - 1) + 0.5:
        ax_a.axhline(y, color="white", lw=2.2)
    ax_a.set_yticks(range(len(ind_cols)))
    ax_a.set_yticklabels(ind_names, fontsize=6.5)
    ax_a.set_xlabel("时间片")
    ax_a.set_xticks(np.linspace(0, tmax, 8).astype(int))
    ax_a.set_title("多指标对齐热力图（逐行归一化）", fontsize=8, pad=5)
    fig.colorbar(im, ax=ax_a, pad=0.012, fraction=0.025).set_label("行内归一化值", fontsize=6)

    for c, name, color in zip(ind_cols, ind_names,
                              [PALETTE["blue_main"], PALETTE["orange_main"], PALETTE["green_strong"]]):
        ax_b.plot(centers, norm(grp[c].mean().fillna(0).values), color=color, lw=1.4,
                  marker="o", ms=2.4, label=name)
    ax_b.set_xlabel("时间"); ax_b.set_ylabel("行内归一化值")
    ax_b.set_title("对齐后的指标均值时序", fontsize=8)
    ax_b.legend(fontsize=5.8, loc="upper right")

    xcol = ind_cols[0]
    per_entity = df.groupby(args.entity_col) if args.entity_col else df.groupby(df.index)
    ent = per_entity.agg({xcol: "first", ind_cols[1]: "max"}).dropna()
    x, yv = ent[xcol].values, ent[ind_cols[1]].values
    xy = np.vstack([x, yv])
    dens = gaussian_kde(xy)(xy)
    idx = dens.argsort()
    sc = ax_c.scatter(x[idx], yv[idx], c=dens[idx], s=8, cmap="YlOrBr", linewidths=0, alpha=0.85)
    k, b0 = np.polyfit(x, yv, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax_c.plot(xs, k * xs + b0, color=PALETTE["orange_main"], lw=1.3)
    r, p = pearsonr(x, yv)
    ax_c.annotate("r = %.2f\np %s %.3f" % (r, "<" if p < 0.001 else "=", min(p, 0.999)),
                  xy=(0.03, 0.90), xycoords="axes fraction", fontsize=6,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["neutral_mid"], lw=0.5))
    ax_c.set_xlabel(ind_names[0]); ax_c.set_ylabel(ind_names[1])
    ax_c.set_title("指标相关（密度着色）", fontsize=8)
    fig.colorbar(sc, ax=ax_c, pad=0.02, fraction=0.055).set_label("点密度", fontsize=6)

    glo = args.zoom_lo if args.zoom_lo is not None else 0
    ghi = args.zoom_hi if args.zoom_hi is not None else tmax / 7
    if args.group_col and args.group_col in df.columns and args.group_a_value is not None:
        vals = [args.group_a_value, args.group_b_value]
        names = [args.group_a_name, args.group_b_name]
        fine_edges = np.linspace(glo, ghi, 7)
        df["fine"] = pd.cut(df["t"], bins=fine_edges, include_lowest=True, right=False)
        fine_index = pd.interval_range(fine_edges[0], fine_edges[-1], periods=6, closed="left")
        zrows = []
        for v in vals:
            sel = df[(df[args.group_col].astype(str) == str(v)) & (df["t"] >= glo) & (df["t"] < ghi)]
            ser = sel.groupby("fine", observed=True)[ind_cols[1]].mean()
            zrows.append(norm(ser.reindex(fine_index).fillna(0).values))
        mat_d = np.vstack(zrows)
        im2 = ax_d.imshow(mat_d, aspect="auto", cmap=CMAP,
                          extent=[glo, ghi, 1.5, -0.5], vmin=0, vmax=1)
        ax_d.axhline(0.5, color="white", lw=2.0)
        ax_d.set_yticks([0, 1]); ax_d.set_yticklabels(names, fontsize=6.5)
    else:
        med_split = df[ind_cols[1]].median()
        hi_m = mat[1] >= np.nanmedian(mat[1])
        lo_m = ~hi_m
        m2 = np.vstack([norm(np.where(hi_m, mat[1], np.nan).astype(float)),
                        norm(np.where(lo_m, mat[1], np.nan).astype(float))])
        im2 = ax_d.imshow(np.nan_to_num(m2), aspect="auto", cmap=CMAP,
                          extent=[0, tmax, 1.5, -0.5], vmin=0, vmax=1)
        ax_d.axhline(0.5, color="white", lw=2.0)
        ax_d.set_yticks([0, 1])
        ax_d.set_yticklabels(["%s 高值片" % ind_names[1], "%s 低值片" % ind_names[1]], fontsize=6.5)
    ax_d.set_xlabel("时间片")
    ax_d.set_title("关键窗放大：%.0f–%.0f" % (glo, ghi), fontsize=8, pad=5)
    fig.colorbar(im2, ax=ax_d, pad=0.02, fraction=0.055).set_label("组内归一化均值", fontsize=6)

    for ax, tag, dx in [(ax_a, "A", -0.045), (ax_b, "B", -0.11), (ax_c, "C", -0.11), (ax_d, "D", -0.11)]:
        ax.text(dx, 1.06, tag, transform=ax.transAxes, fontsize=9, fontweight="bold")

    kw = dict(boxstyle="round,pad=0.5", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    fig.text(0.5, 0.026, "主要发现    " + args.finding, ha="center", va="center",
             fontsize=6.4, bbox=kw, wrap=True)

    fig.suptitle(args.title, fontsize=11, y=0.99)
    fig.text(0.5, 0.958, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
