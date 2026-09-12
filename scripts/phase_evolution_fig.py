# -*- coding: utf-8 -*-
"""动态演化与相空间轨迹：单峰过程（蓄积-达峰-消退）四面板 hero 图。"""
import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def gamma_curve(t, a, b, c):
    return a * np.power(np.maximum(t, 1e-9), b) * np.exp(-c * t)


def gamma_rate(t, a, b, c):
    return a * np.exp(-c * t) * np.power(np.maximum(t, 1e-9), b - 1.0) * (b - c * t)


def main():
    ap = argparse.ArgumentParser(description="动态演化与相空间轨迹四面板")
    ap.add_argument("--data", required=True)
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--time-max", type=float, default=None)
    ap.add_argument("--curve-params", default="", help="a,b,c；留空自动拟合")
    ap.add_argument("--stages", default="0,48,168,336", help="阶段边界（时间）")
    ap.add_argument("--stage-names", default="蓄积区,演进区,消退区")
    ap.add_argument("--stage-colors", default="#FBE9E7,#E8F0FB,#EAF7EC")
    ap.add_argument("--peak-label", default="最大水平")
    ap.add_argument("--base-label", default="首诊基线")
    ap.add_argument("--value-name", default="负荷水平")
    ap.add_argument("--time-name", default="时间 (h)")
    ap.add_argument("--flow-boxes", default="", help="JSON：[[框1,框2,框3],[箭头1,箭头2]]")
    ap.add_argument("--finding", default="过程呈单峰演化，蓄积与消退两阶段边界清晰。")
    ap.add_argument("--conclusions", default="① 过程呈单峰演化；② 早期为变化最快窗口；③ 相空间轨迹呈双阶段环。")
    ap.add_argument("--title", default="负荷水平动态演化与风险状态相空间轨迹")
    ap.add_argument("--subtitle", default="—— 基于人群曲线拟合与重复观测的动态演化分析 ——")
    ap.add_argument("--out", default="fig_phase")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["t"] = pd.to_numeric(df[args.time_col], errors="coerce") * 1.0
    df["v"] = pd.to_numeric(df[args.value_col], errors="coerce") * args.value_scale
    tmax = args.time_max if args.time_max else float(np.nanpercentile(df["t"], 99))
    df = df.dropna(subset=["t", "v"])
    df = df[df["t"].between(0, tmax)]

    # 曲线拟合
    t = df["t"].values; v = df["v"].values
    if args.curve_params:
        a0, b0, c0 = [float(x) for x in args.curve_params.split(",")]
        params = (a0, b0, c0)
    else:
        p0 = [max(np.nanmedian(v), 1.0), 0.6, 3.0 / max(tmax, 1.0)]
        params, _ = curve_fit(gamma_curve, t, v, p0=p0, maxfev=20000)
    a_g, b_g, c_g = params
    t_star = b_g / c_g
    e_star = gamma_curve(t_star, *params)
    stages = [float(x) for x in args.stages.split(",")]
    stage_names = [x for x in args.stage_names.split(",")]
    stage_colors = [x for x in args.stage_colors.split(",")]

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(122)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.5, wspace=0.36,
                          left=0.06, right=0.965, top=0.845, bottom=0.20)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # ---------- A ----------
    tt = np.linspace(max(1e-3, df["t"].min() * 0.5), tmax, 600)
    curve = gamma_curve(tt, *params)
    for lo, hi, name, c in zip(stages[:-1], stages[1:], stage_names, stage_colors):
        ax_a.axvspan(lo, hi, color=c, zorder=0)
        ax_a.text((lo + hi) / 2, e_star * 1.22 + v.max() * 0.03, name.replace("\n", ""),
                  ha="center", va="top", fontsize=6.3, color=PALETTE["neutral_dark"])
    rng = np.random.default_rng(7)
    ax_a.scatter(df["t"] + rng.uniform(-0.01, 0.01, len(df)) * tmax,
                 v + rng.uniform(-1, 1, len(df)) * v.std() * 0.02,
                 s=3.5, color=PALETTE["neutral_mid"], alpha=0.28, linewidths=0, zorder=1,
                 label="观测点（不确定性区间）")
    ax_a.plot(tt, curve, color=PALETTE["blue_main"], lw=1.8, zorder=3, label="人群拟合演化曲线")
    ax_a.scatter([t_star], [e_star], s=26, color=PALETTE["orange_main"], edgecolor="white",
                 lw=0.7, zorder=6)
    ax_a.annotate(f"{args.peak_label}\n({e_star:.1f}, {t_star:.1f} h)", xy=(t_star, e_star),
                  xytext=(t_star * 0.55, e_star * 1.55), fontsize=6.2,
                  arrowprops=dict(arrowstyle="->", color=PALETTE["black"], lw=0.7),
                  bbox=dict(boxstyle="round,pad=0.25", fc="#FFF3E0", ec=PALETTE["orange_main"], lw=0.6))
    t_base = max(df["t"].min(), 0.5)
    e_base = gamma_curve(t_base, *params)
    ax_a.scatter([t_base], [e_base], s=26, color=PALETTE["green_strong"], edgecolor="white",
                 lw=0.7, zorder=6)
    ax_a.annotate(f"{args.base_label}\n({e_base:.1f}, {t_base:.1f} h)", xy=(t_base, e_base),
                  xytext=(t_base + 0.06 * tmax, e_base + v.max() * 0.18), fontsize=6.2,
                  arrowprops=dict(arrowstyle="->", color=PALETTE["black"], lw=0.7),
                  bbox=dict(boxstyle="round,pad=0.25", fc="#EAF7EC", ec=PALETTE["green_strong"], lw=0.6))
    ax_a.set_xlim(0, tmax)
    ax_a.set_ylim(-v.max() * 0.03, v.max() * 1.08)
    ax_a.set_xlabel(args.time_name); ax_a.set_ylabel(f"平均{args.value_name}")
    ax_a.set_title("负荷水平随时间的动态演化与阶段划分", fontsize=8, pad=5)
    ax_a.legend(handles=[
        Line2D([0], [0], color=PALETTE["blue_main"], lw=1.8, label="拟合演化曲线"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["neutral_mid"],
               alpha=0.4, markersize=4, label="观测点"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["orange_main"], markersize=6,
               label=args.peak_label + "点"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["green_strong"], markersize=6,
               label=args.base_label + "点")] +
        [Patch(facecolor=c, label=n) for n, c in zip(stage_names, stage_colors)],
        loc="upper right", fontsize=5.4, ncol=2, handlelength=1.3, columnspacing=0.8)

    # ---------- B: 机理流向 ----------
    ax_b.set_xlim(0, 10); ax_b.set_ylim(0, 10); ax_b.axis("off")
    ax_b.set_title("过程机理流向", fontsize=8, pad=5)
    if args.flow_boxes:
        cfg = json.loads(args.flow_boxes)
        boxes, arrows = cfg[0], cfg[1]
    else:
        boxes = ["驱动因素", "核心过程", args.value_name + "演化"]
        arrows = ["驱动传导", "累积效应"]
    xs = [1.0, 3.9, 6.9]
    for x, txt in zip(xs, boxes):
        ax_b.add_patch(FancyBboxPatch((x, 4.0), 2.5, 2.2, boxstyle="round,pad=0.12",
                                      facecolor="#EAF3FD", edgecolor=PALETTE["blue_main"], lw=1.0))
        ax_b.text(x + 1.25, 5.1, txt, ha="center", va="center", fontsize=6.2)
    for x, lab in zip([3.55, 6.55], arrows):
        ax_b.add_patch(FancyArrowPatch((x - 0.15, 5.1), (x + 0.35, 5.1), arrowstyle="-|>",
                                       mutation_scale=12, color=PALETTE["black"], lw=1.1))
        ax_b.text(x + 0.1, 5.75, lab, fontsize=5.6, ha="center", color=PALETTE["neutral_dark"])
    ax_b.text(5.0, 1.2, "驱动力传导与累积效应共同决定过程的形态与节律。",
              fontsize=5.8, ha="center", color=PALETTE["neutral_dark"])

    # ---------- C: 相空间 ----------
    tt = np.linspace(max(1e-3, tmax * 0.002), tmax * 6, 800)
    ee = gamma_curve(tt, *params)
    dd = gamma_rate(tt, *params)
    dd_c = np.clip(dd, -v.max() * 0.05, dd.max())
    pts = np.array([ee, dd_c]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap="viridis", array=tt, lw=1.8)
    ax_c.add_collection(lc)
    ax_c.plot(ee, dd_c, color=PALETTE["neutral_mid"], lw=0.4, alpha=0.4)
    ax_c.scatter([e_star], [0], s=30, color=PALETTE["red_strong"], edgecolor="white", lw=0.7, zorder=5)
    ax_c.annotate("达峰点\n(%.1f h, 速率=0)" % t_star, xy=(e_star, 0),
                  xytext=(e_star * 0.45, dd.max() * 0.55), fontsize=6.2,
                  arrowprops=dict(arrowstyle="->", color=PALETTE["black"], lw=0.7),
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["black"], lw=0.5))
    ax_c.text(ee[10] * 0.8, dd_c[10] * 0.86, "蓄积阶段\n(dX/dt > 0)", fontsize=6.2,
              color=PALETTE["orange_main"])
    ax_c.text(e_star * 0.62, -dd_c.max() * 0.12, "消退阶段 (dX/dt < 0)", fontsize=6.2,
              color=PALETTE["green_strong"])
    ax_c.set_xlim(0, ee.max() * 1.05)
    ax_c.set_ylim(-dd_c.max() * 0.18, dd_c.max() * 1.12)
    ax_c.set_xlabel(f"{args.value_name}水平"); ax_c.set_ylabel("瞬时增速 dX/dt")
    ax_c.set_title("相空间轨迹", fontsize=8, pad=5)
    cbar = fig.colorbar(lc, ax=ax_c, pad=0.02, fraction=0.055)
    cbar.set_label("时间", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    # ---------- D: 速率热图 ----------
    pairs = []
    gs = df.sort_values("t")
    for j in range(1, len(gs)):
        dt = gs["t"].iloc[j] - gs["t"].iloc[j - 1]
        if dt <= 0:
            continue
        pairs.append(((gs["t"].iloc[j] + gs["t"].iloc[j - 1]) / 2,
                      (gs["v"].iloc[j] + gs["v"].iloc[j - 1]) / 2,
                      (gs["v"].iloc[j] - gs["v"].iloc[j - 1]) / dt))
    P = pd.DataFrame(pairs, columns=["tm", "el", "rate"])
    t_bins = np.linspace(0, tmax, 13)
    e_bins = np.linspace(0, np.nanpercentile(P["el"], 97), 6)
    P["tb"] = pd.cut(P["tm"], bins=t_bins)
    P["eb"] = pd.cut(P["el"], bins=e_bins)
    piv = P.pivot_table(index="eb", columns="tb", values="rate", aggfunc="mean", dropna=False)
    mat = piv.values.astype(float)
    from matplotlib.colors import LinearSegmentedColormap as LSC
    cmap_d = LSC.from_list("div", ["#1B5FAA", "#7FA8D9", "#F2F2F2", "#F0A860", "#C0392B"])
    masked = np.ma.masked_invalid(mat[::-1])
    cmap_d.set_bad("#FAFAFA")
    im = ax_d.imshow(masked, aspect="auto", cmap=cmap_d,
                     extent=[0, tmax, mat.shape[0] - 0.5, -0.5], vmin=-v.max() * 0.02, vmax=v.max() * 0.02)
    for y in range(1, mat.shape[0]):
        ax_d.axhline(y - 0.5, color="white", lw=1.4)
    for x in range(1, mat.shape[1]):
        ax_d.axvline(x - 0.5, color="white", lw=1.4)
    ax_d.set_yticks(range(mat.shape[0]))
    ax_d.set_yticklabels([str(iv) for iv in piv.index], fontsize=5.6)
    ax_d.set_xlabel(args.time_name)
    ax_d.set_ylabel(f"{args.value_name}水平")
    ax_d.set_title("瞬时速率：时间 × 水平", fontsize=8, pad=5)
    cbar = fig.colorbar(im, ax=ax_d, pad=0.02, fraction=0.055)
    cbar.set_label("瞬时速率 (单位/h)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)

    for ax, tag, dx in [(ax_a, "A", -0.035), (ax_b, "B", -0.06), (ax_c, "C", -0.09), (ax_d, "D", -0.09)]:
        ax.text(dx, 1.05, tag, transform=ax.transAxes, fontsize=9, fontweight="bold")

    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.935, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])
    kw = dict(boxstyle="round,pad=0.45", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    fig.text(0.5, 0.035, "关键结论    " + args.conclusions, ha="center", va="center",
             fontsize=6.3, bbox=kw, wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()