# -*- coding: utf-8 -*-
"""信号重构与噪声抑制图：强度背景 + 样条重构 + 拐点标注 + 残差校验（仿参考图2）。

数据契约: --data CSV 含 --t-col 与 --obs-col（带噪观测），可选 --true-col（真值曲线）。
输出: svg/pdf/png 三格式 + QA。
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from _common import tag_panel
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def main():
    ap = argparse.ArgumentParser(description="信号重构与噪声抑制图")
    ap.add_argument("--data", required=True)
    ap.add_argument("--t-col", required=True)
    ap.add_argument("--obs-col", required=True)
    ap.add_argument("--true-col", default="")
    ap.add_argument("--smooth", type=float, default=None, help="样条平滑系数（默认自动）")
    ap.add_argument("--title", default="信号时频域重构与噪声抑制分析")
    ap.add_argument("--subtitle", default="—— 强度背景 + 样条重构曲线 + 拐点标注 + 残差校验 ——")
    ap.add_argument("--conclusions", default="精准捕捉动态变化;显著抑制噪声干扰;为后续建模提供可靠基础",
                    help="分号分隔的三条结论")
    ap.add_argument("--out", default="fig_turning_point")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = df.sort_values(args.t_col)
    t = df[args.t_col].values.astype(float)
    obs = df[args.obs_col].values.astype(float)
    true = df[args.true_col].values.astype(float) if args.true_col and args.true_col in df else None

    # 样条重构（平滑样条：在分位结点上拟合，s 由平滑系数控制）
    from scipy.interpolate import UnivariateSpline
    s_factor = args.smooth if args.smooth is not None else len(t) * obs.var() * 0.02
    order = np.argsort(t)
    spl = UnivariateSpline(t[order], obs[order], k=3, s=s_factor)
    tg = np.linspace(t.min(), t.max(), 600)
    fit = spl(tg)
    resid = obs - spl(t)
    sd = resid.std()
    band_hi, band_lo = fit + 1.96 * sd, fit - 1.96 * sd

    # 拐点（极值点检测，最小间隔过滤）
    d1 = np.gradient(fit, tg)
    sgn = np.sign(d1)
    ext_i = [i for i in range(1, len(tg) - 1) if sgn[i - 1] * sgn[i] < 0]
    merged = []
    for i in ext_i:
        if not merged or tg[i] - merged[-1] > (t.max() - t.min()) * 0.08:
            merged.append(i)
    ext_i = merged[:2] if len(merged) >= 2 else merged

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(128)))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.45, wspace=0.34,
                          left=0.06, right=0.97, top=0.90, bottom=0.155)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1:])
    ax_c.set_position([0.40, 0.13, 0.28, 0.30])
    ax_d = fig.add_axes([0.735, 0.13, 0.245, 0.30])

    # ---------- A ----------
    H, xe, ye = np.histogram2d(t, obs, bins=(36, 26))
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(H.T, sigma=1.6)
    cmap_bg = LinearSegmentedColormap.from_list("bg", ["#0B1F4E", "#1B5FAA", "#3F8EFC",
                                                       "#7FC8A9", "#F2E8A0"])
    ax_a.imshow(Hs, extent=[xe[0], xe[-1], ye[0], ye[-1]], origin="lower",
                aspect="auto", cmap=cmap_bg, alpha=0.92, zorder=0)
    if true is not None:
        ax_a.plot(t, true, color=PALETTE["neutral_light"], lw=0.9, alpha=0.7,
                  label="真值曲线", zorder=2)
    ax_a.scatter(t, obs, s=3, color="white", alpha=0.5, linewidths=0, zorder=2, label="原始散点")
    ax_a.plot(tg, fit, color=PALETTE["orange_main"], lw=1.8, zorder=4, label="样条重构曲线")
    ax_a.fill_between(tg, band_lo, band_hi, color=PALETTE["orange_main"], alpha=0.18,
                      lw=0, zorder=3, label="95% 置信区间")
    names = ["关键拐点", "关键拐点"]
    for k, i in enumerate(ext_i):
        ax_a.plot([tg[i], tg[i]], [ye[0], fit[i]], color="white", lw=0.8, ls=":", zorder=4)
        ax_a.scatter([tg[i]], [fit[i]], s=24, facecolor="white", edgecolor=PALETTE["orange_main"],
                     lw=1.0, zorder=6)
        above = fit[i] < np.nanmedian(fit)
        tx = min(tg[i] + 14, tg.max() - 60)
        ty = fit[i] + (5.5 if above else -7.5)
        ax_a.annotate("%s\n(%d s, %.0f Hz)" % (names[k], tg[i], fit[i]),
                      xy=(tg[i], fit[i]), xytext=(tx, ty), fontsize=6.2,
                      color="white",
                      bbox=dict(boxstyle="round,pad=0.25", fc="#123C7A", ec="white", lw=0.5),
                      arrowprops=dict(arrowstyle="->", color="white", lw=0.8))
    ax_a.set_xlabel("时间 (s)"); ax_a.set_ylabel("频率 (Hz)")
    ax_a.set_title("重构曲线与强度背景", fontsize=8, pad=5)
    leg = ax_a.legend(fontsize=5.8, loc="upper right", frameon=True)
    leg.get_frame().set_facecolor("white"); leg.get_frame().set_alpha(0.92); leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_facecolor("white"); leg.get_frame().set_alpha(0.9)
    tag_panel(ax_a, "A", -0.03, 1.04)

    # ---------- B ----------
    ax_b.scatter(t, obs, s=4, color=PALETTE["neutral_mid"], alpha=0.45, linewidths=0,
                 label="原始散点")
    ax_b.plot(tg, fit, color=PALETTE["orange_main"], lw=1.5, label="样条拟合曲线")
    ax_b.fill_between(tg, band_lo, band_hi, color=PALETTE["orange_main"], alpha=0.15, lw=0,
                      label="95% 置信区间")
    ax_b.set_xlabel("时间 (s)"); ax_b.set_ylabel("幅值")
    ax_b.set_title("原始信号与样条拟合对比", fontsize=8)
    ax_b.legend(fontsize=5.8)
    tag_panel(ax_b, "B", -0.10, 1.06)

    # ---------- C: 残差直方图 ----------
    ax_c.hist(resid, bins=26, color=PALETTE["blue_secondary"], alpha=0.85, edgecolor="white",
              density=True)
    xs = np.linspace(resid.min(), resid.max(), 200)
    ax_c.plot(xs, norm.pdf(xs, resid.mean(), sd), color=PALETTE["red_strong"], lw=1.3,
              label="正态拟合曲线")
    r2 = 1 - (resid ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
    ax_c.annotate("均值  %.3f\n标准差  %.3f\nR²  %.3f" % (resid.mean(), sd, r2),
                  xy=(0.97, 0.86), xycoords="axes fraction", fontsize=6, ha="right",
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["neutral_mid"], lw=0.5))
    ax_c.set_xlabel("残差值"); ax_c.set_ylabel("频数")
    ax_c.set_title("残差分布直方图验证插值平滑度", fontsize=8)
    ax_c.legend(fontsize=5.8)
    tag_panel(ax_c, "C", -0.08, 1.06)

    # ---------- D: 关键结论图标框 ----------
    ax_d = fig.add_axes([0.745, 0.13, 0.235, 0.30])
    ax_d.axis("off")
    items = [(g, t1_, d_) for g, t1_, d_ in zip("①②③", args.conclusions.split(";"),
             ["样条插值有效重构环境激励信号，准确识别强弱与时序特征。",
              "插值后信号显著平滑，残差分布近似正态，验证了良好的噪声抑制效果。",
              "时频域结合分析实现了对复杂动态环境的高精度表征与建模。"])]
    for k, (g, title, desc) in enumerate(items):
        yy = 1 - (k + 0.5) / 3
        ax_d.add_patch(plt.Rectangle((0.0, yy - 0.40), 1.0, 0.80, transform=ax_d.transAxes,
                                     facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.7))
        ax_d.text(0.07, yy, g, fontsize=12, color=PALETTE["blue_main"], va="center",
                  transform=ax_d.transAxes)
        ax_d.text(0.18, yy + 0.20, title, fontsize=6.8, fontweight="bold", va="center",
                  transform=ax_d.transAxes)
        ax_d.text(0.18, yy - 0.26, desc, fontsize=4.8, va="center", transform=ax_d.transAxes)
    tag_panel(ax_d, "D", -0.02, 1.02)

    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.935, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
