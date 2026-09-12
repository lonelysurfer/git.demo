# -*- coding: utf-8 -*-
"""灰色/时序预测图：GM(1,1) 或指数拟合 + 预测带 + 后验差检验。"""
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
from _common import tag_panel
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def gm11(y, n_total):
    """GM(1,1)：返回还原后的拟合/预测序列与参数 (a, b)。"""
    y1 = np.cumsum(y)
    z1 = 0.5 * (y1[1:] + y1[:-1])
    B = np.column_stack([-z1, np.ones(len(z1))])
    a, b = np.linalg.lstsq(B, y[1:], rcond=None)[0]
    k = np.arange(n_total)
    xhat = (y[0] - b / a) * np.exp(-a * k) + b / a
    yhat = np.concatenate([[y[0]], np.diff(xhat)])
    return yhat, (a, b)


def main():
    ap = argparse.ArgumentParser(description="灰色/时序预测图")
    ap.add_argument("--data", required=True, help="CSV: 时间列 + 数值列（等距序列）")
    ap.add_argument("--t-col", default="t")
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--method", default="gm11", choices=["gm11", "exp"], help="gm11=灰色预测, exp=指数拟合")
    ap.add_argument("--forecast", type=int, default=6, help="向前预测步数")
    ap.add_argument("--t-name", default="期数")
    ap.add_argument("--value-name", default="数值")
    ap.add_argument("--title", default="灰色预测 GM(1,1) 的拟合、预测与精度检验")
    ap.add_argument("--subtitle", default="—— 拟合曲线、预测带与后验差检验 ——")
    ap.add_argument("--out", default="fig_grey_pred")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = pd.to_numeric(df[args.value_col], errors="coerce").dropna().values
    tv = np.arange(1, len(y) + 1)
    n = len(y)
    nf = args.forecast
    t_all = np.arange(1, n + nf + 1)

    if args.method == "gm11":
        yhat_all, (a_g, b_g) = gm11(y, n + nf)
        fit = yhat_all[:n]
        pred = yhat_all[n:]
        model_name = "灰色预测 GM(1,1)"
    else:
        coef = np.polyfit(tv, y, 1)
        fit = np.polyval(coef, tv)
        pred = np.polyval(coef, n + np.arange(1, nf + 1))
        a_g, b_g = coef[0], coef[1]
        model_name = "指数趋势拟合"

    resid = y - fit
    se = resid.std(ddof=1) if n > 2 else resid.std()
    # 预测带：残差 σ 按 sqrt(1+1/n) 放大
    band = se * np.sqrt(1 + 1.0 / n) * 1.96
    # 后验差检验
    c_ratio = se / (y.std() + 1e-12)
    p_prob = float(np.mean(np.abs(resid - resid.mean()) < 0.6745 * y.std() + 1e-12))

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(100)))
    ax_a = fig.add_axes([0.07, 0.16, 0.52, 0.68])
    ax_b = fig.add_axes([0.68, 0.16, 0.29, 0.68])

    ax_a.plot(tv, y, "o-", color=PALETTE["blue_main"], lw=1.4, ms=4, label="原始序列")
    ax_a.plot(tv, fit, color=PALETTE["orange_main"], lw=1.5, label="拟合曲线 (%s)" % model_name)
    ax_a.plot(np.arange(n + 1, n + nf + 1), pred, color=PALETTE["red_strong"], lw=1.5,
              marker="s", ms=3.5, ls="--", label="预测 (%d 步)" % nf)
    ax_a.fill_between(np.arange(n, n + nf + 1),
                      np.concatenate([[fit[-1]], pred - band]),
                      np.concatenate([[fit[-1]], pred + band]),
                      color=PALETTE["red_strong"], alpha=0.14, lw=0, label="95% 预测带")
    ax_a.axvline(n + 0.5, color=PALETTE["neutral_mid"], lw=0.8, ls=":")
    ax_a.text(n + 0.6, ax_a.get_ylim()[1] * 0.95, "→ 预测区", fontsize=6, va="top")
    ax_a.set_xlabel(args.t_name); ax_a.set_ylabel(args.value_name)
    ax_a.set_title("原始序列、拟合与预测", fontsize=8, pad=5)
    ax_a.legend(fontsize=5.8, loc="upper left")
    tag_panel(ax_a, "A", -0.05, 1.04)

    # B: 残差 + 后验差检验
    ax_b.bar(tv, resid, width=0.55, color=PALETTE["blue_secondary"], alpha=0.85, edgecolor="white")
    ax_b.axhline(0, color=PALETTE["black"], lw=0.7)
    grade = "优" if c_ratio < 0.35 and p_prob > 0.95 else ("良" if c_ratio < 0.5 and p_prob > 0.8 else "合格")
    ax_b.annotate("后验差比 C = %.3f\n小误差概率 P = %.3f\n模型等级：[%s]"
                  % (c_ratio, p_prob, grade),
                  xy=(0.97, 0.92), xycoords="axes fraction", fontsize=6.4, ha="right", va="top",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["black"], lw=0.6))
    ax_b.set_xlabel(args.t_name); ax_b.set_ylabel("残差")
    ax_b.set_title("残差与后验差检验", fontsize=8, pad=5)
    ax_b.set_xticks(tv[::max(1, n // 10)])
    tag_panel(ax_b, "B", -0.08, 1.04)

    # 发展系数结论（GM11 专用）
    extra = ("发展系数 -a = %.4f（%s），灰作用量 b = %.1f。"
             % (-a_g, "序列增长" if a_g < 0 else "序列衰减", b_g)) if args.method == "gm11" else ""
    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.935, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])
    kw = dict(boxstyle="round,pad=0.45", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    fig.text(0.5, 0.028, "关键结论    相对误差均值 %.2f%%；%s拟合质量等级 [%s]，可用于中短期预测。"
             % (np.mean(np.abs(resid / y)) * 100, extra, grade),
             ha="center", va="center", fontsize=6.3, bbox=kw, wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
