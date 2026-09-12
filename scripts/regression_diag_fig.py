# -*- coding: utf-8 -*-
"""回归诊断图：预测vs真实 + 残差 + 正态QQ + 指标框。"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import tag_panel
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def main():
    ap = argparse.ArgumentParser(description="回归诊断四面板")
    ap.add_argument("--data", required=True)
    ap.add_argument("--y-col", required=True, help="真实值列")
    ap.add_argument("--pred-col", default="", help="预测值列（给出则直接用）")
    ap.add_argument("--feature-cols", default="", help="不给 pred 时，对这些特征做 OLS")
    ap.add_argument("--y-name", default="真实值")
    ap.add_argument("--pred-name", default="预测值")
    ap.add_argument("--title", default="回归模型拟合效果与残差诊断")
    ap.add_argument("--subtitle", default="—— 预测对比、残差结构、正态性与分位检验 ——")
    ap.add_argument("--out", default="fig_regression_diag")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    y = pd.to_numeric(df[args.y_col], errors="coerce").values
    if args.pred_col and args.pred_col in df.columns:
        pred = pd.to_numeric(df[args.pred_col], errors="coerce").values
        method = "给定预测列"
    else:
        cols = [c for c in args.feature_cols.split(",") if c]
        X = df[cols].values.astype(float)
        X = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        method = "OLS(%s)" % ", ".join(cols)
    resid = y - pred
    ss_res = (resid ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt((resid ** 2).mean())
    mae = np.abs(resid).mean()

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(100)))
    ax_a = fig.add_axes([0.08, 0.15, 0.38, 0.68])
    ax_b = fig.add_axes([0.585, 0.15, 0.38, 0.68])
    ax_c = fig.add_axes([0.08, 0.62, 0.22, 0.24])
    ax_c2 = fig.add_axes([0.36, 0.62, 0.22, 0.24])
    ax_d = fig.add_axes([0.63, 0.62, 0.22, 0.24])

    # A: 预测 vs 真实
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    pad = (lim[1] - lim[0]) * 0.05
    ax_a.scatter(y, pred, s=14, color=PALETTE["blue_secondary"], alpha=0.7, linewidths=0)
    ax_a.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad],
              color=PALETTE["red_strong"], lw=1.2, ls="--", label="理想线 y = x")
    k, b0 = np.polyfit(y, pred, 1)
    ax_a.plot(xs := np.linspace(lim[0], lim[1], 50), k * xs + b0, color=PALETTE["orange_main"],
              lw=1.2, label="拟合线")
    ax_a.set_xlim(lim[0] - pad, lim[1] + pad); ax_a.set_ylim(lim[0] - pad, lim[1] + pad)
    ax_a.set_xlabel("真实 " + args.y_name); ax_a.set_ylabel("预测 " + args.pred_name)
    ax_a.set_title("预测 vs 真实（%s）" % method, fontsize=8, pad=5)
    ax_a.annotate("R² = %.4f\nRMSE = %.3f\nMAE = %.3f" % (r2, rmse, mae),
                  xy=(0.04, 0.90), xycoords="axes fraction", fontsize=6.2,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=PALETTE["neutral_mid"], lw=0.5))
    ax_a.legend(fontsize=5.8, loc="lower right")
    tag_panel(ax_a, "A", -0.07, 1.03)

    # B: 残差 vs 拟合
    ax_b.scatter(pred, resid, s=12, color=PALETTE["teal_main"], alpha=0.7, linewidths=0)
    ax_b.axhline(0, color=PALETTE["black"], lw=0.8, ls="--")
    bins = pd.cut(pred, 12)
    mres = pd.Series(resid).groupby(bins, observed=True).mean()
    ax_b.plot([iv.mid for iv in mres.index], mres.values, color=PALETTE["red_strong"], lw=1.3,
              marker="o", ms=3, label="分箱均值")
    ax_b.set_xlabel("拟合值"); ax_b.set_ylabel("残差")
    ax_b.set_title("残差 vs 拟合值（异方差检验）", fontsize=8, pad=5)
    ax_b.legend(fontsize=5.6)
    tag_panel(ax_b, "B", -0.07, 1.03)

    # C: 残差直方 + 正态
    ax_c.hist(resid, bins=18, density=True, color=PALETTE["blue_secondary"], alpha=0.8,
              edgecolor="white")
    xs = np.linspace(resid.min(), resid.max(), 150)
    ax_c.plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), color=PALETTE["red_strong"], lw=1.2)
    ax_c.set_title("残差分布", fontsize=7.5, pad=4)
    ax_c.tick_params(labelsize=5)

    # C2: QQ 图
    (osm, osr), _ = stats.probplot(resid, dist="norm")
    ax_c2.scatter(osm, osr, s=6, color=PALETTE["blue_secondary"], alpha=0.7)
    ax_c2.plot(osm, osm * resid.std() + resid.mean(), color=PALETTE["red_strong"], lw=1.1)
    ax_c2.set_title("正态 Q-Q", fontsize=7.5, pad=4)
    ax_c2.tick_params(labelsize=5)

    # D: 指标框
    sw = stats.shapiro(resid[:5000])[1] if len(resid) <= 5000 else np.nan
    ax_d.axis("off")
    rows = [("RMSE", "%.4f" % rmse), ("MAE", "%.4f" % mae), ("R²", "%.4f" % r2),
            ("Shapiro p", "%.3f" % sw if np.isfinite(sw) else "—")]
    for k, (t1_, v1_) in enumerate(rows):
        yy = 1 - (k + 0.5) / 4
        ax_d.text(0.0, yy, t1_, fontsize=6.5, fontweight="bold", va="center")
        ax_d.text(1.0, yy, v1_, fontsize=6.5, va="center", ha="right")
        ax_d.plot([0, 1], [yy - 0.28, yy - 0.28], color=PALETTE["neutral_light"], lw=0.6,
                  transform=ax_d.transAxes)
    ax_d.set_xlim(0, 1); ax_d.set_ylim(0, 1)
    ax_d.set_title("精度指标", fontsize=7.5, pad=4)
    ax_d.tick_params(labelsize=5)

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
