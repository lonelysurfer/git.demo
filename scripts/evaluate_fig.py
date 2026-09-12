# -*- coding: utf-8 -*-
"""综合评价图：熵权法 + TOPSIS 得分 + 权重灵敏度（评价类赛题通用）。"""
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


def entropy_weights(X, neg_cols):
    """X: (n, m) 正向化后的矩阵（全部越大越优）。返回熵权。"""
    P = X / X.sum(axis=0)
    P = np.clip(P, 1e-12, None)
    k = 1 / np.log(len(X))
    e = k * (P * np.log(P)).sum(axis=0)
    d = 1 - e
    return d / d.sum()


def topsis(X, w, neg_mask):
    """负向列先正向化。返回相对接近度。"""
    Xp = X.copy()
    Xp[:, neg_mask] = X[:, neg_mask] * -1
    Xp = Xp / np.sqrt((Xp ** 2).sum(axis=0))
    weighted = Xp * w
    ideal_p, ideal_n = weighted.max(0), weighted.min(0)
    dp = np.sqrt(((weighted - ideal_p) ** 2).sum(1))
    dn = np.sqrt(((weighted - ideal_n) ** 2).sum(1))
    return dn / (dp + dn + 1e-12)


def main():
    ap = argparse.ArgumentParser(description="综合评价图（熵权+TOPSIS）")
    ap.add_argument("--data", required=True, help="CSV：一列方案名 + 若干指标列（数值）")
    ap.add_argument("--name-col", required=True)
    ap.add_argument("--negative-cols", default="", help="成本型（越低越好）指标列，逗号分隔")
    ap.add_argument("--x-name", default="指标", help="指标轴名称（用于热图标注）")
    ap.add_argument("--title", default="基于熵权-TOPSIS 的综合评价与权重灵敏度")
    ap.add_argument("--subtitle", default="—— 熵权法客观赋权 + TOPSIS 相对接近度排序 ——")
    ap.add_argument("--out", default="fig_evaluate")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    names = df[args.name_col].astype(str).values
    crit_cols = [c for c in df.columns if c != args.name_col]
    X = df[crit_cols].values.astype(float)
    neg_mask = np.array([c in [x.strip() for x in args.negative_cols.split(",") if x.strip()]
                         for c in crit_cols])
    # 正向化
    Xp = X.copy()
    Xp[:, neg_mask] = X[:, neg_mask].min() * 1.0001 - X[:, neg_mask]

    w = entropy_weights(Xp, neg_mask)
    score = topsis(Xp, w, neg_mask)
    order = np.argsort(-score)

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(118)))
    ax_w = fig.add_axes([0.07, 0.16, 0.26, 0.70])
    ax_s = fig.add_axes([0.40, 0.16, 0.24, 0.70])
    ax_h = fig.add_axes([0.70, 0.16, 0.27, 0.70])

    # A: 权重条形 + 累计线
    ow = np.argsort(-w)
    ax_w.barh(range(len(crit_cols)), w[ow], color=PALETTE["blue_main"], alpha=0.85, height=0.6)
    for k, oi in enumerate(ow):
        ax_w.text(w[oi] + w.max() * 0.02, k, "%.3f" % w[oi], va="center", fontsize=5.6)
    ax_w.set_yticks(range(len(crit_cols)))
    ax_w.set_yticklabels([crit_cols[i] for i in ow], fontsize=6.2)
    ax_w.set_xlabel("熵权")
    ax_w.set_title("指标权重（熵权法）", fontsize=8)
    ax_w.set_xlim(0, w.max() * 1.18)
    tag_panel(ax_w, "A", -0.16, 1.04)

    # B: 得分排序
    ax_s.barh(range(len(score)), score[order], color=PALETTE["orange_main"], alpha=0.88, height=0.62)
    for k, oi in enumerate(order):
        ax_s.text(score[oi] + 0.012, k, "%.3f" % score[oi], va="center", fontsize=5.6)
    ax_s.set_yticks(range(len(order)))
    ax_s.set_yticklabels([names[i] for i in order], fontsize=6.2)
    ax_s.set_xlabel("TOPSIS 相对接近度")
    ax_s.set_title("方案综合得分排序", fontsize=8)
    ax_s.set_xlim(0, score.max() * 1.16)
    best = names[order[0]]
    ax_s.annotate("最优方案", xy=(score[order[0]], 0), xytext=(score.max() * 0.55, len(order) * 0.55),
                  fontsize=6.2, arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=0.8),
                  color=PALETTE["red_strong"])
    tag_panel(ax_s, "B", -0.15, 1.04)

    # C: 归一化热图
    Xn = (X - X.min(0)) / np.where(X.max(0) - X.min(0) == 0, 1, X.max(0) - X.min(0))
    Xn[:, neg_mask] = 1 - Xn[:, neg_mask]
    im = ax_h.imshow(Xn, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax_h.set_xticks(range(len(crit_cols)))
    ax_h.set_xticklabels([c if len(c) < 8 else c[:6] + "…" for c in crit_cols], fontsize=5.4, rotation=30, ha="right")
    ax_h.set_yticks(range(len(names)))
    ax_h.set_yticklabels([names[i] for i in order], fontsize=6)
    for i in range(len(names)):
        for j in range(len(crit_cols)):
            ax_h.text(j, i, "%.2f" % Xn[i, j], ha="center", va="center", fontsize=5.2,
                      color="white" if Xn[i, j] > 0.6 else "#333333")
    ax_h.set_title("正向化指标热图（按得分排序）", fontsize=8)
    fig.colorbar(im, ax=ax_h, pad=0.02, fraction=0.045).ax.tick_params(labelsize=5)
    tag_panel(ax_h, "C", -0.13, 1.04)

    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.935, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])
    kw = dict(boxstyle="round,pad=0.45", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    fig.text(0.5, 0.028, "关键结论    最优方案为 %s（接近度 %.3f）；权重最大的指标为 %s（%.3f）；"
             "权重结构由数据信息量客观决定。" % (best, score.max(), crit_cols[ow[0]], w[ow[0]]),
             ha="center", va="center", fontsize=6.3, bbox=kw, wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
