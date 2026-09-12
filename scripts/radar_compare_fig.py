# -*- coding: utf-8 -*-
"""雷达对比：多模型多指标对比 + 小倍数 + 汇总柱（仿参考图7）。"""
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
from matplotlib.patches import FancyBboxPatch
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def main():
    ap = argparse.ArgumentParser(description="雷达对比图")
    ap.add_argument("--metrics-csv", required=True, help="CSV: 模型列, 指标列, 数值列")
    ap.add_argument("--model-col", default="model")
    ap.add_argument("--metric-col", default="metric")
    ap.add_argument("--value-col", default="value")
    ap.add_argument("--model-names", default="", help="逗号分隔，按数据出现顺序")
    ap.add_argument("--lower-better", default="", help="逗号分隔的'越低越好'指标名（雷达上仍按相对优劣绘制）")
    ap.add_argument("--detail-csv", default="", help="小倍数数据: 模型, 指标, 点, 数值")
    ap.add_argument("--point-col", default="point")
    ap.add_argument("--point-name", default="位置")
    ap.add_argument("--conclusions", default="", help="分号分隔的右侧结论（2-3 条）")
    ap.add_argument("--title", default="多模型偏差量化对比雷达图")
    ap.add_argument("--subtitle", default="—— 多指标相对优劣对比 ——")
    ap.add_argument("--out", default="fig_radar")
    args = ap.parse_args()

    md = pd.read_csv(args.metrics_csv)
    model_order = list(dict.fromkeys(md[args.model_col].astype(str)))
    mnames = ([x for x in args.model_names.split(",")] if args.model_names else model_order)
    name_map = dict(zip(model_order, mnames))
    metrics = list(dict.fromkeys(md[args.metric_col].astype(str)))
    piv = md.pivot_table(index=args.metric_col, columns=args.model_col, values=args.value_col, aggfunc="mean")
    piv = piv.reindex(index=metrics, columns=model_order)
    # 每个指标按最优=1、最差归一（数值越低越好假设下，雷达外圈=该指标更优）
    norm_mat = piv.copy()
    for m in metrics:
        row = piv.loc[m]
        span = row.max() - row.min()
        norm_mat.loc[m] = 1 - (row - row.min()) / span if span > 0 else 1.0

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(125)))

    # ---------- A: 雷达 ----------
    ax_a = fig.add_axes([0.04, 0.38, 0.46, 0.56], polar=True)
    K = len(metrics)
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    for k, mo in enumerate(model_order):
        vals = norm_mat[mo].values.astype(float)
        vals_c = np.concatenate([vals, vals[:1]])
        ax_a.plot(ang_c, vals_c, lw=1.6, color=PALETTE["orange_main"] if k == 0 else PALETTE["blue_main"],
                  marker="o", ms=3.5, label=mnames[k])
        ax_a.fill(ang_c, vals_c, color=PALETTE["orange_main"] if k == 0 else PALETTE["blue_main"], alpha=0.10)
    ax_a.set_xticks(ang)
    ax_a.set_xticklabels(metrics, fontsize=6.3)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_a.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=5.2)
    ax_a.set_title("雷达对比分析（外圈=该指标更优）", fontsize=8, pad=12)
    ax_a.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    tag = fig.add_axes([0.015, 0.90, 0.05, 0.05]); tag.axis("off")
    tag.text(0, 0.5, "A", fontsize=9, fontweight="bold")

    # ---------- 右侧关键结论 ----------
    ax_r = fig.add_axes([0.55, 0.40, 0.43, 0.50])
    ax_r.axis("off")
    if len(model_order) >= 2:
        a0, a1 = model_order[0], model_order[1]
        impr = []
        for m in metrics:
            v0, v1 = piv.loc[m, a0], piv.loc[m, a1]
            impr.append((m, (v0 - v1) / v0 * 100 if v0 else np.nan))
    items = args.conclusions.split(";") if args.conclusions else \
        ["%s 平均优于 %.1f%%" % (mnames[1], np.nanmean([i[1] for i in impr]))]
    ax_r.text(0.02, 0.97, "关键结论", fontsize=8, fontweight="bold", transform=ax_r.transAxes)
    for k, c in enumerate(items):
        yy = 0.80 - k * 0.22
        ax_r.add_patch(FancyBboxPatch((0.0, yy - 0.10), 1.0, 0.22, boxstyle="round,pad=0.02",
                                      facecolor="#EAF3FD", edgecolor=PALETTE["blue_main"], lw=0.7,
                                      transform=ax_r.transAxes))
        ax_r.text(0.04, yy, "✓", fontsize=10, color=PALETTE["green_strong"], va="center",
                  transform=ax_r.transAxes)
        ax_r.text(0.13, yy, c, fontsize=6.2, va="center", transform=ax_r.transAxes)
    if len(model_order) >= 2:
        for k, (m, imp) in enumerate(impr[:3]):
            ax_r.text(0.13, 0.60 - k * 0.22, "%s: %+.1f%%" % (m, imp), fontsize=5.8,
                      transform=ax_r.transAxes)

    # ---------- B: 小倍数 ----------
    if args.detail_csv and Path(args.detail_csv).exists():
        dt = pd.read_csv(args.detail_csv)
        pts = list(dict.fromkeys(dt[args.point_col].astype(str)))
        for k, m in enumerate(metrics):
            axm = fig.add_axes([0.05 + (k % 3) * 0.32, 0.30 - (k // 3) * 0.155,
                                0.24, 0.20]) if k < 6 else None
            if axm is None:
                continue
            for j, mo in enumerate(model_order):
                sel = dt[(dt[args.metric_col].astype(str) == m) & (dt[args.model_col].astype(str) == mo)]
                sel = sel.sort_values(args.point_col)
                axm.plot(sel[args.point_col].astype(str), sel[args.value_col],
                         marker="o", ms=2.5, lw=1.1,
                         color=PALETTE["orange_main"] if j == 0 else PALETTE["blue_main"],
                         label=mnames[j] if k == 0 else None)
            axm.set_title("%s" % m, fontsize=6.5)
            axm.tick_params(labelsize=4.8)
            if k == 0:
                axm.legend(fontsize=4.6)
    tag = fig.add_axes([0.02, 0.36, 0.05, 0.05]); tag.axis("off")
    tag.text(0, 0.5, "B", fontsize=9, fontweight="bold")

    # ---------- C: 汇总柱 ----------
    ax_c2 = fig.add_axes([0.56, 0.075, 0.40, 0.30])
    xpos = np.arange(len(metrics))
    w = 0.8 / len(model_order)
    for j, mo in enumerate(model_order):
        ax_c2.bar(xpos + (j - (len(model_order) - 1) / 2) * w, piv[mo].values, width=w,
                  color=PALETTE["orange_main"] if j == 0 else PALETTE["blue_main"],
                  alpha=0.88, label=mnames[j])
        for xp, v in zip(xpos + (j - (len(model_order) - 1) / 2) * w, piv[mo].values):
            ax_c2.text(xp, v, "%.1f" % v, ha="center", va="bottom", fontsize=4.8)
    ax_c2.set_xticks(xpos)
    ax_c2.set_xticklabels([m if len(m) < 10 else m[:8] + "…" for m in metrics], fontsize=5.5)
    ax_c2.set_title("指标对比汇总", fontsize=8)
    ax_c2.legend(fontsize=5.4)
    tag = fig.add_axes([0.53, 0.36, 0.05, 0.05]); tag.axis("off")
    tag.text(0, 0.5, "C", fontsize=9, fontweight="bold")

    fig.suptitle(args.title, fontsize=11, y=0.995)
    fig.text(0.5, 0.955, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
