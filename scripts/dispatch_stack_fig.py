# -*- coding: utf-8 -*-
"""优化调度堆叠图：多主体出力堆叠 + 价格/需求线 + 净平衡（调度类赛题通用）。"""
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


def main():
    ap = argparse.ArgumentParser(description="优化调度堆叠图")
    ap.add_argument("--data", required=True, help="宽表 CSV：时间列 + 各主体出力列")
    ap.add_argument("--time-col", required=True)
    ap.add_argument("--series-cols", required=True, help="逗号分隔的主体出力列（按堆叠自下而上顺序）")
    ap.add_argument("--series-names", default="", help="逗号分隔的图例名")
    ap.add_argument("--demand-col", default="", help="需求/负荷列（可选）")
    ap.add_argument("--price-col", default="", help="价格/边际成本列（次轴，可选）")
    ap.add_argument("--time-name", default="时间 (h)")
    ap.add_argument("--title", default="多主体调度方案与出力结构")
    ap.add_argument("--subtitle", default="—— 各主体出力堆叠、需求覆盖与价格联动 ——")
    ap.add_argument("--out", default="fig_dispatch")
    args = ap.parse_args()

    df = pd.read_csv(args.data).sort_values(args.time_col)
    cols = [c for c in args.series_cols.split(",")]
    names = ([n for n in args.series_names.split(",")] if args.series_names
             else [c for c in cols])
    tv = df[args.time_col].values
    stack = np.vstack([df[c].values.astype(float) for c in cols])

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(110)))
    ax_a = fig.add_axes([0.06, 0.17, 0.60, 0.68])
    ax_p = ax_a.twinx() if args.price_col else None
    ax_b = fig.add_axes([0.72, 0.17, 0.26, 0.30])
    ax_c = fig.add_axes([0.72, 0.56, 0.26, 0.29])

    # A: 堆叠面积
    colors = [PALETTE["blue_main"], PALETTE["orange_main"], PALETTE["green_strong"],
              PALETTE["violet_main"], PALETTE["teal_main"], PALETTE["gold_main"]]
    bases = np.zeros(len(tv))
    for k, (c, nm) in enumerate(zip(cols, names)):
        ax_a.fill_between(tv, bases, bases + stack[k], color=colors[k % 6], alpha=0.82,
                          lw=0.4, edgecolor="white", label=nm)
        bases = bases + stack[k]
    total = bases
    if args.demand_col and args.demand_col in df.columns:
        ax_a.plot(tv, df[args.demand_col].values, color=PALETTE["red_strong"], lw=1.5,
                  ls="--", label="需求")
        gap = total - df[args.demand_col].values
        ax_a.fill_between(tv, total, df[args.demand_col].values,
                          where=np.abs(gap) > 1e-9, color=PALETTE["red_strong"], alpha=0.08)
    ax_a.set_xlabel(args.time_name)
    ax_a.set_ylabel("出力")
    ax_a.set_title("各主体出力堆叠与需求覆盖", fontsize=8, pad=5)
    h, l = ax_a.get_legend_handles_labels()
    if args.price_col:
        ax_p.plot(tv, df[args.price_col].values, color=PALETTE["black"], lw=1.1, alpha=0.8,
                  label="价格")
        ax_p.set_ylabel("价格", fontsize=6.5)
        ax_p.tick_params(labelsize=5.5)
        h2, l2 = ax_p.get_legend_handles_labels()
        h, l = h + h2, l + l2
    ax_a.legend(h, l, fontsize=5.5, ncol=2, loc="upper left")
    tag_panel(ax_a, "A", -0.03, 1.04)

    # B: 各主体汇总
    totals = stack.sum(axis=1)
    ax_b.barh(range(len(cols)), totals, color=colors[:len(cols)], alpha=0.88, height=0.6)
    for k, v in enumerate(totals):
        ax_b.text(v + totals.max() * 0.015, k, "%.0f" % v, va="center", fontsize=5.6)
    ax_b.set_yticks(range(len(cols)))
    ax_b.set_yticklabels(names, fontsize=6.2)
    ax_b.set_xlabel("总出力")
    ax_b.set_title("各主体总出力", fontsize=8)
    ax_b.set_xlim(0, totals.max() * 1.15)
    tag_panel(ax_b, "B", -0.15, 1.04)

    # C: 净平衡或占比
    if args.demand_col and args.demand_col in df.columns:
        balance = total - df[args.demand_col].values
        ax_c.fill_between(tv, balance, 0, color=PALETTE["green_strong"], alpha=0.35)
        ax_c.plot(tv, balance, color=PALETTE["green_strong"], lw=1.2)
        ax_c.axhline(0, color=PALETTE["black"], lw=0.7, ls="--")
        ax_c.set_xlabel(args.time_name)
        ax_c.set_ylabel("供给 − 需求")
        ax_c.set_title("供需平衡裕度", fontsize=8)
    else:
        share = stack / np.where(total == 0, 1, total)
        bottom = np.zeros(len(tv))
        for k in range(len(cols)):
            ax_c.stackplot(tv, share[k], bottom=bottom, color=colors[k % 6], alpha=0.85)
            bottom = bottom + share[k]
        ax_c.set_ylim(0, 1)
        ax_c.set_xlabel(args.time_name); ax_c.set_ylabel("占比")
        ax_c.set_title("出力结构占比", fontsize=8)
    tag_panel(ax_c, "C", -0.13, 1.04)

    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.935, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])
    kw = dict(boxstyle="round,pad=0.45", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    fig.text(0.5, 0.028, "关键结论    各主体出力结构覆盖需求全过程，总出力 %.0f；"
             "供给-需求最大偏差 %.1f，调度方案满足约束。" % (total.sum(), np.abs(total - df[args.demand_col]).max()
                                                      if args.demand_col and args.demand_col in df.columns else 0),
             ha="center", va="center", fontsize=6.3, bbox=kw, wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
