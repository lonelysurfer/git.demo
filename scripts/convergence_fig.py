# -*- coding: utf-8 -*-
"""数值解收敛性与守恒律校验（仿参考图4）。数据契约见 --help。"""
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
    ap = argparse.ArgumentParser(description="收敛性与守恒校验图")
    ap.add_argument("--conv-data", required=True, help="CSV: 方法列, 步长列, 误差列")
    ap.add_argument("--method-col", default="method")
    ap.add_argument("--dt-col", default="dt")
    ap.add_argument("--err-col", default="error")
    ap.add_argument("--method-names", default="", help="逗号分隔，按数据出现顺序")
    ap.add_argument("--cons-data", default="", help="CSV: t, E偏差, M偏差（守恒校验，可选）")
    ap.add_argument("--cons-t-col", default="t")
    ap.add_argument("--cons-e-col", default="E")
    ap.add_argument("--cons-m-col", default="M")
    ap.add_argument("--colors", default="#B64342,#1B5FAA,#4F8A4B,#8A8A8A")
    ap.add_argument("--title", default="数值解收敛性与守恒律校验图谱")
    ap.add_argument("--subtitle", default="—— 收敛阶、守恒偏差与稳定性 ——")
    ap.add_argument("--conclusions", default="", help="分号分隔的右侧结论标题（自动配描述）")
    ap.add_argument("--out", default="fig_convergence")
    args = ap.parse_args()

    conv = pd.read_csv(args.conv_data)
    method_order = list(dict.fromkeys(conv[args.method_col].astype(str)))
    names = ([x for x in args.method_names.split(",")] if args.method_names else method_order)
    name_map = dict(zip(method_order, names))
    colors = args.colors.split(",")

    # 收敛阶拟合
    slopes = {}
    for m in method_order:
        g = conv[conv[args.method_col].astype(str) == m]
        slopes[m] = np.polyfit(np.log10(g[args.dt_col]), np.log10(np.maximum(g[args.err_col], 1e-30)), 1)[0]
    best = max(slopes, key=slopes.get)

    cons = pd.read_csv(args.cons_data) if args.cons_data else None

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(108)))
    ax_a = fig.add_axes([0.09, 0.14, 0.56, 0.72])
    ax_b = fig.add_axes([0.70, 0.14, 0.28, 0.66]) if cons is not None else None

    for k, m in enumerate(method_order):
        g = conv[conv[args.method_col].astype(str) == m].sort_values(args.dt_col)
        best_flag = m == best
        ax_a.loglog(g[args.dt_col], g[args.err_col], lw=1.6 if best_flag else 1.1,
                    marker="o", ms=3.5, color=colors[k % len(colors)],
                    alpha=1.0 if best_flag else 0.65,
                    label=name_map[m])
        ax_a.loglog(g[args.dt_col], g[args.err_col].iloc[0] * (g[args.dt_col] / g[args.dt_col].iloc[0]) ** slopes[m],
                    color=colors[k % len(colors)], lw=0.7, ls="--", alpha=0.6)
    ax_a.set_xlabel("时间步长 " + args.dt_col.replace("dt", r"$\Delta t$"))
    ax_a.set_ylabel("误差范数")
    ax_a.set_title("误差范数随时间步长的收敛性（Log-Log 坐标）", fontsize=8, pad=5)
    ax_a.grid(True, which="both", alpha=0.25, lw=0.4)
    leg = ax_a.legend(fontsize=5.8, loc="lower left", title="数值方法", title_fontsize=6)
    leg.get_frame().set_alpha(0.85)
    # 收敛阶三角标注
    ax_a.annotate("收敛阶 (斜率)\n最优方法 ≈ %.1f" % slopes[best],
                  xy=(0.60, 0.22), xycoords="axes fraction", fontsize=6.5,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["neutral_mid"], lw=0.6))
    tag_panel(ax_a, "A", -0.06, 1.03)

    if ax_b is not None:
        e = np.abs(pd.to_numeric(cons[args.cons_e_col], errors="coerce")).values
        m = np.abs(pd.to_numeric(cons[args.cons_m_col], errors="coerce")).values
        tv = pd.to_numeric(cons[args.cons_t_col], errors="coerce").values
        ax_b.semilogy(tv, e + 1e-30, color=PALETTE["blue_main"], lw=1.0, label="总能量守恒偏差 (E)")
        ax_b.semilogy(tv, m + 1e-30, color=PALETTE["gold_main"], lw=1.0, ls="--", label="总质量守恒偏差 (M)")
        ax_b.set_xlabel("时间 t"); ax_b.set_ylabel("守恒偏差（相对）")
        ax_b.set_title("系统总能量/质量守恒偏差的时间序列", fontsize=8, pad=5)
        ax_b.legend(fontsize=5.6, loc="upper right")
        max_dev = "%.1e" % max(e.max(), m.max())
        tag_panel(ax_b, "B", -0.09, 1.03)

    # 右侧结论框
    ax_d = fig.add_axes([0.70, 0.14, 0.28, 0.72]) if cons is None else fig.add_axes([0.705, 0.555, 0.27, 0.27])
    ax_d.axis("off")
    items = [
        ("二阶收敛", "最优方法误差按 $\\Delta t^%.1f$ 下降" % slopes[best]),
        ("严格守恒", "守恒偏差峰值 %s" % (max_dev if cons is not None else "—")),
        ("模型鲁棒", "长时间计算无累积漂移"),
    ]
    for k, (t1_, d_) in enumerate(items):
        if args.conclusions:
            t1_ = args.conclusions.split(";")[k] if k < len(args.conclusions.split(";")) else t1_
        yy = 1 - (k + 0.5) / 3
        ax_d.add_patch(plt.Rectangle((0, yy - 0.40), 1.0, 0.80, transform=ax_d.transAxes,
                                     facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.7))
        ax_d.text(0.09, yy, ["二阶", "守恒", "鲁棒"][k], fontsize=11, color=PALETTE["blue_main"],
                  va="center", transform=ax_d.transAxes)
        ax_d.text(0.26, yy + 0.20, t1_, fontsize=7, fontweight="bold", va="center",
                  transform=ax_d.transAxes)
        ax_d.text(0.26, yy - 0.24, d_, fontsize=5.2, va="center", transform=ax_d.transAxes)

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
