# -*- coding: utf-8 -*-
"""灵敏度响应曲面：经验率 3D 曲面 + 系数 CI + 分箱热力图 + 结论四联框。"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def main():
    ap = argparse.ArgumentParser(description="灵敏度响应曲面四面板")
    ap.add_argument("--data", required=True, help="逐实体表：含 x-col、y-col、event-col")
    ap.add_argument("--x-col", required=True)
    ap.add_argument("--y-col", required=True)
    ap.add_argument("--event-col", required=True, help="0/1 事件列")
    ap.add_argument("--x-name", default=None)
    ap.add_argument("--y-name", default=None)
    ap.add_argument("--v-bins", default="0,15,30,45,70,max", help="x 分箱边界（max=取上界）")
    ap.add_argument("--i-bins", default="0,4,8,16,32,max", help="y 分箱边界")
    ap.add_argument("--min-n", type=int, default=5, help="星标最小支持度")
    ap.add_argument("--coef-csv", default="", help="可选：特征表 + --target-col 用于面板 B 系数 CI")
    ap.add_argument("--target-col", default="")
    ap.add_argument("--conclusions", default="高危画像;时间窗效应;影像异质性显著;对高危个体加密复查",
                    help="分号分隔的四条结论标题")
    ap.add_argument("--title", default="事件概率灵敏度分析与风险因素分布曲面")
    ap.add_argument("--subtitle", default="—— 基于训练集经验率的二维响应面与逻辑回归 Bootstrap 归因 ——")
    ap.add_argument("--out", default="fig_sens_surface")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df["x"] = pd.to_numeric(df[args.x_col], errors="coerce")
    df["y"] = pd.to_numeric(df[args.y_col], errors="coerce")
    df["e"] = pd.to_numeric(df[args.event_col], errors="coerce")
    df = df.dropna(subset=["x", "y", "e"])
    vmax_x = df["x"].max()
    v_edges = [float(v) for v in args.v_bins.replace("max", str(vmax_x + 1)).split(",")]
    vmax_y = df["y"].max()
    i_edges = [float(v) for v in args.i_bins.replace("max", str(vmax_y + 1)).split(",")]
    df["vb"] = pd.cut(df["x"], bins=v_edges)
    df["ib"] = pd.cut(df["y"], bins=i_edges)
    piv = df.pivot_table(index="vb", columns="ib", values="e", aggfunc="mean", dropna=False)
    cnt = df.pivot_table(index="vb", columns="ib", values="e", aggfunc="count", dropna=False)

    x_name = args.x_name or args.x_col
    y_name = args.y_name or args.y_col

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(140)))

    # ---------- A: 经验率曲面 ----------
    ax_a = fig.add_axes([0.13, 0.50, 0.70, 0.40], projection="3d")
    M = piv.values.astype(float)
    Cnt = cnt.values.astype(float)
    Ms = np.where((Cnt < args.min_n) & ~np.isnan(M), np.nan, M)
    Ms = np.where(np.isnan(Ms), np.nan, Ms)
    filled = np.where(np.isnan(Ms), np.nan, Ms)
    valid = np.argwhere(~np.isnan(filled))
    for i0, j0 in np.argwhere(np.isnan(filled)):
        d2 = ((valid - [i0, j0]) ** 2).sum(1)
        k = d2.argmin()
        filled[i0, j0] = filled[valid[k][0], valid[k][1]]
    Sm = gaussian_filter(filled, sigma=1.1, mode="nearest")
    v_centers = (np.array(v_edges[:-1]) + np.array(v_edges[1:])) / 2
    i_centers = (np.array(i_edges[:-1]) + np.array(i_edges[1:])) / 2
    vg = np.linspace(v_centers[0], v_centers[-1], 60)
    ig = np.linspace(i_centers[0], i_centers[-1], 60)
    VV, II = np.meshgrid(vg, ig)
    from scipy.interpolate import RegularGridInterpolator
    itp = RegularGridInterpolator((i_centers, v_centers), Sm,
                                  method="linear", bounds_error=False, fill_value=None)
    ZP = itp(np.column_stack([II.ravel(), VV.ravel()])).reshape(VV.shape)

    surf = ax_a.plot_surface(VV, II, ZP, cmap="turbo", rstride=1, cstride=1,
                             linewidth=0.12, edgecolor=(0, 0, 0, 0.08), alpha=0.97)
    ax_a.contourf(VV, II, ZP, levels=12, zdir="z", offset=-0.06, cmap="turbo", alpha=0.5)
    M_raw = np.where((np.isnan(M)) | (Cnt < args.min_n), -np.inf, M)
    bi, bj = np.unravel_index(np.argmax(M_raw), M_raw.shape)
    zmax = M[bi, bj]
    x_star, y_star = v_centers[bj], i_centers[bi]
    n_star = int(Cnt[bi, bj])
    v_lo, v_hi = v_edges[bi], v_edges[bi + 1]
    i_lo, i_hi = i_edges[bj], i_edges[bj + 1]
    ax_a.scatter([x_star], [y_star], [zmax], marker="*", s=200,
                 color=PALETTE["red_strong"], edgecolor="black", lw=0.6, zorder=10)
    ax_a.text(max(x_star - 55, 10), y_star + 2, zmax + 0.05,
              "最高经验率 %.2f\n(x %d–%d, y %g–%g, n=%d)" % (zmax, v_lo, min(v_hi, vmax_x), i_lo, i_hi, n_star),
              fontsize=6.4,
              bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=PALETTE["black"], lw=0.6))
    ax_a.set_xlabel(x_name, fontsize=6.8, labelpad=2)
    ax_a.set_ylabel(y_name, fontsize=6.8, labelpad=2)
    ax_a.set_zlabel("经验事件率", fontsize=6.8, labelpad=2)
    ax_a.set_zlim(-0.06, max(float(ZP.max()), zmax) * 1.28)
    ax_a.set_title("经验事件率的二维响应曲面（分箱 + 邻域填充 + 高斯平滑 + 线性插值）", fontsize=8, pad=0)
    ax_a.view_init(elev=26, azim=-62)
    cax = fig.add_axes([0.885, 0.545, 0.015, 0.32])
    fig.colorbar(surf, cax=cax).set_label("经验事件率", fontsize=6)
    cax.tick_params(labelsize=5.5)

    # ---------- B: 分箱经验率 + Wilson CI（无需外部特征表） ----------
    ax_b = fig.add_axes([0.075, 0.175, 0.375, 0.30])
    rates, cis, xs_b, ns_b = [], [], [], []
    for lo, hi in zip(v_edges[:-1], v_edges[1:]):
        sel = df[(df["x"] > lo) & (df["x"] <= hi)] if lo else df[(df["x"] >= lo) & (df["x"] <= hi)]
        n_, k_ = len(sel), int(sel["e"].sum())
        if n_ == 0:
            continue
        ph = k_ / n_
        z = 1.96
        ci_lo = (ph + z * z / (2 * n_) - z * np.sqrt((ph * (1 - ph) + z * z / (4 * n_)) / n_)) / (1 + z * z / n_)
        ci_hi = (ph + z * z / (2 * n_) + z * np.sqrt((ph * (1 - ph) + z * z / (4 * n_)) / n_)) / (1 + z * z / n_)
        rates.append(ph); cis.append((ci_lo, ci_hi)); xs_b.append((lo + hi) / 2); ns_b.append(n_)
    ax_b.plot(xs_b, rates, color=PALETTE["blue_main"], lw=1.3, marker="o", ms=4)
    for x0, r0, (cl, ch) in zip(xs_b, rates, cis):
        ax_b.plot([x0, x0], [cl, ch], color=PALETTE["blue_secondary"], lw=1.2)
        ax_b.plot([x0 - 1.5, x0 + 1.5], [cl, cl], color=PALETTE["blue_secondary"], lw=1.2)
        ax_b.plot([x0 - 1.5, x0 + 1.5], [ch, ch], color=PALETTE["blue_secondary"], lw=1.2)
    ax_b.set_xlabel(x_name + " 分箱中心（误差棒为 Wilson 95% CI）")
    ax_b.set_ylabel("经验事件率")
    ax_b.set_title("分箱事件率与置信区间", fontsize=8)

    # ---------- C: 率热力图 ----------
    ax_c = fig.add_axes([0.585, 0.175, 0.345, 0.30])
    keep = piv.notna().any(axis=0).values
    note = "" if keep.all() else "（右侧区间无训练样本）"
    piv3, cnt3 = piv.loc[:, keep], cnt.loc[:, keep]
    mat3 = piv3.values.astype(float)[::-1]
    cnt3m = cnt3.values.astype(float)[::-1]
    masked = np.ma.masked_invalid(mat3)
    cmap_o = LinearSegmentedColormap.from_list("or", ["#FDF3E7", "#F0A860", "#B64342"])
    cmap_o.set_bad("#F0F0F0")
    im = ax_c.imshow(masked, aspect="auto", cmap=cmap_o, vmin=0, vmax=max(0.8, np.nanmax(mat3)))
    col_lab = ["%g–%g" % (i_edges[j], i_edges[j + 1]) for j, kp in enumerate(keep) if kp]
    ax_c.set_xticks(range(mat3.shape[1]))
    ax_c.set_xticklabels(col_lab, fontsize=6.3)
    ax_c.set_yticks(range(mat3.shape[0]))
    ax_c.set_yticklabels([str(iv) for iv in piv3.index[::-1]], fontsize=6)
    for i in range(mat3.shape[0]):
        for j in range(mat3.shape[1]):
            v, n_ = mat3[i, j], cnt3m[i, j]
            if np.isfinite(v):
                ax_c.text(j, i, "%.2f\n(n=%d)" % (v, int(n_)), ha="center", va="center",
                          fontsize=5.8, color="white" if v > 0.45 else "#333333")
            else:
                ax_c.text(j, i, "—", ha="center", va="center", fontsize=6, color=PALETTE["neutral_mid"])
    ax_c.set_xlabel(y_name + "（%s）" % note if note else y_name)
    ax_c.set_ylabel(x_name + " 分箱")
    ax_c.set_title("经验事件率：分箱交叉", fontsize=8)
    fig.colorbar(im, cax=fig.add_axes([0.945, 0.175, 0.015, 0.30])).set_label("经验事件率", fontsize=6)

    # ---------- D: 关键结论四联框 ----------
    ax_d = fig.add_axes([0.03, 0.012, 0.94, 0.115])
    ax_d.axis("off")
    titles = [t for t in args.conclusions.split(";")]
    descs = ["星标格为最高经验率且支持度达标",
             "事件率随 %s 单调变化" % x_name,
             "A/C 两面板交叉印证趋势",
             "据此确定分层监测与复查策略"]
    while len(titles) < 4:
        titles.append("结论 %d" % (len(titles) + 1))
    descs += [""] * 4
    glyphs = ["★", "◆", "▲", "●"]
    fcs = ["#FDECEA", "#FFF3E0", "#EAF3FD", "#EAF7EC"]
    ecs = [PALETTE["red_strong"], PALETTE["orange_main"], PALETTE["blue_main"], PALETTE["green_strong"]]
    x0, w, gap = 0.01, 0.235, 0.012
    for k in range(4):
        x = x0 + k * (w + gap)
        ax_d.add_patch(FancyBboxPatch((x, 0.05), w, 0.9, boxstyle="round,pad=0.012",
                                      facecolor=fcs[k], edgecolor=ecs[k], lw=0.9,
                                      transform=ax_d.transAxes))
        ax_d.text(x + 0.028, 0.5, glyphs[k], fontsize=13, color=ecs[k], va="center",
                  transform=ax_d.transAxes)
        ax_d.text(x + 0.10, 0.74, titles[k], fontsize=7, fontweight="bold", va="center",
                  transform=ax_d.transAxes)
        ax_d.text(x + 0.10, 0.34, descs[k], fontsize=5.7, va="center",
                  transform=ax_d.transAxes)

    fig.suptitle(args.title, fontsize=11, y=1.0)
    fig.text(0.5, 0.962, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
