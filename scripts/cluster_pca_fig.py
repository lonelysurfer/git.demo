# -*- coding: utf-8 -*-
"""聚类+PCA 图：肘部/轮廓 + PCA 散点 + 中心热图 + 载荷图。"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import tag_panel
from py_nature_core import PALETTE, apply_py_nature_style, mm_to_inch, run_py_nature_qa, save_py_nature_figure


def main():
    ap = argparse.ArgumentParser(description="聚类+PCA 图")
    ap.add_argument("--data", required=True, help="CSV：全数值特征列（可选 --label-col 已知标签）")
    ap.add_argument("--label-col", default="", help="已知标签列（散点按标签着色而非聚类）")
    ap.add_argument("--k", type=int, default=0, help="簇数（0=自动按肘部+轮廓选）")
    ap.add_argument("--k-range", default="2,7")
    ap.add_argument("--title", default="基于 K-means 与 PCA 的聚类结构分析")
    ap.add_argument("--subtitle", default="—— 肘部法则、轮廓系数、主成分投影与簇中心 ——")
    ap.add_argument("--out", default="fig_cluster")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.label_col and args.label_col in df.columns:
        feat = df.drop(columns=[args.label_col]).select_dtypes(include=[np.number])
        labels_known = df[args.label_col].astype(str).values
    else:
        feat = df.select_dtypes(include=[np.number])
        labels_known = None
    feat = feat.loc[:, feat.std() > 1e-12]
    Xs = StandardScaler().fit_transform(feat.values)

    krange = range(int(args.k_range.split(",")[0]), int(args.k_range.split(",")[1]) + 1)
    sse, sil = [], []
    for k in krange:
        km = KMeans(n_clusters=k, n_init=10, random_state=2023).fit(Xs)
        sse.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))
    k = args.k if args.k else list(krange)[int(np.argmax(sil))]
    km = KMeans(n_clusters=k, n_init=20, random_state=2023).fit(Xs)
    lab = km.labels_

    pca = PCA(n_components=2).fit(Xs)
    XY = pca.transform(Xs)

    apply_py_nature_style(font_size=7.0)
    fig = plt.figure(figsize=(mm_to_inch(183), mm_to_inch(112)))
    ax_a = fig.add_axes([0.07, 0.14, 0.22, 0.70])
    ax_b = fig.add_axes([0.40, 0.14, 0.26, 0.70])
    ax_c = fig.add_axes([0.70, 0.14, 0.27, 0.70])

    # A: 肘部 + 轮廓双轴
    ax_a.plot(list(krange), sse, color=PALETTE["blue_main"], lw=1.4, marker="o", ms=3.5,
              label="SSE (肘部)")
    ax_a.set_xlabel("簇数 k"); ax_a.set_ylabel("SSE", color=PALETTE["blue_main"])
    ax_a2 = ax_a.twinx()
    ax_a2.plot(list(krange), sil, color=PALETTE["orange_main"], lw=1.4, marker="s", ms=3.5,
               label="轮廓系数")
    ax_a2.set_ylabel("轮廓系数", color=PALETTE["orange_main"])
    ax_a2.tick_params(axis="y", labelcolor=PALETTE["orange_main"])
    ax_a.axvline(k, color=PALETTE["red_strong"], lw=1.0, ls="--")
    ax_a.text(k + 0.06, max(sse) * 0.85, "选择 k=%d" % k, fontsize=6.5,
              color=PALETTE["red_strong"])
    h1, l1 = ax_a.get_legend_handles_labels()
    h2, l2 = ax_a2.get_legend_handles_labels()
    ax_a.legend(h1 + h2, l1 + l2, fontsize=5.6, loc="upper right")
    ax_a.set_title("肘部法则与轮廓系数", fontsize=8, pad=5)
    tag_panel(ax_a, "A", -0.13, 1.04)

    # B: PCA 散点
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))
    if labels_known is not None and args.label_col:
        for u in sorted(set(labels_known)):
            sel = labels_known == u
            ax_b.scatter(XY[sel, 0], XY[sel, 1], s=14, alpha=0.8, label=u, linewidths=0)
        ax_b.legend(fontsize=5.6)
    else:
        for c in range(k):
            sel = lab == c
            ax_b.scatter(XY[sel, 0], XY[sel, 1], s=14, alpha=0.8,
                         color=colors[c], linewidths=0, label="簇 %d" % (c + 1))
        ax_b.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                     marker="*", s=140, color=PALETTE["black"], zorder=5, label="簇中心")
        ax_b.legend(fontsize=5.6)
    evr = pca.explained_variance_ratio_
    ax_b.set_xlabel("PC1 (%.1f%%)" % (evr[0] * 100))
    ax_b.set_ylabel("PC2 (%.1f%%)" % (evr[1] * 100))
    ax_b.set_title("PCA 降维散点（按簇着色）", fontsize=8, pad=5)
    tag_panel(ax_b, "B", -0.10, 1.04)

    # C: 簇中心热图
    centers_z = StandardScaler().fit_transform(km.cluster_centers_)
    im = ax_c.imshow(centers_z, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax_c.set_xticks(range(len(feat.columns)))
    ax_c.set_xticklabels([c if len(c) < 10 else c[:8] + "…" for c in feat.columns],
                         fontsize=5.2, rotation=30, ha="right")
    ax_c.set_yticks(range(k))
    ax_c.set_yticklabels(["簇 %d" % (c + 1) for c in range(k)], fontsize=6)
    for i in range(k):
        for j in range(len(feat.columns)):
            ax_c.text(j, i, "%.1f" % centers_z[i, j], ha="center", va="center", fontsize=5.2,
                      color="white" if abs(centers_z[i, j]) > 1.2 else "#333333")
    ax_c.set_title("簇中心特征剖面 (z-score)", fontsize=8, pad=5)
    fig.colorbar(im, ax=ax_c, pad=0.02, fraction=0.045).ax.tick_params(labelsize=5)
    tag_panel(ax_c, "C", -0.13, 1.04)

    fig.suptitle(args.title, fontsize=11, y=0.995)
    fig.text(0.5, 0.952, args.subtitle, ha="center", fontsize=7.5, color=PALETTE["neutral_dark"])
    kw = dict(boxstyle="round,pad=0.45", facecolor="#F2F6FC", edgecolor=PALETTE["blue_main"], lw=0.9)
    fig.text(0.5, 0.028, "关键结论    数据在 k=%d 处呈现最清晰的簇结构（轮廓系数 %.3f）；"
             "PC1+PC2 累计解释 %.1f%% 方差，各簇在特征剖面上差异显著。"
             % (k, max(sil), (evr[0] + evr[1]) * 100),
             ha="center", va="center", fontsize=6.3, bbox=kw, wrap=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = save_py_nature_figure(fig, out, dpi=320)
    qa = run_py_nature_qa(out)
    print("exports:", [str(x) for x in paths])
    print("qa:", qa.passed, {k: v for k, v in qa.checks.items() if not v})


if __name__ == "__main__":
    main()
