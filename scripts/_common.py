# -*- coding: utf-8 -*-
"""hero-figures-cn 各模板共享的小工具。"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")


def blend_cmap(color, name="blend"):
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    r, g, b = to_rgb(color)
    return LinearSegmentedColormap.from_list(name, [(1, 1, 1), (r, g, b)])


def norm(v):
    import numpy as np
    v = np.asarray(v, float)
    span = v.max() - v.min()
    return (v - v.min()) / span if span > 0 else v * 0


def icon_boxes(fig, boxes, rect=(0.66, 0.16, 0.32, 0.30)):
    """右侧/局部图标结论框。boxes: [(glyph, title, desc, fc, ec), ...]，纵向排列。"""
    import matplotlib.pyplot as plt  # noqa
    x, y0, w, h = rect
    n = len(boxes)
    ax = fig.add_axes([x, y0, w, h])
    ax.axis("off")
    for k, (glyph, title, desc, fc, ec) in enumerate(boxes):
        yy = 1 - (k + 0.5) / n
        ax.add_patch(FancyBboxPatchAx(ax, (0.0, yy - 0.42), w, 0.84, fc, ec))
        ax.text(0.06, yy, glyph, fontsize=12, color=ec, va="center", transform=ax.transAxes)
        ax.text(0.16, yy + 0.18, title, fontsize=7, fontweight="bold", va="center",
                transform=ax.transAxes)
        ax.text(0.16, yy - 0.22, desc, fontsize=5.7, va="center", transform=ax.transAxes)
    return ax


def FancyBboxPatchAx(ax, xy, w, h, fc, ec):
    from matplotlib.patches import FancyBboxPatch
    return FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                          facecolor=fc, edgecolor=ec, lw=0.9, transform=ax.transAxes)


def tag_panel(ax, tag, dx=-0.035, dy=1.05):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontsize=9, fontweight="bold")
