# -*- coding: utf-8 -*-
"""生成其余模板的演示数据：场切片 / 雷达对比。"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(31)
times = [0, 30, 60, 90, 120]
rows = []
for tv in times:
    prog = tv / 120.0
    for xi in np.linspace(0, 1, 25):
        for yi in np.linspace(0, 1, 25):
            r = np.sqrt((xi - 0.5) ** 2 + (yi - 0.5) ** 2) * 2
            front = 1.0 - prog
            v = 100 * np.exp(-((r - (1 - front)) ** 2) / 0.08) + rng.normal(0, 4)
            t2 = 30 + 40 * np.exp(-((xi - 0.5) ** 2) / 0.2) + prog * 25 + rng.normal(0, 2)
            rows.append((tv, xi, yi, np.clip(v, 0, 105), t2))
pd.DataFrame(rows, columns=["time", "x", "y", "v1", "v2"]).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_field.csv", index=False)

models = ["变物性模型", "传统常物性模型"]
metrics = ["最大相对误差 (%)", "RMSE (单位)", "峰值延迟 (s)", "边缘区域误差 (%)", "中心区域误差 (%)"]
mrows = []
base = {"变物性模型": [12.3, 8.7, 6.5, 10.1, 9.4], "传统常物性模型": [18.2, 12.1, 11.1, 15.3, 12.6]}
drows = []
for mi, mv in enumerate(metrics):
    for mo in models:
        mrows.append((mo, mv, base[mo][mi]))
pd.DataFrame(mrows, columns=["model", "metric", "value"]).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_metrics.csv", index=False)

points = ["中心", "内中", "中间", "外中", "边缘"]
drows = []
for mv in metrics:
    for pi, pt in enumerate(points):
        for mo in models:
            v = base[mo][mi] * (0.3 + 0.18 * pi) + rng.normal(0, 1.2)
            drows.append((mo, mv, pt, max(v, 0.5)))
pd.DataFrame(drows, columns=["model", "metric", "point", "value"]).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_detail.csv", index=False)
print("field/radar demo written")
