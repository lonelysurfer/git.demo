# -*- coding: utf-8 -*-
"""收敛性+守恒校验演示数据。"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(21)
dts = np.logspace(-4, -1, 8)
rows = []
for m, base, order in [("我们的方法（物理约束）", 0.02, 2.0), ("传统显式方法", 0.05, 1.0),
                       ("传统隐式方法", 0.08, 1.0), ("无物理约束", 0.12, 0.85)]:
    for dt in dts:
        err = base * dt ** order * np.exp(rng.normal(0, 0.05))
        rows.append((m, dt, err))
pd.DataFrame(rows, columns=["method", "dt", "error"]).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_conv.csv", index=False)

t = np.linspace(0, 2.0, 200)
E = 3e-3 * np.exp(-6 * t) + 1.2e-4 + rng.normal(0, 2e-5, len(t)).cumsum() * 0.02
M = 8e-3 * np.exp(-3 * t) + 2.2e-4 + rng.normal(0, 3e-5, len(t))
pd.DataFrame({"t": t, "E": E, "M": M}).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_cons.csv", index=False)
print("conv/cons demo written")
