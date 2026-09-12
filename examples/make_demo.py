# -*- coding: utf-8 -*-
"""生成演示数据 demo_long.csv 与 demo_events.csv。"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(11)


def gamma_curve(t, a=25.0, b=0.9, c=0.02):
    return a * np.power(np.maximum(t, 1e-9), b) * np.exp(-c * t)


rows = []
for i in range(1, 61):
    tmax = rng.uniform(120, 360)
    ts = np.sort(rng.uniform(1, tmax, rng.integers(4, 9)))
    grp = int(rng.random() < 0.45)
    for t in ts:
        v = gamma_curve(t) * (1.25 if grp else 0.85) + rng.normal(0, 6)
        hm = max(60 * np.exp(-t / 200) + rng.normal(0, 4), 0)
        res = np.clip(1 - np.exp(-t / 300) + rng.normal(0, 0.08), 0, 1)
        rows.append((f"id{i:03d}", t, v, hm, res, grp))
pd.DataFrame(rows, columns=["id", "t", "value", "hm", "res", "grp"]).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_long.csv", index=False, encoding="utf-8")

rows = []
for i in range(1, 121):
    vol = rng.gamma(2.2, 18) + rng.random() * 40
    interval = rng.uniform(0.5, 48)
    p = 1 / (1 + np.exp(-(0.018 * vol - 0.09 * interval - 0.3)))
    rows.append((vol, interval, int(rng.random() < p)))
pd.DataFrame(rows, columns=["volume", "interval", "event"]).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_events.csv", index=False, encoding="utf-8")
print("demo data written")
