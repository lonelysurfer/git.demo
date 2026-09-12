# -*- coding: utf-8 -*-
"""生成其余模板演示数据：评价矩阵 / 调度宽表 / 蒙特卡洛样本 / 灰色序列 / 聚类特征 / 回归数据。"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(77)

# 评价矩阵：6 方案 × 5 指标（含 1 个成本型）
plans = ["方案一", "方案二", "方案三", "方案四", "方案五", "方案六"]
crit = ["经济性(万元)", "可靠性(%)", "碳排放(t)", "建设周期(月)", "社会效益"]
X = np.column_stack([
    rng.uniform(200, 500, 6),      # 经济性（成本，负）
    rng.uniform(85, 99, 6),        # 可靠性
    rng.uniform(10, 60, 6),        # 碳排放（负）
    rng.uniform(6, 24, 6),         # 周期（负）
    rng.uniform(50, 95, 6),        # 社会效益
])
pd.DataFrame(X, index=plans, columns=crit).rename_axis("方案").to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_topsis.csv", encoding="utf-8")

# 调度宽表
t = np.arange(1, 25)
demand = 40 + 25 * np.exp(-((t - 8) / 3) ** 2) + 30 * np.exp(-((t - 19) / 3) ** 2)
price = 0.3 + 0.5 * np.exp(-((t - 19) / 2.5) ** 2) + 0.1 * (t < 7)
gas = np.clip(demand * 0.7 + rng.normal(0, 1.5, 24), 0, None)
wind = np.clip(np.abs(rng.normal(12 * np.exp(-((t - 14) / 5) ** 2) + 3, 2)), 0, None)
solar = np.clip(28 * np.exp(-((t - 13) / 3.5) ** 2) + rng.normal(0, 1.2, 24), 0, None)
storage = np.where((t >= 22) | (t <= 6), -6.0, 0.0) + np.where((t >= 8) & (t <= 20), 5.0, 0.0) + rng.normal(0, 1, 24)
pd.DataFrame({"t": t, "燃气机组": gas, "风电": wind, "光伏": solar, "储能": storage,
              "需求": demand, "价格": price}).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_dispatch.csv", index=False)

# 蒙特卡洛样本
mc = rng.normal(100, 15, 4000) + rng.gamma(2, 8, 4000) - 16
pd.DataFrame({"npv": mc}).to_csv(r"D:\华为杯\hero-figures-cn\examples\demo_mc.csv", index=False)

# 灰色序列
tg = np.arange(1, 13)
y_g = 80 * np.exp(0.12 * tg) + rng.normal(0, 4, 12)
pd.DataFrame({"t": tg, "y": y_g}).to_csv(r"D:\华为杯\hero-figures-cn\examples\demo_grey.csv", index=False)

# 聚类特征
n = 180
c1 = rng.normal([20, 80, 30, 55, 10], [3, 4, 3, 5, 2], (60, 5))
c2 = rng.normal([45, 40, 60, 30, 40], [4, 5, 4, 4, 3], (60, 5))
c3 = rng.normal([70, 15, 20, 75, 70], [3, 3, 5, 3, 4], (60, 5))
feats = ["厚度", "强度", "浓度", "温度", "纯度"]
pd.DataFrame(np.vstack([c1, c2, c3]), columns=feats).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_cluster.csv", index=False)

# 回归数据
x1 = rng.uniform(10, 100, 120)
x2 = rng.uniform(0, 50, 120)
y_reg = 30 + 1.8 * x1 - 0.9 * x2 + rng.normal(0, 8, 120)
pd.DataFrame({"x1": x1, "x2": x2, "y": y_reg}).to_csv(
    r"D:\华为杯\hero-figures-cn\examples\demo_reg.csv", index=False)
print("all extra demos written")
