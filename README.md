# hero-figures-cn

数学建模论文顶刊风 hero 大图模板库（10 类场景 × 16 个参数化模板）。

详细的数据契约、参数说明与选图决策表见 [SKILL.md](SKILL.md)。

## 快速使用

```bash
# 密度水母图（重复观测分布演变）
python scripts/jellyfish_fig.py --data data.csv --time-col t --value-col v \
  --entity-col id --group-col grp --windows "0,24,72,168,336"

# 时空对齐热力图（多源指标一致性）
python scripts/alignment_fig.py --data data.csv --time-col t \
  --indicator-cols "hm,ed,res" --indicator-names "血肿,水肿,消退率" --bins 12

# 动态演化与相空间轨迹（单峰过程）
python scripts/phase_evolution_fig.py --data data.csv --time-col t --value-col v \
  --stages "0,48,168,336" --stage-names "蓄积区,演进区,消退区"

# 灵敏度响应曲面（双因素事件率）
python scripts/sens_surface_fig.py --data events.csv --x-col volume \
  --y-col interval --event-col event --min-n 5
```

其余 6 类：`turning_point_fig.py`（信号重构拐点）、`phase_spiral_fig.py`（耦合相空间螺旋）、`convergence_fig.py`（数值收敛守恒）、`param_surface_fig.py`（参数敏感性拓扑）、`field_slices_fig.py`（场演化切片）、`radar_compare_fig.py`（雷达对比）、`evaluate_fig.py`（熵权-TOPSIS 评价）、`dispatch_stack_fig.py`（调度堆叠）、`monte_carlo_fig.py`（蒙特卡洛）、`grey_pred_fig.py`（灰色预测）、`cluster_pca_fig.py`（聚类+PCA）、`regression_diag_fig.py`（回归诊断）。

## 输出

每图自动输出 `.svg / .pdf / .png` 三格式（183mm 双栏宽、中文标签、可编辑文本），并执行中文字体渲染与完整性 QA。

## 依赖

Python ≥3.9：numpy、pandas、scipy、matplotlib。
