---
name: hero-figures-cn
description: 数学建模论文顶刊风 hero 大图模板库（10 种）：密度水母图、时空对齐热力图、动态演化+相空间、灵敏度曲面、信号重构拐点图、相空间螺旋、收敛守恒校验、参数敏感性拓扑、场演化切片云图、雷达对比。当用户需要为数学建模/物理/工程类论文绘制多面板大幅插图，或询问"什么数据该配什么大图"时使用；输入为长表 CSV 与列映射参数，输出 svg/pdf/png 三格式并自动做中文字体与可编辑文本 QA。
---

# hero-figures-cn

## 定位

把十种"顶刊竞赛风"多面板 hero 图做成**数据驱动模板**：换论文时只需整理长表 CSV、指定列名与少量参数即可复现同版式。默认 183 mm 双栏宽、白底、中文标签、可编辑文本，输出 svg + pdf + png 三格式并自动 QA。

## 选图决策引擎（什么情况用什么图）

| 数据特征 / 论文场景 | 用哪个模板 | 理由 |
|---|---|---|
| 噪声观测、插值/拟合方法论证、拐点识别 | `turning_point_fig.py` | 强度背景+样条重构+残差正态校验，直观证明方法可靠 |
| 重复观测 "实体×时间×数值"，要展示分布演变、组间差异、来源分解 | `jellyfish_fig.py` | 水母图呈现分布形态，方差分解回答"差异从哪来" |
| 多源指标按同一时间轴清洗对齐、交叉验证 | `alignment_fig.py` | 三行对齐热力图+均值时序+相关散点+关键窗放大 |
| 单峰过程（蓄积-达峰-消退、充放电、负荷峰谷） | `phase_evolution_fig.py` | 演化曲线+阶段分区+相空间环+速率热图 |
| 两个耦合状态变量、滞后/闭环动力学（如温度-湿度） | `phase_spiral_fig.py` | 相空间螺旋轨迹按时间着色，滞后与耦合强度定量 |
| 数值模拟的收敛性验证、守恒律检查 | `convergence_fig.py` | log-log 收敛线+收敛阶标注+守恒偏差时序 |
| 参数随双状态变量非线性变化、2 因子敏感性 | `param_surface_fig.py` | 双 3D 拓扑+梯度剧烈区域+剖面族 |
| 时空场演化（PDE 解、浓度/温度场、锋面推进） | `field_slices_fig.py` | 多时刻场快照行+剖面族+推进示意 |
| 多模型/多方案在多指标上的对比评价 | `radar_compare_fig.py` | 雷达+小倍数+汇总柱+关键结论框 |
| 事件响应找曲面（双参数响应） | `sens_surface_fig.py` | 经验率曲面+Bootstrap 系数 CI+分箱交叉热图 |
| 综合评价/方案选优（多指标多方案） | `evaluate_fig.py` | 熵权条形+TOPSIS 得分排序+正向化热图 |
| 优化调度（多主体出力覆盖需求） | `dispatch_stack_fig.py` | 出力堆叠+需求线+价格次轴+供需裕度 |
| 蒙特卡洛模拟（大量随机样本） | `monte_carlo_fig.py` | 累积均值收敛+分布直方+KDE+正态 QQ |
| 灰色预测/时序外推（小样本等距序列） | `grey_pred_fig.py` | GM(1,1) 拟合+预测带+后验差 C/P 检验 |
| 聚类分析/降维（无标签多特征） | `cluster_pca_fig.py` | 肘部+轮廓+PCA 散点+簇中心热图 |
| 回归建模诊断 | `regression_diag_fig.py` | 预测vs真实+残差异方差+QQ+指标框 |

选择口诀：**看噪声用重构，看分布用水母，看对齐用热力，单峰过程相空间，两态耦合螺旋走，数值模拟查收敛，参数扫描画曲面，场演化用切片，模型对比雷达见，事件响应找曲面，评价用熵权排序，调度看堆叠，随机模拟看收敛分布，小样本外推用灰色，聚类降维画散点，回归诊断四联图。**

## 数据契约（长表 CSV，UTF-8，首行列名）

各脚本 `--help` 有完整参数；共同约定：

- 所有脚本：`--title --subtitle --out` 自定义标题与输出前缀；输出自动 `.svg/.pdf/.png`
- `jellyfish_fig.py`：必需 `--time-col --value-col`；可选 `--entity-col --group-col --windows --value-scale --decomp-cols`（C 面板顺序方差分解）、`--surface-csv/--surface-x/--surface-y/--surface-z`（D 面板响应面，缺省为二维 KDE）
- `alignment_fig.py`：必需 `--time-col --indicator-cols(2-3个) --indicator-names`；可选 `--entity-col --bins --max-time --group-col --group-a-value/--group-b-value`（D 面板分组对比，值按字符串比较）`--zoom-lo/--zoom-hi --finding`
- `phase_evolution_fig.py`：必需 `--time-col --value-col`；自动拟合 Gamma 曲线（可 `--curve-params a,b,c` 覆盖）；可选 `--stages --stage-names --stage-colors --peak-label --base-label --value-name --time-name --flow-boxes(JSON) --finding --conclusions`
- `sens_surface_fig.py`：必需 `--x-col --y-col --event-col(0/1)`；可选 `--v-bins --i-bins`（"max" 表示取上界）`--min-n`（星标最小支持度，默认 5）`--conclusions "四条标题;..."`；空数据区间自动剔除并在轴标签注明
- `turning_point_fig.py`：必需 `--t-col --obs-col`；可选 `--true-col --smooth`（样条平滑系数）
- `phase_spiral_fig.py`：必需 `--t-col --x-col --y-col --series-col`；可选 `--series-names --annots "每序列标注;..." --couple-name --lag-name`
- `convergence_fig.py`：必需 `--conv-data(--method-col --dt-col --err-col)`；可选 `--cons-data(--cons-t-col --cons-e-col --cons-m-col) --method-names --colors --conclusions`
- `param_surface_fig.py`：必需规则网格 `--grid-data(--x-col --y-col --z1-col)`；可选 `--z2-col --x-name --y-name --z1-name --z2-name`
- `field_slices_fig.py`：必需 `--field-data(--time-col --x-col --y-col --v1-col)`；可选 `--v2-col --times --n-snap --x-name --y-name --v1-name --v2-name`
- `radar_compare_fig.py`：必需 `--metrics-csv(--model-col --metric-col --value-col)`；可选 `--detail-csv`（小倍数数据）`--model-names --lower-better --conclusions`

## 运行示例

```bash
python scripts/jellyfish_fig.py --data examples/demo_long.csv \
  --time-col t --value-col value --entity-col id --group-col grp \
  --windows "0,72,144,216,288,360" --title "数值概率密度演变与水母图"
python scripts/turning_point_fig.py --data examples/demo_signal.csv \
  --t-col t --obs-col obs --true-col true --conclusions "捕捉动态;抑制噪声;提供建模基础"
python scripts/radar_compare_fig.py --metrics-csv examples/demo_metrics.csv \
  --detail-csv examples/demo_detail.csv --conclusions "新模型误差全面更低;RMSE 降低 28%"
```

## 质量纪律

- 输出前自动执行中文字体渲染、可编辑文本、三格式完整性 QA，不通过不交付（渐变密集的图 svg 颜色数超限为已知非阻塞提示，以 pdf/png 为准）。
- 极端值导致纵轴失真时按 P97.5 截断并图内注明。
- 热力图/曲面星标只在支持度 n≥5 的格中选取，小样本格由邻域填充承接；标注文本动态生成（含 n），禁止编造。
- 颜色用内置低饱和家族色板；同一序列跨面板不变色。
- 全部标签中文；统计量（r、p、n、斜率）必须来自真实计算。
- 与 math-paper-cn 联用：产物复制到 `Qn/figures/` 并登记 `figures/manifest.json`（generator=python、三格式 exports、qa 字段）后入文。

## 依赖

Python ≥3.9：numpy、pandas、scipy、matplotlib。样式引擎 `scripts/py_nature_core.py` 随技能分发。
