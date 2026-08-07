# 改进版 XRD-PPO：完整训练与独立验证报告

日期：2026-08-07

## 1. 最终结论

四阶段修改已经完成，改进版训练也已完整、稳定地跑完；但核心科学结论是：**PPO 实现比旧版本更正确、更可诊断，训练过程没有跑崩，`peak_penalized` 也修正了部分 XRD 尺度捷径，但这些改动仍没有可靠提高正确拓扑的生成率。**

最关键的独立测试结果是：

| 比较（每项 1200 个未见样本） | StructureMatcher | Top-10 命中 | primitive strata | 目标空间群候选 |
|---|---:|---:|---:|---:|
| `peak` 基座 | 41 | 6（按 peak）/ 19（按 penalized） | 454 | 89 |
| `peak` epoch 25 | 41 | 6 / 19 | 457 | 91 |
| `peak_penalized` 基座 | 41 | 6 / 19 | 454 | 89 |
| `peak_penalized` epoch 25 | 42 | 6 / 20 | 459 | 92 |

`peak` 完全没有增加结构命中；`peak_penalized` 只多 1 个，而且精确符号翻转检验 `p=1.000`、配对 t 检验 `p=0.339`、Wilcoxon `p=1.000`，不能认为它带来了可重复改进。这个额外命中只来自 NaNiH3；NbS2 和 Li2TeC2 仍然是 0。

因此，下一轮的第一优先级不应是继续微调 XRD reward 或盲目延长 epoch，而应是**扩大并均衡空间群、Wyckoff 序列和离散结构骨架的候选覆盖**。

## 2. 四阶段修改及其实际作用

### 阶段一：修正 PPO 学习信号

旧流程先用累计历史最好分数平移当前 reward，再用 EMA baseline 构造 advantage。这使 reward 的参照随时间移动；历史偶然高分会令后续 reward 整体变负，而且 batch 8 时方差很大。旧 replay buffer 又只保留一个候选，并按平移后的 reward 排序，两个先后刷新历史最佳的样本都可能得到 0，导致更高 raw score 无法替换旧样本。

现在的 XRD 流程直接对当前 batch 的原始分数做标准化：

```text
A_i = (S_i - mean(S)) / (std(S) + eps)
```

每个 batch 的 `advantage_mean` 约为 0、`advantage_std` 为 1。参考策略项从 advantage 中拆出，直接作为独立正则项作用于目标；XRD replay 默认 `gamma=0`，若以后显式启用，也按 raw score 而非历史最佳平移后的 reward 排序。

这项修改解决的是“学习信号非平稳、replay 排序错误、KL 含义混杂”，并不保证 batch 中一定包含正确结构。若一个 batch 全是错误拓扑，标准化仍只会学习“错误候选中的相对较好者”。这也是本轮训练稳定但恢复率不升的一个重要边界。

### 阶段二：让 PPO 真正发生多次可观测更新

旧设置每轮 `ppo_epochs=1`，第一次重算时 `pi_theta=pi_old`，ratio 从 1 开始，clip 很难在同一次更新中发挥约束作用。新设置每个 outer epoch 采样 32 个候选，切成 8 个样本的 microbatch，并针对同一批旧策略样本进行 4 次 PPO 更新：

```text
25 outer epochs x 32 candidates = 800 candidates / run
25 outer epochs x 4 PPO epochs = 100 optimizer updates / run
```

同时记录 `clip_fraction`、`ratio_mean/max`、`approx_kl_old`、`grad_norm`、分数分位数和 replay 来源。训练中 600 个 epoch 行全部为有限值，ratio 均值总体为 `0.9990`，最大 ratio 的全局最大值为 `1.2531`，近似 KL 均值为 `0.000234`。只有 3/600 轮出现非零 clip fraction，说明 clip 路径确实可触发，但学习率和 heads-only 范围下更新非常保守。

### 阶段三：修正奖励尺度捷径并限制更新范围

新增 `peak_penalized`：峰对齐仍允许有限的 q 尺度变换和零点移动，但在选择最佳对齐时同时施加平滑尺度先验：

```text
penalty(a) = exp[-0.5 * (log(a) / log(1.15))^2]
S_penalized = S_peak * penalty(a)
```

这样，候选不能先靠很大的晶格尺度变换得到高峰匹配，再在评分结束后才被惩罚。离线注入 CIF 的同池验证显示，它能把 NaNiH3 的错误大尺度捷径从前几名推后，同时保留正确结构的 Top-10；但注入 CIF 只证明 ranker 更合理，不证明生成器能够产生正确拓扑。

训练默认改为 heads-only，仅更新 `linear`、`linear_1`、`linear_2`、`linear_3` 和最终 `linear_41` 输出头，冻结 Transformer 主体。实测为 307,373 / 13,836,461 个参数，约 2.22%。这降低了小样本 RL 破坏预训练表征的风险，也使训练更稳定；代价是如果缺失拓扑需要改变深层离散表示，heads-only 的表达能力可能不足。

### 阶段四：把排序、覆盖和结构恢复分开验证

新增候选覆盖审计：先转 primitive cell，再按 `(primitive space group, primitive site count)` 分层；使用 StructureMatcher 去近重复，并可对不同 strata 做 round-robin 选择。评估同时报告：

- 所有 StructureMatcher oracle 命中，而不是只看最高 XRD 分；
- 正确结构是否进入 Top-10；
- 目标空间群和晶格接近子集是否存在；
- primitive strata 数与重复候选数；
- 注入目标晶格后的命中，单独标记为诊断，不当作生成成功。

训练矩阵与独立评估都使用 manifest、固定 seed 和断点完整性校验。pickle 必须能反序列化，采样 CSV 必须恰好 100 行且索引连续，`common_metrics.json` 与 `coverage.json` 都有效才允许复用。这样解决的是证据链和恢复可靠性，不会直接改变模型能力。

## 3. 完整训练协议与稳定性

| 项目 | 设置/结果 |
|---|---|
| 材料 | Zr、NbS2、Li2TeC2、NaNiH3 |
| reward | `peak`、`peak_penalized` |
| 重复 | 每材料、每 reward 3 个训练 seed，共 24 组 |
| 每组训练 | 25 epoch、batch 32、PPO epochs 4、microbatch 8 |
| 每组预算 | 800 次采样尝试、100 次 optimizer update、5 个 checkpoint |
| 总量 | 600 epoch、19,200 次采样尝试、18,806 个 CIF、120 个 checkpoint |
| 参数范围 | heads-only，约 2.22% 参数 |
| 总耗时 | 19,341 秒，约 5.37 小时 |
| manifest | 24/24 `complete`，24/24 validation `complete` |

数值与资源审计：

- 600/600 行 `xrd_mean`、PPO objective、ratio、KL、grad norm 与分位数全部有限；
- 600/600 行 `advantage_std=1.0`，说明新标准化路径实际生效；
- 24/24 内存守护正常退出，`memory_stopped=0`，return code 均为 0；
- 守护期间最小 Linux available memory 为 2.58 GiB，最小空闲 Swap 为 8.0 GiB；
- 最终检查时约 8.9 GiB 内存可用，Swap 使用为 0，未发现残留训练进程。

这里的“训练成功”只表示训练完整、数值有限、资源安全，不等于方法在结构恢复上有效。

### 3.1 逐 epoch 训练数据与 PPO objective 曲线

24 组 `data.txt` 的 600 行训练数据已统一导出为：

- `reports/improved_ppo_20260807/IMPROVED_PPO_EPOCH_METRICS.csv`
- `reports/improved_ppo_20260807/IMPROVED_PPO_FIRST_LAST_SUMMARY.csv`

![改进版 PPO objective 完整曲线](improved_ppo_20260807/IMPROVED_PPO_OBJECTIVE_ALL_RUNS.png)

图中每个面板有 3 条 seed 曲线，黑线为三 seed 均值。对比前 5 和后 5 个 outer epoch：

| 材料 | reward | PPO objective | 训练 batch XRD 分数 |
|---|---|---:|---:|
| Zr | peak | `0.00676 -> 0.00401` | `0.41278 -> 0.41855` |
| Zr | penalized | `0.00599 -> 0.00327` | `0.33721 -> 0.34399` |
| NbS2 | peak | `0.01195 -> -0.00868` | `0.35707 -> 0.35704` |
| NbS2 | penalized | `0.01354 -> -0.00379` | `0.28338 -> 0.27969` |
| Li2TeC2 | peak | `0.00407 -> 0.00223` | `0.30046 -> 0.30509` |
| Li2TeC2 | penalized | `0.00439 -> 0.00134` | `0.21065 -> 0.21197` |
| NaNiH3 | peak | `0.00584 -> -0.00009` | `0.42800 -> 0.43590` |
| NaNiH3 | penalized | `0.00639 -> -0.00266` | `0.33476 -> 0.33817` |

![改进版训练 batch XRD 分数完整曲线](improved_ppo_20260807/IMPROVED_PPO_RAW_XRD_ALL_RUNS.png)

这些曲线并不支持“PPO 越训越好”：8 个材料/reward 组合的 outer-epoch objective 均没有持续上升；训练 batch XRD 分数多数只有小幅波动，NbS2-penalized 还下降。需要注意，每个 outer epoch 都换了一批候选并重新把 advantage 标准化为均值 0、标准差 1，因此跨 epoch 的 `ppo_objective` 不是同一固定数据集上的 loss，不能要求它单调上升。当前日志只保存每轮第 4 次 PPO 更新后的 objective，没有保存该轮内部 4 次更新各自的轨迹；最终有效性仍以固定 test seed 的独立 XRD 与 StructureMatcher 为准。

## 4. 固定条件独立评估

每个训练模型评估基座、epoch 5 和 epoch 25；24 个训练单元共形成 72 个池。每池使用未参与训练的固定 test seed 生成 100 个样本，总计 7,200 个样本。每个池均重新计算公共 XRD 分数、StructureMatcher、Top-10、去重和覆盖指标。

- 72/72 池完成，manifest 为 `complete`；
- 7,200 个 sampler 轨迹全部满足目标化学式；
- 生成 7,078 个 CIF，其中 7,034 个有有效 XRD；
- 结构命中按 4 材料 x 3 seed 共 12 个配对单元进行统计，而不是把 1,200 个候选错误地当作相互独立重复。

### 4.1 三个核心配对比较

| 比较 | StructureMatcher 变化 | exact p | paired t p | Wilcoxon p | 结论 |
|---|---:|---:|---:|---:|---|
| `peak` epoch25 - base | 0 | 1.000 | 不可定义（差值全 0） | 1.000 | 完全无变化 |
| `peak_penalized` epoch25 - base | +1 | 1.000 | 0.339 | 1.000 | 不显著 |
| epoch25 penalized - peak | +1 | 1.000 | 0.339 | 1.000 | 不显著 |

连续分数也没有显著提升。`peak_penalized` epoch25 相对 base 的 common peak 均值每池增加 `0.002287`，exact `p=0.105`；common penalized score 增加 `0.001075`，exact `p=0.250`。这是很小的排序/尺度偏好变化，不是拓扑恢复证据。

覆盖指标也只是轻微变化：`peak_penalized` 的 primitive strata `454 -> 459`、目标空间群候选 `89 -> 92`，对应 exact p 分别为 `0.375` 和 `0.500`。所有方法的“目标空间群且晶格接近”子集仍为 0，说明候选没有进入正确拓扑附近的关键局部区域。

### 4.2 分材料结果

| 材料（每项 300 候选） | `peak` base -> epoch25 | `penalized` base -> epoch25 | 关键覆盖观察 |
|---|---:|---:|---|
| Zr | 5 -> 5 | 5 -> 5 | 目标 SG 6 -> 6；无恢复提升 |
| NbS2 | 0 -> 0 | 0 -> 0 | 目标 SG 4 -> 4；始终无正确拓扑 |
| Li2TeC2 | 0 -> 0 | 0 -> 0 | 目标 SG 始终为 0；reward 无从学习正确骨架 |
| NaNiH3 | 36 -> 36 | 36 -> 37 | penalized 目标 SG 79 -> 82；唯一 +1 命中来源 |

Top-10 结论同样保守：`peak` 前后保持 6 个，`peak_penalized` 排序下由 19 增至 20；新增的 Top-10 命中仍只是 NaNiH3 的同一个额外结构。NbS2 与 Li2TeC2 没有候选可供任何 ranker 排到前面。

## 5. 为什么机制更正确，最终效果仍差

### 5.1 首要瓶颈是“正确候选不存在”，不是“已有正确候选排错”

PPO 只能提高自己采样到的动作序列概率。Li2TeC2 的目标空间群候选仍为 0，NbS2 虽有少量目标空间群候选但 StructureMatcher 仍为 0。XRD reward 再精细，也不能从未访问过的空间群/Wyckoff/骨架中产生梯度。

### 5.2 XRD 相似度与拓扑正确性不是一一对应

粉末 XRD 对晶格尺度和主峰较敏感，但不同原子排列、不同超胞或相近晶格可能得到相似峰。`peak_penalized` 消除了一个明显捷径，却没有把分数变成 StructureMatcher 的等价指标。因此 PPO 可以略微提高平均 XRD 分或覆盖数，而结构命中保持不变。

### 5.3 batch 内相对学习无法解决“全错 batch”

标准化 advantage 修复了非平稳参照，但它只回答当前 32 个候选中谁相对较好。对于 Li2TeC2 这类没有正确骨架的 batch，模型仍在错误结构之间做排序学习。下一版需要质量门控或分层目标，而不应把任意 batch 的最高 XRD 分都当作值得强化的正例。

### 5.4 更新稳定，但过于保守且作用范围有限

只有 3/600 轮出现非零 clipping，approx KL 很小；heads-only 又只更新 2.22% 参数。这说明训练没有发散，也说明策略分布移动很有限。直接提高学习率或解冻全模型风险较高：在没有先解决候选覆盖前，更大的更新可能只是更快地强化错误高分骨架。

### 5.5 当前样本量只能排除“大幅提升”，不能证明 +1 有效

统计单元只有 12 个配对池。`+1/1200` 的总候选变化集中在一个材料和一个配对方向，三种检验均不显著。不能把它解释成“趋势已成立，只需多跑 epoch”。

## 6. 下一轮具体方案

### 方案 A：先做离散骨架覆盖实验，不训练 PPO

目标是确认生成器能否访问正确结构附近。对每个材料固定总预算，按目标空间群、邻近空间群、primitive site count 和 Wyckoff skeleton 分层采样；primitive 归一化后去重，并报告每个 stratum 的尝试数、唯一 skeleton 数、目标 SG 数、晶格接近数和 StructureMatcher oracle 命中。

建议先做每材料 5,000 个候选的串行安全试验，按 500 个一批落盘和审计。接受标准：

- Li2TeC2 必须从目标 SG `0` 提高到至少 20 个唯一候选；
- NbS2、Li2TeC2 每个都至少出现 1 个 StructureMatcher oracle 命中，或至少出现明确的目标 SG + 晶格接近子集；
- 唯一 `(SG, primitive sites, Wyckoff skeleton)` 覆盖随样本数持续增长，而不是主要重复旧骨架。

若达不到该标准，应改候选生成/条件模型，而不是进入 PPO。

### 方案 B：对 zero-coverage 与 already-covered 材料分流

- Li2TeC2、NbS2：使用目标 SG/邻近 SG 配额、Wyckoff skeleton 均衡和 primitive cell-size 分层，先解决覆盖；
- NaNiH3、Zr：正确拓扑已经能生成，重点检验 `peak_penalized` 是否提高正确候选的排序和采样频率；
- 分层采样只用于搜索和评估时可以直接重加权；若用于 PPO 采样，必须把 proposal/行为策略概率纳入 log-prob 或 importance correction，不能无记录地硬拒绝重复样本。

### 方案 C：在覆盖通过后再做质量门控 PPO

一个 batch 只有同时满足以下条件才更新：有效 CIF/XRD 比例达标、分数超过材料固定的 base-checkpoint 分位阈值、目标/邻近 SG 或晶格接近候选达到最小数量。否则记录并跳过更新，避免强化“全错 batch 的相对赢家”。

仍保留单一训练标量，例如：

```text
S_train = S_peak_penalized * validity_gate * coverage_weight
```

其中 gate 和 weight 使用少量预先固定的材料无关规则；StructureMatcher 只用于离线验证，不作为真实未知结构场景中的训练 reward。

### 方案 D：最小化对照矩阵与停止线

覆盖通过后只做两个条件：`peak_penalized heads-only` 与 `peak_penalized heads-only + quality gate/stratified proposal`。每种材料 3 个 paired seed，保持相同候选和 optimizer-update 预算。不要同时改学习率、解冻范围、reward 和采样器。

主要成功标准：

- 4 材料 x 3 seed 的 StructureMatcher 配对提升方向一致，exact sign-flip `p<0.05`；
- NbS2 和 Li2TeC2 不再都是 0；
- Top-10 命中和 oracle 命中同时提高，而不是只有平均 XRD 分提高；
- primitive strata/唯一 skeleton 不下降，ratio、KL、grad norm 有限，无资源守护停止。

提前停止：完成一半预算后，若 Li2TeC2 目标 SG 仍为 0，或两个复杂材料仍无任何晶格接近子集，则停止 PPO 矩阵，返回候选生成阶段。

## 7. 验证与产物

- 训练 manifest：`issue68_xrd_recovery/experiments/improved_ppo_20260806_full/training_manifest.json`
- 训练日志和 checkpoint：`issue68_xrd_recovery/experiments/improved_ppo_20260806_full/`
- 72 池评估汇总：`issue68_xrd_recovery/experiments/improved_ppo_evaluation_20260806/CHECKPOINT_EVALUATION_SUMMARY.csv`
- 配对统计：`issue68_xrd_recovery/experiments/improved_ppo_evaluation_20260806/PAIRED_EVALUATION_SUMMARY.json`
- 评估 manifest：`issue68_xrd_recovery/experiments/improved_ppo_evaluation_20260806/checkpoint_evaluation_manifest.json`

代码回归验证：Issue #68 测试 `48 passed`；加入组成约束测试后的最终相关回归为 `86 passed`；核心脚本均通过 `py_compile`；`git diff --check` 通过。测试只有已有的 spglib/JAX 弃用警告，没有失败。训练和大体积评估产物保留在 experiments 目录，未提交或回滚用户工作区中的其他改动。
