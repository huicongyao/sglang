# PCTree 论文技术分析

> **From Chains to Trees: Parent-Conditioned Drafting for Semi-Autoregressive Speculative Decoding**
> Zixian Li, Tong Li, Chi Xie, Xiaohui Song, Haonan Lu — OPPO AI Center
> arXiv:2608.02123v1 [cs.CL], 2026-08-03

---

## 1. 一句话总结

PCTree 是一个**训练无关（training-free）的推理策略改动**：它复用 DSpark 已经训练好的 Markov head，
把 DSpark 的**单链草稿**改造成**固定预算的草稿树**。关键点是"每个具体的父节点各自打分"
（parent-conditioned scoring）——不增加 backbone forward、不重训任何权重，仅把 Markov head 的
batch 从 1 扩到 k。在 Qwen3-{4B,8B,14B} × 9 个 benchmark 上，B=7 时端到端 AR 加速提升 3.1%–29.5%；
Qwen3-4B GSM8K B=16 时平均接受长度从 9.41 → 11.16，AR 加速从 6.14× → 6.60×。

---

## 2. 问题背景

### 2.1 投机解码的收益结构

> 以下的延迟公式与 `∝ B` 的记法是我为了对比而整理的，**不是 PCTree 原文表述**。
> 论文 §3.1 只用文字说「效率取决于 draft 延迟、target 验证延迟与每轮推进 token 数三者的权衡」。
> 另注：PCTree 全文用 `B` 表示块大小，不使用 `γ`（`γ` 是 DSpark / 本仓库侧的记号）。

每 token 延迟 `L = (T_draft + T_verify) / τ`，其中 `τ` 是每轮平均接受长度（含 target bonus token）。
三条技术路线各有短板：

| 路线 | 代表 | τ | T_draft | 短板 |
|---|---|---|---|---|
| 自回归 drafter | EAGLE/EAGLE-2/3 | 高 | `∝ B`（每深度一次 forward） | 草稿延迟随深度线性增长 |
| 纯并行 drafter | DFlash、block diffusion | 低 | 1 次 forward | 块内无依赖 → **suffix decay**（后缀多模态碰撞） |
| 半自回归 | **DSpark** | 中高 | 1 backbone + B 次轻量 Markov | 解码器只产出**一条链** |

### 2.2 DSpark 的具体瓶颈

DSpark（arXiv:2607.05147，DeepSeek-AI）= DFlash 风格并行 backbone + 轻量 Markov head：

- **并行 backbone**：输入 anchor token `x_t` + `B-1` 个 MASK，一次 forward 得到共享 base logits
  `L = (L_0, ..., L_{B-1})`。因为并行计算，`L_d` **不依赖**块内先前选出的具体 token。
- **序列 Markov head**：从 `y_{-1} := x_t` 出发，`d = 0..B-1` 顺序执行

  ```
  z_d = L_d + Markov(y_{d-1}),    y_d = argmax_v softmax(z_d)_v      (eq.1)
  ```

问题出在 eq.1 的 **top-1**：整个块被解码成唯一一条链，一旦靠近根部发生 mismatch，
**整条后缀全部作废**（rejection cascade）。块越大（B=16）浪费越严重。

### 2.3 单纯分支不够：parent-independent stitching 失效模式

如果像 DDTree（Ringel & Romano 2026）那样直接从 per-position 分布 `q_d` 建树，
则深度 `d-1` 的**所有父节点共享同一个 `q_d``**——论文称之为 *parent-independent per-position scoring*。
图 2 的反例：

- 位置 0：`p_0(A)=0.55`, `p_0(B)=0.45`
- 位置 1（共享分布）：`p_1(C)=0.60`, `p_1(D)=0.25`, `p_1(E)=0.10`
- 结果：`B→C` 的联合分 0.27 排在 `B→E`（0.045）前面，尽管 **C 在 B 之后语义不成立**；
  真正 parent-consistent 的 `B→E` 被压制。

而 PCTree 用 Markov head 对每个父节点重新条件化后：`A→C=0.50`、`B→E=0.40`，两条都是
parent-consistent 的分支。**这就是 DSpark 已经具备但没有被利用的条件信号。**

---

## 3. PCTree 方法

### 3.1 记号

| 符号 | 含义 |
|---|---|
| `B` | 块大小 = Markov 顺序阶段数 = 树的最大非根层数 |
| `k` | 局部分支因子（每个父节点取 top-k 子节点） |
| `N` | 验证节点预算（**含 root**），主实验 `N=32` |
| `F_{d-1}` | 阶段 `d` 使用的父节点 frontier，`F_{-1} = {x_t}` |
| `b_d = \|F_{d-1}\|` | frontier batch size，`b_0=1`，之后 `1 ≤ b_d ≤ k` |

### 3.2 Markov 树扩展（batched parent conditioning）

同一深度的所有父节点共享 base logits `L_d`，但各自拿到自己的 Markov bias：

```
z_d(p) = L_d + Markov(p),    log π_d(· | p) = log softmax(z_d(p))      (eq.2)
```

把 frontier 堆叠成 `P_{d-1} = (p_1,...,p_{b_d})`，**一次 batched 调用**完成整层：

```
Z_d = 1_{b_d} L_d^T + Markov(P_{d-1})  ∈ R^{b_d × |V|}                (eq.3)
```

节点排序用**联合路径分数**（root 到该节点的 log 概率和）：

```
s(c) = Σ_{(u→v) ∈ path(c)} log π(v | u)                               (eq.4)
```

### 3.3 逐层剪枝（frontier pruning）

不剪枝的话，进入阶段 `d` 的 frontier 最多 `k^d` 个父节点，扩展产生 `k^{d+1}` 个节点 —— 指数爆炸，
对大 B 不可行。PCTree 沿用 EAGLE-2 的 score-guided dynamic tree 思路：

1. 每个 frontier 父节点扩展 local top-k 子节点，push 进 candidate pool（分数 `s(child) = s(p) + log π_d(child|p)`）；
2. **只有当前层联合分最高的 k 个节点**成为下一层 frontier。

于是：frontier 恒定被 `k` 界住，**加入 pool 的候选节点数** `≤ k + (B-1)k²`
（原文措辞是 "added to the pool"，不含 root；总节点数为 `1 + k + (B-1)k²`）—— 线性于 B，而非指数。
Markov head 仍然是**每深度调用一次、共 B 次**，只是 batch 从常数 1 变成 `b_d ∈ {1..k}`。

> 与 beam search 的区别：beam search 返回终端序列；PCTree **保留被打分的内部节点**，
> 因为它们本身就是验证候选、也是更深节点的祖先。

### 3.4 预算选择与树验证

B 个阶段结束后，对**所有候选节点（含 root）**按联合分降序排序，
tie-break 顺序为：分数降序 → 深度升序 → 稳定节点 ID，取全局 top-`N`。

- root 分数 `s(x_t) = 0`，因此**必然入选**；
- 因为 `s(c) = s(p) + log π(c|p) ≤ s(p)`，**任一祖先分数不低于后代** ⇒ 选中集合天然是
  **prefix-closed（ancestor-closed）树**，无需额外闭包修补。这是该设计最漂亮的一点。

选中集合导出三样东西：(1) 扁平 draft token tensor；(2) 枚举 root-to-leaf 路径的 retrieve indices；
(3) ancestor-only tree attention mask。然后**一次 target forward** 用标准 greedy tree 投机规则验证。

`k=1` 时退化为每深度唯一 greedy child，即恢复 DSpark（在全局预算 N 之下）。

### 3.5 算法 1（原文伪代码）

```
Require: 草稿模型 M_D, root token x_t, 块大小 B, top-k, 预算 N
 1: L ← 一次并行块 forward: M_D(x_t, MASK, ...)
 2: frontier ← {x_t}, joint score 0;  pool ← {x_t}
 3: for d = 0 .. B-1 do
 4:     一次 Markov 调用 batch 全部 p ∈ frontier: z_d(p) ← L_d + Markov(p)
 5:     扩展 local top-k children，按 eq.4 更新联合分后 push 进 pool
 6:     frontier ← 当前层按联合分的 top-k 节点
 7: end for
 8: S ← pool 全局 top-N（score 降 / depth 升 / 稳定 ID）
 9: return draft tokens, retrieve indices, 由 S 导出的 tree attention mask
```

### 3.6 复杂度

设 `C_bb` = 一次 B 位置 backbone forward 成本，`C_mk(b)` = batch 为 b 的 Markov 调用成本：

- DSpark：`C_bb + Σ_{d=0}^{B-1} C_mk(1)`
- PCTree：`C_bb + Σ_{d=0}^{B-1} C_mk(b_d)`，`b_0=1`、之后 `b_d ≤ k`，外加可忽略的选择开销

**顺序阶段数不变（仍是 B），只是 Markov batch 变宽（上界 k）。** 论文对这份额外草稿成本的定性是
「相对 target 验证仍然很小，且可以被更长的接受路径摊销」（§4.3）——原文没有给出与自回归 drafter
的量化倍数比较；定性上，自回归 drafter 每深度要为每个活跃父节点付一次完整 forward
（或一次 mask 不断增长的 batched tree forward），而 PCTree 只需一次 batched Markov 评估 + top-k 选择。

### 3.7 Tensor 实现细节（附录 A）

- 选中节点存为扁平 token tensor；
- 每节点 **position id = 树深度 + 已验证前缀长度** ⇒ 同深度兄弟共享 position，但祖先链不同；
- tree mask 允许节点 attend 到「已验证前缀 + 自身 + root-to-node 祖先」；
- retrieve indices 枚举 padded root-to-leaf 行，用于把每个候选 child 与前一个 target logit 比对；
- 选出最长接受行后，沿该行 gather KV tensor 与 target hidden states，并在已验证前缀之后做 compact。

---

## 4. 与相关工作的定位

论文 Table 1 的概念对比：

| | DFlash | DDTree | DSpark | **PCTree** |
|---|---|---|---|---|
| 草稿模型 | block diffusion | block diffusion | parallel + Markov | parallel + Markov |
| 候选结构 | chain | budgeted tree | chain | **budgeted tree** |
| 打分来源 | per-position `p_d` | per-position `p_d` | Markov-refined logits | Markov-refined logits |
| parent-specific 打分 | no | no | 单个 active parent | **batched parents** |
| 建树需额外训练 | – | – | – | **no** |
| 每轮顺序草稿阶段 | 1 backbone | 1 backbone | 1 backbone + B Markov | 1 backbone + B Markov |

其他相关线：TAPS（Wang et al. 2026，target-aware scorer 选 prefix-closed 子树）、
OPT-Tree、Medusa 多头、EAGLE 系列特征级草稿。PCTree 的独特性在于
**不引入新打分器、不改架构，纯粹把已有的条件容量（Markov head）从 1 个父节点用到 k 个父节点。**

---

## 5. 实验

### 5.1 设置

- **Target**：Qwen3-4B / 8B / 14B。`B=7` 用 DeepSeek 官方 DSpark checkpoint；
  `B=16` 用官方 DSpark recipe 在 `mlabonne/open-perfectblend` 上自训 target-matched draft。
- **超参**：`k=4`, `N=32`（含 root），greedy target 验证，**confidence scheduling 关闭**（保证两者验证策略相同）。
- **硬件/软件**：单卡 NVIDIA H20，bfloat16，PyTorch 2.11 / Transformers 5.5 / CUDA 12.8 / SDPA attention。
- **协议**：chat template、thinking 关闭、batch size 1、512 token 输出上限、greedy、EOS 停止。
  confidence scheduling 与 **near-tie re-verification** 均关闭。
  Qwen3-4B B=16 全配置与 Qwen3-4B GSM8K B=7 跑 **3 次重复**取均值，其余主表配置目前只有一次运行。
- **计时细节（附录 A）**：模型与数据集加载在计时之外，但**没有 prompt 级 warmup**，
  第一个 prompt 计入测量循环；B=7 的重复实验**交替 DSpark/PCTree 的执行顺序**；
  GPU persistence mode 与 application clocks 保持主机默认；
  AR 加速在**每次重复内部**先算出、再对重复取 mean±std。
- **Benchmark**：Math（GSM8K, MATH-500, AIME25）、Code（MBPP, HumanEval, LiveCodeBench）、
  Chat（MT-Bench, Alpaca, Arena-Hard）。评测集规模（Table 7）差异很大，对解读小数位有实际影响：

  | 数据集 | 上游 release | 行数 |
  |---|---|---|
  | GSM8K | `openai/gsm8k` | 1,319 |
  | MATH-500 | `HuggingFaceH4/MATH-500` | 500 |
  | AIME25 | `MathArena/aime_2025` | **30** |
  | MBPP | `google-research-datasets/mbpp` | 257 |
  | HumanEval | `openai/openai_humaneval` | 164 |
  | LiveCodeBench | `livecodebench/code_generation_lite` | 1,055 |
  | MT-Bench | `HuggingFaceH4/mt_bench_prompts` | 80 |
  | Alpaca | `tatsu-lab/alpaca` | **52,002** |
  | Arena-Hard | `lmarena-ai/arena-hard-auto` | 750 |

  AIME25 只有 30 题、MT-Bench 只有 80 条 —— 这两列的单次运行数字波动空间最大；
  Alpaca 则用了全部 52k 行。

### 5.2 主结果：接受长度 τ（Table 2 摘选）

**PCTree 在全部 9 个 benchmark × 3 个 target × 2 个 block size 上都提升 τ。** Qwen3-4B：

| Drafter | GSM8K | MATH-500 | AIME25 | MBPP | HumanEval | LCB | MT-Bench | Alpaca | Arena-Hard |
|---|---|---|---|---|---|---|---|---|---|
| EAGLE-3† | 5.14 | 4.62 | 3.92 | 3.69 | 4.16 | 3.77 | 2.39 | 2.26 | 2.55 |
| DFlash† | 5.40 | 4.85 | 4.15 | 4.40 | 4.74 | 4.18 | 3.07 | 2.96 | 2.83 |
| DSpark (B=7) | 6.31 | 6.24 | 5.47 | 5.33 | 5.60 | 5.30 | 3.82 | 3.67 | 3.81 |
| **PCTree (B=7)** | **7.24** | **7.22** | **6.77** | **6.53** | **6.78** | **6.30** | **5.07** | **4.83** | **4.83** |
| DSpark (B=16) | 9.41 | 9.03 | 6.87 | 6.90 | 7.42 | 6.71 | 4.28 | 3.99 | 4.30 |
| **PCTree (B=16)** | **11.16** | **10.85** | **8.80** | **8.43** | **9.21** | **8.05** | **5.50** | **5.21** | **5.44** |

†：转引自 DSpark 论文，非同环境严格对比。

### 5.3 端到端加速（Table 3）

| Suite | B | τ_DS | τ_PC | Δτ | AR 加速 (DSpark/PCTree) | Δ 加速 |
|---|---|---|---|---|---|---|
| GSM8K | 7 | 6.31 | 7.24 | +0.93 (+14.8%) | 4.24× / 4.50× | **+6.1%** |
| GSM8K | 16 | 9.41 | 11.16 | +1.75 (+18.6%) | 6.14× / 6.60× | **+7.5%** |
| HumanEval | 7 | 5.60 | 6.78 | +1.18 (+21.2%) | 3.74× / 4.27× | **+14.3%** |
| HumanEval | 16 | 7.42 | 9.21 | +1.79 (+24.1%) | 4.89× / 5.47× | **+11.8%** |
| MT-Bench | 7 | 3.82 | 5.07 | +1.24 (+32.5%) | 2.48× / 3.20× | **+29.3%** |
| MT-Bench | 16 | 4.28 | 5.50 | +1.23 (+28.8%) | 2.82× / 3.25× | **+15.1%** |

全 9 任务 × 3 模型的加速表见附录 Table 8：14B B=16 GSM8K 达 6.93× → 7.21×。
重复性检验（Table 9）：Qwen3-4B GSM8K B=16 三次 6.09/6.22/6.10× vs 6.64/6.57/6.57×，
增益 +7.5%±1.7%，**标准差远小于增益**。

### 5.4 机制隔离实验（核心可信度证据，Table 4 + Figure 4）

为了把「分支带来的收益」和「parent-specific 重新条件化带来的收益」分开，构造三个
共享 checkpoint / prompt / B / verifier / packing 代码的变体：

1. **DSpark**：原始 top-1 链；
2. **Shared-Markov tree**：先用 eq.1 建 greedy 参考链，然后**每个深度复用该链的 Markov 分布 `q_d`**
   给所有树父节点 —— 即 `q_d(·|p) = q_d(·|p') = q_d(·)`。保留了联合训练的 Markov head 与树搜索，
   **只去掉 parent-specific 条件化**；
3. **PCTree**：按 eq.2 对每个具体父节点单独算分布。

GSM8K（B=16, k=4, N=32）：

| 构造 | 平均 τ | verified-node 利用率 | 未提交节点占比 | Rounds/sample |
|---|---|---|---|---|
| DSpark | 9.410 | 53.78% | 46.22% | 26.812 |
| Shared-Markov tree | 10.225 | 29.88% | 70.12% | 24.718 |
| **PCTree** | **11.156** | 32.90% | 67.10% | **22.632** |
| DFlash+DDTree (外部 checkpoint) | 7.485 | 20.92% | 79.08% | 33.826 |

**结论**：PCTree vs Shared-Markov 差异**仅在于是否按父节点重新条件化**，
τ 从 10.225 → 11.156（**+9.1%**），利用率 29.88% → 32.90%，轮数 24.718 → 22.632（**−8.4%**）。
接受深度生存曲线 `S(d) = Pr[A ≥ d]`：depth 8 从 57.1% → 64.8%，depth 12 从 40.3% → 46.8%。

> 注意 DFlash+DDTree 用了不同的 draft 架构与 checkpoint，只作系统级参照，
> **不能**把差异归因于树打分本身。其曲线在 d=15 终止，因为官方 B=16 约定是
> 1 个固定 root 位置 + 15 个 mask 位置。

### 5.5 消融

**分支因子 k**（B=16, N=32, GSM8K, Qwen3-4B）：

| k | 1 (≡DSpark) | 2 | 4 | 8 |
|---|---|---|---|---|
| 平均 τ | 9.41 | 10.85 | **11.16** | 11.14 |
| AR 加速 | 6.14× | 6.58× | **6.60×** | 6.59× |

`k: 1→2` 拿到大部分收益；`k=4` 基本饱和；`k=8` 无额外接受收益，加速差异在计时噪声内。
（注：`k=1` 与 `k=4` 两个端点直接复用主实验的三次均值，`k=2`/`k=8` 则是单独测量 ——
两组的测量条件不完全对等。）

**验证预算 N**（Figure 5，k=4）：增大 N 会延长接受前缀，但**改进逐渐饱和**
（τ 缓慢升到 N=128 的约 12）；AR 加速则**非单调**：B=7 峰值在 N=32，B=16 峰值在 N=64，
之后建树与验证开销超过边际接受收益。主实验保留预先指定的 N=32（跨 block size 的稳健工作点），
而非按 GSM8K 测试集回溯调参；论文也提醒 sweep 的最优点应视为硬件与负载相关的工作点。
**这份 sweep 是独立计时的**（图注明确说明 "timed independently from the main tables"），
所以图中 B=16 的峰值（约 6.95×）**不能与主表的 6.60× 直接相减比较**。

**成本分解（Table 6，GSM8K, k=4, N=32）**：

| B | 方法 | τ | rounds/sample | backbone (ms/rd) | Markov+tree (ms/rd) | AR 加速 |
|---|---|---|---|---|---|---|
| 7 | DSpark | 6.31 | 39.9 | 3.46 | 0.65 | 4.24× |
| 7 | **PCTree** | 7.24 | 35.1 | 3.46 | 3.52 | **4.50×** |
| 16 | DSpark | 9.41 | 26.8 | 3.55 | 1.09 | 6.14× |
| 16 | **PCTree** | 11.16 | 22.6 | 3.55 | 5.24 | **6.60×** |

**backbone 成本完全不变**（PCTree 只改候选扩展）。Markov+tree 从 0.65→3.52ms 和 1.09→5.24ms
（该测量含选择、剪枝、bookkeeping、packing，所以 `k=4` 并不等于 4× 的 batch-1 Markov 延迟），
但轮数从 39.9→35.1 / 26.8→22.6，**轮数下降盖过单轮变贵**。

---

## 6. 局限（论文自陈 + 我的补充）

论文列出 4 条：

1. 树质量受预训练草稿分布上界约束 —— 训练无关的扩展策略**修不了系统性偏弱的 Markov head**；
2. 逐层 top-k 剪枝是启发式，可能在预算 N 下丢掉全局最优路径；
3. `k` 或 `N` 过大时，建树与 verify attention 在小 target / 显存紧张设备上不再可忽略；
4. AR 加速增益依赖硬件、精度、attention kernel、树预算（当前是 BF16 + SDPA + H20 的结论）。

补充几条需要注意的：

- **全部实验 batch size = 1**（附录 A 明确）。这是单请求延迟场景的结论。生产 serving 下 batch 大、
  target forward 已被算力打满时，树验证多出的节点会直接和其他请求争算力。
  论文关闭 confidence scheduling 的**自陈理由**是让 DSpark 与 PCTree 用完全相同的固定验证策略
  （§5.1），与 batch 规模无关；但客观结果是
  **PCTree 与 confidence-scheduled verification 的交互完全没有被评估**。
- **只报告 greedy 验证**。eq.2 保留了精确 softmax，理论上兼容 lossless rejection sampling，
  但论文未给采样路径（temperature > 0）的实验。
- 除 Qwen3-4B B=16 和 GSM8K B=7 外，其他主表配置**只有一次运行**，作者自己提醒
  小数位不应解读为统计置信度。结合 Table 7，AIME25（30 题）与 MT-Bench（80 条）的单次数字最脆弱。
- 「training-free」的准确范围是**树扩展不需要训练**（`k=1` 与 `k=4` 共用同一 checkpoint）。
  B=7 用 DeepSeek 官方 checkpoint，B=16 的 draft 由作者按官方 recipe 自训 ——
  但这份训练成本是 **DSpark 基线与 PCTree 共担**的前提，不是 PCTree 独有的开销。
  只是摘要里没提 B=16 需要自训，复现 B=16 那组数字的门槛因此不低。

---

## 7. 在本仓库 SGLang DSpark 上的可行性分析

结论先说：**草稿侧几乎是现成的，验证侧分两种情况 —— 非 V4 目标（Qwen3 系，即论文自己的实验设置）可行；
DeepSeek-V4 的 `dsv4` backend 上被 CSA 压缩语义堵死，不是补一个 mask 参数能解决的。**

以下全部基于静态代码阅读（未运行 server 或测试）。

### 7.1 草稿侧：树搜索逻辑在仓库里已经存在

最重要的发现是 PCTree 算法 1 的第 4–6 行（batched 扩展 + 联合分 + 逐层剪枝）与 SGLang
EAGLE 路径的 `_select_top_k_tokens_later`（`python/sglang/srt/speculative/spec_utils.py:307`）**结构同构**：

```python
expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)    # 联合路径分数 (eq.4)
topk_cs_p, topk_cs_index = fast_topk(expand_scores.flatten(1), topk) # 逐层 frontier 剪枝到 k
tree_info = (expand_scores, topk_index, topk_cs_index + ...)         # k² 候选入 pool + parent 索引
```

算法 1 的第 8–9 行同样已有：

- `organize_draft_results`（`eagle_utils.py:107`）做全局 top-N（`torch.topk(score_list, N-1)`，
  索引升序排 ≈ 论文的 depth-asc tie-break），产出 `parent_list` / `top_scores_index` / `draft_tokens`；
- `build_tree_kernel_efficient`（`eagle_utils.py:151`）产出 `tree_mask` / `positions` / `retrieve_index`
  / `retrieve_next_token` / `retrieve_next_sibling`。其 `positions` 注释直接写着
  「depth of each draft token is [0,1,1,2] and prompt length 7 → positions = [7,8,8,9]」——
  **正是论文附录 A 的 position 规则（兄弟共享 position）**。

两者唯一差别正是 PCTree 的卖点：EAGLE 每深度要跑一次完整 draft forward 才拿到 `topk_p`，
PCTree 用 `L_d + Markov(P_{d-1})` 一次 batched Markov 调用替代。

**DSpark 侧要动的很少**：

- 草稿 backbone forward **完全不用改**。`dspark_draft.py:352` 的 `draft_forward_batch` 只依赖
  `draft_block_ids (bs, gamma)` + 线性 positions + gamma 个 KV slot；PCTree 复用同一份 `base_logits`。
- eq.3 的 broadcast 天然成立：`VanillaMarkov.compute_step_bias`（`models/dspark.py:100`）对前置维度是
  elementwise 的，`logits [bs,1,V] + bias [bs,k,V]` 就是 `1_{b_d} L_d^T + Markov(P_{d-1})`。
- 要改的是 `run_markov_block`（`models/dspark.py:46`）的循环体：`prev_tokens` 从 `[bs]` 扩成 `[bs, b_d]`，
  并在每步后插入 top-k / 剪枝。RNN head（`models/dspark.py:176`）需要按 frontier 展开状态张量 ——
  这也解释了论文为何明确沿用一阶 Markov 默认值。

### 7.2 验证侧（`dsv4`）：结构性阻塞

DSpark 的 verify **完全不用 attention mask** 表达块内因果性，而是「每个 query 一个标量 KV 前缀长度」：

```python
# deepseek_v4_backend.py:2147
seq_lens_casual = seq_lens[:, None] + torch.arange(-qo_len + 1, 1, ...)
```

page table 由此从 `req_to_token[req, :len]` 切**连续前缀**（`dsv4_attn_metadata_kernels.py:99`）。
树的祖先集合不是连续前缀，metadata 里没有任何字段能表达它。三条 verify 构造路径一律
`custom_mask=None`（`dspark_verify.py:202/248/360`），且 `DFlashVerifyInput` 自己注明
「DFLASH verify is linear (non-tree), so this is always 1」（`dflash_info.py:37`）。

按难度排序的阻塞点：

1. **稀疏 top-k 与树 mask 不可组合（根本阻塞）**。DSA/lightning indexer 的 top-k 选择发生在
   **c4 压缩页**粒度上（`c4_seq_lens = seq_len // 4`），而 indexer 的因果性是前缀谓词
   `positions < seq_lens.unsqueeze(1)`（`dsv4/indexer.py:122`、`:291`；`seq_lens <= TOPK` 的行
   还直接合成顺序 index `0..seq_len-1`，`:319`）。祖先集不是 4 对齐的，**共享同一个 4-token
   压缩块的兄弟 draft token 在压缩域里不可区分**。kernel 签名（`kernels/ops/attention/dsv4/topk.py:48`）
   连 indptr 都没有。这需要重新定义 verify 阶段的压缩语义，或对 draft 窗口关闭压缩走 dense。
2. **compressor 对 verify 根本没实现**：`dsv4/compressor.py:244` 对 `is_target_verify()` 直接
   `raise NotImplementedError("target verify mode to be implemented")`。
3. **over-allocation 免 compact 的内存模型**。`dflash_info_v2.py:114` 预留 `2 * block_size` 连续槽，
   被拒绝的 tail 由下一步直接覆写，因此 DSpark **完全没有** KV compact 逻辑
   （`move_accept_tokens_to_target_kvcache` 在 `dspark_components/` 零命中；EAGLE 版在
   `eagle_worker_common.py:406`）。树上必须搬移，而 DSV4 要搬的是 4~5 个 layout 各异的子池
   （`swa_kv_pool` / `c4_kv_pool` / `c128_kv_pool` / `c4_indexer_kv_pool` / `CompressStatePool`），
   其中 ring buffer 的 `pos % ring` 寻址（`dspark_verify_window.py:634`）在 position 不再单调递增时失效。
4. **index list 的 64 对齐 + page_size 256**（`deepseek_v4_backend.py:1683`、`:544`）。
   祖先列表长度不规则，pad 到 64 倍数在 γ=5~8 的小窗口上开销占比很高。
5. **indexer 的 RoPE position 也来自这条链**（`dsv4/indexer.py:662`），树深度需要独立 position 通路。

上游其实早就把树排除了：`deepseek_v4_backend.py:579` 有
`assert self.topk in [0, 1], "MTP Topk > 1 not supported for DeepSeek V4"`，
`arg_groups/deepseek_v4_hook.py:150` 有同样的 `speculative_eagle_topk == 1` 断言。

**唯一的绕行路径**（仅覆盖 ratio 0/128 层）：学 FA3 的做法（`flashattention_backend.py:923`），
不给 kernel 传 mask，而把祖先集直接烧进 per-query 的 gather 列表 + `topk_length`
（`deepseek_v4_backend.py:1743` 的 `indices=` / `topk_length=` 接口本来就是 per-query 的）。
core attention 侧不需要改 kernel。但 ratio-4 的 CSA 层因阻塞 1 无解。

### 7.3 验证侧（非 V4 目标）：这条路是通的

DSpark 不只服务 V4。`models/dspark.py` 的 `DSparkDraftModel` 面向 Qwen3 系目标，
CI 的 `test/registered/core/test_basic_sanity_dspark.py:21` 就是
`Qwen/Qwen3-14B` + `deepseek-ai/dspark_qwen3_14b_block7`，attention backend 为
**`fa3`**（Blackwell 上 `trtllm_mha`）。

而 `fa3` 是**已支持树 mask**的 backend —— `flashattention_backend.py:2292` 的
`is_read=self.topk > 1`，机制是把 mask 转成「排序压紧的 page table + 每行 `cache_seqlens`」
再走两级 cascade attention（`:923-954`）。`triton` 与 `flashinfer` 同样支持（`triton_backend.py:1073`
`is_read=True`）。

**这正好是论文自己的实验设置（Qwen3-{4B,8B,14B}）。** 所以要复现论文，落地目标应该是
非 V4 的 DSpark 路径，而不是 V4。

仓库里 backend 的树 mask 能力现状（`is_read=` 全仓取值）：

| backend | `is_read` | 树 mask |
|---|---|---|
| `triton_backend.py:1073` | `True` | 支持 |
| `flashattention_backend.py:2292` | `topk > 1` | 支持 |
| `deepseek_v4_backend.py:1520` | `False` | **不支持**（注释：`Verify metadata never extracts the mask.`） |
| `flashmla_backend.py:297` | `False` | 不支持 |
| `trtllm_mla_backend.py:453` | `False` | 不支持 |

注意 `supports_ragged_verify_graph`（`base_attn_backend.py:63`）声明的是 ragged-verify graph 能力，
**与 mask 无关**；目前**没有**任何「backend 支持树形 verify」的能力声明机制，
唯一保护是散落各处的 `assert topk == 1`。要做树 verify，第一步应把 `is_read`
提升为与 `supports_ragged_verify_graph` 同级的显式类属性并加测试。

### 7.4 与 ragged verify / confidence scheduling 的关系

`RaggedVerifyLayout`（`ragged_verify.py:46`）只有 `verify_lens` / `extend_start_loc` / `qo_indptr_device`,
**没有任何 parent / depth / sibling 字段** —— 它表达的是「链的前缀截断」，不是拓扑。
容器本身与树正交（可复用），但语义必须扩展：`verify_lens` 从「链长」变成「树节点数 + 拓扑」。

`ScheduleVerifyLensTopk`（`kernels/ops/speculative/dspark/dspark_schedule.py:109`）输出的是标量长度，
`survival = cumprod(confidence, dim=1)` 沿链累乘。树上要选的是一个连通子树，累乘要改成
沿 parent 的 scan。**一个正面结论**：`compute_verify_token_budget` 的贪心最优性依赖
「存活概率沿链单调非增 ⇒ 全局排序尊重前缀依赖」，而 PCTree 的 `s(c) ≤ s(p)` 恰好在树上给出同一性质
（子树的祖先必被选中），所以最优性论证仍然成立 —— 只是实现从 cumsum 变成 parent-scan。

confidence head 本身是 parent-conditioned 的（输入 `[h_k; W1[x_{k-1}]]`），与 PCTree 同构，可直接复用；
但 STS 校准（`dspark_sts.py`）的 ECE 是在**链的**累积乘积上拟合的，树上累积路径不唯一，
校准目标需要重新定义。

另外 `dspark_worker_v2.py:812` 的 mamba/KDA state 提交有
`assert get_spec().speculative_eagle_topk in (None, 1)`，注释明说
「A tree (topk > 1) layout would need the accept-index mapping」。

### 7.5 两个论文 batch=1 完全没暴露的成本

1. **`verify_num_draft_tokens = gamma + 1` 是硬编码的**（`dspark_config.py:122`）。PCTree 要把 `N`
   与 `gamma+1` 解耦 —— γ=5 时 verify token 数 6 → 32（**5.3×**）。`run_dsv4_dspark.sh` 用
   `--max-running-requests 24`，即每批 verify token 144 → 768，直接压到 MoE dispatch
   （`SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`，默认 128）和 CUDA graph tier 上。
2. **图内静态缓冲按 k 放大**。`DsparkDraftSampler.corrected_out` 是 `(max_bs*gamma, vocab)`
   （`dspark_draft_sampler.py:80`），采样路径下 PCTree 需要 k× 或按选中节点重算；
   `_resolve_folded_sampling` 的显存探测阈值要跟着改。V4 的 `markov_w2` TP-shard 每步有一次
   vocab 维 all-gather（`models/deepseek_v4_dspark.py:385`），通信量同样 ×k。

### 7.6 建议的分阶段方案

**Phase 1（可做，复现论文）**：在**非 V4** DSpark 路径上实现 PCTree。
目标 Qwen3-14B + `fa3`，`SGLANG_RAGGED_VERIFY_MODE=static`（固定验证窗口，与论文一致 ——
论文也关闭了 confidence scheduling）。改动集中在：
`run_markov_block` 的父节点维扩展、复用 `organize_draft_results` + `build_tree_kernel_efficient`
产出树 mask/positions/retrieve、把 `DFlashVerifyInput` 的 `topk`/`custom_mask` 从写死的 1/None 打开、
把退化的 `_get_or_create_chain_verify_buffers`（`dflash_utils.py:309`，`next_sibling` 全 `-1`）
换成真树 buffer、accept 改用已有的 `verify_tree_greedy` / `tree_speculative_sampling_target_only`。
**这一阶段不需要写新 kernel。**

**Phase 2（重，收益不确定）**：V4 的 ratio 0/128 层走 per-query gather 列表绕行 mask；
ratio-4 的 CSA 层需要先解决压缩域的兄弟可分性，以及 `compressor.py:244` 的 verify 未实现。
外加 DSV4 多子池的 accept-index KV compact —— 这是最重的单项。

**Phase 3（研究性）**：树 + confidence scheduling 的联合调度（论文完全没做），
含 parent-scan 存活概率、树上 STS 重新校准。

### 7.7 测试护栏

按 `.claude/rules/unit-test-admission.md`（每个用例要能回答「什么 diff 会让它变红」）：

1. `k=1` 必须**逐 token 复现**现有 DSpark 链输出 —— 任何 batched 改写引入的 broadcast bug 会打红。
   参考 `test/registered/spec/dspark/test_dspark_kernel_parity.py`。
2. batched Markov（eq.3）vs 逐父节点 for 循环的数值 parity。
3. ancestor-closed 性质：随机分数下全局 top-N 选出的集合，任一节点的所有祖先都在集合内。
4. backend 能力：把 `is_read` 提成显式类属性后，加一个「声明支持树 verify 的 backend 必须
   真的读 mask」的测试 —— 现在漏声明是**静默**退化。
5. 端到端：`test/registered/spec/dspark/` 全套 + `test/registered/core/test_basic_sanity_dspark.py`。

### 7.8 顺带发现的文档问题

`CLAUDE.md` 的 DSpark 章节写 CUDA kernel 在 `dspark_components/kernels/`，该目录下已无 `.py`
（只剩 `__pycache__`），实际位置是 `python/sglang/kernels/ops/speculative/dspark/`。

## 8. 本仓库上的实现与实测（Qwen3-4B, B=16）

已在本仓库实现 PCTree 的建树算法并用真实 checkpoint 实测。**结论：算法完全复现，
论文的三条定性结论全部成立。**

### 8.1 落地范围

| 文件 | 内容 |
|---|---|
| `python/sglang/srt/speculative/dspark_components/dspark_pctree.py` | 算法 1 全部：eq.2/eq.3 batched parent conditioning、eq.4 联合分、逐层 top-k 剪枝、全局 top-N（score desc / depth asc / stable id）、ancestor-closed 导出（tokens/parents/depths/scores）、ancestor-only mask、`next_token`/`next_sibling`、`position = depth + prefix_len`。纯 PyTorch，无新 CUDA kernel。 |
| `test/registered/spec/dspark/test_dspark_pctree.py` | 8 个用例（CPU CI）：k=1 逐 token 复现 DSpark 链、batched vs 逐父节点 parity、ancestor-closed、分数沿路径单调、pool 上界与 padding、mask 与 parent walk 一致、retrieve 链覆盖每节点恰好一次、兄弟共享 position。 |
| `dspark_pctree_eval.py` + `python/sglang/benchmark/dspark_pctree_eval.py` | 离线评测：env 门控的录制钩子 + 打分器。 |
| `dspark_draft.py`（3 行）、`environ.py`（3 个 env） | 钩子接入点。env 未设置时 `recorder is None`，folded 条件与原来完全一致。 |

**回归验证**：钩子关闭时同一 prompt 的 `spec_accept_length` = 5.2631578947368425，
与改动前逐位相同。

### 8.2 评测方法（为什么不需要树形 attention）

greedy 验证下，草稿 token 被接受 iff 它等于 target 在已接受前缀下的 argmax。所以
**一个候选集的接受长度 = target 自身 greedy 续写的最长前缀在树中作为 root 路径存在的长度**。
DSpark 是 lossless 的，服务端 greedy 跑出来的 committed token 流就是这个续写。于是：

1. 正常跑 DSpark 服务，记录每轮的 `base_logits`、anchor、`prefix_len`、链草稿，
   并用**同一份 base_logits** 建 PCTree；
2. 事后用完整输出 token 流当 ground truth 给两者打分。

链与树在同一轮、同一 anchor、同一 base logits 下比较，差异只来自 parent-conditioned 分支。
校验：本方法算出的链 τ=4.289 与服务端上报的 `spec_accept_length` 均值一致，
且逐轮的 `prefix_len` 前进量恰好等于算出的接受长度 +1。

### 8.3 实测结果

配置：Qwen3-4B（本地）+ `Qwen3-4B-DSpark-b16`（B=16, vanilla Markov, rank 256），
单卡 SM100，`trtllm_mha` target / `fa4` draft，`SGLANG_RAGGED_VERIFY_MODE=static`，
20 条 GSM8K 风格 prompt × 256 token greedy，共 **1192 轮**。

```
DSpark chain tau = 4.289

k            N=8      N=16      N=32      N=64     N=128
k=1        3.911     4.284     4.289     4.289     4.289
k=2        4.071     4.847     5.160     5.275     5.275
k=4        4.082     4.933     5.445     5.794     6.021
k=8        4.083     4.935     5.482     5.893     6.268
```

三条与论文一致的结论：

1. **`k=1` 在 `N ≥ 17` 时 τ 与 DSpark 链完全相等（4.289 = 4.289）** —— 论文 §4.2
   「k=1 退化为 DSpark」在 1192 轮真实数据上逐位成立。这是整条流水线最强的正确性闸门。
   （`N=16 < B+1=17` 时树被预算截断成 15 层，τ 略降到 4.284；`N=8` 降到 3.911。）
2. **`k: 1→2` 拿到大部分收益，`k=4` 基本饱和，`k=8` 增益边际**（N=32 时
   4.289 → 5.160 → 5.445 → 5.482）—— 与论文 Table 5 的形状一致。
3. **τ 随 N 单调上升并逐渐饱和** —— 与论文 Figure 5 右panel 一致。

论文操作点（k=4, N=32）本仓库实测 **4.289 → 5.445（+27.0%）**；
论文 Qwen3-4B GSM8K B=16 是 **9.41 → 11.16（+18.6%）**。

绝对 τ 低于论文、相对增益高于论文，原因是我用的是裸 prompt（未套 chat template、
非 GSM8K 官方 few-shot 格式），链基线更弱因而 headroom 更大。**方向与量级一致，
但这不是论文数字的严格复现**（prompt 集、模板、样本量都不同）。

另一个与论文注记吻合的现象：N 较大时 k=8 反超 k=4（N=128 时 6.268 vs 6.021）——
候选空间更大需要更大预算才能兑现。

### 8.4 真实树形验证（端到端）

上面 §8.2/§8.3 用的是「greedy 等价性」离线打分，**不需要树形 attention**。之后我把真正的树形
验证也实现了：target 用 ancestor-only mask 做一次真实 forward，真实 greedy 树验证，真实 KV 提交。

落地文件 `dspark_pctree_verify.py` + worker 分支 `_forward_decode_pctree`：

| 环节 | 实现 |
|---|---|
| verify 窗口 | `verify_num_draft_tokens` 从 `gamma+1` 解耦为 `N`（`dspark_config.py::pctree_node_budget`），KV over-allocation 随 `speculative_num_draft_tokens` 自动变成 `2N` |
| tree mask | SGLang `TreeMaskMode.FULL_MASK` 布局：每请求 N 行 ×(prefix+N) 位，prefix 全开、树块 ancestor-or-self，triton backend 在 TARGET_VERIFY 直接消费 `spec_info.custom_mask` |
| position id | `prefix + depth`（兄弟共享 position） |
| accept | 真实 greedy 树验证：沿 root 走，接受 token 等于父节点 argmax 的那个子节点 |
| KV 提交 | **置换 `req_to_token` 而非搬 KV 字节**：`swap(prefix+j, prefix+c_j)`，保证该行槽位集合仍是分配到的那一份（无重复、无泄漏）。路径列号严格递增，所以升序 j 不会破坏已定的条目 |
| backend 能力位 | 新增 `AttentionBackend.supports_tree_verify_mask`（triton / flashinfer / fa3 声明 True）。不声明就拒绝启动 —— 否则树会被当成链验证并**静默**接受错 token |

**正确性证据**：greedy 验证是无损的，所以树和链必须产出同一 token 流。5 条 prompt × 128 token：
**4/5 逐 token 完全相同**，第 5 条在第 121 个 token 才分叉（attention 从 17 token 变成 32 token +
mask，浮点归约顺序不同，near-tie 处 argmax 翻转）。作为对照，链的 graph 与 eager 输出 5/5 相同。

**实测（单卡 SM100，triton backend，`--disable-cuda-graph`，5 prompt × 128 token × 2 轮）**：

| 配置 | tok/s | vs AR | 平均接受长度 | verify 轮数 |
|---|---|---|---|---|
| AR（无投机） | 30.86 | 1.00× | – | – |
| DSpark 链 | 101.53 | 3.29× | 4.799 | 272 |
| **PCTree 树 (k=4, N=32)** | 93.34 | 3.02× | **5.914 (+23.2%)** | **222 (−18.4%)** |

**接受长度 +23.2%、轮数 −18.4%，与论文一致；但 wall-clock 反而慢 8%。** 这是我的实现问题，不是算法问题：

- 全程 eager：建树、accept、KV 置换各有 16 次 Python 迭代的 per-depth 循环，mask 每步用
  `torch.cat` 逐请求拼接，外加每步一次 D2H 同步取 prefix 长度；
- verify token 从 17 涨到 32，target forward 计算量近 2×；
- 参照量级：**同样的链在 cuda graph 下是 949 tok/s，是其 eager 的 9.3×** —— 说明 eager 路径几乎完全被
  Python 开销支配，这个对比测不出算法的真实代价。

**cuda graph 下的树验证是错的，已禁止**。graph replay 把 mask 拷进 backend 预分配 buffer，
但 replay 的 metadata 没有复现 eager 的树几何：实测输出在 ~20 个 token 内就偏离 greedy
（5/5 全部早期分叉，接受长度也从 5.914 掉到 5.122）。所以 `_resolve_pctree_config` 在
未加 `--disable-cuda-graph` 时直接抛错，而不是静默服务一棵被当作链验证的树。

**回归**：原链配置（trtllm_mha + graph）同一 prompt 的 `spec_accept_length`
仍为 5.2631578947368425，与改动前逐位相同。

### 8.5 端到端测试

**功能正确性 —— GSM8K 5-shot 准确率（120 题，parallel=8，eager，同一 target）**：

| 路径 | run 1 | run 2 | Invalid | 吞吐 (tok/s) | 延迟 (s) |
|---|---|---|---|---|---|
| DSpark 链 | 0.858 | 0.875 | 0.000 | 648 / 662 | 21.1 / 19.8 |
| **PCTree 树** | 0.883 | 0.875 | 0.000 | 566 / 584 | 23.3 / 22.4 |

准确率完全落在同一区间（链自身两次就差 0.858→0.875，即 120 题里的 2 题 —— 并发调度导致
batch 组成变化，输出本身不是逐位可复现的）。**Invalid 均为 0.000。**
这是树路径没有损坏输出的决定性证据。

吞吐上树慢 12%，与 §8.4 的单请求结论（慢 8%）一致。

**并发（bs>1）测试** —— 这是 mask 布局最容易出错的地方（flat FULL_MASK 在 prefix 长度上是 ragged 的，
per-request stride 算错只会在 bs>1 且 prefix 不等长时暴露）。8 条不同长度 prompt 并发：

| 路径 | 聚合 tok/s | 接受长度 | verify 轮数 |
|---|---|---|---|
| DSpark 链 | 219.7 | 4.955 | 213 |
| **PCTree 树** | **300.9** | **6.122 (+23.6%)** | **172 (−19.2%)** |

并发下树反而**快 37%** —— 因为 eager 的 Python 开销被 batch 摊薄，接受长度的收益才显出来。

**token 流一致性与 batch 不确定性**：并发时树 vs 链只有 3/8 逐位相同，但**链 vs 链的对照组
（同一服务器跑两遍）也只有 7/8 相同**，且分叉点相同（token 58）—— 说明并发下输出本身不是
batch-invariant 的。单请求（bs=1，确定性）时树 vs 链是 4/5，唯一分叉在第 121 个 token。
结合准确率不变，可以判断分叉来自 near-tie 处的浮点差异（verify 从 17 token 变 32 token +
mask，归约顺序不同），而不是验证逻辑错误。

**回归测试**：
- `test/registered/spec/dspark/` 全套 **169 passed**（含我新增的 14 个用例）；
- 原链配置（trtllm_mha + cuda graph）同一 prompt 的 `spec_accept_length` 仍为
  5.2631578947368425，与改动前逐位相同。

### 8.6 没有做什么

- **没有可用的端到端加速**。要拿到论文的 AR 加速，需要：树构建/accept/KV 置换的 kernel 化、
  GPU 侧 scatter 生成 mask（消掉每步的 D2H）、以及把树拓扑正确地带过 cuda graph replay。
- **cuda graph 下的树验证不可用**（已在启动时拒绝，见 §8.4）。链在 graph 下是 eager 的 9.3×，
  所以「树比链慢 12%」这个数字只在 eager-vs-eager 下成立，不是生产结论。
- 采样路径未实现（`corrected_logits` 需按 k 扩展；当前只走 greedy）。
- grammar 约束下直接抛 `NotImplementedError`（`GrammarTree.from_linear_chain` 描述不了树）。
- 没有测过：多 GPU（TP>1）、DP attention、radix cache 前缀复用下的树路径、长上下文。
- confidence scheduling 在树路径上被完全绕过（论文也关闭了它）。

---

## 9. 我的评价

**扎实的地方**：

- Shared-Markov tree 这个控制变量做得很干净。共享 checkpoint、共享搜索策略、共享 packing 代码、
  共享 verifier，唯一差异是 parent-specific 重新条件化 —— 这让 +9.1% τ 的归因可信。
  大多数投机解码论文都缺这一层隔离。
- 明确区分「同环境 matched 对比」（DSpark）与「转引数字」（EAGLE-3/DFlash 带 †）、
  以及「外部 checkpoint 参照」（DFlash+DDTree）。
- ancestor-closed 由 `s(c) ≤ s(p)` 自动保证，而不是后处理补闭包 —— 干净的设计。
- 主动报告 N 的非单调性并保留预先指定的 N=32，而不是回溯挑最优点。

**弱的地方**：

- 方法本身是 EAGLE-2 的 score-guided dynamic tree 直接套到 DSpark 的 Markov 分布上，
  算法新颖性有限。论文的实际贡献是「指出 DSpark 有未被利用的条件容量」这个观察，
  以及把它工程化落地。作者自己在贡献列表里也是这么写的（"a decoding perspective"）。
- batch=1、confidence scheduling 关闭、只有 greedy 验证 —— 离生产 serving 还有距离。
  MT-Bench 上 +29.3% 这类最漂亮的数字都来自 τ 基数很低（3.82）的场景，
  绝对加速只有 2.48×→3.20×。
- 「training-free」的宣称本身没有夸大（树扩展确实零训练），但摘要没有提到 B=16 的 draft
  需要自训，容易让读者以为两个 block size 都能拿官方 checkpoint 直接复现。

**对本仓库的实际价值**：中等偏高。DSpark 已经是 DeepSeek-V4 生产路径的投机解码框架，
PCTree 是一个**不需要重训任何 checkpoint**就能试的推理侧改动，收益上界看 Table 3 大约 +7%~+15% 加速。
主要风险在 7.3 节 —— 树布局与 ragged verify / confidence scheduling / CUDA graph 的交互，
工程量集中在 mask 与 KV 位置计算，而不是算法本身。

---

## 附：关键数据速查

- 论文核心数字：Qwen3-4B GSM8K B=16，τ 9.41→11.16，AR 6.14×→6.60×（三次均值，+7.5%±1.7%）
- 默认超参：`k=4`，`N=32`（含 root）
- 候选 pool 新增节点上界：`k + (B-1)k²`（k=4,B=16 → 244；含 root 共 245），frontier 恒 ≤ k
- 顺序阶段数：`B`（与 DSpark 相同，未增加）
- Backbone 延迟：**完全不变**（3.46ms@B=7 / 3.55ms@B=16，两法相同）
- 机制隔离：Shared-Markov 10.225 → PCTree 11.156（+9.1%，纯 parent-conditioning 贡献）
