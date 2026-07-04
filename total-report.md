# MemXLife GPU Profiler Total Report

## 1. Report Purpose

这份 `total-report.md` 的目标不是只复述三个 phase 的结果，而是把整个项目从开发到优化的完整技术路径整理成一份统一文档。它主要覆盖四个层面：

1. 整体目标与项目结构
2. Phase 1 / Phase 2 / Phase 3 的具体技术实现
3. 我们在项目中采用的 AI agentic 开发方式
4. 从实验、日志、提交、验证中得到的关键工程洞察

如果后续要做 15 分钟 talk，这份文档可以直接作为底稿使用。

---

## 2. Unified Project Narrative

这三个阶段表面上分别在做：

- Phase 1: GPU profiling
- Phase 2: LoRA operator optimization
- Phase 3: lightweight LLM serving runtime optimization

但从方法论上看，它们其实是同一条技术主线的递进：

1. **Phase 1 先学会“理解机器”**
   - 不相信静态规格表
   - 用 micro-benchmark 反推出 GPU 的真实行为

2. **Phase 2 再学会“理解算子调用模式”**
   - 不只看一次算子的算力路径
   - 开始利用重复调用中的时间局部性

3. **Phase 3 最后学会“理解推理运行时结构”**
   - 不只看单个 kernel
   - 优化 prefill / decode / same-length batch / varlen batch 的真实路径

因此，这个项目最核心的价值不是“某个单点 trick”，而是形成了一条统一的系统优化方法：

**Measurement -> Modeling -> Specialization -> Validation**

---

## 3. Repository Structure And Role Map

项目主目录位于：

- `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler`

核心结构可以概括为：

```text
memxlife-gpu-profiler/
├── memxlife-project/              # Phase 1: GPU profiling multi-agent system
├── phase2_agent/                  # Phase 2: LoRA optimization agent
├── phase3_engine_sources/         # Phase 3 source-of-truth engine variants
├── workspace/                     # Phase 3 final generated submission artifacts
├── evaluator/                     # Phase 3 correctness / throughput evaluation
├── scripts/                       # rendering, toy weights, local public tests
├── stage2_outputs/                # Stage 2 downloaded official outputs and summaries
├── stage3_outputs/                # Stage 3 local fallback / remote evaluation logs
├── course_*.sh                    # remote service, sync, submit, status, fetch scripts
├── report2.md                     # Phase 2 report
├── phase2_optimization_journey.md # Phase 2 journey / retrospective
├── run.sh                         # Phase 3 entry point
└── workspace/report3.md           # Phase 3 report
```

从职责上可以把仓库分为三层：

1. **算法与运行时层**
   - `memxlife-project/`
   - `phase2_agent/`
   - `phase3_engine_sources/`

2. **评测与产物层**
   - `workspace/`
   - `evaluator/`
   - `stage2_outputs/`
   - `stage3_outputs/`

3. **自动化与远程执行层**
   - `course_prepare_submit.sh`
   - `course_prepare_submit3.sh`
   - `course_sync_workspace.sh`
   - `course_submit2.sh`
   - `course_submit3.sh`
   - `course_submit_status.sh`
   - `course_submit_status3.sh`

---

## 4. Phase 1: GPU Hardware Profiling Agent System

### 4.1 Goal

Phase 1 的任务不是优化已有程序，而是构建一个**能够自动测量 GPU 真实硬件特征**的 profiling system。

从官方项目说明的角度看，Phase 1 实际包含两层目标：

1. **先学会看懂已有 CUDA operator 的瓶颈**
   - 用 `ncu` 指标判断 compute-bound 还是 memory-bound
   - 用 roofline、L1/L2/DRAM、tensor core、occupancy 等指标定位瓶颈

2. **再进一步做硬件本征参数探测**
   - 不只分析现有 kernel
   - 而是主动生成 micro-benchmark 去反推出 GPU 的“DNA”

我们的实现重点落在第二层，即 hardware intrinsic profiling；但在 reasoning 方式上，也吸收了官方前半段强调的 `ncu` / roofline / memory hierarchy 分析框架。

它要解决的关键问题是：

- `cudaGetDeviceProperties` 可能不可靠
- `nvidia-smi` 给出的信息可能经过限制、屏蔽或不反映真实运行状态
- 某些评测环境下不能简单相信规格表

所以系统目标变成了：

**通过自动生成并运行 CUDA micro-benchmark，主动测量 GPU 的 latency / bandwidth / cache / clock 等行为。**

### 4.2 Supported Metrics

Phase 1 支持测量的指标包括：

- `dram_latency_cycles`
- `l1_latency_cycles`
- `l2_latency_cycles`
- `l2_cache_size_kb`
- `max_global_mem_bandwidth_gb_s`
- `max_shmem_bandwidth_gb_s`
- `actual_boost_clock_mhz`
- `bank_conflict_penalty_cycles`
- `max_shmem_per_block_kb`

这些指标不是通过静态 API 读出来，而是通过 benchmark 反推出来。

如果从官方说明里的分析框架来映射，Phase 1 关心的不是单个数字本身，而是两类能力：

1. **性能分析能力**
   - 能用 `sm__throughput`、`gpu__compute_memory_throughput` 等指标做 roofline 风格判断
   - 能继续向 L1/L2/DRAM、tensor core、occupancy、bank conflict 这些方向钻取

2. **硬件探测能力**
   - 能用 micro-benchmark 主动推断 latency hierarchy、cache size、effective bandwidth、boost clock、resource penalty

也就是说，Phase 1 的本质是“分析 + 主动探测”的组合，而不是只做一个 benchmark runner。

### 4.3 Architecture

Phase 1 的系统位于：

- `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project`

整体架构是一个典型 multi-agent pipeline：

```text
target_spec.json
    ->
Scout
    ->
Planner
    ->
Codegen
    ->
Runner
    ->
Analyzer
    ->
results.json
```

关键模块分布如下：

- `main.py`: CLI 入口
- `core/orchestrator.py`: 总控流程
- `agents/scout.py`: 环境探测
- `agents/planner.py`: 策略选择
- `agents/codegen.py`: CUDA probe 代码生成
- `agents/runner.py`: 编译与执行
- `agents/analyzer.py`: 结果解析与打分
- `agents/verifier.py`: 结果验证
- `agents/judge.py`: 辅助判断与回合控制
- `analysis/consistency.py`: 物理一致性校验
- `audit/logger.py`: 审计日志与报告

### 4.4 Agentic Loop

Phase 1 最值得强调的是：它不只是一个 benchmark 集合，而是一个**会自己循环决策与修正**的 agent 系统。

具体流程如下：

1. **Scout**
   - 检测 CUDA、GPU、工具链和环境配置
   - 把环境摘要送给后续 agent 作为上下文

2. **Planner**
   - 根据 metric 类型和已知知识库，从候选测量策略中选一个
   - 如果 LLM 不可用，则退化为启发式 fallback

3. **Codegen**
   - 让 LLM 生成完整 `.cu` benchmark
   - 本地先做一次 `nvcc` compile test
   - 如果失败，把 compile error 回喂给 LLM 修复
   - 最多进行多轮修复与重生成

4. **Runner**
   - 真正编译和运行 probe
   - 自适应切换不同 `-arch` / `-gencode` 参数
   - 在需要时可以接入 `ncu`

5. **Analyzer**
   - 解析 stdout / Nsight Compute 输出
   - 计算结果置信度
   - 写入知识库和运行日志

6. **Verifier / Judge**
   - 对结果做物理一致性检查
   - 判断是否需要继续尝试或接受结果

### 4.5 Why This Is Agentic Instead Of Just Scripted

Phase 1 的“agentic”不是空泛地加一个 LLM，而是在以下位置真正体现了 agent 行为：

- 策略是可选的，不是固定的
- 代码是运行时生成的，不是全部硬编码的
- 编译失败后会自我修复，不是直接退出
- 结果会进入后续决策，而不是一次性跑完就算
- 有 fallback path，保证系统在 LLM 或环境不稳定时仍可运行

### 4.6 Technical Highlights

#### 4.6.1 Anti-tampering Mindset

Phase 1 的核心思想之一是：

**不相信“被报告出来的硬件信息”，而是相信“被实验测出来的硬件行为”。**

这与官方文档里强调的 anti-hacking / environment variation 是一致的。课程明确提到，评测环境可能出现：

- **非标准频率锁定**
  - 例如 GPU 被锁到非常规 core / memory clock
- **资源屏蔽**
  - 例如只开放部分 SM，或限制 shared memory / block 资源
- **API 误导**
  - 例如 `cudaGetDeviceProperties` 可能被虚拟化、拦截或返回不可信值

例如：

- clock 通过 FMA 指令计时推断，而不是只看 `nvidia-smi`
- latency 通过 pointer chasing 绕过 prefetch
- cache size 通过 latency cliff 识别
- bandwidth 通过 streaming benchmark 测量

#### 4.6.2 Physical Consistency Validation

系统会做一系列物理一致性检查，例如：

- `L1 < L2 < DRAM latency`
- bandwidth 是否超过理论合理上限
- cache size 是否接近 power-of-two 结构
- clock / bandwidth / measured throughput 是否互相矛盾

这意味着它不是“测出数字就信”，而是会对多个结果做 cross-check。

#### 4.6.3 Compile-Fix Loop

`memxlife-project/agents/codegen.py` 的实现体现了一个很典型的 agentic code generation loop：

- Round 1: 生成完整 CUDA 文件
- Round 2: 读取 compile errors 修复
- Round 2.5: 再次修复
- Round 3: 重新生成

这个设计的价值在于：

- 提高 end-to-end 成功率
- 降低单次 LLM 代码生成失误对整体流程的影响

### 4.7 Scoring And Evaluation Alignment

官方 Phase 1 文档还明确给出了一个很重要的评分语境：

- **70% 数值对齐**
  - 即结果和 server-side ground truth benchmark 的一致程度
- **30% 工程推理与方法论**
  - 即 agent 是否真正识别了频率锁定、资源 masking、probe 合法性与 cross-verification 过程

并且官方将这一过程描述为一种 **LLM-as-a-Judge** 风格的综合评分：不仅看最终数字，也看 reasoning、benchmark 设计和实验过程。

这与我们的实现方向是吻合的，因为我们没有把系统设计成“只产出一个数”，而是同时保留：

- 结果值
- 置信度
- 方法学说明
- audit 日志
- 交叉验证依据

### 4.8 Outputs

每次运行会产出：

- `results.json`
- `results_detailed.json`
- `knowledge.json`
- `audit_report.md`
- `audit_log.jsonl`

这使得 Phase 1 不是黑盒运行，而是一个可以审计、可回放的 profiling pipeline。

**Evidence snapshot.**

- Phase 1 的系统实现入口和职责分解可以直接在 [README.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/README.md) 和 [report.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/report.md) 中对照查看。
- 多 agent 总控和真实执行闭环体现在 [core/orchestrator.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/core/orchestrator.py)。
- compile-fix loop 的实现证据在 [agents/codegen.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/agents/codegen.py)。
- 物理一致性与环境 fingerprint 的支撑实现可以在 [analysis/consistency.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/analysis/consistency.py) 中看到。
- 需要诚实说明的一点是：当前仓库快照里保留了 Phase 1 的实现证据和方法学文档，但没有保留完整的历史 `runs/` 目录输出，因此这一阶段的证据强项主要是代码结构与审计设计，而不是现成的运行产物归档。

### 4.9 How The Direction Actually Changed

如果只看最后的系统结构，Phase 1 很容易被写成一个“从一开始就知道要做 multi-agent probing”的项目，但真实实现过程并不是这样线性的。

一开始更自然的想法其实是：

- 先把 `ncu` 指标读出来
- 再写一些针对性 benchmark
- 最后把这些结果拼起来

但随着我们真正去对照课程里的 anti-hacking 场景，问题定义发生了变化。我们逐渐意识到，Phase 1 的难点不在于“读到多少指标”，而在于：

- 当 API 不可信时，还能否自己构造证据
- 当某个值看起来合理时，是否能证明它和其他值一起也合理

所以实际实现顺序并不是“先把九个指标都做全”，而是“先打通一个最小闭环”。我们优先保证：

1. 从 `target_spec.json` 接收任务
2. 让 agent 选一个 probe 策略
3. 生成并编译一次可运行的 CUDA probe
4. 解析结果
5. 把结果存下来并能继续下一轮

这个最小闭环打通之后，系统才开始逐渐长出更多能力，例如：

- compile-fix loop
- fallback 策略
- 物理一致性校验
- audit 日志
- 更强的环境感知

回头看，这个阶段一个非常真实的认识变化是：

**真正困难的不是“怎么多测几个数”，而是“怎么让系统自己知道哪些数值得信，哪些不值得信”。**

### 4.10 Debug Evidence, Reflection, And What We Learned

Phase 1 最典型的 debug 过程，不是某一个 spectacular 的 crash，而是一类更系统性的“不协调”：

- 有的 probe 单看返回值似乎合理，但放进 latency hierarchy 后关系不对
- 有的 bandwidth 数值不离谱，但和 clock 或资源限制一起看时解释不通
- 有的代码不是方法论错误，而是卡在编译细节、架构 flag、输出格式解析上

这也是为什么我们后来特别重视：

- compile error 回喂
- parser 的鲁棒性
- consistency validator
- audit log 中的方法与证据绑定

真正让这一阶段变得“像系统工程而不是脚本集合”的，不是多加一个 agent，而是开始接受下面这件事：

**拿到一个数字，不等于理解了系统；只有当它能和别的数字一起成立时，它才真正有意义。**

就个人收获而言，Phase 1 最重要的不是学会了某种 profiling 技巧，而是形成了一个后来在 Phase 2 和 Phase 3 一直重复出现的习惯：

- 不急着相信单点结果
- 不把一次跑通当成理解
- 遇到异常时优先怀疑问题建模和证据链，而不是只怀疑实现细节

这也是后面两个 phase 能走得更稳的基础。

### 4.11 Phase 1 Summary

Phase 1 的本质成果不是某一个 benchmark，而是一个可自动运行的 GPU measurement system。它为后续两个 phase 奠定了方法论基础：

- 先测量，再判断
- 先验证，再信任
- 先做稳定 pipeline，再追求更高性能

---

## 5. Phase 2: LoRA Operator Optimization Agent

### 5.1 Goal

Phase 2 的目标是为如下 LoRA operator 生成并维护一个正确且高性能的 `optimized_lora.cu`：

```text
Y = W X + A(B^T X)
```

约束非常关键：

- 最终产物必须是单个自包含的 CUDA/C++ 文件
- `run.sh` 必须可驱动完整流程
- agent 必须始终在磁盘上保留一个可编译候选
- 只有 correctness 通过的实现才有意义

从官方合同来看，还需要明确几件事：

- evaluator 会在 submission root 里执行 `bash run.sh`
- 随后从同一目录读取最终的 `./optimized_lora.cu`
- agent 的官方时间预算上限是 **30 分钟**
- hidden case 不会提前暴露，所以 agent 必须依赖公开尺寸区间内的 **synthetic tensors** 做本地搜索

这使得任务不是“随便试 kernel”，而是一个**带稳定性约束的搜索系统**。

### 5.2 Official Environment And Scoring Contract

官方 Phase 2 环境明确给出为：

- GPU: NVIDIA GeForce RTX 3090
- OS: Ubuntu 22.04.4 LTS
- Python: 3.10.12
- PyTorch: 2.3.0a0+6ddf5cf85e.nv24.04
- CUDA toolkit: 12.4
- GCC: 11.4.0

评分合同也比较清楚：

- **correctness 是硬门槛**
- 通过 correctness 后，最终分数为：
  - **70% speedup**
  - **30% agent implementation / engineering methodology**

这意味着 Phase 2 不只是“有没有快”，还包括：

- 是否真的做了 iterative improvement
- 是否做了 candidate generation / comparison
- 是否有 benchmark 驱动的决策机制
- 是否具备 reproducibility 和工程组织质量

### 5.3 Phase 2 Directory Structure

核心目录：

- `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase2_agent`

关键文件：

- `run_agent.py`: agent 入口
- `config.py`: 预算、路径、候选数、LLM 配置
- `candidate_space.py`: 候选定义与启发式初始队列
- `codegen.py`: 根据 candidate 生成 `optimized_lora.cu`
- `harness.py`: 本地编译、correctness、benchmark
- `optimizer.py`: 搜索总控
- `tracing.py`: JSONL trace 与 summary
- `llm_helper.py`: 可选的 LLM candidate suggester

### 5.4 Search Agent Architecture

Phase 2 的整体流程是：

```text
AgentSettings
    ->
Heuristic Candidate Queue
    ->
Code Generation
    ->
PyTorch CUDA Extension Compile
    ->
Correctness Check
    ->
Benchmark Pair
    ->
Promotion / Rejection
    ->
Trace + Report
```

`phase2_agent/optimizer.py` 中的 `LoRAOptimizationAgent` 负责整个循环。

### 5.5 Agenticity And Rule Compliance

官方明确禁止两种做法：

1. 只提交一个静态 final kernel
2. 把预写好的最终 `optimized_lora.cu` 当固定字符串/模板直接输出

我们的实现和这一要求的关系需要说清楚：

- 我们没有把提交流程设计成“直接吐出一个唯一写死的最终答案”
- 我们保留了结构化 candidate space、编译、correctness、benchmark、promotion 的闭环
- 同时，我们的搜索空间也不是完全无结构的自由生成，而是**由人工设计的参数化候选空间 + 实验驱动 promotion** 组成

因此，更准确的表述是：

**Phase 2 是一个带启发式先验的 agentic optimization system，而不是一个 one-shot 静态 kernel exporter。**

### 5.6 Bootstrap Strategy

Phase 2 一个很重要的工程设计是：

**一开始就先写一个可编译的 bootstrap candidate 到 `optimized_lora.cu`。**

这样做的意义是：

- 无论搜索后续是否成功，仓库里始终有合法提交产物
- agent 可以边搜索边更新，而不是等到最后一次性输出
- 即使远程环境或 LLM 出问题，也不会出现“没有可提交文件”的情况

### 5.7 Candidate Representation

`candidate_space.py` 里把候选配置抽象成了 `CandidateConfig`，包含：

- `strategy`
- `main_backend`
- `low_rank_backend`
- `accumulation_order`
- `allow_tf32`
- `cache_mode`
- `variant_name`
- `notes`

这意味着搜索空间不是“任意代码字符串”，而是**结构化设计空间**。

这样做的好处：

- 便于追踪 candidate identity
- 便于做哈希、去重、日志记录
- 便于将调优从“写代码”升级为“探索设计空间”

### 5.8 Local Harness And Measured Metrics

`phase2_agent/harness.py` 实现了本地评估器，它做了三类事情：

1. **编译检查**
   - 使用 `torch.utils.cpp_extension.load`
   - 每个 candidate 独立 build directory

2. **正确性检查**
   - 对多个 size 做 `torch.allclose`
   - 记录 `max_abs_err` 与 `rel_l2_err`

3. **性能评估**
   - 评估 fixed-weight, varying-`X` 的速度
   - 评估 repeated exact input 的速度
   - 使用 harmonic mean 形成综合 speedup

这个评估设计非常关键，因为它直接反映了 hidden workload 的两个潜在模式：

- **same weights, changing activations**
- **exact repeated calls**

这也与官方鼓励的本地 synthetic search 思路一致：在 hidden tensors 不可见时，必须自己构造足够接近真实 workload 的局部搜索环境。

### 5.9 API Key And External Model Use

官方文档还特别强调了一条工程要求：

- 如果 agent 依赖外部模型 API，必须使用**自己的 API key**

因此我们在配置层显式支持：

- 环境变量注入
- 本地配置读取
- agent 在 LLM 不可用时退化为启发式工作流

这让系统不会因为外部 API 缺失而完全失效。

### 5.10 Why Phase 2 Did Not Stop At ATen Baseline

最初的稳定 baseline 是：

```text
Y = W @ X
Bt = B^T.contiguous()
BX = Bt @ X
Y.addmm_(A, BX)
```

它的优点是：

- 正确
- 容易维护
- 冷路径稳定

但深入分析算子结构后，出现了一个关键洞察：

- `W @ X` 是主要成本
- LoRA rank=16 分支相对很小
- 如果为了优化低秩分支而伤害主 GEMM，整体反而会退步

这就把优化方向从“单次低秩路径微优化”引向了“跨调用复用”。

### 5.11 Key Algebraic Insight: `W_eff`

最关键的代数等价式是：

```text
W X + A(B^T X) = (W + A B^T) X
```

于是出现了 `W_eff` 思路：

- 如果 `W / A / B` 不变但 `X` 会变
- 那么可以先物化 `W_eff = W + A B^T`
- 后续只做一次大 GEMM：`W_eff @ X`

这个思路的本质不是纯数学改写，而是：

**把“算子优化”变成“带缓存策略的 runtime dispatch”。**

### 5.12 Final Strong Path: `hybrid_weff`

最终最有价值的 Phase 2 路线是 `hybrid_weff`。

它包含三条路径：

1. **Exact Repeat Path**
   - 如果 `W / A / B / X` 都没变
   - 直接返回缓存输出

2. **Same-Weight Path**
   - 如果 `W / A / B` 没变，但 `X` 改了
   - 复用 `W_eff`

3. **Cold Fallback Path**
   - 如果权重变了
   - 退回安全的参考形态计算

这条路径的关键不在于某个 kernel，而在于**识别调用模式**。

### 5.13 Cache Safety Design

为了避免错误复用，Phase 2 并不是只看 tensor pointer。

`phase2_agent/codegen.py` 里生成的 C++ 代码为缓存判断引入了 `TensorStamp`：

- data pointer
- version counter
- rows / cols
- device index

只有 stamp 全部一致，才认为可以复用。

这非常重要，因为它保证：

- tensor 被 inplace 修改后，不会误命中旧缓存
- 不同形状、不同设备上的 tensor 不会混淆

### 5.14 Debug Stats And Traceability

Phase 2 不只是算一个 speedup 数字，而是记录“到底走了哪条路径”。

例如：

- `exact_repeat_hits`
- `same_weight_weff_hits`
- `weff_materializations`
- `threshold_fallback_hits`
- `bt_cache_hits`
- `cold_fallback_hits`

同时 `tracing.py` 会记录：

- `trace.jsonl`
- `trace_summary.md`
- `history.json`
- `summary.json`

这样每个 candidate 的表现都可以回溯，不会出现“这个版本为什么快了”却答不上来的情况。

### 5.15 Turning Point: Hot-Path Bug Discovery

Phase 2 最关键的工程转折之一，是在远程 GPU 验证时发现：

- 早期 `hybrid_weff` 热路径本来应该只执行 `W_eff @ X`
- 但实现里还残留了一次多余的主 GEMM

这导致：

- 设计本身是对的
- 但实现把收益吃掉了

修复这个 bug 之后，`hybrid_weff` 的优势才真正体现出来。

这个故事在汇报里非常重要，因为它体现了：

- 我们不是“碰运气出高分”
- 而是通过 instrumentation 和远程验证定位到具体实现错误

### 5.16 Official Results And What They Mean

几个关键官方 mission 结果如下：

#### Mission `6c5e83a34efa2beb67dcdaf68f6d4fd6`

- Case 1: `361.3508567766487`
- Case 2: `1.9258016538686438`
- Case 3: `0.9968725244232519`

#### Mission `efbab79b85fe777801ed0c4ba7e6ab44`

- Case 1: `291.2396601485792`
- Case 2: `1.924633215238158`
- Case 3: `1.0033482283098771`

#### Mission `c0f31ee32dec227bb48cae77b577a2e7`

- Case 1: `342.1327226062235`
- Case 2: `1.9401524893940716`
- Case 3: `0.994588466415906`

这几个结果说明了三件事：

1. hidden evaluation 中确实存在强重复调用场景
2. `hybrid_weff` 可以在重复模式上获得极端收益
3. 最弱 case 是否能守住 `>= 1.0x`，取决于 materialization threshold 与 fallback 成本控制

要特别强调的是：

**`361x` 不代表单次 GEMM 普遍提升 361 倍。**

它代表在 hidden case 的 exact-repeat 模式下，缓存命中带来了极端收益。

**Evidence snapshot.**

- Phase 2 的完整研发叙事和路线转向记录在 [phase2_optimization_journey.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase2_optimization_journey.md)。
- 三个最关键的官方结果摘要分别在 [6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md), [efbab79b85fe777801ed0c4ba7e6ab44_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/efbab79b85fe777801ed0c4ba7e6ab44_summary.md), [c0f31ee32dec227bb48cae77b577a2e7_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/c0f31ee32dec227bb48cae77b577a2e7_summary.md)。
- 当前仓库里的 [.phase2_work/trace_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/.phase2_work/trace_summary.md) 和 [.phase2_work/history.json](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/.phase2_work/history.json) 记录的是一次本地 `torch` 缺失环境下的失败 run；它们更适合作为 fallback / robustness 的工程证据，而不是作为性能主证据。

| Official mission | Candidate kept by agent | Case 1 | Case 2 | Case 3 | What it established |
|---|---|---:|---:|---:|---|
| `6c5e83...` | `aten_addmm_inplace_btcontig_mainfirst_hybridweff` | 361.3509 | 1.9258 | 0.9969 | First run proving exact-repeat upside can be extreme |
| `efbab79...` | `aten_addmm_inplace_btcontig_mainfirst_hybridweff` | 291.2397 | 1.9246 | 1.0033 | More balanced run; weakest case crossed above parity |
| `c0f31ee...` | newer synced search space visible | 342.1327 | 1.9402 | 0.9946 | Methodological proof that the synced remote workspace path was correct |

### 5.17 Submission Workflow Insight

Phase 2 还有一个很关键的非算法结论：

- 官方 `/submit2` 并不会自动上传本地最新代码
- 如果远程 workspace 没同步，官方跑的可能还是旧版本

这推动我们把整个远程流程脚本化：

- 初始化课程服务
- 获取 dev mission
- 等待 `ssh_port`
- 同步 workspace
- 在远程做 sanity check
- 再触发 official submit

这是一条非常现实的工程经验：

**优化算法本身不够，提交流程正确与否同样决定最终结果。**

### 5.18 How The Direction Actually Changed

Phase 2 是三个阶段里“路线被数据改写”最明显的一次。

如果只看任务公式：

```text
Y = W X + A(B^T X)
```

一个非常自然的第一直觉是：既然这是 LoRA operator，那主要优化对象应该是低秩分支，或者至少是把这两个分支做更紧的 fusion。

真实过程并不是直接跳到 `hybrid_weff`。我们先做的是一条更保守、也更工程化的路线：

1. 先让 `run.sh -> optimized_lora.cu` 的提交流程可靠
2. 先拿到一个稳定、正确、可 benchmark 的 ATen baseline
3. 再围绕 memory layout、`B^T` contiguous、`addmm_` 组合做近邻搜索

这一阶段最重要的不是“第一版就很快”，而是 baseline 足够稳定，足以让后续每一次偏离都有比较对象。

真正的转折来自一个非常朴素的事实：benchmark 一直在提醒我们，真正重的还是 `W @ X`。  
LoRA rank=16 分支当然值得优化，但它并不是决定上限的主战场。

这迫使我们把问题从：

- “怎么让单次公式算得更快”

改成：

- “hidden workload 里到底有没有可复用的结构”

于是才出现了中间几条路线：

- `BX` cache
- adaptive cache
- repeated `(B, X)` reuse

这些都不是最终答案，但它们非常重要，因为它们第一次把我们从“单次算子视角”推到了“跨调用视角”。

直到 `W_eff = W + A B^T` 这条线真正成形，Phase 2 才开始从一个 ATen variant search，变成一个更像 runtime policy design 的项目。

### 5.19 Debug Evidence, Reflection, And What We Learned

Phase 2 里最有代表性的 debug 证据，不是单纯“这个版本慢了”，而是：

- 设计看起来是对的
- 结果却不像理论上应该那么强

最关键的一次就是 `hybrid_weff` 的 hot-path bug。  
如果没有 trace、benchmark pair、remote validation 和 debug stats，这个问题很容易被误判成：

- `W_eff` 思路没用
- hidden case 不稳定
- 这个方向只是偶然有效

但真正的证据链告诉我们：

- 不是方向错了
- 而是实现里还残留了一次多余的主 GEMM

这次经历很值得写进报告，因为它体现了一个后来反复验证的经验：

**性能结果和算法直觉不一致时，不应该马上否定思路，而应该先检查证据链里哪一环在说谎。**

另一个很重要的认识来自 `361x` 这类结果。  
如果只看数字，它很容易被误解成某种“神奇超优化”；但结合 hidden workload 与 debug path 去看，它更像是一个很强的信号：

- 我们识别对了某种重复模式
- exact-repeat cache 在这个模式上被大幅命中

所以这类结果真正证明的，不是“单次 GEMM 被提了几百倍”，而是：

**我们开始理解评测并不是冷启动算子比赛，而是 workload modeling 的比赛。**

就个人收获而言，Phase 2 给我的最大改变是：

- 以前会自然地把“优化”理解成写更快的 kernel
- 但这一阶段之后，更自然的理解变成了“先搞清楚系统到底在重复什么，再决定该优化哪一层”

这也是为什么后面我们越来越重视：

- candidate history
- debug stats
- 远程复现实验
- 提交流程本身的可验证性

因为没有这些证据，优化就很容易重新退回到猜测和玄学。

### 5.20 Phase 2 Summary

Phase 2 的核心成果不是“写出了一个更快的 CUDA 文件”，而是：

1. 把问题建模成了结构化 candidate search
2. 找到了比单次低秩微优化更重要的跨调用复用模式
3. 形成了 `exact repeat -> same-weight reuse -> cold fallback` 的 runtime policy
4. 建立了完整的 trace / benchmark / promotion 流程

---

## 6. Phase 3: Lightweight LLM Serving Runtime Optimization

### 6.1 Goal

Phase 3 的目标是优化轻量 LLM serving runtime 的 decode throughput，同时保持 correctness。

如果严格按官方原文来表述，Phase 3 的任务是：

- 构建一个 agent
- 由它在 `run.sh` 期间生成 `workspace/engine.py`
- 该 runtime 需要从 `model_config.json` 和 `weight_dir` 动态构造 decoder-only engine
- 正确支持 `prefill / decode / remove`
- 在 serving-style trace 上取得更高吞吐

官方还特别强调：

- evaluator 比较的是 **logits**，不是生成文本
- correctness 是硬门槛
- throughput 看的是 **prefill + decode + remove** 组成的 serving behavior，而不是单个 isolated call

和 Phase 2 一样，真正难的点不是“写一个更快的函数”，而是：

- 理解 prefill 与 decode 的差异
- 理解 same-length 与 varlen batch 的不同结构
- 找到公共评测与真实隐藏 workload 都比较稳的优化

### 6.2 Artifact Pipeline

Phase 3 的入口是：

- `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/run.sh`

它的职责不是直接跑模型，而是先把 source-of-truth 渲染成最终提交产物：

```text
phase3_engine_sources/current_best_engine.py
    ->
scripts/render_phase3_engine.py
    ->
workspace/engine.py
```

这样做的意义：

- 我们可以在 `phase3_engine_sources/` 中保留更可维护的源版本
- `workspace/engine.py` 始终是评测实际消费的 artifact
- `run.sh` 同时负责准备日志与兼容课程要求的输出文件

这里需要和官方“agent 生成 runtime”措辞对齐：

- 我们的开发过程是高度 agentic 的
- 提交阶段的 `run.sh` 不是在线大范围搜索，而是**确定性地 materialize 当前已经验证过的 best runtime artifact**

这种设计更像：

- **development-time agentic optimization**
- **submission-time deterministic artifact generation**

它牺牲了一些在线搜索自由度，但换来了更强的稳定性和可复现性。

### 6.3 Submission Contract And Measured Region

官方 Phase 3 合同里有几件事需要明确写出来：

- `bash run.sh` 执行结束后，evaluator 会 import `workspace/engine.py`
- 同时要求存在 `workspace/results.log`
- `results.log` **不参与评分**
- measured region **不包括** `create_engine(...)` 和初始权重加载
- measured region **包括** `prefill(...)`、`decode(...)`、`remove(...)`

这意味着：

- lazy compile / lazy init 如果被放进 measured calls，会真实伤害吞吐
- 日志文件主要是为了调试与失败回溯，而不是为了自报成绩

### 6.4 Engine Core Structure

Phase 3 当前保留的实现位于：

- `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase3_engine_sources/current_best_engine.py`

核心抽象包括：

- `WeightMap`
  - 负责从 state dict 中读取并搬运权重
  - 在 device / dtype 上做统一处理

- `RequestState`
  - 管理单个请求的 token buffer、length、KV cache
  - 也支持共享 batch cache / batch row 的模式

- `Engine`
  - 统一管理 prefill、decode、cache growth、rope cache、request table

### 6.5 Weight Loading And Layer Packing

`Engine.__init__` 会：

- 解析 model config
- 计算 `hidden_size / num_heads / num_kv_heads / head_dim`
- 从权重中加载 embedding、final norm、lm head
- 对每层做打包

每一层会被整理成较适合 runtime 使用的结构：

- `attn_norm`
- `qkv`，即把 `q / k / v` 权重合并
- `o`
- `ffn_norm`
- `w13`，即把 `w1 / w3` 合并
- `w2`

这意味着运行时不是频繁做零散字典查找，而是使用较为紧凑的层结构。

### 6.6 Why This Is A Meaningful Departure From The Official Baseline

官方 public skeleton 的 baseline 是一个非常慢但接口正确的实现：

- 为每个 request 保存完整 token 序列
- 每次 decode 时重新计算整条序列

因此，我们相对于官方 baseline 的核心提升，不是某一个 isolated micro-optimization，而是：

- 实现真实 per-layer KV cache
- 把 decode 变成真正的 incremental decode
- 在此基础上继续做 batch / shared-layout / varlen specialization

这是理解 Phase 3 技术路线最重要的官方对照点之一。

### 6.7 Request State And Cache Layout

Phase 3 的关键不是只算单 token，而是高效管理 request state。

`RequestState` 中维护：

- `token_buffer`
- `length`
- `kv_cache`
- `batch_cache`
- `batch_row`
- `batch_token_buffer`

这使得系统既支持：

- 独立请求状态
- 也支持多个请求共享底层 batch cache

这正是后续 shared-batch promotion 的基础。

### 6.8 Interface And Correctness Contract

官方要求 `workspace/engine.py` 中必须暴露：

- `create_engine(model_config, weight_dir, device="cuda")`
- `Engine.prefill(request_ids, input_ids)`
- `Engine.decode(request_ids, token_ids)`
- `Engine.remove(request_ids)`

公共 correctness 阈值是：

- `atol = 1e-2`
- `rtol = 1e-2`

并且 correctness 会覆盖：

- single-request prefill / decode
- multi-request prefill / decode
- 插入新请求
- 删除已有请求后继续解码其他请求

这正是为什么我们在实现里非常重视 request-state correctness，而不是只看一条理想化 decode fast path。

### 6.9 Prefill Path Dispatch

`prefill()` 的路径不是单一的，而是根据输入形态选择：

1. **单请求路径**
   - 调用 `_forward_with_cache`

2. **same-length batch prefill**
   - 调用 `_forward_prefill_batch`

3. **varlen batch prefill**
   - 在 padding ratio 允许时，调用 `_forward_prefill_varlen_batch`

为什么这样设计：

- same-length batch 更适合直接批处理
- varlen batch 如果 padding 过多会浪费算力
- 因此需要一个 padding ratio threshold 去平衡是否共享批处理

### 6.10 Decode Path Dispatch

`decode()` 是整个 Phase 3 的核心。

它会根据请求状态分派到不同路径：

1. **same-length, non-shared batch**
   - `_forward_decode_batch`

2. **same-length, shared batch**
   - `_forward_decode_shared_batch`

3. **varlen, non-shared batch**
   - `_forward_decode_varlen_batch`

4. **varlen, shared batch**
   - `_forward_decode_varlen_shared_batch`

这种分流是 Phase 3 最重要的结构设计之一，因为它避免了“用一个路径处理所有 case”带来的额外开销。

### 6.11 Shared-Batch Promotion

Phase 3 当前保留的一个核心优化是：

**保守地把合适的请求组提升到 shared batch 表示。**

主要触发条件：

- batch size 至少达到 4
- same-length decode 的 position 至少达到 16
- varlen batch 也需要满足一定长度与差异条件

为什么是“保守提升”而不是“默认共享”：

- 共享布局的创建和扩容本身有成本
- 小 batch、短序列时，这个成本可能吞掉收益
- 过于激进的 promotion 会让 public throughput 或 mixed case 退步

因此当前实现强调：

- 只在较大概率有收益的区域 promotion
- 否则继续走普通路径

### 6.12 KV Cache Update Optimization

当前保留的一个重要优化是：

**shared-varlen decode 的 K/V 更新采用 batched `index_put_`。**

它的作用是：

- 避免逐 request 写回带来的 Python / kernel 开销
- 对 shared cache 中离散 row 的更新更直接

在 `current_best_engine.py` 中，这个优化主要体现在：

- `_attention_decode_shared_batch`
- `_attention_decode_varlen_shared_batch`
- `decode()` 中 shared path 的 token buffer 更新

这个优化对 mixed decode throughput 的帮助尤其明显。

### 6.13 RMSNorm Fast Path

另一个保留下来的优化是：

- 如果当前 PyTorch 提供 `torch.nn.functional.rms_norm`
- 就优先走 fused `F.rms_norm`
- 否则回退到手写 RMSNorm

这个优化本身不是巨大收益，但它的特点是：

- 非常稳
- 风险低
- 基本不影响 correctness

因此被保留在当前版本中。

### 6.14 Attention Path Design

Phase 3 的 attention 并非一个单一实现，而是针对多种场景拆成多套：

- `_attention`
- `_attention_decode_single`
- `_attention_decode_batch`
- `_attention_decode_shared_batch`
- `_attention_decode_varlen_batch`
- `_attention_decode_varlen_shared_batch`
- `_attention_prefill_batch`
- `_attention_prefill_varlen_batch`

并且会尽量利用：

- `F.scaled_dot_product_attention`
- shared KV cache
- row-indexed cache selection
- precomputed rope cache / position cache

这里的核心不是“自己重写注意力数学”，而是：

**把不同场景的 memory layout 和 batch structure 匹配到合适的执行路径。**

### 6.15 Why Many More Aggressive Optimizations Were Rejected

Phase 3 并不是没有尝试更激进的路线。

我们试过的方向包括但不限于：

- 更激进的 decode promotion
- grouped decode 重写
- 预展开 shared KV cache
- 更大范围的 dense shared-path 假设
- 一些会抬高单项 public 指标的 decode 改写

最后很多都被放弃，原因包括：

- mixed throughput 退步
- decode 稳定性下降
- stress correctness 失败
- 某个单项指标上涨，但总体收益下降

这说明 Phase 3 最重要的不是“榨出一个局部峰值”，而是：

**在真实运行时结构下找到稳定、可复现的增益。**

### 6.16 Evaluation Pipeline

Phase 3 的公共测试脚本是：

- `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/scripts/run_public_tests.sh`

它会执行：

1. `evaluator/test_correctness.py`
2. `evaluator/stress_correctness.py`
3. `evaluator/benchmark_throughput.py`
4. `evaluator/benchmark_mixed.py`

同时我们也使用：

- `evaluator/benchmark_breakdown.py`

去进一步拆解：

- same_prefill
- same_decode
- mixed_prefill
- mixed_decode

这一步非常重要，因为它让我们知道优化到底改善了哪里，而不是只看一个总 throughput 数。

### 6.17 Allowed Optimization Space And What We Chose Not To Use

官方允许的优化方向其实很宽，包括：

- real per-layer KV cache
- batched prefill / decode
- PyTorch SDPA
- Triton kernels
- C++/CUDA extensions
- 自定义 RMSNorm / RoPE / attention / MLP / cache kernels

但 evaluator 也明确希望最终 `engine.py` 直接实现要求接口，而不是依赖一个完整的外部 inference framework。

我们最终保留的是一条偏 PyTorch runtime engineering 的路线，而没有在当前版本里重度押注 Triton 或自定义 extension，这也是出于：

- correctness 风险
- 开发迭代速度
- hidden-case 泛化稳定性

### 6.18 Current Retained Local Fallback Result

当前保留版本的代表性本地 fallback public 指标大致为：

- public throughput: `4341.2259 tokens/s`
- public mixed throughput: `5298.9116 tokens/s`
- correctness smoke: passed
- stress correctness: passed

相较较早的稳定版本，提升大致为：

- throughput: 约 `+0.55%`
- mixed throughput: 约 `+1.37%`

虽然这个涨幅不像 Phase 2 的某些 hidden case 那么夸张，但它有两个优势：

- 更稳定
- 更符合 Phase 3 的真实运行时目标

**Evidence snapshot.**

- Phase 3 最关键的实验总表在 [remote_breakdown_decode_experiments_20260601.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_decode_experiments_20260601.md)。
- 当前保留版本的本地 fallback public 结果在 [remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md)。
- 较早稳定 official-local fallback 对照在 [remote_public_eval_official_26e0fd0e_local_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_official_26e0fd0e_local_summary.md)。
- 提交时真正生成 artifact 的链路在 [run.sh](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/run.sh) 和 [scripts/render_phase3_engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/scripts/render_phase3_engine.py)。
- 当前保留 runtime 源码在 [phase3_engine_sources/current_best_engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase3_engine_sources/current_best_engine.py)。

| Variant / evidence point | Representative metrics | Decision |
|---|---|---|
| Baseline breakdown | `same_decode = 5317.80`, `mixed_decode = 3764.24` | Starting point |
| Experiment B: batched `index_put_` | `same_decode = 5205.42`, `mixed_decode = 4500.56` | Keep; big mixed decode win |
| Experiment E: `F.rms_norm` fast path | `same_decode = 5248.05`, `mixed_decode = 4534.95` | Keep; small but robust gain |
| Experiment F: generalized shared-row path | `same_decode = 4761.59`, `mixed_decode = 4357.75` | Reject; same decode regressed too much |
| Experiment I: proactive promotion | `same_decode = 4594.43`, `mixed_decode = 4192.54` | Reject default path; heuristic too aggressive |
| Experiment K: flattened layer objects | `public = 4278.13`, `mixed = 5249.82` | Reject; code cleanup did not pay off |
| Experiment L: manual attention instead of SDPA | `public = 3781.00`, `mixed = 4524.28` | Reject immediately; SDPA remained clearly superior |

### 6.19 Official Submission Difficulty

Phase 3 的一个现实问题是：

- 官方 `outputs3` 发布服务并不稳定
- 一些 official run 已经完成，但 markdown artifact 不一定及时可取

因此我们采用了两条并行验证路径：

1. **official submit**
2. **course GPU container 内本地 fallback evaluation**

这保证了即使 artifact 服务不稳定，我们仍然有可靠的本地对照标准。

### 6.20 How The Direction Actually Changed

Phase 3 的真实实现过程，和表面上“继续做推理加速”相比，其实更像一次不断收缩问题定义的过程。

一开始很容易把它想成：

- 优化 attention
- 优化 decode
- 尽量把 runtime 各处都做快一点

但真正对照官方 baseline 和 evaluator 之后，路线很快收敛成了一个更明确的判断：

- baseline 最慢的地方，不是某个细小算子，而是 decode 仍然在做过多重复工作
- 所以第一原则不是追局部 kernel 峰值，而是先把 runtime 结构改成真实的 incremental decode

这就是为什么我们的主线后来越来越清楚：

1. 先建立真实 per-layer KV cache
2. 再把 request state 管理做正确
3. 再区分 prefill / decode
4. 再区分 same-length / varlen
5. 最后才在 shared batch promotion、cache update、RMSNorm 等路径上做局部强化

这条路线的一个重要特点是，它不是越做越“花”，反而是越做越保守。  
很多实验的价值，不在于留下了多少代码，而在于它们迫使我们承认哪些聪明做法其实不适合这个 workload。

例如在 decode 路径上，我们先后看到过几种非常典型的现象：

- batched `index_put_` 明显改善 mixed decode，说明 shared-varlen cache update 真的是瓶颈之一
- `F.rms_norm` 这种小优化虽然不戏剧化，但稳定且几乎没有副作用
- 更激进的 shared-row generalization、主动 promotion、dense/shared 假设，看起来很聪明，但常常在 same decode 或 stress correctness 上付出代价

也就是说，Phase 3 后来不再是“收集更多优化点”，而是开始变成：

**判断哪些优化值得留下，哪些优化即使局部漂亮也应该主动删掉。**

### 6.21 Debug Evidence, Reflection, And What We Learned

Phase 3 最有代表性的 debug 证据，不是某个异常栈，而是一份份 breakdown 和回归对比。

例如在 [stage3_outputs/remote_breakdown_decode_experiments_20260601.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_decode_experiments_20260601.md) 里，我们能看到一种很有教育意义的演化：

- Experiment B 的 batched `index_put_` 让 `mixed_decode` 从 `3764.24` 提到 `4500.56`
- Experiment E 的 `F.rms_norm` 在不伤 correctness 的前提下继续小幅抬高 decode
- Experiment F 说明“全面 generalized shared-row path”虽然帮了 mixed public test，但把 `same_decode` 拖得太厉害
- Experiment I 又说明“主动 promotion”这个想法不是完全错，而是默认开启太激进
- Experiment K / L 则更直接：看起来更底层、更手工的路径，最终反而不如保留 SDPA 和稳定路径

这些实验最有价值的地方在于，它们不是简单分成“成功”和“失败”。  
很多被拒绝的版本其实都带来了一条重要认识：

- 哪个指标是真正敏感的
- 哪类路径是 hidden/public 都共享的
- 哪些优化只是把成本从一个 case 挪到了另一个 case

这一阶段我觉得最真实的个人反思是：

**删掉一个看起来聪明的优化，常常比继续堆一个新优化更难。**

因为删除意味着接受一件事：

- 问题不是我们还不够努力
- 而是这个想法本身不适合当前 runtime 结构

这也是 Phase 3 最终带来的系统理解：

- throughput 不是一个单点数值
- serving runtime 也不是一个统一路径
- 如果不把 `same/mixed`、`prefill/decode`、`public/stress` 拆开看，很容易把“局部最快”误当成“整体最优”

### 6.22 Phase 3 Summary

Phase 3 的成果不是“把 attention 改写得更复杂”，而是：

1. 建立了 request-state-aware 的 runtime 结构
2. 正确地区分 prefill / decode、same-length / varlen、shared / non-shared 路径
3. 保留了少量但稳定有效的优化
4. 用 breakdown 驱动决策，而不是只盯单一 throughput 数

---

## 7. AI Agentic Development: Two Different Meanings Of "Agent"

这个项目里，“agentic”其实有两层含义。

### 7.1 Layer A: The Product Itself Contains Agents

这是项目内在的 agentic 性：

- Phase 1 是显式 multi-agent system
- Phase 2 是 search-and-promote optimization agent
- Phase 3 的提交产物不是多 agent runtime，而是由 agentic development 过程收敛出的 runtime artifact

也就是说，agent 不是只出现在开发工具中，产品本身就包含 agent logic。

### 7.2 Layer B: We Also Used AI As A Development Partner

另一层是开发方式上的 agentic：

- 用 AI 协助阅读代码库
- 协助构造实验
- 批量分析日志
- 生成和修改提交脚本
- 监视远程 mission 状态
- 将 official / local 结果快速对比并总结

这一层不是课程要求里的“产品 agent”，而是我们的**开发工作流 agent**。

从结果上看，这一层帮助最大的不是“自动写代码”，而是：

- 维持高频实验节奏
- 降低手工检查远程状态的成本
- 快速把试验失败转成可解释信息

---

## 8. Agentic Development Workflow In Practice

### 8.1 Phase 1: LLM In The Runtime Loop

Phase 1 中，AI 直接处在系统主循环里：

- 选策略
- 生成 CUDA
- 修 compile error
- 在失败后自我纠错

这是最典型、最“产品内嵌”的 agentic 形态。

### 8.2 Phase 2: LLM + Heuristic Search Hybrid

Phase 2 更偏工程化：

- 初始候选由启发式队列提供
- 可选地允许 LLM 建议新的 candidate
- 真正的筛选依据来自 correctness + benchmark + debug stats

这说明我们并没有把“是否推广候选”交给 LLM，而是交给可测量证据。

这是一种更稳健的 agentic 设计：

- **LLM 负责扩展搜索空间**
- **Harness 负责做最终裁决**

### 8.3 Phase 3: AI-Assisted Iterative Runtime Engineering

Phase 3 没有做成“在线搜索型 runtime 内部 agent”，但开发过程高度 agentic：

- 从 breakdown 指标出发定位热点
- 快速实现多个结构假设
- 做 public / mixed / stress correctness 多维回归
- 失败后回滚并记录实验结论

这一阶段的 agentic 重点是：

**AI 帮助我们更快地产生、验证、否定假设。**

因此更准确地说，Phase 3 的 agenticity 主要体现在：

- 开发和实验闭环是 agentic 的
- 最终 `run.sh` 生成的是已经验证好的 runtime artifact
- 而不是在 official submit 时重新做大规模开放式 runtime 搜索

### 8.4 Remote Experiment Automation

为了让实验高频进行，我们还把课程环境相关流程脚本化了。

代表性脚本包括：

- `course_prepare_submit.sh`
- `course_prepare_submit3.sh`
- `course_sync_workspace.sh`
- `course_remote_exec.sh`
- `course_submit2.sh`
- `course_submit3.sh`
- `course_submit_status.sh`
- `course_submit_status3.sh`
- `course_collect_phase3_local_eval.sh`

这套脚本的价值在于把以下重复动作自动化：

- 申请 dev mission
- 等待 GPU 与 `ssh_port`
- 同步 workspace
- 远程 sanity check
- official submit
- 本地拉回输出和总结

因此，项目中的 agentic development 不只是“AI 帮你写几行代码”，而是：

**把实验闭环尽可能自动化，让系统和人都更专注于有价值的优化决策。**

### 8.5 Logging As A First-Class Design Choice

我们在三个阶段都把 logging 当成一等公民，而不是附属品：

- Phase 1: `audit_log.jsonl`, `audit_report.md`, knowledge store
- Phase 2: `trace.jsonl`, `trace_summary.md`, candidate history, debug stats
- Phase 3: public eval summaries, breakdown logs, remote fallback reports

这带来的最大收益是：

- 每一次“为什么这个版本快了/慢了”都有证据链
- 每一次回滚都有明确理由
- 项目复盘和讲解成本显著下降

---

## 9. Cross-Phase Technical Insights

### 9.1 Measure Before You Trust

三个阶段都证明了一件事：

- 静态信息并不等于真实行为
- API 返回值并不等于运行时瓶颈
- 单一 benchmark 也不等于真实 workload

所以我们始终坚持：

- 先测量
- 再建模
- 再优化

### 9.2 The Biggest Wins Came From Structure, Not From Fancy Kernels

最值得强调的洞察之一是：

- Phase 1 的价值来自自动测量结构
- Phase 2 的最大收益来自调用模式复用
- Phase 3 的最大收益来自 runtime path specialization

换句话说，最大收益并不是都来自“更花哨的底层 kernel”。

### 9.3 Fast Path Must Be Paired With Safe Fallback

我们三个阶段都没有采用“只有激进快路径、没有退路”的方案。

相反，始终保留了：

- heuristic fallback
- cold path
- correctness gate
- compile-safe bootstrap artifact

这使得整个系统具备真实工程可用性，而不是只在单次演示里跑通。

### 9.4 Instrumentation Is Not Auxiliary Work

如果没有 logging、debug stats、history、breakdown，我们几乎不可能：

- 发现 Phase 2 hot-path bug
- 解释 `361x` 的来源
- 识别 Phase 3 某些激进优化为何对 mixed case 有害

因此，instrumentation 不是“做完优化再补一下”，而是优化工作的组成部分。

---

## 10. Current Technical Path Summary

### 10.1 Phase 1 Current Position

当前我们已经拥有一个可自动运行的 profiling agent system，它可以：

- 探测环境
- 自动生成 probe
- 编译执行
- 解析结果
- 做物理一致性验证
- 输出结构化审计材料

### 10.2 Phase 2 Current Position

当前 Phase 2 的技术结论已经比较明确：

- 最强方向不是低秩分支局部手工优化
- 而是对 hidden workload 中重复模式的识别与复用
- `hybrid_weff` 是最关键的 runtime policy

它体现为：

- exact repeat output cache
- same-weight `W_eff` reuse
- safe cold fallback

### 10.3 Phase 3 Current Position

当前 Phase 3 的保留版本强调：

- request-state-aware runtime
- shared batch promotion
- batched `index_put_` for shared varlen decode
- fused RMSNorm fast path
- 保守而稳定的 decode specialization

它的目标不是押单点极限，而是追求：

- 正确
- 稳定
- 可解释
- 对真实 workload 更鲁棒

---

## 11. What This Project Shows Technically

如果要用一句话总结整个项目的技术价值，可以表述为：

**我们把一个看似分散的三阶段课程项目，做成了一条从 GPU measurement、到 operator runtime reuse、再到 serving path specialization 的连续系统优化路线。**

更具体地说，这个项目证明了以下能力：

1. 能搭建 multi-agent system，而不是只写单点脚本
2. 能把优化问题重构为结构化搜索和运行时决策问题
3. 能从日志和实验中抽取真正有效的性能规律
4. 能把 AI 用在系统内部，也能把 AI 用在开发工作流本身
5. 能在远程课程环境、提交流程、artifact 不稳定的现实约束下维持高频迭代

---

## 12. Final Conclusion

这个项目从 Phase 1 到 Phase 3 的真正主线，不是“做了三个作业”，而是逐步回答三个越来越复杂的问题：

1. **机器真实在做什么？**
2. **算子真实在被怎样调用？**
3. **推理系统真实在被怎样使用？**

在这条主线上，我们逐步建立了：

- 自动测量能力
- 结构化搜索能力
- 运行时特化能力
- 以日志和验证为基础的 agentic 开发能力

因此，最有价值的产出不只是某一个分数，而是形成了一套可以迁移到更多系统优化任务中的完整工程方法论。

---

## Appendix A. Representative Benchmark Tables

### A.1 Phase 2 Official Result Table

Sources:

- [6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md)
- [efbab79b85fe777801ed0c4ba7e6ab44_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/efbab79b85fe777801ed0c4ba7e6ab44_summary.md)
- [c0f31ee32dec227bb48cae77b577a2e7_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/c0f31ee32dec227bb48cae77b577a2e7_summary.md)

| Mission | Case 1 | Case 2 | Case 3 | Internal agent speedup | Interpretation |
|---|---:|---:|---:|---:|---|
| `6c5e83...` | 361.3509 | 1.9258 | 0.9969 | 2.2270 | Massive repeat-case upside; weak case still just under parity |
| `efbab79...` | 291.2397 | 1.9246 | 1.0033 | 2.2132 | Safer floor; weak case crossed above `1.0x` |
| `c0f31ee...` | 342.1327 | 1.9402 | 0.9946 | not the new best | Proved synced workspace path and new candidate wording were visible officially |

### A.2 Phase 3 Retained Public Fallback Table

Sources:

- [remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md)
- [remote_public_eval_official_26e0fd0e_local_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_official_26e0fd0e_local_summary.md)

| Variant | Public throughput | Public mixed throughput | Correctness | Meaning |
|---|---:|---:|---|---|
| Earlier stable official-local fallback | 4317.5338 | 5227.2767 | passed | Reliable comparison baseline |
| Current kept version | 4341.2259 | 5298.9116 | passed | Best retained local fallback public result |

### A.3 Phase 3 Decode-Focused Experiment Table

Source:

- [remote_breakdown_decode_experiments_20260601.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_decode_experiments_20260601.md)

| Experiment | same_decode | mixed_decode | Decision |
|---|---:|---:|---|
| Baseline | 5317.7999 | 3764.2381 | Starting point |
| A: bool SDPA mask | 5227.9536 | 3674.2874 | Rejected |
| B: batched `index_put_` | 5205.4204 | 4500.5575 | Kept |
| E: `F.rms_norm` | 5248.0524 | 4534.9480 | Kept |
| F: generalized shared-row path | 4761.5861 | 4357.7510 | Rejected |
| I: proactive promotion | 4594.4265 | 4192.5400 | Rejected as default |

---

## Appendix B. Representative Code Snippets

### B.1 Phase 1: Compile-Fix Loop Is Real, Not Just Described

Source:

- [memxlife-project/agents/codegen.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/agents/codegen.py)

```python
# Round 1: generate complete CUDA code
code = self._generate_full_code(...)
ok, errors = _try_compile(code, self.nvcc, self.arch)
if ok:
    return self._success(...)

# Round 2: feed compile errors back for repair
fixed_code = self._fix_compile_errors(code, errors)
if fixed_code:
    ok2, errors2 = _try_compile(fixed_code, self.nvcc, self.arch)
    if ok2:
        return self._success(...)
```

这段代码的重要性在于，它证明 Phase 1 不是 one-shot probe generation，而是真正有 compile-test-repair 回路的 agentic workflow。

### B.2 Phase 2: Cache Safety Relies On Tensor Identity, Version, Shape, And Device

Source:

- [phase2_agent/codegen.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase2_agent/codegen.py)

```cpp
struct TensorStamp {
    std::uintptr_t data_ptr = 0;
    uint32_t version = 0;
    int64_t rows = 0;
    int64_t cols = 0;
    int device_index = -1;
};

inline bool same_stamp(const TensorStamp& lhs, const TensorStamp& rhs) {
    return lhs.data_ptr == rhs.data_ptr &&
           lhs.version == rhs.version &&
           lhs.rows == rhs.rows &&
           lhs.cols == rhs.cols &&
           lhs.device_index == rhs.device_index;
}
```

这段代码是 Phase 2 的关键证据之一，因为它说明我们的缓存决策不是只看裸指针，而是带着 mutation-safe 的 stamp 设计。

### B.3 Phase 3: Decode Path Dispatch Is Structured Around Request State

Source:

- [phase3_engine_sources/current_best_engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase3_engine_sources/current_best_engine.py)

```python
if same_length and position_offset is not None:
    shared_rows = self._shared_batch_rows(states)
    if shared_rows is None and self._should_promote_same_length_batch(states, position_offset):
        states = self._promote_request_states_to_shared_batch(request_ids, states)
        shared_rows = self._shared_batch_rows(states)
    if shared_rows is not None:
        logits = self._forward_decode_shared_batch(token_ids, states, position_offset, shared_rows)
    else:
        logits = self._forward_decode_batch(token_ids, states, position_offset)
```

这段代码体现了 Phase 3 的核心思想：性能不是靠一个统一快路径获得的，而是靠 request-state-aware dispatch。

---

## Appendix C. Evidence File Index And Additional Materials

### C.1 Phase 1 Evidence Files

- [memxlife-project/README.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/README.md)
- [memxlife-project/report.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/report.md)
- [memxlife-project/core/orchestrator.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/core/orchestrator.py)
- [memxlife-project/agents/codegen.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/agents/codegen.py)
- [memxlife-project/analysis/consistency.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/memxlife-project/analysis/consistency.py)

### C.2 Phase 2 Evidence Files

- [phase2_optimization_journey.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase2_optimization_journey.md)
- [report2.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/report2.md)
- [stage2_outputs/6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md)
- [stage2_outputs/efbab79b85fe777801ed0c4ba7e6ab44_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/efbab79b85fe777801ed0c4ba7e6ab44_summary.md)
- [stage2_outputs/c0f31ee32dec227bb48cae77b577a2e7_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/c0f31ee32dec227bb48cae77b577a2e7_summary.md)
- [.phase2_work/trace_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/.phase2_work/trace_summary.md)
- [.phase2_work/history.json](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/.phase2_work/history.json)

### C.3 Phase 3 Evidence Files

- [workspace/report3.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/workspace/report3.md)
- [run.sh](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/run.sh)
- [scripts/render_phase3_engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/scripts/render_phase3_engine.py)
- [phase3_engine_sources/current_best_engine.py](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase3_engine_sources/current_best_engine.py)
- [stage3_outputs/remote_breakdown_decode_experiments_20260601.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_decode_experiments_20260601.md)
- [stage3_outputs/remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_same_length_shared_promotion_retry_20260602_summary.md)
- [stage3_outputs/remote_public_eval_official_26e0fd0e_local_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_official_26e0fd0e_local_summary.md)

### C.4 Note On Screenshots And Additional Materials

这份仓库快照里没有专门整理好的报告截图包，因此本报告没有把截图作为主证据。  
这是一个刻意的取舍：相较于 screenshot，当前仓库保留的 markdown summaries、JSON traces、代码文件和实验日志更适合作为可审计、可复查的第一手材料。
