# Phase 1-2-3 Talk Script

这份文档是面向 **15 分钟 talk** 的逐页讲稿版本。它和 [total-report.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/total-report.md) 的关系是：

- `total-report.md` 负责完整、详细、可追溯
- `phase123_talk_script.md` 负责口头表达、节奏、重点取舍

建议页数：`10` 页主讲 + `2` 页 backup  
建议时长：`14` 到 `15` 分钟

---

## Overall Advice

这场 talk 最好的讲法不是“我们做了三个作业”，而是：

> 我们沿着一条连续的系统优化路径，从硬件测量，走到算子复用建模，再走到推理运行时特化。

全场统一主线建议写在第一页，也建议你口头反复回到这四个词：

`Measure -> Model -> Specialize -> Validate`

这份讲稿的推荐重心也很明确：

- 多讲实现思路、方向变化、调试证据、个人反思
- 少讲排名、数字堆砌、结果炫耀
- 如果提到结果，也把它当成“支撑理解的证据”，不要当成 talk 主角

---

## Slide 1. Title And Thesis

### Slide content

标题建议：

`From GPU Profiling to Runtime Specialization: An Agentic Systems Optimization Journey`

副标题建议：

`Measure -> Model -> Specialize -> Validate`

页面元素建议：

- 题目
- 姓名 / 学号
- 一张横向三阶段主线图
  - Phase 1: Understand the GPU
  - Phase 2: Understand operator reuse
  - Phase 3: Understand serving runtime structure

### Recommended on-slide copy

- `From GPU Profiling to Runtime Specialization`
- `Measure -> Model -> Specialize -> Validate`
- `One project, three layers of system understanding`

### Speaker notes

“这次我想讲的不是三个分开的 phase，而是一条连续的系统优化路线。  
Phase 1 我们在回答，GPU 真正在做什么。  
Phase 2 我们在回答，一个算子在真实 workload 里到底怎样被重复调用。  
Phase 3 我们在回答，一个推理 runtime 怎样根据请求结构做特化。  
所以这三个阶段虽然题目不同，但我们的方法其实很一致，就是先测量，再建模，再做特化，最后用证据验证。  
回头看，这个项目最可迁移的产出也不是某一个分数或者某一个优化点，而是这套不断重写问题定义、不断用证据收缩方向的工作方法。  
如果现在重来一次，我会从一开始就把 talk 和报告的主线写成‘理解与反思’，而不是默认按结果来组织材料。”  

### Time

`1 min`

---

## Slide 2. One Project, Three Questions

### Slide content

标题建议：

`One Project, Three Increasingly Realistic Questions`

页面主体建议放一个三行表：

| Phase | Question | What changed in our thinking |
|---|---|---|
| Phase 1 | What is this GPU really like? | API values are not enough |
| Phase 2 | What is this operator workload really repeating? | Single-call optimization is not enough |
| Phase 3 | What is this serving runtime really spending time on? | Local fast path is not enough |

底部再放一行统一方法论：

`Measure -> Model -> Specialize -> Validate`

### Recommended on-slide copy

| Phase | Core question | What changed |
|---|---|---|
| 1 | What is the GPU really doing? | API values were not enough |
| 2 | What is the operator really repeating? | Single-call speed was not enough |
| 3 | What is the runtime really spending time on? | Local fast paths were not enough |

Bottom line:

`From local signals to system structure`

### Speaker notes

“这一页是为了把三个阶段统一起来。  
Phase 1 不是简单做 benchmark，而是去判断哪些硬件信息值得相信。  
Phase 2 不是简单把公式写快一点，而是去识别隐藏 workload 里的复用模式。  
Phase 3 不是简单堆优化点，而是去理解整个 serving runtime 的结构瓶颈。  
所以每一阶段我们都经历了一个认知升级：从看局部数值，到理解整体结构。  
而且真实过程都不是一开始就知道答案，反而是每一阶段都被实验和证据逼着重新定义问题，这也是我觉得这三个 phase 最像同一个项目的地方。  
如果重新设计整个项目，我会更早把‘问题是怎么被改写的’专门记录下来，因为这部分其实比最终结果更能体现理解。”  

### Time

`1 min`

---

## Slide 3. Phase 1 Problem And System

### Slide content

标题建议：

`Phase 1: Hardware Profiling Under Untrusted Signals`

左边放任务目标：

- cache size
- latency hierarchy
- bandwidth
- boost clock
- bank conflict penalty

右边放 agent pipeline：

- Scout
- Planner
- Codegen
- Runner
- Analyzer

最适合配一张 pipeline 图。

### Recommended on-slide copy

Problem:

- `APIs may be incomplete or untrustworthy`
- `We needed active probing, not passive reading`

System idea:

- `Build a minimal closed-loop profiling agent`
- `Generate -> compile -> run -> validate`

### Speaker notes

“Phase 1 的难点不是把 CUDA API 调出来，而是官方环境里这些 API 本身未必可靠。  
所以我们没有把任务理解成‘读出几个值’，而是把它理解成‘主动探测 GPU 行为’。  
我们的系统是一个多 agent 流水线：Scout 先识别目标和约束，Planner 设计 probe，Codegen 生成 microbenchmark，Runner 执行，Analyzer 做结果解释和一致性检查。  
这里 agentic 的点不只是用了 LLM，而是系统能围绕失败、编译错误和不一致结果继续推进。  
而且真实实现并不是一上来就知道要做完整 multi-agent probing，我们其实先打通了一个最小闭环：接任务、生成一个 probe、编译运行、解析结果、再继续下一轮，然后才逐渐长出 compile-fix、fallback 和 consistency validation。  
如果重来一次，Phase 1 我会更早把环境感知、parser 鲁棒性和 audit log 作为一级设计目标，因为这些东西后来证明并不是附属品，而是系统能不能持续工作的基础。”  

### Time

`1.5 min`

---

## Slide 4. Phase 1 Turning Point And What We Learned

### Slide content

标题建议：

`Phase 1 Turning Point: Evidence Matters More Than A Single Number`

做一个左右对照：

左侧 `Initial intuition`

- run probes
- read values
- output metrics

右侧 `What we later learned`

- values must be physically consistent
- anti-hacking matters
- compile-fix loop matters
- evidence beats raw numbers

右下可以加一句 takeaway：

`A metric is useful only if we can explain why we trust it.`

### Recommended on-slide copy

Initially:

- `Read metrics`
- `Run probes`
- `Collect outputs`

What we learned:

- `Build evidence`
- `Check consistency`
- `Repair failing probes`

Takeaway:

`Trust the explanation, not just the number`

### Speaker notes

“Phase 1 最重要的收获不是某个具体 cache 数字，而是我们从‘拿到一个值’转向了‘判断这个值为什么可信’。  
比如 API 可以给提示，但不能作为最终证据。  
编译修复循环看起来像工程细节，但实际上它直接影响 probe 能不能稳定跑起来。  
最后我们形成的不是一个简单脚本，而是一套能持续修 probe、交叉验证结果、过滤可疑结论的 profiling workflow。  
这也变成了后面两个 phase 的习惯：如果一个结果单看很漂亮，但和别的证据合不起来，我们宁愿先怀疑证据链，而不是急着相信这个结果。  
如果重做这一阶段，我会更早把 consistency validator 独立出来，因为后来回头看，它其实比多写几个 probe 更能体现系统理解。”  

### Time

`1.5 min`

---

## Slide 5. Phase 2 Problem: Why Naive LoRA Optimization Was Not Enough

### Slide content

标题建议：

`Phase 2: The First Optimization Target Was Not The Best One`

页面上方放公式：

`Y = W X + A(B^T X)`

左边放 `Our initial path`

- correctness-first ATen baseline
- optimize contiguous `B^T`
- look at low-rank branch first

右边放 `What measurements showed`

- main `W @ X` dominates
- low-rank branch is smaller
- hidden cases reward reuse across calls

### Recommended on-slide copy

Initial intuition:

- `Stabilize the baseline`
- `Improve B^T layout`
- `Start from the low-rank branch`

What changed:

- `W @ X stayed dominant`
- `The hidden workload repeated`
- `Reuse mattered more than local fusion`

### Speaker notes

“Phase 2 一开始最自然的想法，是盯着 LoRA 这条低秩分支去做优化，因为它看起来最特别。  
但我们真正开始测以后发现，主成本往往还是在 `W @ X` 上。  
这就逼着我们从‘把一个公式局部写快’转向‘这个 workload 在多次调用之间有没有可以利用的结构’。  
也就是说，决定结果的不是某个局部 kernel 花活，而是我们有没有理解 hidden workload 的重复模式。  
而且真实路线也不是直接跳到 `hybrid_weff`，我们先做的是稳定 baseline、可靠提交流程、还有围绕 `B^T` contiguous 和 `addmm_` 组合的近邻搜索。正是因为 baseline 够稳，后面每次改方向才有参照物。  
如果重新设计，Phase 2 我会更早搭好 candidate history 和 benchmark ledger，因为这类问题特别需要记录‘为什么我们离开了一条路线’，而不只是保存最后留下的路线。”  

### Time

`1.5 min`

---

## Slide 6. Phase 2 Core Insight: W_eff And Runtime Reuse

### Slide content

标题建议：

`Phase 2 Core Insight: Optimize The Reuse Pattern, Not Just The Formula`

左侧放等式：

`W X + A(B^T X) = (W + A B^T) X`

右侧放三路策略：

- exact repeat -> return cached output
- same weights, new X -> use cached `W_eff`
- cold case -> safe decomposition fallback

右下角小字建议：

`The headline win came from workload reuse, not from a universal 361x GEMM speedup.`

### Recommended on-slide copy

Core rewrite:

`W X + A(B^T X) = (W + A B^T) X`

Policy:

- `Exact repeat -> reuse output`
- `Same weights -> reuse W_eff`
- `Cold case -> safe fallback`

Takeaway:

`Optimize repeated structure, not one isolated call`

### Speaker notes

“Phase 2 的真正转折点是我们意识到，这题最值钱的不是单次算子优化，而是复用建模。  
如果 `W/A/B/X` 都一样，我们其实可以直接复用输出。  
如果权重一样但输入 `X` 变了，我们就可以复用已经构造好的 `W_eff = W + AB^T`。  
只有冷路径才回退到安全的原始分解。  
所以 `hybrid_weff` 这条路线本质上是在做 runtime policy，而不只是做代数变形。  
这也解释了为什么有时会看到特别高的 hidden-case speedup，它反映的是重复 workload 被我们抓住了，而不是说所有 GEMM 都突然快了几百倍。  
更准确地说，这更像是一种 benchmark-aware runtime optimization，而不是普适算子提速，因为官方 benchmark 本来就在对同一组 `W/X/A/B` 做 repeated-call 测量。  
在这之前我们也试过 `BX` cache、adaptive cache 这些中间路线，它们不是最终答案，但很重要，因为它们第一次把我们从单次算子视角推到了跨调用视角。  
如果重来一次，我会更早把这件事显式表述成 runtime policy design，而不是把它包装成一个算子小优化，因为那样更符合问题本质，也更容易指导后续实现。”  

### Time

`2 min`

---

## Slide 7. Phase 2 Debug Reality And Why The Result Was Believable

### Slide content

标题建议：

`Phase 2: Debug Evidence Was As Important As The Idea`

页面只保留三点：

- the hot path once still computed an unnecessary main GEMM
- remote validation exposed the mismatch
- submission workflow and code sync affected final results

底部放一个小 timeline：

`good idea -> mediocre result -> trace and remote check -> bug found -> result flipped`

### Recommended on-slide copy

What went wrong:

- `The fast path still did extra work`
- `Local intuition and remote behavior diverged`
- `Submission workflow also mattered`

Lesson:

`When theory and performance disagree, inspect the evidence chain`

### Speaker notes

“这一页我想强调，Phase 2 的结果不是一上来就成立的。  
我们曾经有一个 hot-path bug，逻辑上以为走到了复用快路径，但实际上还偷偷多做了一次主 GEMM。  
这个问题不是光看代码就很容易发现的，是在远端验证和 trace 证据里逐渐暴露出来的。  
这对我们很重要，因为它说明高分不是‘公式猜对了’这么简单，而是 idea、logging、remote validation、submission workflow 这些部分一起闭环了，结果才真正可信。  
我觉得这里一个很重要的经验是，当性能结果和算法直觉不一致时，不要马上否定思路，而是先检查证据链里哪一环在说谎。像 `361x` 这种结果，真正说明的也不是某个 GEMM 神奇提速，而是 repeated hidden workload 被强命中了。  
所以我们后来会更谨慎地表述它：这不是“普适快几百倍”，而是官方 repeated-call harness 与 exact-repeat runtime cache 共同放大的结果。  
如果重做，我会更早加上 hot-path assertions、path counters 和远端对账脚本，因为这类 bug 不是‘代码能不能跑’，而是‘代码到底跑到了哪条路径’。”  

### Time

`1.5 min`

---

## Slide 8. Phase 3 Main Structural Upgrade

### Slide content

标题建议：

`Phase 3: Throughput Became A Runtime Structure Problem`

左边放 baseline：

- stores full sequence
- decode recomputes too much
- limited specialization

右边放我们的主线：

- per-layer KV cache
- request state tracking
- prefill / decode split
- shared-batch promotion

建议做一张 before/after 图：

- Baseline: decode -> recompute full sequence
- Ours: decode -> reuse KV cache -> compute only new token

### Recommended on-slide copy

Baseline:

- `Too much repeated decode work`
- `Limited runtime specialization`

Our redesign:

- `Per-layer KV cache`
- `Request-state tracking`
- `Prefill / decode split`

Takeaway:

`Throughput is a runtime-structure problem`

### Speaker notes

“到 Phase 3，我们面对的问题不再是单个算子，而是整个推理 runtime。  
官方 baseline 在 decode 上有比较明显的全量重算倾向，所以最先要做的不是微调某个小函数，而是把结构改成更真实的 incremental decode。  
我们的主升级包括每层 KV cache、请求状态管理、prefill/decode 分离，以及在可控范围内做 shared-batch promotion。  
这里最核心的变化是：吞吐不再只是算子问题，而是 runtime 对请求结构的适配问题。  
真实过程里我们其实也是不断收缩问题定义，一开始很容易想把 attention、decode、各种 kernel 都一起做快，但后来越来越清楚，第一原则不是追局部峰值，而是先把 runtime 结构改成真实 incremental decode，再谈局部强化。  
如果重来，我会从第一天就把 same/mixed workload、prefill/decode、public/stress 这些 evaluation 维度拆开，因为不拆开看，很容易把局部加速误当成整体进步。”  

### Time

`2 min`

---

## Slide 9. Phase 3 Experiment Logic: Why We Rejected Some Faster-Looking Ideas

### Slide content

标题建议：

`Phase 3: We Kept The Most Robust Runtime, Not The Most Exciting Local Trick`

左边放 `Retained optimizations`

- batched `index_put_`
- fused `F.rms_norm`
- conservative same-length promotion

右边放 `Rejected or limited ideas`

- generalized shared-row path
- overly aggressive promotion
- manual attention rewrites without stable end-to-end gain

底部放一句总结：

`Same-decode best point and mixed-decode best point were often different.`

### Recommended on-slide copy

Retained:

- `Batched index_put_`
- `F.rms_norm`
- `Conservative promotion`

Rejected:

- `Generalized shared-row path`
- `Over-aggressive promotion`
- `Fragile manual rewrites`

Takeaway:

`Local fastest did not mean overall best`

### Speaker notes

“Phase 3 一个很重要的成熟点，是我们开始主动放弃一些看起来更激进、局部更快的优化。  
因为我们发现 same-decode 的最优点和 mixed workload 的最优点经常不是同一个。  
所以我们最后保留的是 batched `index_put_`、`F.rms_norm` 和比较保守的 shared promotion，而不是把所有局部快路径都硬塞进去。  
这页其实想表达的是，系统优化的目标不是把每个局部 benchmark 都堆高，而是让 end-to-end runtime 在真实 workload 下更稳、更可解释。  
我觉得这阶段最真实的收获之一，是删掉一个看起来聪明的优化，常常比继续堆一个新优化更难，因为那意味着承认它虽然局部漂亮，但不适合当前 runtime 结构。  
如果重新设计，我会更早建立明确的 accept/reject criteria，比如 same-decode、mixed-decode、correctness margin 和 complexity cost 一起看，这样实验筛选会更清楚，也更容易解释为什么某些优化被删掉。”  

### Time

`1.5 min`

---

## Slide 10. Final Takeaways

### Slide content

标题建议：

`What We Actually Built Across The Three Phases`

放三条 takeaway：

1. The biggest wins came from understanding structure, not from fancy low-level tricks alone.
2. Logging, traces, and debug evidence were part of the optimization itself.
3. Agentic development was valuable because it supported fast iteration, validation, and rollback.

最后再放一句总结束语：

`From probing hardware, to modeling operator reuse, to specializing serving runtime, we ended up with a reusable systems optimization methodology.`

### Recommended on-slide copy

What we actually built:

- `Structure-first optimization`
- `Evidence-first debugging`
- `Fast path + safe fallback`

If redesigning the whole project:

- `Standardize instrumentation earlier`
- `Track why directions changed`
- `Optimize for understanding, not just score`

### Speaker notes

“最后总结一下，我觉得这三个 phase 最重要的产出不是三组分数，而是一套我们可以迁移到别的系统问题上的工作方法。  
第一，真正的大收益往往来自结构理解，而不是一开始就追求最花的底层优化。  
第二，logging、trace、breakdown 和 debug evidence 不是附属工作，它们就是优化过程的一部分。  
第三，AI 在这个项目里的价值，不只是帮我们写代码，而是帮助我们维持一个高频、可验证、可回滚的 agentic development loop。  
所以从硬件探测，到算子复用，再到 serving runtime 特化，我们最后拿到的是一套可复用的系统优化方法论。  
如果再压成两句，我会说，一句是最大的收益来自结构，不只来自更花的 kernel；另一句是 fast path 必须和 safe fallback、instrumentation 一起存在，不然优化很容易变成不可验证的猜测。  
如果整件事重来一次，我会更早把实验台账、调试证据和失败尝试正式纳入主材料，因为这些内容最能展示理解和反思，而不只是展示结果。”  

### Time

`1 min`

---

## Backup Slide A. Repository Structure

### Slide content

- `memxlife-project/`
- `phase2_agent/`
- `phase3_engine_sources/`
- `workspace/`
- `evaluator/`

### Speaker notes

“如果老师问项目结构，我会说我们没有把三个 phase 混成一个仓库逻辑，而是把 Phase 1 的 profiling agent、Phase 2 的 candidate search 与 operator path、Phase 3 的 runtime engine 和 evaluator 分开组织。这样每一阶段既能独立验证，也能保留演进证据。  
这个结构其实也对应了我们的工作方式：先把每一阶段做成能单独闭环的系统，再在报告和 talk 里把它们串成同一条方法论主线。”  

---

## Backup Slide B. Key Numbers

### Slide content

建议只放最有解释性的数字，不要放满页：

- Phase 2 选 `6c5e83`, `efbab79`, `c0f31ee` 三个官方结果做对照
- Phase 3 选 retained local fallback 和 decode-focused experiment 做对照

### Speaker notes

“如果老师追问结果，我会把数字讲成‘证据点’，而不是把 talk 变成报表。Phase 2 重点讲为什么 `hybrid_weff` 说明 hidden workload 有重复结构。Phase 3 重点讲为什么我们最终保留了更稳的 runtime 版本，而不是最激进的实验版本。  
也就是说，数字不是孤立结论，而是用来支撑我们前面那条思路转变和工程判断链条的。”  

---

## Delivery Tips

### 你在讲的时候最值得强调的三次认知转折

1. Phase 1：从读指标到信证据
2. Phase 2：从单次算子优化到 workload reuse
3. Phase 3：从局部快路径到 runtime 整体最优

### 你要避免的讲法

- 不要逐页报分
- 不要把三个 phase 讲成三个孤立作业
- 不要把 agentic development 讲成“AI 帮我写代码”

### 一个很好用的结尾句

“这三个 phase 表面上分别在做 profiling、算子优化和推理优化，但更深一层看，我们一直在做同一件事：找到系统真正重复的结构，然后用测量和验证把它变成稳定收益。”  
