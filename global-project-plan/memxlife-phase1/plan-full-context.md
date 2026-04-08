# Phase 1 完整讨论记录

## Session 1 — 2026-04-08 架构设计讨论

### 背景
- 项目: MLSys 课程 Phase 1 — GPU Performance Analysis Agent
- Deadline: 2026-04-21 8am（13天）
- 需求文档: memxlife-origin-static/mlsys-project-phase1.md
- 参考示例: GPUArchitect（简单循环）、GPUProfiler（成熟多Agent）

### 需求核心总结
Phase 1 有两个层次：
1. 基础层 (1.1-1.6): Agent 分析 ncu 输出的 metrics，判断 CUDA kernel 瓶颈类型
2. 进阶层 (1.7): Agent 作为 Hardware Probe，自主生成微基准测试，反向探测 GPU 物理特性

评测方式：
- 输入: target_spec.json（要探测的硬件指标列表）
- 输出: results.json（探测到的数值）
- 评分: 70分数值精度 + 30分工程推理方法论
- 反作弊: GPU频率锁定到非标准值、SM masking、cudaGetDeviceProperties 可能返回假数据

### Q&A 记录

**Q1: 架构复杂度边界？**

对比了 ~2000行基础方案 vs ~3500行完整方案，逐模块分析了效果提升：
- 自适应重试引擎 (+200行) → 效果高，应对非标准CUDA配置
- 环境探测Agent (+300行) → 效果高，反作弊关键
- 结果校准层 (+200行) → 效果中高，抓住LLM分析错误
- Probe变体生成器 (+300行) → 效果中，基础模板够用（暂不加）
- 多轮Agent对话 (+400行) → 效果中低，13天风险大（暂不加）
- 完整审计日志 (+500行) → 对30分工程推理评分有帮助

结论: 采用前三个 + 审计日志，总量 ~3500行。代码量上限 4000行Python + 500行CUDA。

**Q2: LLM选择？**

- Codegen + Analyzer → Claude Opus（CUDA生成更强，长上下文推理优势）
- Planner → Claude Sonnet（决策相对简单，省成本）
- 可配置 + fallback 到 OpenAI GPT-4o

**Q3: 开发环境？**
- 本地: MacBook Air M1，无GPU
- GPU服务器: 远程，有 NVIDIA GPU
- 策略: 本地搭框架用 mock runner，GPU服务器接真实执行

**Q4: 为什么借鉴 GPUProfiler 但大幅简化？**

借鉴的部分：
- Agent 基类 (can_handle + run) 设计干净
- Orchestrator 的 _run_with_retry 机制
- Task 数据模型 (id, kind, payload, result, status)
- AgentContext (run_id, run_dir) 产物管理

简化的部分（GPUProfiler agents.py 2199行中约1200行不需要）：
- CommunicationMonitorAgent — 不需要Agent对话记录
- BookBuilderAgent — 900行知识库整理，用简单JSON够了
- 两层嵌套的 BenchmarkCycle/Executor — 一个Runner够了
- _apply_negotiation_policy — 200行benchmark准入评分不需要

知识库学 GPUArchitect 的 JSONL claims（结构化数值数据更合适）。
总结：骨架学GPUProfiler，知识库学GPUArchitect，业务逻辑全部重写。

### 最终架构决策

5个Agent + 完整审计日志，~3500行

| 模块 | 说明 |
|:---|:---|
| Planner | 策略选择 + 任务规划 |
| Codegen | CUDA代码生成（模板+LLM） |
| Runner | 编译+执行+ncu profiling |
| Analyzer | 结果解读+置信度评估+KB更新 |
| Environment Scout | 环境侦察（频率、SM数、工具可用性） |
| + 自适应重试引擎 | 编译/执行失败时智能切换策略 |
| + 结果校准层 | 物理约束 sanity check |
| + 审计日志 | Markdown报告生成（给30分工程推理评分用） |

LLM配置: Codegen + Analyzer 用 Opus，Planner 用 Sonnet，可配置 + fallback

CUDA微基准: 混合方案 — 预置高质量模板 + LLM调参/修改

3个创新点（记录备用，暂不实现）：
1. 环境指纹识别 — 标定探针反推环境篡改
2. 物理一致性验证 — 指标间关系交叉检查
3. 渐进式精度提升 — 先粗后细保底策略

### 项目结构

```
memxlife-project/
├── main.py
├── config.py
├── core/
│   ├── orchestrator.py
│   ├── models.py
│   └── state.py
├── agents/
│   ├── base.py
│   ├── planner.py
│   ├── codegen.py
│   ├── runner.py
│   ├── analyzer.py
│   └── scout.py
├── llm/
│   ├── client.py
│   └── prompts.py
├── knowledge/
│   ├── store.py
│   └── metrics_catalog.py
├── probes/
│   ├── templates/
│   └── registry.py
├── parser/
│   ├── ncu_parser.py
│   └── probe_parser.py
├── audit/
│   └── logger.py
├── runs/
└── tests/
```

### 核心流程

```
target_spec.json → Environment Scout（环境侦察）
  → 对每个target指标:
    Planner（选策略）→ Codegen（生成CUDA）→ Runner（编译执行）→ Analyzer（解读+校准）
    → 置信度不够？回到Planner换策略
  → 汇总 results.json + 审计报告
```
