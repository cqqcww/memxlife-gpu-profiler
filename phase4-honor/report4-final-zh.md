# Phase 4 Honor Report：Agentic Mini Training Framework
23302010089 王丰淼

## 1. 项目目标与范围

Phase 4 的目标是构建一个由 agent 组织和推动的 mini training framework。
我把这个阶段理解成一个训练系统项目，而不是一个追求最终模型质量的项目：
最重要的成果不是训练出一个很强的语言模型，而是做出一个可以构建模型、
准备数据、运行训练、记录证据、保存/恢复 checkpoint、测量速度，并且支持
持续实验迭代的训练框架。

这个框架的核心路径是：

```text
config -> data -> model -> optimizer -> scheduler -> trainer -> logger -> checkpoint -> resume
```

随着实验推进，我对项目定位的理解也变得更清楚。它最开始是一个 mini
training framework，后来逐渐变成了一个面向 causal language model 的
**agentic training feasibility and optimization harness**。它不试图替代
HuggingFace Trainer、DeepSpeed、FSDP 或 Megatron，而是位于这些重型训练
系统之前：先判断某个 model/data/config 组合能不能跑，记录速度、loss、
显存和失败证据，分类问题，再决定下一步是否值得做更长训练或迁移到更重的
训练系统。

我刻意把 claim 控制得比较窄：当前框架支持的是经过测试的 causal-LM
profiles，而不是所有 HuggingFace 任务和模型架构。这个限制让系统更诚实，
也让 debug 更可控。

## 2. Agent 设计与工作流

这个 agent 是 rule-based 和 auditable 的。它不假装能自动发明新的训练算法，
而是做 harness engineering：

- 选择下一个受控实验；
- 用指定 config 启动训练；
- 解析 `events.jsonl`、`summary.json` 和 preflight artifacts；
- 写出 summary 和 recommendation；
- 把 run metadata 追加到 ledger；
- 根据观测到的瓶颈和风险提出下一个配置建议。

最早的 planner ladder 是：

```text
configs/debug.yaml
configs/baseline_tinystories.yaml
configs/cached_tinystories.yaml
configs/mixed_precision.yaml
```

我在远端用下面的命令验证过这个 loop：

```bash
python3 -m agent.planner --goal improve_tokens_per_sec --run-next
```

planner 选择了 TinyStories baseline，跑完后解析结果，并写出
`agent_patch_proposal.md`。proposal 发现 optimizer overhead 相对 forward
time 很大，于是建议做 batch size / gradient accumulation sweep。这个建议和
timing breakdown 是一致的，所以 agent 的价值不是“魔法优化”，而是把实验变成
一个有证据、有 rollback、有下一步判断的结构化循环。

后面我又加入了两个 agent 模块：

- `agent/auto_probe.py`：沿着有边界的 token-budget ladder 扩张，失败时停止，
  并写出 recommendation artifact。
- `agent/stability_runner.py`：把被选中的 recommendation 转成更长的
  stability run。

最成熟的例子是 DeepSeek stretch path。agent 从 `64 -> 128 -> 256 -> 512`
tokens/optimizer step 开始扩张，然后根据自己的 recommendation 继续到
`1024` 和 `2048`，最后只把 `2048` profile 提升到更长 stability test，而不是
直接宣称这是 final quality result。

## 3. 框架结构

实现结构基本遵循官方 guide 建议：

| 模块 | 作用 |
|---|---|
| `training_framework/config.py` | YAML 到 dataclass config，并做校验 |
| `training_framework/data.py` | 文本加载、tokenization、token blocks、cache、dataloader |
| `training_framework/model.py` | HuggingFace tokenizer/config/model 构建 |
| `training_framework/optim.py` | optimizer 构建和参数分组 |
| `training_framework/scheduler.py` | warmup + cosine scheduler |
| `training_framework/trainer.py` | 显式 train/validation loop |
| `training_framework/logger.py` | console、JSONL、TensorBoard logging |
| `training_framework/checkpoint.py` | model/optimizer/scheduler/global-step/RNG state |
| `training_framework/timing.py` | 每一步 timing breakdown |
| `agent/` | planner、runner、analyzer、matrix runner、auto-probe、recommendation |
| `selfcmd-workflow/` | 本地到远端的开发、验证和证据拉回流程 |

训练循环刻意保持可见：

```text
load batch -> forward -> backward -> gradient clip -> optimizer step
-> scheduler step -> log -> validate -> checkpoint
```

这个“可见性”对 debug 很重要。出现问题时，我可以定位到具体阶段：data time、
forward time、backward time、optimizer time、validation、checkpointing 或
resume，而不是只看到一个总耗时或一个最终 loss。

## 4. Profile-Based Extensibility

第一版框架跑通之后，我加入了 profile composition layer。直接 config 路径仍然可用：

```bash
python train.py --config configs/debug.yaml
```

组合式路径支持 base/model/data profiles 和 override：

```bash
python train.py \
  --base configs/base/causal_lm_debug.yaml \
  --model-profile configs/model_profiles/tiny_gpt2.yaml \
  --data-profile configs/data_profiles/local_fixture.yaml \
  --override trainer.max_steps=4
```

每次 composed run 都会写出 resolved config 和 `preflight.json` /
`preflight.md`。preflight 包含 model/profile metadata、tokenizer IDs、
dataset 信息、package availability、CUDA availability、参数量、tokens per
optimizer step 和显存风险估计。

已经测试过的 profiles 包括：

- `tiny_gpt2`：非常快的 smoke tests；
- `distilgpt2`：TinyStories 主实验；
- `gpt2`：更大但仍然可迭代的 stress profile；
- `Qwen/Qwen2.5-0.5B`：跨模型家族 stretch profile；
- `deepseek-ai/deepseek-coder-1.3b-base`：显存风险 stretch profile；
- local fixture、TinyStories、WikiText-2 数据 profiles。

这一层也让我更清楚地意识到边界：我可以舒适地切换经过测试的 causal-LM
profiles，但不能宣称无缝支持 encoder-only、seq2seq、multimodal 或任意
remote-code 模型。真正的任意架构切换还需要 task adapters、collators、losses
和 metrics。

## 5. 数据加载与 Tokenization

我使用了三类数据：

- local fixture corpus：快速 smoke testing；
- TinyStories subset：baseline 和优化实验；
- WikiText-2：DeepSeek real-data feasibility validation。

数据模块支持 HuggingFace datasets、本地文本文件、train/validation split、
固定长度 token blocks 和可选 token-block cache。cache key 包括 dataset
settings、tokenizer name、sequence length、max samples、validation split 和
cache version。

即使 cache 没有提高端到端吞吐，它仍然有价值。它让数据路径可复现，并且让我
能够把 data-loading time 和 compute/optimizer time 分开观察。

## 6. Model、Optimizer 与 Scheduler

smoke tests 使用 `sshleifer/tiny-gpt2`。第一批主实验使用 `distilgpt2`。默认
情况下，模型从 HuggingFace config 初始化，而不是加载 pretrained weights，
因为报告重点是 framework behavior 和 training-system evidence，而不是最终
模型质量。

基础 optimizer 是 AdamW，并区分 decay / no-decay 参数组。bias 和 normalization
参数不做 weight decay。scheduler 是 warmup + cosine decay，训练时会记录
learning rate。

后面我加入了 Adafactor，因为 DeepSeek stretch profile 暴露了 AdamW 的显存
成本。这不是为了“多支持一个 optimizer”而添加功能，而是因为它改变了 24GB GPU
上哪些 model/config 是可行的。

## 7. Logging、Checkpointing 与 Resume

每个 run 会写出：

```text
runs/<run_name>-YYYYMMDDTHHMMSSZ/
  copied_config.yaml
  events.jsonl
  summary.json
  summary.md
  tb/
  checkpoints/
```

框架会记录 train loss、validation loss、learning rate、tokens/sec、timing
breakdown，以及 CUDA 可用时的显存指标。

checkpoint 包含：

- model state；
- optimizer state；
- scheduler state；
- global step；
- config 和 metadata；
- 启用时的 RNG state。

一次远端 resume 检查加载了：

```text
/workspace/runs/debug-20260616T090134Z/checkpoints/step_000006.pt
```

恢复后的 run 到达 `global_step=8`，并记录：

```json
{
  "resume": {
    "global_step": 6,
    "optimizer_lr": 5e-05,
    "scheduler_lrs": [5e-05, 5e-05],
    "rng_restored": true
  }
}
```

这说明 resume 不只是加载模型权重。一个有用的 training checkpoint 必须保存足够
状态，让训练能从正确 step、正确 LR、正确 optimizer/scheduler/RNG 状态继续。

虽然每个 run 的 `tb/` 目录下都生成了 TensorBoard logs，但最终报告里我主要使用
从 JSONL/TensorBoard scalars 导出的表格，而不是截图。这样更容易复现和比较：
同一份 `events.jsonl` 可以重新解析成 loss、throughput、timing 和 memory 表格，
不依赖某个 UI 截图角度。TensorBoard 在开发中仍然有用，尤其是快速检查 validation
loss 是否和 train loss 同方向变化，以及配置改动后吞吐是否发生变化。

## 8. Remote-First Development Workflow

本地 laptop 不适合反复保存模型 cache、checkpoints 和远端训练 artifacts，所以我
构建了 `selfcmd-workflow/`。

原则是：

- local `phase4-honor/` 是代码和报告文本的 source of truth；
- remote `/workspace` 是 GPU 执行区域；
- 大 artifacts 留在远端；
- 本地只拉回轻量 summaries、logs 和 markdown evidence。

典型流程是：

```bash
./selfcmd start
./selfcmd deploy-clean
./selfcmd install-deps
./selfcmd test
./selfcmd smoke
./selfcmd evidence
./selfcmd fetch
```

这个 workflow 不只是 convenience。前几个 phase 让我意识到，官方 evaluation
经常读取 server-side `/workspace`，所以代码同步、远端验证和 artifact collection
本身就是系统的一部分。

### 8.1 开发迭代过程

实际开发并不是从“写框架”直线走到“最终实验”。它是一串小 probe、失败、修复和
新 harness feature 的循环。

第一个里程碑是最小 end-to-end training path。`debug.yaml` 必须能构建
tokenizer/model，创建 token blocks，训练几步，运行 validation，写 TensorBoard
和 JSONL logs，保存 checkpoint，并从 checkpoint resume。这个阶段先确立了
framework contract，之后才继续加大模型和优化实验。

第二个里程碑是环境修复。课程容器里有 PyTorch、PyYAML 和 TensorBoard，但没有
需要的 HuggingFace packages。最新 `transformers` 和容器里的 PyTorch 版本不匹配，
所以我 pin 了兼容版本。后面 `torch.optim.AdamW` 通过 Dynamo/ONNX 触发了
protobuf compatibility issue，我通过在 downstream torch imports 之前设置
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` 解决。它们不是模型优化，但它们是
可复现训练系统的前提。

第三个里程碑是 extensibility。我加入 base/model/data profile composition、
preflight reports 和 matrix runner。这个阶段产生了很有价值的 bug：PyYAML 把
未加引号的 `mixed_precision: off` 解析成 boolean `False`，而 trainer 期待的是
string enum。修掉这个 bug 后，config layer 更稳健。随后 profile-smoke 证明
`tiny_gpt2 + local_fixture` composed config 可以在远端训练；WikiText smoke 证明
数据 profile 可以从 local fixture 切到 HuggingFace dataset。

第四个里程碑是受控扩展。我没有直接从 DistilGPT2 跳到 DeepSeek，而是先跑 Qwen
smoke，再跑 GPT2/Qwen 的 60-step TinyStories sanity runs，最后做 Qwen throughput
probe。这个过程发现，第一版 Qwen 慢并不是“框架不适合 Qwen”，而是 tokens per
optimizer step 太少，同时 gradient checkpointing 带来额外 recomputation。这个
观察后来变成了具体 profile recommendation。

第五个里程碑是 DeepSeek safety gating。直接 AdamW smoke 在 `optimizer.step()`
CUDA OOM 后，我没有立刻硬调参数，而是先加入 `--preflight-only` 和 safety matrix，
把 model compatibility 和 optimizer memory 分开。只有在 preflight 证明模型路径可行、
AdamW 被定位为显存瓶颈后，我才加入 Adafactor、CUDA memory metrics、token-budget
auto-probing、stability runner，以及最后的 real-data WikiText run。每一个新功能
都来自上一次 run 的具体失败或不确定性。

最后一个里程碑是 workflow hardening。`selfcmd` 从方便脚本逐渐变成小型远端开发
harness：`deploy-clean`、dependency repair、remote tests、DeepSeek probes、
evidence fetches，以及避免 macOS metadata 污染的干净 tar archive。到这个阶段，
workflow 已经不是 shell-script detail，而是 Phase 4 系统的一部分。

## 9. Tests And Validation

测试覆盖 config loading、data shape、cache keys、checkpoint errors、planner
behavior、matrix selection、preflight reports、CUDA memory metrics、auto-probe、
stability-runner logic 和 recommendation generation。

远端测试演进如下：

| 阶段 | 远端结果 |
|---|---:|
| 初始 framework tests | `16 passed in 0.09s` |
| profile composition、matrix、preflight regressions | `24 passed in 2.55s` |
| DeepSeek/Adafactor/failure classification | `37 passed in 3.03s` |
| auto-probe 和 stability-runner coverage | `44 passed in 2.92s` |
| WikiText real-data auto-probe regression | `45 passed in 3.10s` |
| memory predictor 和 recommendation-report tests | `48 passed in 3.00s` |

当前代码库最后一次远端 validation 是：

```text
48 passed
```

这让报告证据更强：实验不是手工跑一次就结束，harness 本身也随着 bug 出现而获得
regression coverage。

## 10. Speed Measurement And Optimization

框架记录 tokens/sec，并分解 data loading、forward、backward、optimizer 阶段耗时。
我从 `events.jsonl` 导出了 compact evidence tables。

### 10.1 Cache And Mixed Precision

TinyStories 关键行如下：

| run | cache | final train loss | final val loss | mean tokens/sec | data ms | forward ms | backward ms | optimizer ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline TinyStories | miss | 5.0250 | 4.9362 | 28036.48 | 0.3953 | 10.4002 | 15.5264 | 9.8224 |
| cached TinyStories | hit | 4.9360 | 4.9279 | 27610.90 | 0.2113 | 11.1361 | 15.5508 | 9.8035 |
| mixed precision | hit | 4.8527 | 4.9218 | 27610.99 | 0.2172 | 11.0823 | 15.5793 | 9.8195 |

这个结果有价值，正是因为它不是简单的“优化成功”。cache hit 确实降低了 measured
data time，但整体 throughput 没有明显提高，因为 run 主要被 forward/backward/
optimizer time 主导。mixed precision 稳定，但在这个小 DistilGPT2-style 设置里
没有实质性提升吞吐。

这里的教训是：优化应该跟着 timing evidence 走，而不是跟着直觉走。

### 10.2 Batch Size And Gradient Accumulation

timing breakdown 表明，更大的 direct batch 可能能摊薄 per-step overhead。因此我
跑了 batch / gradient accumulation sweep：

| variant | avg tokens/sec | final val loss | interpretation |
|---|---:|---:|---|
| `bs2_ga1` | 15114.61 | 5.7362 | direct batch 最小，吞吐较低。 |
| `bs4_ga1` | 20284.70 | 5.3667 | direct batch 增大后 utilization 改善。 |
| `bs8_ga1` | 23949.54 | 5.1880 | 本 sweep 中吞吐最高，validation loss 也最好。 |
| `bs4_ga2` | 22139.84 | 5.1907 | effective batch 也是 8，但比 true batch 8 慢，因为做了两次 forward/backward。 |

因此当前 TinyStories/DistilGPT2 路径推荐：

```text
batch_size=8, grad_accum_steps=1
```

gradient accumulation 仍然是 memory fallback，但如果真正的大 batch 放得下，true
larger batch 更快。

## 11. Qwen Stretch Profile

我把 Qwen 当作跨模型家族的 stretch profile。保守 smoke run 使用
`Qwen/Qwen2.5-0.5B`、`seq_len=64`、`batch_size=1` 和 `max_steps=2`，结果成功：

```text
parameters=494032768
device=cuda
final_loss=10.7192
```

这是 compatibility evidence，不是 model-quality evidence。它证明 profile 可以构建
tokenizer/model、构建数据、跑 train/validation steps、checkpoint，并发出 preflight
evidence。

随后我跑了 60-step TinyStories sanity runs：

| run | model | steps | params | tokens/step | val loss path | mean tokens/sec |
|---|---|---:|---:|---:|---|---:|
| GPT2 TinyStories | `gpt2` | 60 | 124.4M | 256 | 7.2021 -> 6.1158 -> 5.9795 | 5460.35 |
| Qwen TinyStories | `Qwen/Qwen2.5-0.5B` | 60 | 494.0M | 64 | 9.4152 -> 7.3863 -> 7.2141 | 348.71 |

Qwen run 证明了这个 profile 可以持续训练和验证，但也说明它不应该作为默认迭代路径：
它慢很多，token budget 保守，而且需要 `trust_remote_code`。

后续 Qwen throughput probe 进一步解释了原因：

| Qwen variant | avg tokens/sec | final val loss | interpretation |
|---|---:|---:|---|
| `s64_b1_gc_on` | 350.41 | NaN | checkpointing 慢，并且这个短 random-init run 里数值不稳定。 |
| `s64_b1_gc_off` | 447.92 | 9.2218 | 关闭 checkpointing 后吞吐提高。 |
| `s64_b2_gc_off` | 897.67 | 8.9437 | 增大 direct batch 后吞吐接近翻倍。 |
| `s128_b1_gc_off` | 897.69 | 8.9839 | 增大 sequence length 得到类似吞吐提升，同时 batch size 仍保守。 |

更新后的 Qwen 推荐配置是：

```text
seq_len=128, batch_size=1, grad_accum_steps=1, gradient_checkpointing=false
```

## 12. DeepSeek Stretch Profile

DeepSeek 是最有价值的 stress test，因为它暴露了真实显存边界。我先把
`deepseek-ai/deepseek-coder-1.3b-base` 作为 random-init LLaMA-style causal LM
加载出来，参数量约 1.346B。直接做 one-step AdamW training smoke 时，在 24GB RTX
3090 上的 `optimizer.step()` 阶段 CUDA OOM。

这个失败很有用，因为它定位了问题。tokenizer/model profile 路径可行；直接阻塞点是
AdamW optimizer-state memory 和 temporary optimizer buffers。

我加入了 `--preflight-only`，让框架可以构建 tokenizer/model，把模型移到 device，
写出 preflight evidence，然后在 optimizer/training 前退出。然后我加入
`deepseek_safety_probe` matrix：

| variant | execution | result | interpretation |
|---|---|---|---|
| `preflight_s16_b1` | preflight only | success | Profile/model/tokenizer/preflight 路径可行。 |
| `adamw_s16_b1` | train | CUDA OOM | 24GB 设置下 full AdamW 不安全。 |
| `adafactor_s16_b1_no_ckpt` | train | success | 低显存 optimizer 路径可以完成真实一步。 |

Adafactor 改变了边界。它说明 DeepSeek 对框架不是普遍不可行；具体问题是 AdamW 的
显存成本。

随后我把它扩展成短 multi-step probes：

| DeepSeek Adafactor shape | avg tokens/sec | val loss | peak allocated / reserved |
|---|---:|---:|---|
| `s16_b1` | 47.38 | 10.8575 | about 10.98GB / 12.29GB |
| `s32_b1` | 84.54 | 10.8434 | about 10.98GB / 12.29GB |
| `s64_b1_gc_off` | 193.80 | 2.7425 | 11.00GB / 12.51GB |
| `s128_b1_gc_off` | 385.16 | 9.9014 | 11.02GB / 12.58GB |
| `s256_b1_gc_off` | 756.78 | 9.5584 | 11.06GB / 13.00GB |

之后 bounded `auto_probe` ladder 达到 `2048` tokens per optimizer step：

| token budget | avg tokens/sec | val loss | peak allocated / reserved | recommendation |
|---:|---:|---:|---:|---|
| 64 | 192.95 | 10.6877 | 11.00GB / 12.51GB | safe |
| 128 | 385.21 | 9.7545 | 11.02GB / 12.58GB | safe |
| 256 | 755.79 | 10.1239 | 11.06GB / 13.00GB | safe |
| 512 | 1343.92 | 8.6315 | 11.14GB / 13.67GB | try 1024 |
| 1024 | 2135.06 | 6.5600 | 11.39GB / 15.14GB | try 2048 |
| 2048 | 3092.42 | 3.5100 | 14.31GB / 18.58GB | run stability check |

我没有把这个当成 final result，因为最初 probe 用的是小 local fixture。正确下一步是更长
stability run 和真实数据 profile。

fixture stability run 通过：

| metric | result |
|---|---:|
| requested/completed steps | `50/50` |
| avg tokens/sec | `3680.77` |
| first -> last train loss | `22.0702 -> 0.0651` |
| last validation loss | `0.0608` |
| peak allocated / reserved | `14.31GB / 18.58GB` |

loss 快速塌缩很可能是因为 fixture 太小而发生 memorization，所以我把它记录为 systems
evidence，而不是 quality evidence。

随后我把同样 recommendation 转到 WikiText-2，并扩展到 100 steps：

| DeepSeek WikiText stability metric | result |
|---|---:|
| requested/completed steps | `100/100` |
| avg tokens/sec | `3650.34` |
| first -> last train loss | `9.5252 -> 6.3307` |
| last validation loss | `6.6539` |
| peak allocated / reserved | `14.31GB / 18.83GB` |

这是项目中最强的 DeepSeek 证据。它仍然不是 convergence claim，但它验证了 harness
claim：agent 在 fixture 上找到 memory-safe 配置，把它迁移到真实数据 profile，并确认
100 个 training steps 内没有 OOM、NaN 或 memory drift。

## 13. Calibrated Memory Prediction

DeepSeek 实验后，我给 preflight 加了 memory predictor。它把显存拆成 parameter
memory、gradient memory、optimizer state、activation proxy 和 reserved allocator
headroom。

100-step DeepSeek WikiText run 上的预测如下：

| memory quantity | predicted | observed | error |
|---|---:|---:|---:|
| allocated peak | 15051 MiB | 14305 MiB | +5.2% |
| reserved peak | 18814 MiB | 18828 MiB | -0.1% |

这解释了主要 DeepSeek 结论。AdamW 不安全，是因为 fp32 weights、gradients 和两个
fp32 moment tensors 让 reserved-memory estimate 在考虑 activations 和 temporary
buffers 前就接近/超过 24GB 设备边界。Adafactor 的 optimizer state 小很多，所以改变了
可行配置的边界。

最后，`agent/recommendation_report.py` 会把 auto-probe、stability、preflight 和
summary artifacts 合并成 compact markdown/json recommendation。它只把 2048-token
Adafactor/WikiText path 提升为“值得更长 probe”，而不是 final model-quality result。

## 14. Bugs、Pitfalls 与 Debugging

很多 bug 比成功 run 更有教育意义。

PyYAML boolean parsing：

- `mixed_precision: off` 被解析成 boolean `False`。
- trainer 期待 string enum。
- 我修复了 config loader，让它 normalize boolean mixed-precision values，并加入回归测试。

Dependency mismatch：

- 课程镜像使用 NVIDIA PyTorch 2.3。
- 最新 `transformers` 需要 PyTorch >= 2.4。
- 我 pin 了 `transformers==4.41.2`、`numpy==1.24.4` 和兼容的 `fsspec`。

ONNX/protobuf import issue：

- `torch.optim.AdamW` 会触发 Dynamo/ONNX import，并遇到 protobuf compatibility issue。
- 框架在 downstream torch imports 前设置 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`。

Gradient accumulation logging：

- 第一版 matrix run 把 accumulated train loss 记录成 microbatch losses 的总和。
- `bs4_ga2` 看起来异常差，但 validation 是正常的。
- 我修复 trainer，让它记录 accumulated microbatches 的平均 loss。

Qwen NaN best-selection：

- 一个 Qwen probe 产生 NaN validation loss。
- matrix selection 不应该让 non-finite validation loss 赢。
- 我修复选择逻辑：只要存在 finite validation losses，就忽略 NaN。

DeepSeek latest-run discovery：

- 第一版 DeepSeek matrix summary 把 `runs/matrix_logs` 误认为 training run。
- 我修复 run discovery，要求目录里有 `copied_config.yaml`，避免 auxiliary directories 污染 summary。

Stability-runner sparse log bug：

- 第一版 stability runner 统计 train log events，而不是 optimizer steps。
- `log_every=5` 时，一个完成 50 steps 的 run 看起来只有 10 steps。
- 我修复为读取 `summary.json.global_step`，并单独记录 `logged_train_events`。

这些 bug 改变了我对 training framework 的理解。framework 不只是 training loop，
还包括 loop 周围的证据系统。如果 logging、run discovery 或 summary logic 错了，
agent 即使训练本身正确，也可能做出错误决策。

## 15. 反思与收获

训练系统和推理优化很不一样。前几个 phase 里，核心问题经常是：
“这个 shape 上哪条 kernel/path 最快？” 到 Phase 4，核心问题变成了：

```text
Can every part of the training system preserve state, expose evidence,
and support reliable iteration?
```

这改变了我对“进展”的判断。一个 checkpoint/resume 正确的小 smoke run，比一个状态不清楚
的大 run 更有价值。一个中性的 cache 结果也有价值，因为它说明瓶颈不是 data loading。
一个简单 rule-based agent 比失控的自动改代码 agent 更可靠，因为它的决策可以追溯到 logs。

我也学会区分 compatibility evidence、feasibility evidence 和 quality evidence：

- smoke run 证明构建和基本执行；
- short stability run 证明配置短期内不会立刻失败；
- longer real-data run 开始说明训练行为；
- 这些都不自动等于最终模型质量。

这个区分在 Qwen 和 DeepSeek 上特别重要。Qwen 证明了跨模型 profile 可行，但不是最快迭代路径。
DeepSeek 的价值在于暴露 AdamW memory boundary，并迫使框架成为更好的 feasibility harness。

## 16. 和 Phase 1-3 的连接

前三个 phase 改变了我做 Phase 4 的方式。

Phase 1 和 Phase 2 让我习惯从 measured operators 和具体 shape 出发思考问题。特别是
Phase 2 里，一个看起来很惊人的 speedup 只有在理解 benchmark pattern 后才有意义；
我需要区分 benchmark-aware runtime optimization 和 universal operator improvement。
这个经验延续到 Phase 4：我不轻易说 cache、mixed precision、Qwen 或 DeepSeek “更好”，
除非 timing、loss 和 memory 证据支持这个判断。

Phase 3 让我更重视 request management、batching 和 remote workflow reliability。
这影响了 `selfcmd-workflow` 的设计：server-side `/workspace` 状态、deploy hygiene、
fetched evidence 和 repeatable tests 都成为 framework 的一部分，而不是事后补的脚本。

我没有把前面 phases 的 custom kernels 直接集成进训练 loop。我的判断是，Phase 4 先展示
一个完整、可观察、可恢复、可迭代的训练系统更重要。kernel-level 思维仍然体现在 timing
breakdown、batch-size sweep 和 memory-bound analysis 里，但主要贡献是 framework
coordination，而不是新的 low-level kernel。

## 17. Final Submission Evidence Index

最终提交建议配合以下 artifacts 阅读：

- `report4-final.md` / `report4-final-zh.md`：最终提交报告。
- `report4.md`：更长的 chronological working record。
- `training_framework/`：生成的 mini training framework。
- `agent/`：planner、matrix runner、auto-probe、stability runner 和 recommendation generator。
- `configs/`：base configs、model profiles、data profiles、matrices 和 auto-probe profiles。
- `tests/`：覆盖 config、data、checkpointing、preflight、matrix selection、auto-probe、stability 和 recommendation reporting 的 regression tests。
- `selfcmd-workflow/`：远端开发和验证 workflow。

最重要的证据点包括：最终远端测试 `48 passed`、带 RNG restoration 的 checkpoint/resume
metadata、TinyStories throughput/timing tables、Qwen stretch-profile comparison、
DeepSeek AdamW OOM classification、DeepSeek Adafactor/WikiText 100-step stability、
以及 calibrated memory prediction。

下面是更具体的 source-file map，路径都相对于 `phase4-honor/`。精简证据副本放在
`final_evidence/`，对应的原始远端 artifact 路径记录在 `final_evidence/README.md`。

| 结论或结果 | 主要源文件 | 可以核验什么 |
| --- | --- | --- |
| 最终 framework regression 状态 | `final_evidence/tests/tests-20260621T095143Z.log` | 远端 pytest 收集 48 个测试，最终结果为 `48 passed in 3.00s`。 |
| 远端证据快照 | `final_evidence/README.md` 和 `final_evidence/phase4-artifacts-20260621T095347Z.tar.gz` | final report 使用的 fetched evidence bundle，以及 curated evidence-copy map。 |
| Checkpoint/resume 和 RNG restoration | `final_evidence/resume/debug-resume-rng-summary.md`、`final_evidence/resume/debug-resume-rng-events.jsonl` 和 `tests/test_checkpoint_resume.py` | resume 行为被测试覆盖，并且远端 run 产生了 resume 相关证据。 |
| TinyStories batch/grad-accumulation sweep | `final_evidence/matrix_summaries/batch_grad_sweep-20260616T144038Z.md` 和对应 `.json` | `bs8_ga1` 因为在 validation-sane candidates 中 throughput 最好而被选中。 |
| Qwen stretch-profile comparison | `final_evidence/matrix_summaries/qwen_throughput_probe-20260619T033125Z.md` 和对应 `.json` | 短 Qwen probe 中 gradient checkpointing 更慢/不稳定，`s128_b1_gc_off` 是 best throughput candidate。 |
| Qwen longer compatibility run | `final_evidence/qwen/qwen-long-tinystories-summary.md`、`final_evidence/qwen/qwen-long-tinystories-preflight.md` 和 `final_evidence/qwen/qwen-long-tinystories-events.jsonl` | Qwen 可以通过 framework 跑起来、checkpoint、输出 metrics，但仍慢于 tiny GPT-style profile。 |
| DeepSeek AdamW failure classification | `final_evidence/deepseek/deepseek_safety_probe-20260619T070456Z.md` 和 `final_evidence/deepseek/deepseek-adamw-oom-agent-summary.md` | AdamW 被分类为 `cuda_oom`，说明 optimizer choice 是 feasibility decision，不只是小参数调优。 |
| DeepSeek Adafactor token-budget auto-probe | `final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-20260621T072201Z.md` 和对应 `.json` | WikiText-2 上 512/1024/2048 tokens per step 都通过，`2048` 被选入 stability validation。 |
| DeepSeek 100-step real-data stability | `final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-stability-20260621T095331Z.md`、对应 `.json` 和 `final_evidence/deepseek/deepseek-wikitext-2048-100step-events.jsonl` | 选中的 DeepSeek/WikiText path 完成 `100/100` steps，平均 `3650.34` tokens/sec，并且没有 OOM/NaN。 |
| 最终 recommendation 和 memory calibration | `final_evidence/recommendations/phase4-current-recommendation.md` 和对应 `.json` | 最终推荐配置、risk notes、next steps，以及 predicted-vs-actual CUDA memory calibration。 |

## 18. 当前限制与未来工作

这个框架刻意保持小而清楚。它没有实现 distributed training、ZeRO、FSDP、Megatron-style
tensor/pipeline parallelism 或 custom CUDA kernels。它也不宣称 cache 或 mixed precision
总能提高速度。在当前设置中，cache 改善了 data-time control，但没有提高整体 throughput；
mixed precision 是稳定的，但基本中性。

如果继续做，我最想改进：

- 更干净的 final workspace layout，避免 Phase 4 artifacts 和旧 phases 混在一起；
- 自动从 JSONL/TensorBoard scalars 生成 plots；
- 每个 run summary 记录 config hash 和 environment hash；
- matrix comparison 加 repeated trials 和 confidence intervals；
- 对 DeepSeek 2048-token path 做 200-500 step real-data run，再判断它是否是 durable recipe；
- 在更多模型家族上校准 memory predictor；
- 可选地把 feasibility harness 的结论导出成 HuggingFace Trainer、DeepSpeed 或 FSDP configs。

最终结果不是替代这些重型训练系统。它更像一个小而可检查的 control plane，用来判断下一步
什么配置安全、什么实验值得做、什么失败需要先修环境或框架。

## 19. Appendix：关键原始证据片段

这个 appendix 的目的，是让最终报告即使单独提交，也能直接看到关键数据和输出。完整证据副本
仍然保存在 `final_evidence/`，这里挑选最能支撑结论的片段。

### 19.1 远端 Regression Test 输出

来源：`final_evidence/tests/tests-20260621T095143Z.log`

```text
collected 48 items
Running 48 items in this shard

tests/test_agent_planner.py .....                                        [ 10%]
tests/test_auto_probe.py ....                                            [ 18%]
tests/test_checkpoint_resume.py ...                                      [ 25%]
tests/test_config.py .....                                               [ 35%]
tests/test_config_merge.py ..                                            [ 39%]
tests/test_cuda_memory_metrics.py .                                      [ 41%]
tests/test_data.py ....                                                  [ 50%]
tests/test_matrix_runner.py .............                                [ 77%]
tests/test_preflight.py ....                                             [ 85%]
tests/test_recommendation_report.py ..                                   [ 89%]
tests/test_smoke_train.py .                                              [ 91%]
tests/test_stability_runner.py ....                                      [100%]

============================== 48 passed in 3.00s ==============================
```

### 19.2 TinyStories Batch Sweep

来源：`final_evidence/matrix_summaries/batch_grad_sweep-20260616T144038Z.md`

| Variant | Status | Avg tokens/sec | Last val loss | Complexity |
| --- | ---: | ---: | ---: | --- |
| `bs2_ga1` | `0` | 15114.61 | 5.7362 | batch=2, grad_accum=1 |
| `bs4_ga1` | `0` | 20284.70 | 5.3667 | batch=4, grad_accum=1 |
| `bs8_ga1` | `0` | 23949.54 | 5.1880 | batch=8, grad_accum=1 |
| `bs4_ga2` | `0` | 22139.84 | 5.1907 | batch=4, grad_accum=2 |

最终选择 `bs8_ga1`。它在 validation-sane candidates 里吞吐最高；`bs4_ga2` 则说明
gradient accumulation 不等价于真正的大 batch，因为它需要额外的 forward/backward pass。

### 19.3 Qwen Stretch-Profile 证据

来源：`final_evidence/matrix_summaries/qwen_throughput_probe-20260619T033125Z.md`
和 `final_evidence/qwen/qwen-long-tinystories-summary.md`

| Variant | Status | Avg tokens/sec | Last val loss | Main difference |
| --- | ---: | ---: | ---: | --- |
| `s64_b1_gc_on` | `0` | 350.41 | nan | seq_len=64, batch=1, checkpointing on |
| `s64_b1_gc_off` | `0` | 447.92 | 9.2218 | seq_len=64, batch=1, checkpointing off |
| `s64_b2_gc_off` | `0` | 897.67 | 8.9437 | seq_len=64, batch=2, checkpointing off |
| `s128_b1_gc_off` | `0` | 897.69 | 8.9839 | seq_len=128, batch=1, checkpointing off |

Qwen longer run 跑到了 60 steps，参数量为 494,032,768，final loss 是 `7.695265`，
final learning rate 是 `3e-05`，token blocks 是 3,510。这个结果证明跨模型 profile
可以跑通、可以 checkpoint、可以输出 metrics，但它不是当前硬件和框架下最快的训练路径。

### 19.4 DeepSeek AdamW Safety Gate

来源：`final_evidence/deepseek/deepseek_safety_probe-20260619T070456Z.md`

| Variant | Execution | Status | Failure | Complexity |
| --- | --- | ---: | --- | --- |
| `preflight_s16_b1` | preflight_only | `0` |  | batch=1, grad_accum=1, seq_len=16, checkpointing=True |
| `adamw_s16_b1` | train | `1` | cuda_oom | batch=1, grad_accum=1, seq_len=16, checkpointing=True |

这个结果改变了后续设计方向。问题不是简单地“把 learning rate 调小”，而是 optimizer state
和 memory boundary 让 AdamW 在这张 GPU 上对 DeepSeek profile 不安全。因此框架后续增加了
更强的 preflight、failure classification，以及基于 Adafactor 的 probes。

### 19.5 DeepSeek Adafactor Auto-Probe 和 Stability

来源：
`final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-20260621T072201Z.md`
和
`final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-stability-20260621T095331Z.md`

| Variant | Status | Tokens/step | Avg tokens/sec | Last val loss | Peak CUDA MB |
| --- | ---: | ---: | ---: | ---: | --- |
| `tok512` | `0` | 512 | 1341.83 | 9.8190 | 11173/13666 |
| `tok1024` | `0` | 1024 | 2137.52 | 10.7984 | 11899/15140 |
| `tok2048` | `0` | 2048 | 3100.38 | 10.0320 | 14305/18828 |

被选中的 `tok2048` 配置随后通过了 100-step stability run：

| Metric | Value |
| --- | ---: |
| Completed train steps | 100/100 |
| Logged train events | 10 |
| Validation events | 5 |
| Average tokens/sec | 3650.34 |
| Last train loss | 6.3306779861450195 |
| Last validation loss | 6.653914451599121 |
| Peak CUDA MB allocated/reserved | 14305/18828 |

### 19.6 最终 Recommendation 和 Memory Calibration

来源：`final_evidence/recommendations/phase4-current-recommendation.md`

| Field | Value |
| --- | --- |
| Model | `deepseek-ai/deepseek-coder-1.3b-base` |
| Data profile | `wikitext2` |
| Optimizer | `adafactor` |
| Tokens/step | `2048` |
| Shape | `seq_len=2048, batch=1, grad_accum=1` |
| Mixed precision | `auto` |
| Gradient checkpointing | `False` |
| Stability status | `pass` |
| Train loss | `9.5252 -> 6.3307` |
| Last validation loss | `6.6539` |

| Memory calibration field | Value |
| --- | ---: |
| Predicted allocated peak | 15051 MiB |
| Actual allocated peak | 14305 MiB |
| Allocated prediction error | 5.2% |
| Predicted reserved peak | 18814 MiB |
| Actual reserved peak | 18828 MiB |
| Reserved prediction error | -0.1% |

最终建议是把这个配置推进到更长 probe，但在没有 offload 或 sharding 前，继续把 AdamW
标记为不安全默认选项。
