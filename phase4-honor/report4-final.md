# Phase 4 Honor Report: Agentic Mini Training Framework
23302010089 王丰淼

## 1. Project Goal And Scope

The goal of this phase was to build an agent-organized mini training framework.
I interpreted the task as a systems project rather than a model-quality project:
the most important output is not a strong trained language model, but a
framework that can construct a model, prepare data, run training, log evidence,
checkpoint, resume, measure speed, and support controlled iteration.

The core pipeline I built is:

```text
config -> data -> model -> optimizer -> scheduler -> trainer -> logger -> checkpoint -> resume
```

The final direction became more precise over time. I started with a small
training framework, but the project gradually became an **agentic training
feasibility and optimization harness** for causal language model experiments.
Its role is before a heavier system such as HuggingFace Trainer, DeepSpeed,
FSDP, or Megatron: it checks whether a model/data/config path is viable, records
speed/loss/memory evidence, classifies failures, and recommends the next safe
experiment.

I intentionally kept the supported claim narrow. The framework supports tested
causal-LM profiles, not every possible HuggingFace task or architecture. This
made the system more honest and easier to debug.

## 2. Agent Design And Workflow

The agent is deliberately rule-based and auditable. It does not pretend to invent
new training algorithms. Its job is harness engineering:

- select the next controlled experiment;
- launch training with a config;
- parse `events.jsonl`, `summary.json`, and preflight artifacts;
- write summaries and recommendations;
- append run metadata to a ledger;
- propose the next configuration based on observed bottlenecks and risks.

The first planner ladder was:

```text
configs/debug.yaml
configs/baseline_tinystories.yaml
configs/cached_tinystories.yaml
configs/mixed_precision.yaml
```

I validated this loop remotely with:

```bash
python3 -m agent.planner --goal improve_tokens_per_sec --run-next
```

The planner selected the TinyStories baseline, ran it, parsed the result, and
wrote an `agent_patch_proposal.md`. The proposal identified optimizer overhead
as large relative to forward time and suggested a batch-size /
gradient-accumulation sweep. This matched the timing data, so the agent was
useful as a structured experiment loop rather than a magic optimizer.

Later, I added two more agent modules:

- `agent/auto_probe.py`: expands a bounded token-budget ladder, stops on failure,
  and writes a recommendation artifact.
- `agent/stability_runner.py`: turns a selected recommendation into a longer
  stability run.

The most mature example is the DeepSeek stretch path. The agent expanded from
`64 -> 128 -> 256 -> 512` tokens per optimizer step, then followed its own
recommendations to `1024` and `2048`, and finally promoted the `2048`-token
profile only to a longer stability test, not to a final quality claim.

## 3. Framework Architecture

The implementation follows the official guide's suggested structure:

| module | role |
|---|---|
| `training_framework/config.py` | YAML-to-dataclass config and validation |
| `training_framework/data.py` | text loading, tokenization, token blocks, cache, dataloaders |
| `training_framework/model.py` | HuggingFace tokenizer/config/model construction |
| `training_framework/optim.py` | optimizer construction and parameter groups |
| `training_framework/scheduler.py` | warmup plus cosine scheduler |
| `training_framework/trainer.py` | explicit train/validation loop |
| `training_framework/logger.py` | console, JSONL, and TensorBoard logging |
| `training_framework/checkpoint.py` | model/optimizer/scheduler/global-step/RNG state |
| `training_framework/timing.py` | per-step timing breakdown |
| `agent/` | planner, runner, analyzer, matrix runner, auto-probe, recommendation |
| `selfcmd-workflow/` | local-to-remote development and validation workflow |

The training loop is intentionally visible:

```text
load batch -> forward -> backward -> gradient clip -> optimizer step
-> scheduler step -> log -> validate -> checkpoint
```

This visibility mattered. When something looked wrong, I could inspect the exact
stage: data time, forward time, backward time, optimizer time, validation,
checkpointing, or resume.

## 4. Profile-Based Extensibility

After the first working version, I added a profile-composition layer. The direct
config path still works:

```bash
python train.py --config configs/debug.yaml
```

The composed path supports base/model/data profiles plus overrides:

```bash
python train.py \
  --base configs/base/causal_lm_debug.yaml \
  --model-profile configs/model_profiles/tiny_gpt2.yaml \
  --data-profile configs/data_profiles/local_fixture.yaml \
  --override trainer.max_steps=4
```

Each composed run writes a resolved config and a `preflight.json` /
`preflight.md` report. Preflight includes model/profile metadata, tokenizer IDs,
dataset information, package availability, CUDA availability, parameter count,
tokens per optimizer step, and memory-risk estimates.

The tested profiles include:

- `tiny_gpt2` for very fast smoke tests;
- `distilgpt2` for the main TinyStories experiments;
- `gpt2` as a larger but still practical stress profile;
- `Qwen/Qwen2.5-0.5B` as a cross-family stretch profile;
- `deepseek-ai/deepseek-coder-1.3b-base` as a memory-risk stretch profile;
- local fixture, TinyStories, and WikiText-2 data profiles.

This layer clarified an important design boundary: I can switch among tested
causal-LM profiles comfortably, but I should not claim universal support for
encoder-only, seq2seq, multimodal, or arbitrary remote-code models without
task-specific adapters, collators, losses, and metrics.

## 5. Data Loading And Tokenization

I used three data modes:

- a local fixture corpus for fast smoke testing;
- a TinyStories subset for baseline and optimization experiments;
- WikiText-2 for real-data DeepSeek feasibility validation.

The data module supports HuggingFace datasets, local text files, train/validation
splits, fixed-length token blocks, and optional token-block caching. The cache
key includes dataset settings, tokenizer name, sequence length, max samples,
validation split, and a cache version.

Caching was still valuable even when it did not improve end-to-end throughput.
It made the data path reproducible and let me separate data-loading time from
compute/optimizer time.

## 6. Model, Optimizer, And Scheduler

For smoke tests I used `sshleifer/tiny-gpt2`. For the first main experiments I
used `distilgpt2`. I initialized models from HuggingFace config by default
rather than using pretrained weights, because the report focuses on framework
behavior and training-system evidence, not final model quality.

The base optimizer is AdamW with decay and no-decay parameter groups. Biases and
normalization parameters are excluded from weight decay. The scheduler is warmup
plus cosine decay, and learning rate is logged during training.

I later added Adafactor because the DeepSeek stretch profile exposed AdamW's
memory cost. This was not just adding another optimizer for completeness: it
changed which model/configs were feasible on a 24GB GPU.

## 7. Logging, Checkpointing, And Resume

Each run writes:

```text
runs/<run_name>-YYYYMMDDTHHMMSSZ/
  copied_config.yaml
  events.jsonl
  summary.json
  summary.md
  tb/
  checkpoints/
```

The framework logs train loss, validation loss, learning rate, tokens/sec,
timing breakdown, and CUDA memory metrics when CUDA is available.

Checkpoints include:

- model state;
- optimizer state;
- scheduler state;
- global step;
- config and metadata;
- RNG state when enabled.

One remote resume check loaded:

```text
/workspace/runs/debug-20260616T090134Z/checkpoints/step_000006.pt
```

The resumed run reached `global_step=8` and recorded:

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

This was important because a useful training checkpoint is not just model
weights. It must preserve enough state for training to continue correctly.

Although TensorBoard logs were generated under each run's `tb/` directory, I
used exported scalar tables in this report instead of screenshots. This made the
evidence easier to reproduce and compare: the same `events.jsonl` records can be
parsed into loss, throughput, timing, and memory tables without depending on a
particular UI view. The TensorBoard output still served as a quick visual sanity
check during development, especially for checking whether validation loss moved
in the same direction as training loss and whether throughput changed after a
configuration change.

## 8. Remote-First Development Workflow

My laptop was not a good place for repeated model caches and GPU runs, so I built
a remote-first workflow under `selfcmd-workflow/`.

The rule was:

- local `phase4-honor/` is the source of truth for code and report text;
- remote `/workspace` is the GPU execution area;
- heavy artifacts stay remote;
- lightweight summaries, logs, and markdown evidence are fetched back.

The typical loop became:

```bash
./selfcmd start
./selfcmd deploy-clean
./selfcmd install-deps
./selfcmd test
./selfcmd smoke
./selfcmd evidence
./selfcmd fetch
```

This workflow was not just convenience. Earlier phases taught me that official
evaluation often uses the server-side `/workspace`, so code synchronization,
remote validation, and artifact collection are part of the system.

### 8.1 Development Iteration Trail

The actual development process was not a straight line from "write framework" to
"run final experiment." It was a sequence of small probes, failures, fixes, and
new harness features.

The first milestone was a minimal end-to-end training path: `debug.yaml` had to
construct a tokenizer/model, create token blocks, train for a few steps, run
validation, write TensorBoard/JSONL logs, save checkpoints, and resume from a
checkpoint. This established the basic framework contract before I added larger
models or optimization experiments.

The second milestone was environment repair. The course container already had
PyTorch, PyYAML, and TensorBoard, but not the HuggingFace packages I needed. The
newest `transformers` release did not match the container's PyTorch version, so
I pinned a compatible stack. Later, an ONNX/protobuf import issue appeared
through `torch.optim.AdamW`; I fixed it by setting
`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before downstream torch imports.
These were not model improvements, but they were necessary for a repeatable
training system.

The third milestone was extensibility. I added the base/model/data profile
composition path, preflight reports, and the matrix runner. This produced useful
new bugs. For example, PyYAML parsed unquoted `mixed_precision: off` as boolean
`False`, which broke the trainer's string enum logic. Fixing that bug made the
config layer more robust. The profile-smoke run then proved that a composed
`tiny_gpt2 + local_fixture` config could train remotely, while the WikiText
smoke proved the data profile could switch away from the local fixture.

The fourth milestone was controlled scale-out. I did not jump directly from
DistilGPT2 to DeepSeek. I first ran a Qwen smoke, then 60-step GPT2/Qwen
TinyStories sanity runs, then a Qwen throughput probe. This revealed that the
first Qwen path was slow partly because it used too few tokens per optimizer
step and had gradient checkpointing enabled. That observation became a concrete
profile recommendation instead of a vague "Qwen is slow" conclusion.

The fifth milestone was DeepSeek safety gating. A direct AdamW smoke failed with
CUDA OOM, so I added `--preflight-only` and a safety matrix before trying to
"optimize" anything. Only after preflight separated model compatibility from
optimizer memory did I add Adafactor, CUDA memory metrics, token-budget
auto-probing, a stability runner, and finally the real-data WikiText run. Each
new feature came from a specific failure or uncertainty in the previous run.

The final milestone was workflow hardening. `selfcmd` grew from a convenience
script into a small remote development harness: deploy-clean, dependency repair,
remote tests, DeepSeek probes, evidence fetches, and clean tar archives that
avoid macOS metadata pollution. This is why I now consider the workflow itself
part of the Phase 4 system rather than a separate shell-script detail.

## 9. Tests And Validation

The tests cover config loading, data shape, cache keys, checkpoint errors,
planner behavior, matrix selection, preflight reports, CUDA memory metrics,
auto-probe, stability-runner logic, and recommendation generation.

Remote test progression:

| stage | remote result |
|---|---:|
| initial framework tests | `16 passed in 0.09s` |
| profile composition, matrix, preflight regressions | `24 passed in 2.55s` |
| DeepSeek/Adafactor/failure classification | `37 passed in 3.03s` |
| auto-probe and stability-runner coverage | `44 passed in 2.92s` |
| WikiText real-data auto-probe regression | `45 passed in 3.10s` |
| memory predictor and recommendation-report tests | `48 passed in 3.00s` |

The final remote validation I used for the current codebase was:

```text
48 passed
```

This made the report evidence stronger because the experiments were not only
manual runs; the harness itself gained regression coverage as bugs appeared.

## 10. Speed Measurement And Optimization

The framework logs tokens/sec and timing breakdowns for data loading, forward,
backward, and optimizer phases. I exported compact evidence tables from
`events.jsonl`.

### 10.1 Cache And Mixed Precision

Key rows from the TinyStories evidence table:

| run | cache | final train loss | final val loss | mean tokens/sec | data ms | forward ms | backward ms | optimizer ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline TinyStories | miss | 5.0250 | 4.9362 | 28036.48 | 0.3953 | 10.4002 | 15.5264 | 9.8224 |
| cached TinyStories | hit | 4.9360 | 4.9279 | 27610.90 | 0.2113 | 11.1361 | 15.5508 | 9.8035 |
| mixed precision | hit | 4.8527 | 4.9218 | 27610.99 | 0.2172 | 11.0823 | 15.5793 | 9.8195 |

This result was useful precisely because it was not a simple win. Cache hit
reduced measured data time, but overall throughput barely changed because the
run was dominated by forward/backward/optimizer time. Mixed precision was stable
but did not materially improve this small DistilGPT2-style setup.

The lesson was: optimization should follow timing evidence, not intuition.

### 10.2 Batch Size And Gradient Accumulation

The timing breakdown suggested that larger direct batches might amortize
per-step overhead. I then ran a batch/gradient-accumulation sweep:

| variant | avg tokens/sec | final val loss | interpretation |
|---|---:|---:|---|
| `bs2_ga1` | 15114.61 | 5.7362 | Smallest direct batch, lower throughput. |
| `bs4_ga1` | 20284.70 | 5.3667 | Larger direct batch improves utilization. |
| `bs8_ga1` | 23949.54 | 5.1880 | Best throughput and best validation loss. |
| `bs4_ga2` | 22139.84 | 5.1907 | Effective batch 8, but slower than true batch 8 because it repeats forward/backward work. |

The current recommendation for this TinyStories/DistilGPT2 path is therefore:

```text
batch_size=8, grad_accum_steps=1
```

Gradient accumulation remains useful as a memory fallback, but when a true
larger batch fits, it is faster.

## 11. Qwen Stretch Profile

I used Qwen as a cross-family stretch profile. A conservative smoke run with
`Qwen/Qwen2.5-0.5B`, `seq_len=64`, `batch_size=1`, and `max_steps=2` succeeded:

```text
parameters=494032768
device=cuda
final_loss=10.7192
```

This was compatibility evidence, not model-quality evidence. It showed that the
profile could construct tokenizer/model, build data, run train/validation steps,
checkpoint, and emit preflight evidence.

I then ran 60-step TinyStories sanity runs:

| run | model | steps | params | tokens/step | val loss path | mean tokens/sec |
|---|---|---:|---:|---:|---|---:|
| GPT2 TinyStories | `gpt2` | 60 | 124.4M | 256 | 7.2021 -> 6.1158 -> 5.9795 | 5460.35 |
| Qwen TinyStories | `Qwen/Qwen2.5-0.5B` | 60 | 494.0M | 64 | 9.4152 -> 7.3863 -> 7.2141 | 348.71 |

The Qwen run proved that the profile could keep training and validating, but it
also showed why Qwen should not be the default iteration path. It was much
slower, used a conservative token budget, and required `trust_remote_code`.

A later Qwen throughput probe clarified the reason:

| Qwen variant | avg tokens/sec | final val loss | interpretation |
|---|---:|---:|---|
| `s64_b1_gc_on` | 350.41 | NaN | Checkpointing was slow and numerically unstable in this short random-init run. |
| `s64_b1_gc_off` | 447.92 | 9.2218 | Removing checkpointing helped. |
| `s64_b2_gc_off` | 897.67 | 8.9437 | Larger direct batch nearly doubled throughput. |
| `s128_b1_gc_off` | 897.69 | 8.9839 | Longer sequence gave the same throughput gain while keeping batch size conservative. |

The updated Qwen recommendation is:

```text
seq_len=128, batch_size=1, grad_accum_steps=1, gradient_checkpointing=false
```

## 12. DeepSeek Stretch Profile

DeepSeek was the most useful stress test because it exposed a real memory
boundary. I first loaded `deepseek-ai/deepseek-coder-1.3b-base` as a random-init
LLaMA-style causal LM with about 1.346B parameters. A direct one-step AdamW
training smoke failed inside `optimizer.step()` with CUDA OOM on the 24GB RTX
3090.

That failure was useful because it localized the problem. The tokenizer/model
profile path worked; the immediate blocker was AdamW optimizer-state memory and
temporary optimizer buffers.

I added `--preflight-only` so the framework can build tokenizer/model, move the
model to device, write preflight evidence, and exit before optimizer/training.
Then I added a `deepseek_safety_probe` matrix:

| variant | execution | result | interpretation |
|---|---|---|---|
| `preflight_s16_b1` | preflight only | success | Profile/model/tokenizer/preflight works. |
| `adamw_s16_b1` | train | CUDA OOM | Full AdamW is unsafe on this 24GB setup. |
| `adafactor_s16_b1_no_ckpt` | train | success | Low-memory optimizer path can complete a real step. |

The Adafactor result changed the boundary. It showed that DeepSeek was not
universally impossible for the framework; AdamW was the specific memory problem.

I expanded this into short multi-step probes:

| DeepSeek Adafactor shape | avg tokens/sec | val loss | peak allocated / reserved |
|---|---:|---:|---|
| `s16_b1` | 47.38 | 10.8575 | about 10.98GB / 12.29GB |
| `s32_b1` | 84.54 | 10.8434 | about 10.98GB / 12.29GB |
| `s64_b1_gc_off` | 193.80 | 2.7425 | 11.00GB / 12.51GB |
| `s128_b1_gc_off` | 385.16 | 9.9014 | 11.02GB / 12.58GB |
| `s256_b1_gc_off` | 756.78 | 9.5584 | 11.06GB / 13.00GB |

Then the bounded `auto_probe` ladder reached `2048` tokens per optimizer step:

| token budget | avg tokens/sec | val loss | peak allocated / reserved | recommendation |
|---:|---:|---:|---:|---|
| 64 | 192.95 | 10.6877 | 11.00GB / 12.51GB | safe |
| 128 | 385.21 | 9.7545 | 11.02GB / 12.58GB | safe |
| 256 | 755.79 | 10.1239 | 11.06GB / 13.00GB | safe |
| 512 | 1343.92 | 8.6315 | 11.14GB / 13.67GB | try 1024 |
| 1024 | 2135.06 | 6.5600 | 11.39GB / 15.14GB | try 2048 |
| 2048 | 3092.42 | 3.5100 | 14.31GB / 18.58GB | run stability check |

I did not treat this as a final result, because the first probe used a small
local fixture. The correct next step was a longer stability run and a real data
profile.

The fixture stability run passed:

| metric | result |
|---|---:|
| requested/completed steps | `50/50` |
| avg tokens/sec | `3680.77` |
| first -> last train loss | `22.0702 -> 0.0651` |
| last validation loss | `0.0608` |
| peak allocated / reserved | `14.31GB / 18.58GB` |

The loss collapse was likely memorization because the fixture was tiny. I
recorded it as systems evidence, not quality evidence.

I then repeated the recommendation on WikiText-2 and extended it to 100 steps:

| DeepSeek WikiText stability metric | result |
|---|---:|
| requested/completed steps | `100/100` |
| avg tokens/sec | `3650.34` |
| first -> last train loss | `9.5252 -> 6.3307` |
| last validation loss | `6.6539` |
| peak allocated / reserved | `14.31GB / 18.83GB` |

This is the strongest DeepSeek evidence in the project. It is still not a
convergence claim, but it validates the harness claim: the agent found a
memory-safe configuration on a fixture, transferred it to a real dataset
profile, and confirmed 100 training steps without OOM, NaN, or memory drift.

## 13. Calibrated Memory Prediction

After the DeepSeek experiments, I added a memory predictor to preflight. It
separates parameter memory, gradient memory, optimizer state, an activation
proxy, and reserved allocator headroom.

For the 100-step DeepSeek WikiText run:

| memory quantity | predicted | observed | error |
|---|---:|---:|---:|
| allocated peak | 15051 MiB | 14305 MiB | +5.2% |
| reserved peak | 18814 MiB | 18828 MiB | -0.1% |

This helped explain the main DeepSeek result. AdamW is unsafe because fp32
weights, gradients, and two fp32 moment tensors push the reserved-memory estimate
beyond the 24GB device before activations and temporary buffers are considered.
Adafactor changes the boundary because its optimizer state is much smaller.

Finally, `agent/recommendation_report.py` merges auto-probe, stability,
preflight, and summary artifacts into a compact markdown/json recommendation. It
promotes the 2048-token Adafactor/WikiText path only for longer probing, not as a
final model-quality result.

## 14. Bugs, Pitfalls, And Debugging

Several bugs were more educational than the successful runs.

PyYAML boolean parsing:

- `mixed_precision: off` was parsed as boolean `False`.
- The trainer expected a string enum.
- I fixed the config loader to normalize boolean mixed-precision values and
  added a regression test.

Dependency mismatch:

- The course image used NVIDIA PyTorch 2.3.
- The newest `transformers` expected PyTorch >= 2.4.
- I pinned `transformers==4.41.2`, `numpy==1.24.4`, and a compatible `fsspec`.

ONNX/protobuf import issue:

- `torch.optim.AdamW` imported Dynamo/ONNX and hit a protobuf compatibility
  issue.
- The framework sets `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before
  downstream torch imports.

Gradient accumulation logging:

- The first matrix run logged accumulated train loss as a sum of microbatch
  losses.
- `bs4_ga2` looked artificially worse even though validation was sane.
- I fixed the trainer to log average accumulated loss.

Qwen NaN best-selection:

- A Qwen probe produced NaN validation loss.
- Matrix selection should not allow non-finite validation losses to win.
- I changed selection to ignore NaN losses when finite losses exist.

DeepSeek latest-run discovery:

- The first DeepSeek matrix summary accidentally treated `runs/matrix_logs` as a
  training run.
- I fixed run discovery to require `copied_config.yaml`, so auxiliary directories
  cannot pollute run summaries.

Stability-runner sparse log bug:

- The first stability runner counted train log events instead of optimizer
  steps.
- With `log_every=5`, a completed 50-step run looked like 10 steps.
- I fixed it to read `summary.json.global_step` and separately record
  `logged_train_events`.

These bugs changed how I think about training frameworks. A framework is not
only the training loop; it is also the evidence machinery around the loop. If
the logging, run discovery, or summary logic is wrong, the agent may make the
wrong decision even when training itself is correct.

## 15. Reflection And Lessons Learned

Training systems feel different from inference optimization. In earlier phases,
the central question was often "which kernel/path is fastest for this shape?" In
Phase 4, the central question became:

```text
Can every part of the training system preserve state, expose evidence,
and support reliable iteration?
```

That changed what counted as progress. A small smoke run with correct
checkpoint/resume was more valuable than a larger run with unclear state. A
neutral cache result was still useful because it showed that the bottleneck was
not data loading. A simple rule-based agent was more trustworthy than an
uncontrolled code-writing loop because its decisions could be traced to logs.

I also learned to separate compatibility evidence, feasibility evidence, and
quality evidence:

- A smoke run proves construction and basic execution.
- A short stability run proves a configuration can train without immediate
  failure.
- A longer real-data run begins to say something about training behavior.
- None of these automatically prove final model quality.

That distinction mattered especially for Qwen and DeepSeek. Qwen was valid as a
cross-family profile, but not the fastest iteration path. DeepSeek was valuable
because it exposed the AdamW memory boundary and forced the framework to become a
better feasibility harness.

## 16. Connection To Earlier Phases

The first three phases changed how I approached Phase 4.

Phase 1 and Phase 2 made me think in terms of measured operators and specific
shapes. In Phase 2, especially, an apparently surprising speedup was only useful
after I understood the benchmark pattern and separated a benchmark-aware runtime
optimization from a universal operator improvement. That lesson carried into
Phase 4: I tried not to call cache, mixed precision, Qwen, or DeepSeek "better"
unless timing, loss, and memory evidence supported the claim.

Phase 3 made me more careful about request management, batching, and remote
workflow reliability. That influenced the `selfcmd-workflow` design: server-side
`/workspace` state, deployment hygiene, fetched evidence, and repeatable tests
became part of the framework rather than afterthoughts.

I did not integrate previous custom kernels directly into the training loop. I
decided that Phase 4 would be stronger if it first demonstrated a complete,
observable training system. The kernel-level mindset still appears in the report
through timing breakdowns, batch-size sweeps, and memory-bound analysis, but the
main contribution is framework coordination rather than a new low-level kernel.

## 17. Final Submission Evidence Index

The final submission should be read together with these project artifacts:

- `report4-final.md`: polished final report for submission.
- `report4.md`: longer working record with more chronological detail.
- `training_framework/`: generated mini training framework.
- `agent/`: planner, matrix runner, auto-probe, stability runner, and
  recommendation generator.
- `configs/`: base configs, model profiles, data profiles, matrices, and
  auto-probe profiles.
- `tests/`: regression tests covering config, data, checkpointing, preflight,
  matrix selection, auto-probe, stability, and recommendation reporting.
- `selfcmd-workflow/`: remote development and validation workflow.

The most important evidence points are the final remote test result
(`48 passed`), checkpoint/resume metadata with RNG restoration, TinyStories
throughput/timing tables, Qwen stretch-profile comparison, DeepSeek AdamW OOM
classification, DeepSeek Adafactor/WikiText 100-step stability, and calibrated
memory prediction.

Concrete source-file map, with paths relative to `phase4-honor/`. The compact
copies are in `final_evidence/`; their original remote artifact paths are listed
in `final_evidence/README.md`.

| Claim or result | Primary source files | What the files verify |
| --- | --- | --- |
| Final framework regression state | `final_evidence/tests/tests-20260621T095143Z.log` | Remote pytest collected 48 tests and ended with `48 passed in 3.00s`. |
| Remote artifact snapshot | `final_evidence/README.md` and `final_evidence/phase4-artifacts-20260621T095347Z.tar.gz` | The fetched evidence bundle used for the final report and the curated evidence-copy map. |
| Checkpoint/resume and RNG restoration | `final_evidence/resume/debug-resume-rng-summary.md`, `final_evidence/resume/debug-resume-rng-events.jsonl`, and `tests/test_checkpoint_resume.py` | Resume behavior was tested and a remote run emitted resume-related evidence. |
| TinyStories batch/grad-accumulation sweep | `final_evidence/matrix_summaries/batch_grad_sweep-20260616T144038Z.md` and matching `.json` | `bs8_ga1` was selected because it had the best throughput among validation-sane candidates. |
| Qwen stretch-profile comparison | `final_evidence/matrix_summaries/qwen_throughput_probe-20260619T033125Z.md` and matching `.json` | Gradient checkpointing was slower/unstable in the short Qwen probe; `s128_b1_gc_off` was the best throughput candidate. |
| Qwen longer compatibility run | `final_evidence/qwen/qwen-long-tinystories-summary.md`, `final_evidence/qwen/qwen-long-tinystories-preflight.md`, and `final_evidence/qwen/qwen-long-tinystories-events.jsonl` | Qwen could run through the framework, checkpoint, and emit metrics, but remained slower than the tiny GPT-style profile. |
| DeepSeek AdamW failure classification | `final_evidence/deepseek/deepseek_safety_probe-20260619T070456Z.md` and `final_evidence/deepseek/deepseek-adamw-oom-agent-summary.md` | AdamW was classified as `cuda_oom`, making optimizer choice a feasibility decision rather than a small tuning detail. |
| DeepSeek Adafactor token-budget auto-probe | `final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-20260621T072201Z.md` and matching `.json` | WikiText-2 probes at 512/1024/2048 tokens per step all passed, with `2048` selected for stability validation. |
| DeepSeek 100-step real-data stability | `final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-stability-20260621T095331Z.md`, matching `.json`, and `final_evidence/deepseek/deepseek-wikitext-2048-100step-events.jsonl` | The selected DeepSeek/WikiText path completed `100/100` steps, averaged `3650.34` tokens/sec, and avoided OOM/NaN. |
| Final recommendation and memory calibration | `final_evidence/recommendations/phase4-current-recommendation.md` and matching `.json` | The final recommended configuration, risk notes, next steps, and predicted-vs-actual CUDA memory calibration. |

## 18. Limitations And Future Work

The framework is intentionally small. It does not implement distributed training,
ZeRO, FSDP, Megatron-style tensor/pipeline parallelism, or custom CUDA kernels.
It does not claim that cache or mixed precision always improves speed. In this
setup, cache improved data-time control but not total throughput, and mixed
precision was neutral.

The most important future improvements would be:

- a cleaner final workspace layout so Phase 4 artifacts are not mixed with older
  phases;
- automatic plots from JSONL/TensorBoard scalars;
- config hashes and environment hashes in every run summary;
- repeated trials and confidence intervals for matrix comparisons;
- a 200-500 step DeepSeek real-data run before treating the 2048-token path as a
  durable recipe;
- memory-predictor calibration across more model families;
- optional export from this feasibility harness into HuggingFace Trainer,
  DeepSpeed, or FSDP configs.

The final result is not a replacement for those heavy training systems. It is a
small, inspectable control plane that helps decide what is safe and worthwhile
to try next.

## 19. Appendix: Selected Raw Evidence

This appendix keeps the final report usable even if only this markdown file is
submitted. I include the most important compact outputs here and keep the full
copied evidence files under `final_evidence/`.

### 19.1 Remote Regression Test Output

Source: `final_evidence/tests/tests-20260621T095143Z.log`

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

Source: `final_evidence/matrix_summaries/batch_grad_sweep-20260616T144038Z.md`

| Variant | Status | Avg tokens/sec | Last val loss | Complexity |
| --- | ---: | ---: | ---: | --- |
| `bs2_ga1` | `0` | 15114.61 | 5.7362 | batch=2, grad_accum=1 |
| `bs4_ga1` | `0` | 20284.70 | 5.3667 | batch=4, grad_accum=1 |
| `bs8_ga1` | `0` | 23949.54 | 5.1880 | batch=8, grad_accum=1 |
| `bs4_ga2` | `0` | 22139.84 | 5.1907 | batch=4, grad_accum=2 |

Selected variant: `bs8_ga1`. It had the highest average tokens/sec among
validation-sane candidates, while `bs4_ga2` showed that gradient accumulation is
not equivalent to a true larger batch when extra forward/backward passes are
needed.

### 19.3 Qwen Stretch-Profile Evidence

Sources: `final_evidence/matrix_summaries/qwen_throughput_probe-20260619T033125Z.md`
and `final_evidence/qwen/qwen-long-tinystories-summary.md`

| Variant | Status | Avg tokens/sec | Last val loss | Main difference |
| --- | ---: | ---: | ---: | --- |
| `s64_b1_gc_on` | `0` | 350.41 | nan | seq_len=64, batch=1, checkpointing on |
| `s64_b1_gc_off` | `0` | 447.92 | 9.2218 | seq_len=64, batch=1, checkpointing off |
| `s64_b2_gc_off` | `0` | 897.67 | 8.9437 | seq_len=64, batch=2, checkpointing off |
| `s128_b1_gc_off` | `0` | 897.69 | 8.9839 | seq_len=128, batch=1, checkpointing off |

The Qwen longer run reached 60 steps with 494,032,768 parameters, final loss
`7.695265`, final learning rate `3e-05`, and 3,510 token blocks. This was useful
compatibility evidence, but not the fastest training path in this hardware and
framework setup.

### 19.4 DeepSeek AdamW Safety Gate

Source: `final_evidence/deepseek/deepseek_safety_probe-20260619T070456Z.md`

| Variant | Execution | Status | Failure | Complexity |
| --- | --- | ---: | --- | --- |
| `preflight_s16_b1` | preflight_only | `0` |  | batch=1, grad_accum=1, seq_len=16, checkpointing=True |
| `adamw_s16_b1` | train | `1` | cuda_oom | batch=1, grad_accum=1, seq_len=16, checkpointing=True |

This result changed the design direction. The issue was not simply "make the
learning rate smaller"; the optimizer state and memory boundary made AdamW an
unsafe default for this DeepSeek profile on the available GPU. That is why the
framework added stronger preflight checks, failure classification, and
Adafactor-based probes.

### 19.5 DeepSeek Adafactor Auto-Probe And Stability

Sources:
`final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-20260621T072201Z.md`
and
`final_evidence/deepseek/deepseek_adafactor_wikitext_realdata-stability-20260621T095331Z.md`

| Variant | Status | Tokens/step | Avg tokens/sec | Last val loss | Peak CUDA MB |
| --- | ---: | ---: | ---: | ---: | --- |
| `tok512` | `0` | 512 | 1341.83 | 9.8190 | 11173/13666 |
| `tok1024` | `0` | 1024 | 2137.52 | 10.7984 | 11899/15140 |
| `tok2048` | `0` | 2048 | 3100.38 | 10.0320 | 14305/18828 |

The selected `tok2048` configuration then passed the 100-step stability run:

| Metric | Value |
| --- | ---: |
| Completed train steps | 100/100 |
| Logged train events | 10 |
| Validation events | 5 |
| Average tokens/sec | 3650.34 |
| Last train loss | 6.3306779861450195 |
| Last validation loss | 6.653914451599121 |
| Peak CUDA MB allocated/reserved | 14305/18828 |

### 19.6 Final Recommendation And Memory Calibration

Source: `final_evidence/recommendations/phase4-current-recommendation.md`

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

The recommendation is to promote this configuration for a longer probe, while
keeping AdamW classified as unsafe unless offload or sharding is introduced.
