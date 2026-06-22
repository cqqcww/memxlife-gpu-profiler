# Phase 4 Honor Report: Agentic Mini Training Framework

## 1. Project Goal And Scope

The goal of Phase 4 is to build an agent-organized mini training framework. I
kept the framework intentionally small so that the training pipeline is explicit
instead of hidden inside HuggingFace Trainer, PyTorch Lightning, or a large
distributed system.

The core path is:

```text
config -> data -> model -> optimizer -> scheduler -> trainer -> logger -> checkpoint -> resume
```

The main deliverable is not a high-quality trained model. The main deliverable is
a working, observable training system and an agent loop that can run experiments,
inspect logs, record evidence, and propose the next controlled change.

## 1.1 Current Positioning

After the Qwen and DeepSeek stretch probes, I would describe the project more
precisely as an agentic training feasibility and optimization harness for
causal-LM experiments. It is not trying to replace HuggingFace Trainer,
DeepSpeed, FSDP, Megatron, or PyTorch Lightning. Its role is earlier in the
workflow: check whether a model/data/config path is viable, run small controlled
experiments, record speed/loss/memory evidence, classify failures, and decide
whether a setup deserves longer training or migration into a heavier training
system.

This refined direction still matches the original scaffold goal of comfortable
model/data switching, but it narrows the claim. The framework can switch among
tested causal-LM profiles, not arbitrary architectures and tasks. The most useful
result is therefore not "this scaffold trains the biggest model"; it is "this
scaffold explains what is safe, what is risky, and what evidence supports the
next step."

The newest implementation step follows this positioning directly: an
`auto_probe` agent module can expand a bounded token-budget ladder, stop on
failure, classify OOM or runtime errors, and write a recommendation artifact.
For the DeepSeek stretch path, the probe ladder started at `64 -> 128 -> 256 ->
512` tokens per optimizer step with Adafactor and checkpointing disabled, then
followed its own recommendations to `1024` and `2048`. A paired
`stability_runner` can now turn a selected recommendation into a longer sanity
run. The important output is not just a faster number; it is a concrete answer
such as "try a larger budget", "promote the last safe config", or "this
configuration is stable enough to test on a real dataset."

## 2. Agent Design And Workflow

The agent is deliberately rule-based and auditable. It does not pretend to
autonomously discover new training algorithms. Its job is harness engineering:

- choose the next experiment from a deterministic ladder,
- launch training with a config,
- parse `events.jsonl` and `summary.json`,
- write `agent_summary.json` and `agent_summary.md`,
- append `runs/ledger.jsonl`,
- produce a small patch/config proposal with bottleneck, expected effect, risk,
  and rollback plan.

The current planner ladder is:

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

The planner selected `configs/baseline_tinystories.yaml`, ran it, parsed the
latest run, and wrote:

```text
/workspace/runs/baseline-tinystories-20260616T090824Z/agent_patch_proposal.md
```

The proposal identified optimizer overhead as large relative to forward time and
suggested a batch-size or gradient-accumulation sweep to amortize per-step
overhead. This matched the timing evidence, so the agent was useful as a
structured experiment loop rather than a magic optimizer.

## 3. Framework Architecture

The implementation follows the official guide's recommended module structure:

- `training_framework/config.py`: YAML-to-dataclass config with validation.
- `training_framework/data.py`: text loading, tokenization, fixed token blocks, cache, dataloaders.
- `training_framework/model.py`: HuggingFace tokenizer/config/model construction.
- `training_framework/optim.py`: AdamW with decay/no-decay parameter groups.
- `training_framework/scheduler.py`: warmup plus cosine decay.
- `training_framework/trainer.py`: explicit train/validate loop with timing and checkpointing.
- `training_framework/logger.py`: console, JSONL, and TensorBoard logging.
- `training_framework/checkpoint.py`: model/optimizer/scheduler/global-step/RNG checkpoint state.
- `training_framework/timing.py`: per-step timing breakdown.
- `agent/`: deterministic planner, runner, analyzer, and ledger.
- `selfcmd-workflow/`: local-to-remote development workflow.

The training loop is intentionally visible:

```text
load batch -> forward -> backward -> gradient clip -> optimizer step
-> scheduler step -> log -> validate -> checkpoint
```

This made debugging easier because every phase had a timing measurement and a
log event.

## 3.1 Profile-Based Extensibility

After the first working framework pass, I added a profile-composition layer so
the scaffold can switch model/data choices without rewriting the training loop.
The supported claim is intentionally narrow:

```text
causal-LM mini training framework with extensible model/data profiles
```

This is different from claiming universal support for every HuggingFace
architecture. A truly seamless framework for decoder-only LM, encoder-only
classification, seq2seq, multimodal, and custom remote-code models would need
task adapters, different collators, different losses, and different evaluation
metrics. I kept the current implementation focused on GPT-style
`AutoModelForCausalLM` training.

The new composition path is:

```bash
python train.py \
  --base configs/base/causal_lm_debug.yaml \
  --model-profile configs/model_profiles/tiny_gpt2.yaml \
  --data-profile configs/data_profiles/local_fixture.yaml \
  --override trainer.max_steps=4
```

The direct config path still works:

```bash
python train.py --config configs/debug.yaml
```

Each composed run saves the resolved config and writes a `preflight.json` /
`preflight.md` report with profile metadata, model parameter count, tokenizer
IDs, data source, package availability, CUDA availability, and tokens per
optimizer step. I also added a YAML-driven matrix runner for cache on/off and
batch-size / gradient-accumulation sweeps. This turns the next optimization
question into an auditable experiment matrix rather than hand-written command
history.

The first remote validation pass surfaced a useful YAML bug: PyYAML parsed
unquoted `mixed_precision: off` as boolean `False`, while the trainer expected a
string enum. I fixed this in the config loader by normalizing boolean
mixed-precision values back to `"off"` / `"auto"` and added a regression test.

After the fix and the later matrix/preflight enhancements, the expanded
extension suite passed remote validation on the course GPU server:

```text
24 passed in 2.55s
```

After adding the DeepSeek safety matrix, failure classification, optional
preflight-only execution, Adafactor optimizer support, checkpoint-disable path,
CUDA memory logging, and the DeepSeek token-budget probes, the newer remote
validation pass remained stable:

```text
37 passed in 3.03s
```

After adding the auto-probe and stability-runner tests, including a regression
test for sparse train logging during long stability runs, the latest remote
validation became:

```text
44 passed in 2.92s
```

After adding the WikiText real-data auto-probe regression, the latest remote
validation became:

```text
45 passed in 3.10s
```

After adding the calibrated memory predictor and recommendation-report generator,
the latest remote validation became:

```text
48 passed in 3.00s
```

The profile-composed smoke run produced:

```text
/workspace/runs/profile-smoke-20260616T140500Z
```

Its preflight report confirmed:

```text
model_profile=tiny_gpt2
data_profile=local_fixture
device=cuda
mixed_precision=off
parameters=102714
```

The batch/gradient-accumulation matrix dry-run also expanded four resolved
configs remotely. I then ran a real WikiText-2 profile smoke and a full
batch/gradient-accumulation matrix. The WikiText-2 smoke proved that the data
profile path can switch from a local fixture to a HuggingFace dataset, build
token blocks, validate, checkpoint, and emit preflight evidence.

Finally, I ran a conservative Qwen stretch smoke with `Qwen/Qwen2.5-0.5B`,
`seq_len=64`, `batch_size=1`, and `max_steps=2`. This was not meant as a quality
experiment; it was a compatibility and memory check. It succeeded:

```text
/workspace/runs/qwen-profile-smoke-20260616T144427Z
parameters=494032768
device=cuda
final_loss=10.7192
```

The Qwen preflight warned explicitly that `trust_remote_code` was enabled. This
is acceptable for a stretch profile, but it should remain explicit in the report
because it changes the safety and reproducibility story compared with GPT-style
profiles.

In this report, a "smoke" result means a compatibility and systems check: the
profile can construct the tokenizer/model, build data, run at least a few
training and validation steps, write checkpoints, and emit preflight evidence.
It is a real execution result, but it is not a quality claim or a long-training
result. Longer runs are needed before comparing model behavior beyond basic
compatibility.

After the smoke path was stable, I ran two controlled longer sanity runs on
TinyStories to compare short smoke evidence against real training-loop evidence:

| run | model | steps | params | tokens/optimizer step | val loss path | mean tokens/sec |
|---|---|---:|---:|---:|---|---:|
| `gpt2-long-tinystories-20260616T145821Z` | `gpt2` | 60 | 124.4M | 256 | 7.2021 -> 6.1158 -> 5.9795 | 5460.35 |
| `qwen-long-tinystories-20260616T145919Z` | `Qwen/Qwen2.5-0.5B` | 60 | 494.0M | 64 | 9.4152 -> 7.3863 -> 7.2141 | 348.71 |

This comparison clarified the role of smoke tests. The Qwen smoke proved that
the cross-family profile could start safely. The 60-step Qwen run proved that it
could keep training, validate repeatedly, and checkpoint. But it also showed why
Qwen should remain a stretch path rather than the default experiment path: it is
much larger, uses a conservative token budget, requires `trust_remote_code`, and
is roughly an order of magnitude slower in this setup. The full `gpt2` run gives
a more practical stress profile for report evidence because it is larger than
DistilGPT2 but still fast enough for iterative experiments.

I later revisited the Qwen result because the first longer run was intentionally
conservative. A targeted Qwen throughput probe separated three factors:
gradient checkpointing, direct batch size, and sequence length. This showed that
the original Qwen setup was not simply "the framework is bad for Qwen"; it was
underfeeding tokens per optimizer step and paying extra recomputation cost.

| Qwen probe variant | avg tokens/sec | final val loss | interpretation |
|---|---:|---:|---|
| `s64_b1_gc_on` | 350.41 | NaN | Matches the slow original path; checkpointing made backward much slower and was numerically unstable in this short random-init run. |
| `s64_b1_gc_off` | 447.92 | 9.2218 | Removing checkpointing improved throughput and restored finite loss. |
| `s64_b2_gc_off` | 897.67 | 8.9437 | Doubling direct batch almost doubled tokens/sec with similar step time. |
| `s128_b1_gc_off` | 897.69 | 8.9839 | Doubling sequence length gave the same throughput gain while keeping batch size conservative. |

The updated Qwen recommendation is therefore `seq_len=128`, `batch_size=1`,
`grad_accum_steps=1`, and `gradient_checkpointing=false` as the first stretch
profile. If it OOMs on a smaller GPU, the safer fallback is to reduce sequence
length or batch size before re-enabling checkpointing.

I then tried the DeepSeek stretch profile in the same staged way. First I loaded
only the config and tokenizer for `deepseek-ai/deepseek-coder-1.3b-base`; this
worked and identified the model as a LLaMA-style causal LM with about 1.346B
parameters in our random-init setup. A direct one-step smoke with `seq_len=16`,
`batch_size=1`, gradient checkpointing enabled, and mixed precision reached
model setup but failed inside `AdamW optimizer.step()` with CUDA OOM. That
failure was useful because it localized the boundary: DeepSeek was not blocked
by tokenizer/profile compatibility, but by optimizer-state memory and temporary
AdamW buffers on the 24GB RTX 3090.

To make that failure actionable instead of just a traceback, I added
`--preflight-only` to `train.py`. This mode builds tokenizer/model, moves the
model to the target device, writes `preflight.json` / `preflight.md`, and exits
before dataloaders, optimizer state, forward/backward, or checkpointing. The
DeepSeek preflight-only run succeeded and produced this recommendation:

```text
Estimated AdamW state is large relative to GPU memory before activations and
temporary optimizer buffers; prefer preflight-only, a low-memory optimizer,
offload, smaller seq_len/batch_size, or a smaller model.
```

I also added a `deepseek_safety_probe` matrix so the agent can record the staged
result:

| DeepSeek probe variant | execution | status | failure | interpretation |
|---|---|---:|---|---|
| `preflight_s16_b1` | `preflight_only` | 0 | none | Profile/model/tokenizer/preflight path works. |
| `adamw_s16_b1` | `train` | 1 | `cuda_oom` | Full AdamW training is not safe on this 24GB setup without a low-memory strategy. |
| `adafactor_s16_b1_no_ckpt` | `train` | 0 | none | A low-memory optimizer plus disabled checkpointing can complete a real one-step DeepSeek train path. |

The Adafactor result is a meaningful boundary shift, not a final model result.
It shows that the earlier DeepSeek failure was not a universal incompatibility
with the framework. The blocker was specifically the memory footprint of AdamW
state and temporary optimizer buffers. With Adafactor and checkpointing disabled,
the same model family completed one training step at about `18.52` tokens/sec
with train loss `10.6670`. This is slow and still only a smoke-level result, but
it gives the next optimization path a concrete starting point: extend the
low-memory DeepSeek profile to a short multi-step run, add validation when memory
allows, and only then consider it a serious stretch profile.

I then ran that follow-up as `deepseek_adafactor_probe`, adding CUDA memory
metrics to every setup/train/validation event. Both longer low-memory variants
completed and validated:

| DeepSeek Adafactor probe | status | avg tokens/sec | val loss | peak allocated / reserved |
|---|---:|---:|---:|---|
| `s16_b1_5step_val` | 0 | 47.38 | 10.8575 | about 10.98GB / 12.29GB |
| `s32_b1_3step_val` | 0 | 84.54 | 10.8434 | about 10.98GB / 12.29GB |

The first step in each run was much slower because it paid warm-up and allocator
costs. After that, `seq_len=16` stabilized around `54.5` tokens/sec and
`seq_len=32` stabilized around `108` tokens/sec. I then expanded the probe in
two steps:

| DeepSeek Adafactor shape | avg tokens/sec | val loss | peak allocated / reserved |
|---|---:|---:|---|
| `s64_b1_gc_on` | 168.99 | 3.3519 | 10.99GB / 12.32GB |
| `s64_b1_gc_off` | 193.80 | 2.7425 | 11.00GB / 12.51GB |
| `s128_b1_gc_off` | 385.16 | 9.9014 | 11.02GB / 12.58GB |
| `s64_b2_gc_off` | 385.81 | 10.0552 | 11.02GB / 12.58GB |
| `s256_b1_gc_off` | 756.78 | 9.5584 | 11.06GB / 13.00GB |
| `s128_b2_gc_off` | 756.23 | 9.9811 | 11.06GB / 12.99GB |
| `s64_b4_gc_off` | 759.65 | 10.7421 | 11.06GB / 13.01GB |

The important result is not the absolute loss, because this is still a tiny
random-init fixture run. The important systems result is that Adafactor can
train and validate DeepSeek for multiple steps with stable memory, and direct
token-budget scaling improved throughput almost linearly up to 256 tokens per
step. At this stage, `seq_len=256,batch_size=1,gradient_checkpointing=false,
optimizer=adafactor` was the best validated DeepSeek stretch shape.

After adding the `auto_probe` agent, I reran the same idea as a bounded
token-budget search. The first ladder tested `64 -> 128 -> 256 -> 512`
tokens/step and wrote a recommendation artifact instead of requiring manual log
reading. Because `512` used only about 13.7GB reserved CUDA memory, the tool
recommended a `1024` follow-up. That also succeeded, so I ran a `2048`
follow-up:

| DeepSeek auto-probe | avg tokens/sec | val loss | peak allocated / reserved | recommendation |
|---|---:|---:|---:|---|
| `tok64` | 192.95 | 10.6877 | 11.00GB / 12.51GB | safe |
| `tok128` | 385.21 | 9.7545 | 11.02GB / 12.58GB | safe |
| `tok256` | 755.79 | 10.1239 | 11.06GB / 13.00GB | safe |
| `tok512` | 1343.92 | 8.6315 | 11.14GB / 13.67GB | try 1024 |
| `tok1024` | 2135.06 | 6.5600 | 11.39GB / 15.14GB | try 2048 |
| `tok2048` | 3092.42 | 3.5100 | 14.31GB / 18.58GB | run stability check |

The updated short-probe conclusion is that `2048` tokens/step is the best
bounded safe point observed so far on the 24GB RTX 3090. However, it should not
be promoted directly to a final training configuration: the probe ran only three
steps on a medium local fixture. The correct next action is a 50-100 step
stability run at `2048` tokens/step, followed by the same probe on a real
TinyStories or WikiText subset.

I then added `agent/stability_runner.py` and ran the recommended 50-step
stability check. The first version of the stability runner found a real
harness bug: it counted train log events instead of actual optimizer steps, so
with `log_every=5` a completed 50-step run looked like only 10 steps. I fixed
the runner to prefer `summary.json.global_step` and keep `logged_train_events`
as a separate diagnostic. After the fix, the same DeepSeek `2048` path passed:

| DeepSeek stability run | result |
|---|---:|
| requested/completed steps | `50/50` |
| logged train events / validation events | `10 / 5` |
| average tokens/sec | `3680.77` |
| first -> last train loss | `22.0702 -> 0.0651` |
| last validation loss | `0.0608` |
| peak allocated / reserved | `14.31GB / 18.58GB` |

This was good framework evidence but not a quality claim. The run used the
medium local fixture, which contains very little text, so the fast loss collapse
was likely memorization. The systems conclusion was narrower and stronger:
`Adafactor + no checkpoint saving + gradient_checkpointing=false + 2048
tokens/step` can train a 1.346B-parameter DeepSeek-family random-init causal LM
for 50 optimizer steps on the RTX 3090 without OOM or NaN.

The next meaningful gate was to repeat the same recommendation on a real data
profile, so I added `deepseek_adafactor_wikitext_realdata`. This uses the
WikiText-2 data profile while keeping the same low-memory optimizer path. The
three short probes all succeeded:

| DeepSeek WikiText probe | avg tokens/sec | val loss | peak allocated / reserved | recommendation |
|---|---:|---:|---:|---|
| `tok512` | 1341.83 | 9.8190 | 11.17GB / 13.67GB | safe |
| `tok1024` | 2137.52 | 10.7984 | 11.90GB / 15.14GB | safe |
| `tok2048` | 3100.38 | 10.0320 | 14.31GB / 18.83GB | run stability check |

The matching 50-step WikiText stability run passed, and I then extended it to a
100-step sanity run after adding the calibrated memory predictor:

| DeepSeek WikiText stability run | result |
|---|---:|
| requested/completed steps | `100/100` |
| logged train events / validation events | `10 / 5` |
| average tokens/sec | `3650.34` |
| first -> last train loss | `9.5252 -> 6.3307` |
| last validation loss | `6.6539` |
| peak allocated / reserved | `14.31GB / 18.83GB` |

This result is much stronger than the fixture-only result. It still is not a
final model-quality claim, because it is only 100 steps on a small WikiText-2
subset. But it does validate the core feasibility-harness claim: the agent loop
found a memory-safe DeepSeek configuration on a toy fixture, transferred it to a
real dataset profile, and confirmed that it trains and validates for 100 steps
without OOM, NaN, or memory drift.

I also added a calibrated memory predictor to preflight. The predictor splits
memory into parameter, gradient, optimizer-state, activation-proxy, and reserved
allocator headroom components. For the 100-step DeepSeek WikiText run, the
prediction was close to the measured CUDA metrics:

| memory quantity | predicted | observed | error |
|---|---:|---:|---:|
| allocated peak | 15051 MiB | 14305 MiB | +5.2% |
| reserved peak | 18814 MiB | 18828 MiB | -0.1% |

This is why the harness can now explain the main DeepSeek finding. AdamW is
unsafe because fp32 weights, gradients, and two fp32 moment tensors push the
reserved-memory estimate beyond the 24GB device before activation and temporary
optimizer buffers are considered. Adafactor changes the boundary because its
factored optimizer state is much smaller, leaving enough room for a 2048-token
training step while still staying under the measured memory limit.

Finally, I added `agent/recommendation_report.py`, which generates a compact
machine-written recommendation from the auto-probe, stability, preflight, and
summary artifacts. The generated recommendation promotes the 2048-token
Adafactor/WikiText path for longer probing, while explicitly keeping it as
feasibility evidence rather than a convergence claim.

## 4. Data Loading And Tokenization

I used three data modes:

- A local fixture corpus for fast smoke testing.
- A TinyStories subset for the main baseline and optimization experiments.
- A WikiText-2 profile for real-data DeepSeek feasibility validation.

The data module supports HuggingFace datasets, local text files, fixed-length
token blocks, train/validation splitting, and optional token-block caching. The
cache key includes dataset settings, tokenizer name, sequence length, max
samples, validation split, and a cache version.

This was useful even when caching did not improve steady-state throughput. The
cache still made the data path reproducible and allowed a direct cache miss vs
cache hit comparison.

## 5. Model, Optimizer, And Scheduler

For smoke tests I used:

```text
sshleifer/tiny-gpt2
```

For the main TinyStories experiments I used:

```text
distilgpt2
```

The model is initialized from HuggingFace config by default rather than using
pretrained weights. This keeps the report focused on framework behavior instead
of model quality.

The optimizer is AdamW with decay and no-decay parameter groups. Biases and norm
parameters are excluded from weight decay. The scheduler is warmup plus cosine
decay, and learning rate is logged during training.

## 6. Logging, Checkpointing, And Resume

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

Checkpoints contain:

- model state,
- optimizer state,
- scheduler state,
- global step,
- extra metadata,
- RNG state when enabled.

I hardened resume during the implementation pass. The loader now gives readable
errors for missing checkpoints, restores RNG state when present, and records
resume metadata in `summary.json`.

Remote resume evidence:

```text
/workspace/runs/debug-resume-rng-20260616T090218Z
```

The resumed run loaded:

```text
/workspace/runs/debug-20260616T090134Z/checkpoints/step_000006.pt
```

The summary recorded:

```json
{
  "global_step": 8,
  "resume": {
    "global_step": 6,
    "optimizer_lr": 5e-05,
    "scheduler_lrs": [5e-05, 5e-05],
    "rng_restored": true
  }
}
```

This shows that resume is not just reloading weights. It also restores optimizer
and scheduler state, resumes from the expected step, and restores RNG state.

## 7. Remote-First Development Workflow

The local laptop does not have enough space for repeated model caches,
checkpoints, and full training artifacts. I therefore built a remote-first
workflow under:

```text
selfcmd-workflow/
```

The principle is:

- local `phase4-honor/` is the source of truth for code and report text,
- remote `/workspace` on the course GPU server is the execution area,
- heavy artifacts stay remote,
- lightweight evidence is fetched back locally.

Typical loop:

```bash
./selfcmd start
./selfcmd deploy-clean
./selfcmd install-deps
./selfcmd test
./selfcmd smoke
./selfcmd evidence
./selfcmd fetch
```

After the DeepSeek experiments, I tightened this workflow further. `./selfcmd
bootstrap` now performs deploy-clean, dependency repair, and remote tests in one
pass; `./selfcmd deepseek-probe` runs the staged DeepSeek matrix; and the source
archive uses a cleaner tar format so macOS metadata files do not pollute the
server workspace. This is a small example of harness engineering: the model code
matters, but the iteration loop has to be reliable too.

This lesson came from earlier phases: official submit/evaluation paths often
operate on server-side `/workspace`, so code synchronization is part of the
system, not an afterthought.

## 8. Debugging Evidence

The remote course container had PyTorch, PyYAML, and TensorBoard installed, but
not HuggingFace `transformers` or `datasets`. Installing the newest
`transformers` first failed because the course image used NVIDIA PyTorch 2.3,
while the newest `transformers` expected PyTorch >= 2.4.

The fix was to pin:

```text
transformers==4.41.2
numpy==1.24.4
fsspec[http]<=2026.4.0,>=2023.1.0
```

Another issue appeared when `torch.optim.AdamW` imported Dynamo/ONNX and hit an
ONNX/protobuf compatibility error. The framework sets this before downstream
torch imports:

```text
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

This is acceptable here because the framework does not use ONNX export.

## 9. Tests And Validation

The tests cover:

- config loading and bad config failures,
- data cache key changes,
- token block shape,
- train/validation split behavior,
- missing checkpoint error messages,
- planner ladder behavior,
- ledger parsing,
- latest-run detection.

Remote test evidence:

```text
/workspace/selfcmd-workflow/logs/remote-tests.log
```

Initial remote result:

```text
16 passed in 0.09s
```

After adding profile composition, matrix selection, preflight column checks, and
the YAML/mixed-precision regression test, the expanded remote suite passed:

```text
24 passed in 2.55s
```

After the DeepSeek/Adafactor pass, the latest remote suite passed:

```text
37 passed in 3.03s
```

After adding auto-probe and stability-runner coverage, including the
`global_step` vs sparse-log-count regression, the latest remote suite passed:

```text
44 passed in 2.92s
```

After adding the WikiText real-data auto-probe config test, the latest remote
suite passed:

```text
45 passed in 3.10s
```

After adding the memory predictor and recommendation-report tests, the latest
remote suite passed:

```text
48 passed in 3.00s
```

The remote smoke run also trained `configs/debug.yaml` end to end and saved
checkpoints:

```text
/workspace/runs/debug-20260616T090134Z
```

## 10. Speed Measurement And Experiment Results

The framework logs tokens/sec and timing breakdowns for data loading, forward,
backward, and optimizer phases. I exported a compact evidence table from
`events.jsonl`:

```text
/workspace/runs/evidence_table.md
```

Key rows from the latest evidence table:

| run | cache | final train loss | final val loss | mean tokens/sec | data ms | forward ms | backward ms | optimizer ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline-tinystories-20260616T090824Z | miss | 5.0250 | 4.9362 | 28036.48 | 0.3953 | 10.4002 | 15.5264 | 9.8224 |
| cached-tinystories-20260616T072645Z | hit | 4.9360 | 4.9279 | 27610.90 | 0.2113 | 11.1361 | 15.5508 | 9.8035 |
| mixed-precision-20260616T072718Z | hit | 4.8527 | 4.9218 | 27610.99 | 0.2172 | 11.0823 | 15.5793 | 9.8195 |
| wikitext-profile-smoke-20260616T141542Z | miss | 10.8217 | 10.8255 | 14182.30 | 0.5275 | 3.1946 | 3.4876 | 1.3888 |
| gpt2-long-tinystories-20260616T145821Z | miss | 5.7937 | 5.9795 | 5460.35 | 0.3395 | 14.6812 | 16.3985 | 14.9579 |
| qwen-long-tinystories-20260616T145919Z | miss | 7.6953 | 7.2141 | 348.71 | 0.2607 | 34.4832 | 90.6663 | 57.4397 |
| qwen-probe-s64-b1-gc-on-20260619T032706Z | hit | NaN | NaN | 350.56 | 0.2628 | 34.0543 | 90.0769 | 57.5021 |
| qwen-probe-s64-b2-gc-off-20260619T032947Z | hit | 9.0163 | 8.9437 | 897.79 | 0.3720 | 33.2704 | 50.6650 | 57.5022 |
| qwen-probe-s128-b1-gc-off-20260619T033035Z | miss | 8.9593 | 8.9839 | 898.61 | 0.3481 | 33.2713 | 50.4937 | 57.5528 |

The result is subtle but useful:

- Cache hit reduced measured data time from about `0.395 ms` to about `0.211 ms`.
- Overall tokens/sec did not improve because the run was dominated by
  forward/backward/optimizer time.
- Mixed precision was stable, but did not materially improve throughput for this
  small DistilGPT2-style setup.

This is an important negative result. The optimization was architecturally
reasonable, but the timing breakdown showed that data loading was not the
dominant bottleneck. The correct next optimization is not "add more tricks"; it
is to test whether larger batch size or gradient accumulation improves
utilization by amortizing optimizer and per-step overhead.

I then ran the batch/gradient-accumulation sweep suggested by the timing
breakdown:

| variant | avg tokens/sec | final val loss | interpretation |
|---|---:|---:|---|
| `bs2_ga1` | 15114.61 | 5.7362 | Smallest direct batch; lower throughput. |
| `bs4_ga1` | 20284.70 | 5.3667 | Larger direct batch improves utilization. |
| `bs8_ga1` | 23949.54 | 5.1880 | Best throughput and best validation loss in this sweep. |
| `bs4_ga2` | 22139.84 | 5.1907 | Effective batch 8, but slower than true batch 8 because it performs two forward/backward passes. |

This gives a more concrete optimization decision: when the 24GB GPU can fit the
larger batch, `batch_size=8, grad_accum_steps=1` is the best current default.
Gradient accumulation remains useful as a memory fallback, but it is not the
fastest path when a true larger batch fits.

This sweep also found a metric bug. The first matrix run logged accumulated
training loss as the sum of microbatch losses, so the `bs4_ga2` train loss looked
roughly doubled even though validation loss and throughput were meaningful. I
fixed the trainer to log the average loss across accumulated microbatches and
reran the matrix; the table above uses the corrected run. This is exactly why
the framework logs both train and validation metrics: a suspicious metric can be
traced and corrected instead of blindly trusted.

The matrix runner now writes an explicit best-config reasoning section:

```text
Selected variant: bs8_ga1
Reason: highest average tokens/sec among validation-sane candidates;
validation loss 5.1880 is within 0.0% of the best observed 5.1880;
complexity=batch=8, grad_accum=1 (direct batch).
```

The Qwen probe also exposed a small matrix-analysis bug: a variant with NaN
validation loss should not be allowed to win best-config selection. I fixed the
selection helper to ignore non-finite losses whenever finite validation losses
exist. The corrected Qwen selection was:

```text
Selected variant: s128_b1_gc_off
Reason: highest average tokens/sec among validation-sane candidates;
validation loss 8.9839 is within 0.4% of the best observed 8.9437;
complexity=batch=1, grad_accum=1, seq_len=128, gradient_checkpointing=False.
```

The DeepSeek probe exposed a different agent-loop issue. The first matrix
version captured the CUDA OOM, but `latest_run()` accidentally treated
`runs/matrix_logs` as if it were a training run. I fixed run discovery to require
`copied_config.yaml`, so auxiliary directories such as `matrix_logs`,
`matrix_configs`, and `matrix_summaries` do not pollute run summaries. The
corrected DeepSeek matrix now shows the real preflight run directory and the real
AdamW-OOM run directory.

The follow-up DeepSeek safety matrix added one more controlled variant:

| DeepSeek variant | result | key evidence |
|---|---|---|
| `preflight_s16_b1` | succeeded | 1.346B parameters loaded and preflight warned about AdamW memory. |
| `adamw_s16_b1` | failed | `torch.cuda.OutOfMemoryError` at `AdamW optimizer.step()`. |
| `adafactor_s16_b1_no_ckpt` | succeeded | one training step, `18.52` tokens/sec, train loss `10.6670`, no final checkpoint. |

This is the most useful DeepSeek result so far because it separates three
questions that were previously tangled together: model/profile compatibility,
optimizer memory, and end-to-end train-loop feasibility. The answer is now:
compatibility works, AdamW is the immediate memory blocker, and a low-memory
optimizer path can make the first real training step possible on the RTX 3090.

The next DeepSeek Adafactor probe improved that conclusion from "one step is
possible" to "short multi-step training and validation are possible":

| DeepSeek Adafactor variant | result | interpretation |
|---|---|---|
| `s16_b1_5step_val` | succeeded, `47.38` avg tok/s, val loss `10.8575` | Multi-step low-memory path is stable. |
| `s32_b1_3step_val` | succeeded, `84.54` avg tok/s, val loss `10.8434` | Larger token budget improves throughput without increasing peak memory materially. |

The memory metrics are the most actionable part: both variants peaked near
`10.98GB` allocated and `12.29GB` reserved on the 24GB RTX 3090. That means the
framework now has enough evidence to try a cautious `seq_len=64` probe, or to
compare gradient checkpointing on/off, instead of treating DeepSeek as only a
preflight-risk example.

The later scale probes completed that next step. `seq_len=64` showed that
disabling gradient checkpointing was faster and still safe. The 128-token and
256-token probes then showed nearly linear throughput scaling while peak
reserved memory stayed around `13GB`. Among the 256-token variants,
`s256_b1_gc_off_3step_val` was selected because it had comparable throughput
(`756.78` tokens/sec) and the best validation loss (`9.5584`) among the
validation-sane candidates.

## 11. Reflection And Lessons Learned

Training framework work feels different from inference optimization. In earlier
phases, the central question was often "which kernel/path is fastest for this
shape?" In Phase 4, the central question became:

```text
Can every part of the training system preserve state, expose evidence,
and support reliable iteration?
```

That changed what counted as progress. A small smoke run with correct
checkpoint/resume was more valuable than a larger run with unclear state. A
neutral cache result was still useful because it taught me that the current
TinyStories setup was compute/optimizer dominated. A simple rule-based agent was
more trustworthy than an uncontrolled code-writing loop because its decisions
could be traced to logs.

If I continued the project, I would improve:

- a cleaner final workspace layout so Phase 4 artifacts are not mixed with older
  phase files,
- a richer run ledger that stores config hashes and environment metadata,
- automatic scalar plotting from JSONL/TensorBoard logs,
- a larger matrix runner that also records config hashes and confidence intervals,
- a longer 200-500 step DeepSeek real-data run before calling the 2048-token
  path a durable training recipe,
- a calibrated memory predictor extended across more model families, not only
  the DeepSeek/Adafactor case.

## 12. Current Limitations

The framework is intentionally small. It does not implement distributed training,
DeepSpeed, Megatron-style parallelism, or custom CUDA kernels. It also does not
claim that cache or mixed precision always improves speed. In the measured
TinyStories setup, cache improved data-time control but not total throughput, and
mixed precision was neutral.

The Qwen smoke, 60-step sanity run, and throughput probe show that the profile
system can reach beyond GPT-2-style models and can be tuned with the same
evidence loop. DeepSeek preflight-only, the Adafactor probes, the auto-probe
ladder, and the 100-step WikiText stability run go one step further: the framework can
safely inspect a much larger model family, explain why full AdamW training is
not viable on the current 24GB GPU, and then validate a low-memory optimizer
alternative with multi-step training, validation, and CUDA memory metrics. These
results are still not a claim of universal architecture support or DeepSeek
convergence quality. DeepSeek-family training remains a higher-risk stretch
because parameter count, optimizer state, tokenizer behavior, `trust_remote_code`,
validation memory, checkpoint storage, data realism, and download/runtime
constraints all become more significant.

These limitations are part of the learning result: the framework is now
observable enough to explain why an optimization helped, did not help, or should
be deferred.
