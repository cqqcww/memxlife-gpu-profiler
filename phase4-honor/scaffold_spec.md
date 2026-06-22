# Phase 4 Honor Scaffold Spec

This document is the implementation guide for the Phase 4 honor project.

It summarizes the current requirements, the intended harness, the framework scaffold, and the evidence we need to collect for `report4.md`.

## 1. Dataset Choice: WikiText-2 vs TinyStories

For the first real dataset, choose:

```text
Primary: TinyStories subset
Secondary/optional: WikiText-2 subset
```

Why TinyStories first:

- It is small enough for fast iteration.
- It is a natural causal language modeling dataset.
- Loss curves are easier to explain because the text distribution is coherent.
- It is better for demonstrating a complete training framework without making the project feel like a benchmark chase.
- It pairs well with tiny GPT-2 and DistilGPT2-style models.

Why keep WikiText-2 as optional:

- It is a familiar language-modeling benchmark.
- It is useful as a second-dataset sanity check.
- It gives the report a more standard reference point if we want one later.

Why not start with WikiText-2:

- It is more benchmark-like but less narratively useful for this project.
- The guide values framework completeness, logs, debugging, and reflection more than benchmark comparability.
- TinyStories makes it easier to keep the first real training run interpretable and lightweight.

Final dataset plan:

1. Local text fixture for smoke tests.
2. TinyStories subset for the main baseline and optimization experiments.
3. WikiText-2 subset only if we want a second-dataset check after the main loop works.

## 2. Confirmed Requirements

Project goal:

- Build an agentic mini training framework.
- The framework must train a small causal language model.
- It must support logging, checkpointing, resume, validation, and speed measurement.
- It must produce evidence suitable for `report4.md`.

Framework emphasis:

- Small and robust.
- Clear implementation details.
- Strong debugging evidence.
- Repeatable experiment harness.
- Reflection over leaderboard-style performance.

Agent emphasis:

- Rule-based first.
- Deterministic and auditable.
- First make run -> parse -> summarize -> propose next config work well.
- Later support controlled patch proposals.
- Do not allow uncontrolled automatic source rewrites.

Model plan:

1. Smoke model: `sshleifer/tiny-gpt2`.
2. Main second-stage model: `distilgpt2`.
3. Later stress model: `gpt2`.
4. DeepSeek-family model: stretch only after the framework is stable.

Dataset plan:

1. Local fixture.
2. TinyStories subset.
3. Optional WikiText-2 subset.

Training features:

- YAML config loaded into dataclasses.
- AdamW with decay/no-decay parameter groups.
- Linear warmup plus cosine decay.
- Gradient clipping.
- Gradient accumulation.
- TensorBoard logging.
- JSONL event logging.
- Checkpoint save and resume.
- Resume checks for global step, learning rate, and plausible loss continuity.
- Tokens/sec as the primary speed metric.
- Timing breakdown for data, forward, backward, optimizer, and logging/checkpoint overhead.

Speed optimization priorities:

1. Pre-tokenized token-block cache.
2. Mixed precision when supported.
3. Dataloader tuning or gradient accumulation depending on timing breakdown.
4. Activation checkpointing as optional, off by default.

Previous phases:

- Do not modify Phase 1-3 files.
- Phase 4 can reference lessons from Phase 1-3 in the report.
- Any operator/kernel reuse is optional and should happen only inside Phase 4 after the baseline framework is stable.

## 2.1 Current Implementation Status

This section records what has already been implemented or validated so the
scaffold stays aligned with the real project instead of drifting into an
aspirational checklist.

Implemented locally:

- `training_framework/config.py`
- `training_framework/config_merge.py`
- `training_framework/data.py`
- `training_framework/model.py`
- `training_framework/optim.py`
- `training_framework/preflight.py`
- `training_framework/scheduler.py`
- `training_framework/trainer.py`
- `training_framework/checkpoint.py`
- `training_framework/logger.py`
- `training_framework/timing.py`
- `agent/planner.py`
- `agent/runner.py`
- `agent/analyzer.py`
- `agent/ledger.py`
- `agent/matrix_runner.py`
- `agent/auto_probe.py`
- `agent/stability_runner.py`
- `configs/debug.yaml`
- `configs/baseline_tinystories.yaml`
- `configs/cached_tinystories.yaml`
- `configs/mixed_precision.yaml`
- `configs/base/*.yaml`
- `configs/model_profiles/*.yaml`
- `configs/data_profiles/*.yaml`
- `configs/matrices/*.yaml`
- `configs/auto_probes/*.yaml`
- `selfcmd-workflow/`
- `configs/auto_probes/deepseek_adafactor_wikitext_realdata.yaml` for the next
  real-data DeepSeek probe.

Validated remotely on the course GPU server:

- `configs/debug.yaml` trained end to end.
- Validation ran during the debug smoke run.
- Checkpoints were saved during the debug smoke run.
- A resume run continued from checkpoint step 6 to step 8.
- TinyStories baseline ran for 100 steps with DistilGPT2-style config initialization.
- Cached TinyStories ran once with a cache miss and once with a cache hit.
- Mixed precision ran stably but did not materially improve throughput.
- Agent runner launched a training run, parsed `events.jsonl`, wrote summaries, and appended `runs/ledger.jsonl`.
- RNG restoration was verified in the resume summary with `rng_restored=true`.
- Latest profile/preflight/matrix extension tests passed remotely: `24 passed in 2.55s`.
- Latest DeepSeek/Adafactor extension tests passed remotely: `37 passed in 3.03s`.
- Latest auto-probe/stability-runner extension tests passed remotely: `44 passed in 2.92s`.
- Latest WikiText real-data auto-probe extension tests passed remotely: `45 passed in 3.10s`.
- Latest memory-predictor and recommendation-report extension tests passed remotely: `48 passed in 3.00s`.
- Profile-composed smoke training ran remotely through `--base --model-profile --data-profile`.
- The profile smoke produced `preflight.json` / `preflight.md`, a final checkpoint, and a run summary.
- Batch/gradient-accumulation matrix dry-run expanded four variants into resolved configs remotely.
- WikiText-2 data-profile smoke trained and validated remotely.
- Full `batch_grad_sweep` trained all four variants remotely.
- Matrix evidence selected `bs8_ga1` as the current best throughput path.
- Matrix analysis exposed and fixed a gradient-accumulation train-loss logging issue.
- Matrix summary now includes best-config reasoning over speed, validation sanity, and complexity.
- Preflight now records inspectable dataset columns and OOM fallback recommendations.
- Qwen small stretch smoke completed remotely with `Qwen/Qwen2.5-0.5B`.
- Full `gpt2` TinyStories sanity run completed for 60 steps with repeated validation and checkpointing.
- Qwen TinyStories sanity run completed for 60 steps with repeated validation and checkpointing.
- Qwen throughput probe completed remotely and showed that `gradient_checkpointing=false` plus a larger token budget improves throughput from about `350` to about `898` tokens/sec.
- Matrix best-selection now ignores NaN validation losses after the Qwen checkpointing-on probe exposed a non-finite-loss edge case.
- DeepSeek config/tokenizer loading succeeded and identified a 1.346B-parameter LLaMA-style stretch model path.
- `train.py --preflight-only` now writes model/tokenizer/preflight evidence without entering dataloaders, optimizer, or training.
- `deepseek_safety_probe` completed remotely: preflight-only succeeded, while the AdamW 1-step smoke was classified as `cuda_oom`.
- The extended `deepseek_safety_probe` now includes `adafactor_s16_b1_no_ckpt`; this low-memory optimizer/no-checkpoint variant completed one real training step at about `18.52` tokens/sec.
- DeepSeek auto-probe followed the token-budget ladder through `64 -> 128 -> 256 -> 512 -> 1024 -> 2048` tokens/step.
- The `2048` DeepSeek Adafactor fixture probe reached `3092.42` avg tokens/sec with about `14.31GB allocated / 18.58GB reserved`.
- `deepseek-stability-run` completed a 50-step stability check from the 2048-token recommendation: `50/50` steps, `3680.77` avg tokens/sec, last validation loss `0.0608`, and the same `14.31GB / 18.58GB` peak memory envelope.
- The stability runner exposed and fixed a log-counting bug: sparse train logs are no longer mistaken for completed optimizer steps.
- Run discovery now ignores auxiliary directories such as `matrix_logs`, `matrix_configs`, and `matrix_summaries`.
- The WikiText real-data DeepSeek probe and stability run completed remotely: `2048` tokens/step reached `3100.38` avg tokens/sec in the short probe, then passed a 100-step stability run at `3650.34` avg tokens/sec, last validation loss `6.6539`, and peak memory around `14.31GB allocated / 18.83GB reserved`.
- The calibrated preflight memory predictor matched the 100-step WikiText run closely: predicted reserved peak `18814 MiB` versus observed `18828 MiB`, and predicted allocated peak `15051 MiB` versus observed `14305 MiB`.
- The recommendation report generator now writes `phase4-current-recommendation.md/json` from the latest auto-probe, stability, preflight, and summary artifacts.

Validated locally after the extensibility pass:

- `train.py --base ... --model-profile ... --data-profile ... --override ... --print-config`.
- Fallback YAML/list parsing without PyYAML.
- Profile composition into a fully resolved config.
- Preflight report construction and markdown rendering.
- Matrix expansion and resolved-config materialization.
- `selfcmd` shell syntax for profile/matrix commands plus `test`, `bootstrap`, and `deepseek-probe`.

Known caveats:

- DeepSeek-family preflight is validated, full AdamW still OOMs, and Adafactor/no-checkpoint now works through both a 2048-token fixture probe and a 2048-token WikiText-2 real-data 100-step stability run. Longer DeepSeek training still requires 200-500 step checks, second-dataset validation, and possibly mixed-precision parameter loading or offload.
- Qwen is smoke-validated, has one 60-step TinyStories sanity run, and has a small throughput probe, but it is not a convergence-quality result.
- The matrix runner has remote evidence, but confidence intervals/repeated trials are future work.
- The planner exposes a deterministic `run-next -> analyze -> propose` command; the newer matrix runner now writes best-config reasoning but does not yet generate source patches.
- `report4.md` is still a working draft and needs final restructuring.
- TensorBoard evidence has been generated, but a compact scalar export or screenshot should be added to the final evidence bundle.

Decision after first evidence:

- Prioritize hardening, tests, report evidence, and agent-loop clarity over adding large optional features.
- Treat cache and mixed precision as measured experiments, including neutral results.
- Keep full `gpt2` as the practical stress profile after its successful 60-step run; use DeepSeek only through staged safety gates until a low-memory training path exists.

## 2.2 Current Positioning After Stretch Probes

The scaffold is now best described as an:

```text
agentic feasibility and optimization harness for causal-LM training experiments
```

This is a refinement of the earlier v0.4 target. The v0.4 direction wanted a
comfortable, extensible, model/data-switching training scaffold. The current
implementation has reached that goal for causal-LM profiles, but the DeepSeek
and Qwen probes made the boundary clearer:

- Switching is reliable when the model follows the `AutoModelForCausalLM` text
  training contract.
- Switching is not yet universal across every HuggingFace architecture or task.
- Larger-model support should start as feasibility probing, not as an immediate
  claim of stable long training.
- Short probes are useful only when they produce decision evidence: speed,
  validation sanity, CUDA memory, failure class, and complexity.

What overlaps with v0.4:

- profile composition through base config, model profile, data profile, and
  overrides,
- cache, mixed precision, gradient accumulation, checkpointing, and low-memory
  optimizer choices as configurable knobs,
- deterministic agent workflow through `selfcmd`, preflight, matrix runs,
  analyzer summaries, and fetched artifacts,
- staged model ambition from GPT-style baseline to Qwen and DeepSeek stretch
  profiles,
- report-ready evidence rather than ad hoc terminal output.

What should be modified from v0.4:

- Replace "seamless any-architecture switching" with bounded causal-LM
  switching plus explicit future adapter interfaces.
- Replace "train the largest model possible" with "decide whether a larger
  model is safe enough for longer training."
- Treat DeepSpeed, FSDP, and Megatron as future downstream execution backends
  that could consume scaffold recommendations, not as systems to reproduce.
- Strengthen preflight into a calibrated memory-risk predictor using observed
  CUDA peak metrics.
- Strengthen matrix summaries into recommendation artifacts that explain why a
  config was selected or rejected.

New implementation step:

- Add `agent/auto_probe.py` for bounded token-budget search.
- Add `agent/stability_runner.py` for turning a selected recommendation into
  a longer sanity run.
- Add `configs/auto_probes/deepseek_adafactor_token_budget.yaml` as the first
  concrete probe: `64 -> 128 -> 256 -> 512` tokens/step with Adafactor,
  checkpoint disabled, gradient checkpointing disabled, and CUDA-memory-aware
  recommendation logic.
- Add `./selfcmd auto-probe` and `./selfcmd deepseek-auto-probe` so the remote
  workflow can execute the probe and fetch recommendation artifacts.
- Store output under `runs/auto_probe_summaries/` and
  `runs/recommendations/`.
- Remote evidence followed the generated recommendations from `512` to `1024`
  and then to `2048` tokens/step. The `2048` follow-up succeeded at about
  `3092` tokens/sec with `18.58GB` reserved CUDA memory.
- The stability runner then completed a 50-step `2048`-token run at about
  `3681` tokens/sec with no detected stability issues. This is fixture-level
  systems evidence, so the next scaffold feature should repeat the same
  recommendation loop on a real data profile.

## 3. Directory Scaffold

```text
phase4-honor/
  PHASE4_HONOR_GUIDE.md
  decisions.md
  implementation_plan.md
  requirements_questions.md
  scaffold_spec.md
  report4.md
  train.py
  configs/
    debug.yaml
    baseline_tinystories.yaml
    cached_tinystories.yaml
    mixed_precision.yaml
  training_framework/
    __init__.py
    config.py
    data.py
    model.py
    optim.py
    scheduler.py
    trainer.py
    checkpoint.py
    logger.py
    timing.py
  agent/
    __init__.py
    planner.py
    runner.py
    analyzer.py
    ledger.py
  tests/
    test_config.py
    test_data.py
    test_checkpoint_resume.py
    test_smoke_train.py
  scripts/
    run_smoke.sh
    run_experiment.sh
  runs/
    .gitkeep
  data_cache/
    .gitkeep
```

## 4. Module Contracts

### `training_framework/config.py`

Responsibilities:

- Load YAML configs.
- Convert YAML into dataclasses.
- Validate required fields.
- Resolve paths.
- Save a frozen copy of the config into each run directory.

Required dataclasses:

- `ModelConfig`
- `DataConfig`
- `OptimizerConfig`
- `SchedulerConfig`
- `TrainerConfig`
- `LoggingConfig`
- `CheckpointConfig`
- `AgentConfig`
- `ExperimentConfig`

Minimum validation:

- `model.name_or_path` is set.
- `data.dataset_name` or `data.local_text_path` is set.
- `trainer.max_steps > 0`.
- `trainer.seq_len > 0`.
- `optimizer.lr > 0`.
- `checkpoint.save_every_steps > 0`.

### `training_framework/data.py`

Responsibilities:

- Load local fixture or HuggingFace dataset.
- Load tokenizer from model config.
- Tokenize text.
- Build fixed-length token blocks.
- Cache token blocks.
- Split train/validation.
- Build PyTorch `Dataset` and `DataLoader`.

Batch contract:

```python
{
    "input_ids": LongTensor[batch, seq_len],
    "labels": LongTensor[batch, seq_len],
    "attention_mask": LongTensor[batch, seq_len] | optional,
}
```

Cache key must include:

- dataset name/path
- dataset split
- tokenizer name
- sequence length
- max samples
- validation split
- cache format version

Data tests:

- batch shape
- dtype
- label shape
- attention mask shape if present
- split size
- deterministic cache hash
- cache invalidation when seq_len changes

### `training_framework/model.py`

Responsibilities:

- Load tokenizer.
- Load model config.
- Construct model from config or pretrained weights.
- Support tiny smoke model and DistilGPT2 baseline.
- Report parameter count.
- Configure gradient checkpointing if enabled.

Default behavior:

- Debug config uses `sshleifer/tiny-gpt2`.
- Baseline config uses `distilgpt2`.
- Pretrained loading is configurable.

### `training_framework/optim.py`

Responsibilities:

- Build AdamW.
- Separate weight-decay and no-weight-decay parameter groups.
- Expose optimizer hyperparameters through config.

No-decay parameters:

- bias
- LayerNorm / norm weights
- embedding weights if we decide to exclude them

### `training_framework/scheduler.py`

Responsibilities:

- Build linear warmup plus cosine decay.
- Log learning rate.
- Restore scheduler state from checkpoint.

Scheduler inputs:

- total training steps
- warmup steps or warmup ratio
- minimum learning-rate ratio

### `training_framework/timing.py`

Responsibilities:

- Measure data time.
- Measure forward time.
- Measure backward time.
- Measure optimizer/scheduler time.
- Measure logging/checkpoint overhead.
- Compute tokens/sec.

Timing output:

```json
{
  "step": 10,
  "tokens_per_sec": 1234.5,
  "step_time_ms": 88.1,
  "data_time_ms": 3.2,
  "forward_time_ms": 25.4,
  "backward_time_ms": 47.0,
  "optim_time_ms": 8.9,
  "log_ckpt_time_ms": 1.5
}
```

### `training_framework/logger.py`

Responsibilities:

- Create TensorBoard writer.
- Write console summaries.
- Write JSONL event logs.
- Log train loss, validation loss, learning rate, tokens/sec, and timing breakdown.
- Keep logs in the run directory.

Run directory format:

```text
runs/<run_name>-YYYYMMDDTHHMMSSZ/
  copied_config.yaml
  events.jsonl
  summary.json
  summary.md
  tensorboard/
  checkpoints/
```

### `training_framework/checkpoint.py`

Responsibilities:

- Save model state.
- Save optimizer state.
- Save scheduler state.
- Save global step.
- Save config.
- Save metadata.
- Optionally save RNG states.
- Load and restore training state.

Resume verification:

- restored global step equals saved global step
- restored scheduler LR matches expected LR
- resumed loss is plausible relative to prior loss
- missing checkpoint produces a readable error

### `training_framework/trainer.py`

Responsibilities:

- Run training loop.
- Run validation loop.
- Support gradient accumulation.
- Support gradient clipping.
- Support mixed precision if enabled.
- Call checkpoint manager.
- Call logger.
- Use timing instrumentation.
- Return a structured summary.

Trainer loop skeleton:

```text
for step:
  time data loading
  autocast if enabled
  forward
  loss / grad_accum
  backward
  clip gradients if enabled
  optimizer step
  scheduler step
  zero_grad
  log metrics
  validate periodically
  checkpoint periodically
```

## 5. Agent Harness

### `agent/planner.py`

Responsibilities:

- Choose next config.
- Start with fixed rule-based plans.
- Use run ledger to decide next experiment.

Initial planner rules:

- If no successful smoke run: run `debug.yaml`.
- If smoke succeeded but no TinyStories baseline: run `baseline_tinystories.yaml`.
- If data time is high: propose cached token blocks.
- If compute dominates and GPU supports it: propose mixed precision.
- If memory limits appear: propose gradient accumulation or activation checkpointing.

### `agent/runner.py`

Responsibilities:

- Launch `train.py --config ...`.
- Capture stdout/stderr.
- Return process status.
- Store command line and environment summary.

### `agent/analyzer.py`

Responsibilities:

- Parse `events.jsonl`.
- Parse `summary.json`.
- Compute last loss, best validation loss, mean tokens/sec, and timing breakdown averages.
- Detect failures.
- Write `summary.md`.

### `agent/ledger.py`

Responsibilities:

- Maintain `runs/ledger.jsonl`.
- Record:
  - run id
  - config path
  - status
  - final train loss
  - final validation loss
  - tokens/sec
  - timing breakdown
  - checkpoint path
  - proposed next step

### Agent command shape

Initial command:

```bash
python -m agent.runner --config configs/debug.yaml
```

Future command:

```bash
python -m agent.planner --goal improve_tokens_per_sec
```

## 6. Config Harness

### `configs/debug.yaml`

Purpose:

- Fast smoke test.
- Local fixture.
- Tiny model.
- Very few steps.

Required settings:

```yaml
model:
  name_or_path: sshleifer/tiny-gpt2
  from_pretrained: false
data:
  local_text_path: fixtures/tiny_corpus.txt
  use_cache: true
  max_samples: 32
trainer:
  seq_len: 64
  max_steps: 6
  batch_size: 2
  grad_accum_steps: 1
  validate_every_steps: 2
checkpoint:
  save_every_steps: 3
```

### `configs/baseline_tinystories.yaml`

Purpose:

- Main baseline.
- TinyStories subset.
- DistilGPT2.
- Cached preprocessing off or cold-cache first.

### `configs/cached_tinystories.yaml`

Purpose:

- Same as baseline, but with token-block cache enabled.
- Used to demonstrate data pipeline speed improvement.

### `configs/mixed_precision.yaml`

Purpose:

- Same as cached baseline, with bf16/fp16 enabled when available.

## 7. Test Harness

Required tests:

1. Config test
   - YAML loads.
   - Dataclasses validate.
   - Bad config fails readably.

2. Data test
   - Dataset creates train/val split.
   - Batch shapes are correct.
   - Labels align with input IDs.
   - Cache key changes when sequence length changes.

3. Checkpoint/resume test
   - Train tiny model for a few steps.
   - Save checkpoint.
   - Resume.
   - Verify global step and LR.
   - Verify loss continuity is plausible.

4. Smoke training test
   - Run tiny fixture end to end.
   - Confirm TensorBoard dir exists.
   - Confirm `events.jsonl`, `summary.json`, and checkpoint exist.

Next hardening tests:

1. Bad config failure
   - Unknown config key fails readably.
   - Invalid `trainer.max_steps` fails readably.
   - Missing model name fails readably.

2. Missing checkpoint failure
   - Resume from a nonexistent checkpoint path produces a clear error.
   - Resume from an empty checkpoint directory produces a clear error.

3. Resume state test
   - Save a debug checkpoint.
   - Resume from it.
   - Assert restored `global_step`.
   - Assert scheduler learning rate continues from checkpoint state.
   - If RNG restore is implemented, assert torch RNG state is restored.

4. Cache invalidation test
   - Same dataset/tokenizer/seq_len gives the same cache key.
   - Changing sequence length changes the cache key.
   - Changing tokenizer name changes the cache key.
   - Changing `max_samples` changes the cache key.

5. Remote test evidence
   - Run tests on the course server, not only locally.
   - Save the test command and output under `selfcmd-workflow/logs/`.
   - Fetch the lightweight test log into `selfcmd-workflow/artifacts/`.

## 8. Experiment Harness

Baseline comparisons:

1. Debug smoke run
   - local fixture
   - tiny-gpt2
   - verifies system mechanics

2. TinyStories baseline
   - DistilGPT2
   - records baseline tokens/sec and loss behavior

3. Cached token-block run
   - compares data time and tokens/sec against baseline

4. Mixed precision run
   - compares compute throughput and loss stability

Optional:

- Full `gpt2` stress run. Completed once as a controlled 60-step sanity run.
- WikiText-2 second dataset sanity check.
- Activation checkpointing if memory becomes the bottleneck.
- `torch.compile` only if the remote PyTorch version supports it cleanly.
- Larger batch or grad accumulation sweep if timing suggests compute utilization is the bottleneck.

Comparison table for report:

| Run | Dataset | Model | Cache | Precision | Tokens/sec | Data ms | Forward ms | Backward ms | Val loss | Resume OK |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|

Current experiment interpretation:

- The baseline TinyStories run is already compute/optimizer dominated.
- Cached token blocks reduce data-control uncertainty, but do not improve steady-state tokens/sec in the current small setup.
- Mixed precision was stable, but the current small DistilGPT2-style run does not show a meaningful speedup.
- The next useful optimization is not automatically "more features"; it is better diagnosis:
  compare mean timing components, GPU memory, variance, and possibly batch-size/grad-accumulation sensitivity.

Candidate next experiments:

1. Batch-size sweep
   - Try batch size 8 if memory allows.
   - Goal: see whether tokens/sec improves when kernel launch/optimizer overhead is amortized.

2. Dataloader worker correction
   - Compare cached TinyStories with `num_workers=0` versus `num_workers=2`.
   - Initial evidence suggests workers may add overhead because cached batches are tiny.

3. Longer run stability
   - Run 200-300 steps for the best simple config.
   - Goal: produce smoother loss and throughput curves for the report.

4. Full `gpt2` stress/sanity run
   - Completed once for 60 TinyStories steps after the core evidence was safe.
   - Treat as a scaling stress test and comparison point, not the core submission.

## 9. Report Evidence Harness

`report4.md` should be created from day one.

Sections to keep live:

1. Goal and scope.
2. Agent workflow.
3. Framework architecture.
4. Data/tokenization path.
5. Model/optimizer/scheduler construction.
6. Training loop.
7. Logging and checkpointing.
8. Timing and speed measurement.
9. Experiments.
10. Bugs and debugging.
11. Reflection and redesign.
12. Connections to Phases 1-3.

Evidence to collect as we go:

- config snippets
- module diagram
- run ledger excerpt
- TensorBoard scalar screenshots or exported scalar tables
- checkpoint/resume log
- throughput comparison table
- timing breakdown table
- failure examples and fixes
- agent proposal examples

Minimum final evidence bundle:

- `report4.md`
- one debug smoke `summary.json`
- one resume `summary.json`
- one TinyStories baseline `summary.json`
- one cached TinyStories cache-hit `summary.json`
- one mixed-precision `summary.json`
- one `events.jsonl`-derived scalar table
- one agent `ledger.jsonl` excerpt
- one remote test log
- one environment-debug note about dependency pins

Report stance:

- Be specific about what improved and what did not.
- Explain why data caching was architecturally useful even when throughput did not improve.
- Explain why mixed precision did not help this small setup.
- Clearly state the boundary between implemented agent automation and human-guided decisions.
- Include limitations instead of hiding them.

## 10. Implementation Exit Criteria

Milestone 1 exit:

- `python train.py --config configs/debug.yaml` works.
- Training logs loss and tokens/sec.
- Validation runs.
- Checkpoint saves.
- Resume works.
- Smoke tests pass.

Milestone 2 exit:

- TinyStories subset trains.
- Token cache works.
- Timing breakdown is populated.
- TensorBoard logs are created.

Milestone 3 exit:

- Agent can run an experiment.
- Agent can parse logs.
- Agent writes `summary.json` and `summary.md`.
- Agent proposes a next config.

Milestone 4 exit:

- Baseline vs cached data comparison exists.
- Baseline vs mixed precision comparison exists if hardware supports it.
- Report includes a clear speed analysis.

Final exit:

- `report4.md` has evidence-backed discussion.
- Framework is complete and reproducible.
- Debugging and reflection sections are specific.
- Previous Phase 1-3 files remain unchanged.

## 11. Immediate Next Implementation Pass

Priority order:

1. Checkpoint hardening
   - Restore RNG state when present.
   - Improve missing-checkpoint error messages.
   - Record resume LR in summary metadata.

2. Agent loop hardening
   - Add a `run-next` command or equivalent planner entrypoint.
   - Read the ledger.
   - Select the next config from the experiment ladder.
   - Run training through `agent.runner`.
   - Analyze the latest run.
   - Write a patch/config proposal with observed bottleneck, expected effect, risk, and rollback.

3. Test hardening
   - Add tests for bad configs, missing checkpoints, cache invalidation, and resume metadata.
   - Execute tests remotely through `selfcmd-workflow`.
   - Fetch the test log as evidence.

4. Evidence export
   - Generate a compact table from `events.jsonl`.
   - Include mean tokens/sec, mean data/forward/backward/optimizer time, final validation loss, and cache status.

5. Report cleanup
   - Reorder sections to match the official guide.
   - Remove "working draft" language.
   - Add specific debugging stories and limitations.
   - Make final submission path `/workspace/report4.md`.

Defer unless time remains:

- Low-memory DeepSeek training beyond preflight.
- Activation checkpointing.
- `torch.compile`.
- Previous-phase kernel integration.

## 12. Switching And Extensibility Scaffold

The next layer of the project is to turn the current working framework into a
more comfortable causal-LM experiment scaffold. The goal is not to claim
universal support for every HuggingFace architecture. The goal is reliable
config-level switching for GPT-style causal language models and standard text
datasets, with clear preflight warnings for riskier model families.

### Scope Boundary

Supported claim for this phase:

```text
causal-LM mini training framework with extensible model/data profiles
```

Not the current claim:

```text
fully seamless training for any HuggingFace architecture
```

"Any architecture" would require adapter layers for different tasks and model
contracts: decoder-only causal LM, encoder-only masked LM or classification,
encoder-decoder seq2seq, multimodal models, custom remote-code models, and their
different collators/losses/evaluation metrics. That is beyond the core Phase 4
scope, but the scaffold should leave room for this future direction.

### New Directory Structure

Add:

```text
configs/
  base/
    causal_lm_debug.yaml
    causal_lm_tinystories.yaml
  model_profiles/
    tiny_gpt2.yaml
    distilgpt2.yaml
    gpt2.yaml
    qwen_small_placeholder.yaml
    deepseek_placeholder.yaml
  data_profiles/
    local_fixture.yaml
    tinystories.yaml
    wikitext2.yaml
    openwebtext_subset.yaml
  matrices/
    batch_grad_sweep.yaml
    cache_on_off.yaml
training_framework/
  preflight.py
  config_merge.py
agent/
  matrix_runner.py
```

The current full configs should keep working. Profile composition is an
additional path, not a replacement for:

```bash
python train.py --config configs/debug.yaml
```

### Model Profile Contract

Each model profile should include:

```yaml
profile:
  name: gpt2
  family: gpt-style-causal-lm
  expected_memory: medium
  caveats:
    - Larger than distilgpt2; may require smaller batch size.

model:
  name_or_path: gpt2
  tokenizer_name:
  from_pretrained: false
  trust_remote_code: false
  gradient_checkpointing: false

trainer_recommendations:
  seq_len: 256
  batch_size: 2
  grad_accum_steps: 1
  mixed_precision: auto
```

First-class stages:

1. `tiny_gpt2`, `distilgpt2`, `gpt2`
   - These are the stable GPT-style causal-LM profiles.

2. Qwen small
   - Profile added and validated through smoke, 60-step sanity training, and a small throughput probe.
   - Current first recommendation is `seq_len=128,batch_size=1,gradient_checkpointing=false`.

3. DeepSeek-family placeholder
   - Preflight-only and safety matrix validated.
   - Full AdamW training remains blocked by optimizer-state memory on the 24GB RTX 3090.

### Data Profile Contract

Each data profile should include:

```yaml
profile:
  name: tinystories
  expected_token_count_note: small subset for fast iteration

data:
  dataset_name: roneneldan/TinyStories
  dataset_config:
  dataset_split: train
  text_field: text
  max_samples: 2000
  validation_split: 0.05
  use_cache: true
  num_workers: 0
  pin_memory: true
  persistent_workers: false
```

First-class data profiles:

1. local fixture
2. TinyStories
3. WikiText-2
4. optional OpenWebText subset

Arbitrary HuggingFace datasets can remain configurable, but should not be
described as fully tested unless a profile and run evidence exist.

### Config Composition

Add a simple merge path:

```bash
python train.py \
  --base configs/base/causal_lm_tinystories.yaml \
  --model-profile configs/model_profiles/gpt2.yaml \
  --data-profile configs/data_profiles/wikitext2.yaml \
  --override trainer.max_steps=50 \
  --override trainer.batch_size=2
```

Rules:

- Full `--config` remains supported.
- Profile composition writes the fully resolved config to the run directory.
- Unknown keys remain hard failures.
- Profile metadata can include `tags` and `notes` for ledger/report grouping.
- Config merge should be simple and explicit; do not adopt Hydra/OmegaConf unless the simple approach becomes too limiting.

### Preflight Report

Every composed or full-config run should be able to emit a lightweight preflight
report before training.

Minimum preflight checks:

- Python package imports.
- Torch device and CUDA availability.
- GPU memory if available.
- Model name, tokenizer name, and `trust_remote_code`.
- Tokenizer pad/eos token status.
- Dataset name/split/text field.
- Available dataset columns for small sample.
- Sequence length, batch size, grad accumulation.
- Mixed precision mode and selected dtype when CUDA is available.
- Cache key and cache path.
- Download/cache availability when detectable.
- Parameter count after model construction.

Preflight output:

```text
runs/<run_name>-.../preflight.json
runs/<run_name>-.../preflight.md
```

OOM handling:

- Catch CUDA OOM where practical.
- Print a recommendation such as smaller batch size, shorter sequence length, or gradient accumulation.
- Do not silently mutate the config or auto-retry in the current version.

### Optimization Toggles As Basic Capabilities

Basic config capabilities:

- token-block cache
- mixed precision `auto`
- gradient accumulation
- gradient clipping

Additional sweep knobs:

- `num_workers`
- `pin_memory`
- `persistent_workers`
- batch size
- gradient accumulation

Deferred but available as config flags:

- activation checkpointing
- `torch.compile`

Interpretation rule:

- These toggles are not promised speedups.
- They are measurement knobs for different model/data/hardware regimes.
- The report should include neutral results when they reveal the actual bottleneck.

### Agent Matrix Runner

Add a YAML-driven matrix runner.

First matrix:

```yaml
name: batch_grad_sweep
base_config: configs/cached_tinystories.yaml
variants:
  - trainer.batch_size: 2
    trainer.grad_accum_steps: 1
  - trainer.batch_size: 4
    trainer.grad_accum_steps: 1
  - trainer.batch_size: 8
    trainer.grad_accum_steps: 1
  - trainer.batch_size: 4
    trainer.grad_accum_steps: 2
```

Secondary matrix:

```yaml
name: cache_on_off
base_config: configs/baseline_tinystories.yaml
variants:
  - data.use_cache: false
  - data.use_cache: true
```

Agent responsibilities:

- Expand matrix variants into resolved configs.
- Run each variant.
- Parse events and summaries.
- Generate an evidence table.
- Name a best config based on tokens/sec plus validation-loss sanity.
- Write short report snippets and observations.
- Continue writing patch proposals with risk and rollback.

Future scoring:

- Consider a weighted score that balances speed, validation loss, stability, and complexity.
- Do not make this score the only decision criterion in the current report.

### Evidence For Switching

Minimum evidence for this extension:

- One successful model-profile switch, such as `distilgpt2 -> gpt2` or `tiny_gpt2 -> distilgpt2`.
- One successful data-profile switch, such as TinyStories -> WikiText-2.
- One preflight report.
- One batch/grad matrix evidence table.
- One agent-generated observation or proposal from the matrix.

Next-stage evidence:

- Qwen small smoke completed with `Qwen/Qwen2.5-0.5B`.
- Full `gpt2` and Qwen small 60-step TinyStories sanity runs completed.
- Qwen throughput probe completed; checkpointing-on was slower and unstable, while larger token budgets reached about `898` tokens/sec.
- DeepSeek safety probe completed; preflight-only works, AdamW smoke OOM is classified and documented, and the Adafactor/no-checkpoint path scales to a 2048-token fixture probe plus a 100-step WikiText-2 real-data stability run.

### Final Workspace Policy

Current decision:

- Keep the current mixed `/workspace` if needed, but ensure `/workspace/report4.md`
  and Phase 4 files are correct.

Future improvement:

- Add a Phase4-only clean final deploy mode if the submission endpoint requires a
  cleaner workspace.

## 13. Extensibility Implementation Pass Status

Status after the first implementation pass:

1. Profile files
   - Done locally.
   - Added model profiles for tiny-gpt2, distilgpt2, gpt2, Qwen placeholder, and DeepSeek placeholder.
   - Added data profiles for local fixture, TinyStories, WikiText-2, and optional OpenWebText subset.

2. Config merge
   - Done locally.
   - Added `training_framework/config_merge.py`.
   - `train.py` supports `--base --model-profile --data-profile --override`.
   - Direct `--config` path remains unchanged.

3. Preflight
   - Done locally.
   - Added `training_framework/preflight.py`.
   - Each training run writes `preflight.json` and `preflight.md` after model/tokenizer construction.
   - Current checks cover model/profile metadata, local path existence, dataset columns when inspectable, packages, CUDA availability, tokenizer IDs, parameter count, tokens per optimizer step, and OOM fallback recommendations.

4. Matrix runner
   - Done locally.
   - Added `agent/matrix_runner.py`.
   - Implemented `cache_on_off` and `batch_grad_sweep` matrix parsing/materialization.
   - Matrix runs generate resolved configs, summary artifacts, and best-config reasoning based on speed, validation sanity, and complexity.
   - Best-config selection ignores NaN validation losses when finite validation losses are available.

5. Remote validation
   - Done for remote tests, profile-smoke, WikiText-2 data switch, batch/grad matrix dry-run, and full batch/grad matrix training.
   - Done for a conservative Qwen small smoke.
   - Done for controlled 60-step TinyStories sanity runs with full `gpt2` and Qwen small.
   - Done for `qwen_throughput_probe`, selecting `seq_len=128,batch_size=1,gradient_checkpointing=false` as the first Qwen stretch recommendation.
   - Done for `deepseek_safety_probe`, validating preflight-only, classifying AdamW smoke as `cuda_oom`, and proving one Adafactor/no-checkpoint training step.
   - Done for `deepseek_adafactor_probe`, proving short multi-step Adafactor training with validation at `seq_len=16` and `seq_len=32`.
   - Done for `deepseek_adafactor_scale_probe`, showing `seq_len=64,gradient_checkpointing=false` improves throughput and remains memory-safe.
   - Done for `deepseek_adafactor_budget_probe`, `deepseek_adafactor_256_probe`, and auto-probe follow-ups through 2048 tokens per step.
   - Done for `deepseek-stability-run` from the 2048-token recommendation: 50 completed fixture steps, `3680.77` avg tokens/sec, and peak memory around `14.31GB allocated / 18.58GB reserved`.
   - Done for `deepseek_adafactor_wikitext_realdata` and its 100-step stability follow-up: `2048` tokens/step, `3650.34` avg tokens/sec, last validation loss `6.6539`, and peak memory around `14.31GB allocated / 18.83GB reserved`.
   - Done for calibrated memory prediction: predicted reserved peak `18814 MiB` versus observed `18828 MiB` on the 100-step WikiText run.
   - Done for automatic recommendation markdown/json via `agent/recommendation_report.py` and `./selfcmd recommendation-report`.
   - Latest unified remote evidence was fetched to `selfcmd-workflow/artifacts/remote-20260621T095347Z`.
   - Still pending: longer 200-500 step real-data runs, second-dataset validation, and memory-predictor calibration across more model families.

6. Report update
   - Updated with profile-based switching, preflight evidence, matrix reasoning, GPT/Qwen sanity-run evidence, Qwen throughput probe, DeepSeek safety probe, auto-probe, fixture stability evidence, WikiText real-data stability evidence, calibrated memory prediction, and automatic recommendation artifacts.
   - Mention Qwen/DeepSeek as staged stretch targets with caveats.
   - Keep neutral optimization results concise but explicit.
