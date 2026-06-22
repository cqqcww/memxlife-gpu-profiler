# Phase 4 Honor Implementation Plan

This plan turns the confirmed decisions in [decisions.md](/Users/amanda/Desktop/School/mlsys/phase4-honor/decisions.md) into a staged implementation path.

## Working Principle

Build the smallest complete training framework first, then let the agent loop inspect runs and propose targeted improvements.

The framework should be understandable enough that the final report can explain:

- what each module does
- how the agent drives experiments
- how training state is logged
- how checkpoint/resume was verified
- how speed was measured
- which optimizations helped or failed

## Proposed Directory Structure

```text
phase4-honor/
  PHASE4_HONOR_GUIDE.md
  decisions.md
  implementation_plan.md
  requirements_questions.md
  report4.md
  configs/
    debug.yaml
    baseline.yaml
    cached_data.yaml
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
  scripts/
    run_smoke.sh
    run_experiment.sh
  tests/
    test_config.py
    test_data.py
    test_checkpoint_resume.py
  runs/
    .gitkeep
  train.py
```

The guide's recommended module structure stays intact. We add `agent/`, `timing.py`, and `tests/` because the project is explicitly about agentic framework construction and evidence.

## Milestone 0: Scaffold And Config

Goal: make the project importable and config-driven before training begins.

Implementation details:

- Create `training_framework/config.py`.
- Define dataclasses for model, data, optimizer, scheduler, trainer, logging, checkpoint, and agent settings.
- Load YAML with validation and useful error messages.
- Add `configs/debug.yaml` with tiny settings.

Tests:

- Config loads.
- Missing required field gives a readable error.
- CLI entrypoint can print resolved config.

Evidence:

- `runs/debug_config/summary.json`
- config snippet for report.

## Milestone 1: Minimal End-To-End Training

Goal: one tiny train run with validation, logging, checkpoint save, and resume.

Implementation details:

- Use `sshleifer/tiny-gpt2` or a tiny GPT-2 config for the first smoke run.
- Implement local text fixture dataset.
- Build token blocks with `input_ids`, `labels`, and optional `attention_mask`.
- Implement `Trainer.fit()`.
- Log loss, learning rate, step time, and tokens/sec.
- Save model, optimizer, scheduler, config, global step, and random states if practical.
- Resume and verify global step and learning rate continue.

Tests:

- Tiny train step passes.
- Validation runs.
- Checkpoint exists.
- Resume starts from expected step.
- Loss continuity is plausible after resume.

Evidence:

- console log
- JSONL event log
- TensorBoard directory
- checkpoint metadata
- resume summary

## Milestone 2: Real Small Dataset Baseline

Goal: train on a real HuggingFace dataset path with cached token blocks.

Implementation details:

- Support one real HF dataset such as WikiText-2 or TinyStories subset.
- Pre-tokenize once.
- Cache token blocks with a deterministic cache key based on:
  - dataset name/path
  - tokenizer name
  - sequence length
  - split seed
  - max samples
- Build train/validation splits.
- Add deterministic token cache hash to data test.

Tests:

- Cache creates.
- Cache reloads.
- Cache invalidates when sequence length or tokenizer changes.
- Batch shape/label shift/attention mask are correct.

Evidence:

- baseline throughput
- data time vs compute time
- token cache hit/miss logs

## Milestone 3: Agent Loop V1

Goal: agent can run experiments, parse logs, and write summaries.

Implementation details:

- `agent/planner.py`: selects next config from a small rule set.
- `agent/runner.py`: launches `train.py`.
- `agent/analyzer.py`: parses JSONL/TensorBoard-derived scalar summaries.
- `agent/ledger.py`: writes a run ledger.
- Agent proposes next experiment but does not auto-rewrite framework modules yet.

First agent loop:

```text
choose config -> run training -> parse metrics -> summarize -> propose next config
```

Artifacts per run:

- copied config
- `events.jsonl`
- `summary.json`
- `summary.md`
- TensorBoard logs
- checkpoints

Evidence:

- agent run ledger
- example `summary.md`
- decision trace explaining why the next run was chosen

## Milestone 4: Speed Optimization Experiments

Goal: compare baseline against 2-3 targeted improvements.

Required baseline:

- on-the-fly or uncached preprocessing baseline if feasible
- cached token blocks as first optimization

Target experiments:

1. Pre-tokenized token-block cache
   - Primary expected win.
   - Helps distinguish data bottleneck from compute bottleneck.

2. Mixed precision
   - Enable bf16/fp16 when hardware supports it.
   - Compare loss behavior and tokens/sec.

3. Dataloader tuning or gradient accumulation
   - Choose based on timing breakdown.
   - If data time dominates, test dataloader workers/pin memory/persistent workers.
   - If compute/memory dominates, test gradient accumulation or activation checkpointing.

Measured metrics:

- tokens/sec
- total step time
- data time
- forward time
- backward time
- optimizer time
- logging/checkpoint overhead
- GPU memory if easy to collect

Decision rule:

- Keep optimizations that improve tokens/sec without breaking resume/correctness.
- Reject optimizations that complicate the system but do not produce measurable evidence.

## Milestone 5: Controlled Code Proposal

Goal: satisfy the chosen `24D` direction without making the system chaotic.

Implementation details:

- First make `24C` reliable.
- Then add a "proposal mode" where the agent writes a markdown patch proposal:
  - observed bottleneck
  - proposed code/config change
  - expected effect
  - risk
  - rollback plan
- Code is still applied manually.

This gives the report a credible agentic coding story without allowing unsafe auto-edits.

## Milestone 6: Final Report

File: `report4.md`

Recommended structure:

1. Project goal and scope
2. Agent design and workflow
3. Framework architecture
4. Data loading and tokenization
5. Model construction
6. Optimizer and scheduler
7. Training loop
8. Logging and checkpointing
9. Speed measurement
10. Experiments and results
11. Bugs and debugging process
12. Reflection and redesign ideas
13. Connections back to phases 1-3

Evidence to include:

- module diagram
- config snippets
- run ledger excerpt
- TensorBoard loss curves or exported scalar table
- checkpoint/resume log
- throughput comparison table
- timing breakdown table
- debugging story
- agent decision trace

## DeepSeek Decision

Do not start with DeepSeek.

Recommended route:

- Smoke test: `sshleifer/tiny-gpt2`
- Baseline real run: small GPT-2 or DistilGPT2-style model
- Stretch run: DeepSeek-family model only after the framework is stable

Reasoning:

- The honor guide emphasizes understanding and framework completeness.
- DeepSeek can create avoidable memory and iteration-speed problems early.
- GPT-2-style models make it easier to demonstrate all required framework features.

## Next Questions

The main remaining choices before implementation are:

1. Which real dataset should we use first?
   - A. WikiText-2 subset
   - B. TinyStories subset
   - C. Custom course/report corpus
   - D. Start with local fixture only, decide after scaffold

2. Which second-stage model should we use after `sshleifer/tiny-gpt2`?
   - A. `distilbert/distilgpt2`
   - B. `gpt2` with reduced training length
   - C. tiny GPT-2 config from scratch only
   - D. Qwen-family small model as a stretch

3. Should the first code implementation include tests immediately?
   - A. Yes, scaffold tests alongside modules
   - B. Implement modules first, tests right after first run
   - C. Only smoke scripts first

4. Should `report4.md` be drafted from day one?
   - A. Yes, keep a live report skeleton and fill evidence as we go
   - B. Start after the first successful run
   - C. Start after speed experiments
