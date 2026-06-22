# Phase 4 Honor Decisions

Date: 2026-06-16

This file records the first confirmed design choices for the Phase 4 honor project.

## Overall Direction

We will build a small but complete agentic mini training framework.

The emphasis is:

- complete framework behavior
- clear agent workflow
- concrete training and resume evidence
- timing and throughput measurement
- debugging and reflection
- targeted speed improvements

The emphasis is not:

- speed leaderboard chasing
- largest possible model
- fully autonomous uncontrolled code rewriting
- modifying previous phase implementations

## Confirmed Choices

1. Agent primary role: `C`
   - Generate, run, inspect logs, and iteratively modify configs/code.

2. First milestone: `A`
   - Minimal end-to-end training loop with one batch, checkpoint save, and resume.

3. Ambition level: `A`
   - Small, robust, easy to explain.

4. Final positioning: `A`
   - Complete mini framework with clear logs and reflection.

5. Model choice: resolved staged GPT-style path
   - Use `sshleifer/tiny-gpt2` for smoke tests and checkpoint/resume debugging.
   - Use `distilgpt2` as the first real TinyStories experiment model.
   - Keep full `gpt2` or DeepSeek-family models as stretch/stress experiments only after the framework evidence is complete.
   - Rationale: Phase 4 is graded on framework completeness, agent workflow, debugging, and reflection. A stable small model gives faster feedback and better evidence than a fragile large-model attempt.

6. Pretrained weights: `C`
   - Support config-from-scratch and pretrained loading, default to config-from-scratch first.

7. Dataset: `E`
   - Tiny local text fixture plus one real HuggingFace dataset path.

8. Sequence length: `B`, with later expansion
   - Start with 256, later expand or sweep.

9. Data preprocessing: `B`
   - Pre-tokenize once and cache token blocks.

10. Validation split: `A`
   - Fixed percentage split from dataset.

11. Module structure: `A`, with later discussion of `C`
   - Start with the guide's structure:
     `data.py`, `model.py`, `optim.py`, `scheduler.py`, `trainer.py`, `checkpoint.py`, `logger.py`.
   - Later we may add explicit agent/profiler modules.

12. Config system: `A`
   - YAML config loaded into dataclasses.

13. Entrypoint: `A`
   - `python train.py --config configs/debug.yaml`.

14. Optimizer: `B`
   - AdamW with weight-decay/no-weight-decay parameter groups.

15. Scheduler: `B`
   - Linear warmup plus cosine decay.

16. Trainer v1: `B`
   - Train, validate, log, checkpoint, resume, gradient clipping, gradient accumulation.

17. Primary speed metric: `C`
   - Tokens/sec.

18. Timing breakdown: `C`
   - Data, forward, backward, optimizer, logging/checkpoint overhead.

19. First optimization: `A`, with later planning for debugging and innovation
   - Pre-tokenization and token-block caching first.
   - Later compare additional options if profiling suggests value.

20. Mixed precision: `B`
   - Enable bf16/fp16 automatically when GPU supports it, after v1 is stable.

21. Activation checkpointing: `B`
   - Optional config toggle, off by default.

22. Previous phase reuse: `A`, with caution
   - Keep Phase 4 focused on new framework engineering first.
   - Do not modify finished Phase 1-3 work.
   - Possible later discussion: optional conceptual reuse only inside Phase 4.

23. Agent implementation style: `A`
   - Rule-based planner/runner/analyzer scripts, no external LLM dependency.

24. Agent loop: `D`, staged through `C`
   - First make `C` work well: run training, parse logs, summarize metrics, propose next config.
   - Later allow controlled source changes or generated patch proposals.

25. Per-run artifacts: `C`
   - JSON summary, markdown summary, TensorBoard logs, copied config.

26. Code modification autonomy: `B`
   - Agent proposes diffs; we apply manually.

27. Smoke tests: `C`
   - Tiny train + validation + checkpoint save + resume.

28. Resume correctness: `C`
   - Check global step, learning rate, and resumed loss continuity.

29. Data tests: `C`
   - Batch shape, label shift, attention mask, split size, deterministic token cache hash.

30. Debug logging: `C`
   - Console + TensorBoard + JSONL events.

31. Intentional failure tests: `D`
   - Bad config, missing checkpoint, dataset/tokenizer cache invalidation.

32. Report emphasis: `D`
   - Balanced across architecture, agent loop, experiments, reflection.

33. Evidence: `E`
   - Config snippets, diagrams, TensorBoard curves, resume logs, throughput tables, timing breakdown.

34. Experiment comparison: `C`
   - Baseline vs 2-3 targeted optimizations.

35. Final report file name: `A`
   - `report4.md`.

36. Redesign reflection: `D`
   - More robust agent autonomy, better run ledger from day one, cleaner config/data cache abstractions.

## Model Recommendation: DeepSeek vs GPT-2

Use GPT-2-style models first.

Recommended staged path:

1. `sshleifer/tiny-gpt2`
   - Best for smoke tests and checkpoint/resume debugging.
   - Very fast iteration.

2. `distilbert/distilgpt2` or a similarly small GPT-2 config
   - Good second-stage real training run.
   - More meaningful than tiny-gpt2 while still manageable.

3. `gpt2` or DeepSeek-family model as stretch experiments
   - Consider only after the framework, logging, checkpointing, and speed measurement are stable.
   - Likely better framed as "scaling stress test" than the baseline.

Rationale:

- The guide values framework completeness, evidence, and reflection over large-model ambition.
- GPT-2-style models make the training framework easier to debug.
- DeepSeek can distract from framework engineering if it causes memory, download, tokenizer, or iteration-speed issues too early.

## Remaining Question Decisions

1. First real dataset: `B`
   - Use a TinyStories subset.
   - Rationale: it is small enough for fast iteration but more semantically natural than a tiny fixture.

2. Second-stage model: leaning `A`
   - Prefer `distilgpt2` before full `gpt2`.
   - Rationale: `distilgpt2` is a better second-stage baseline for repeated experiments because it is faster and less memory-sensitive. Full `gpt2` is still useful later as a stress test once the loop is stable.

3. Tests: `A`
   - Scaffold tests alongside modules from the start.

4. Report: `A`
   - Create a live `report4.md` skeleton from day one and fill evidence as we go.

## Evidence-Backed Decisions After First Remote Runs

These decisions were added after the first course-GPU smoke, resume, TinyStories,
cache, mixed-precision, and agent-runner experiments.

1. Remote-first development is now part of the design.
   - Local code remains the source of truth.
   - `/workspace` on the course GPU server is the execution and validation area.
   - Heavy artifacts such as checkpoints, token caches, and full run folders stay remote.
   - Lightweight evidence is pulled back through `selfcmd-workflow`.

2. Submission safety depends on explicit deploy-before-run.
   - Earlier phases showed that official submit endpoints may evaluate the server-side `/workspace`, not the latest local files.
   - Phase 4 therefore treats `deploy-clean -> remote test -> fetch evidence -> final report sync` as a required loop.

3. Cache optimization is useful evidence even when it does not improve steady-state speed.
   - The TinyStories runs showed data time was already tiny compared with forward/backward/optimizer time.
   - Token-block caching still matters for reproducibility, startup behavior, and proving the data pipeline is controlled.
   - The report should frame this as a measured negative/neutral result, not a failed optimization.

4. Mixed precision should remain an experiment, not a blanket claim.
   - The initial mixed-precision run was stable but did not materially beat the fp32 baseline.
   - The reason appears to be small-model overhead and optimizer/scheduling cost, not data loading.
   - Future mixed-precision discussion should be tied to timing breakdowns and hardware behavior.

5. The agent story should emphasize harness engineering.
   - The valuable automation is run orchestration, log parsing, ledger writing, evidence extraction, and next-step proposal.
   - The agent should not pretend to autonomously invent new training algorithms.
   - Controlled patch proposals are acceptable only when they cite a concrete bottleneck and a rollback plan.

6. Checkpoint/resume claims must be precise.
   - Current evidence confirms model, optimizer, scheduler, and global-step resume.
   - RNG state is saved, but RNG restoration should either be implemented next or documented as a limitation.
   - The report should avoid overstating deterministic resume until this is fixed and tested.

7. Final optimization priority is evidence quality, not feature breadth.
   - Stronger tests, a cleaner agent loop, and a polished `report4.md` are higher value than adding many optional features.
   - Optional features such as `torch.compile`, activation checkpointing, full `gpt2`, or DeepSeek should be attempted only if they produce clear evidence without destabilizing the submission.

## Current Gap Decisions

1. Tests must become executable evidence, not just scaffold files.
   - Run the test suite remotely where PyTorch and HuggingFace dependencies exist.
   - Add or strengthen tests for bad config, missing checkpoint, cache invalidation, and resume LR/global-step behavior.

2. Agent loop should get one stronger command.
   - Add a `run-next` or planner-driven path that reads `runs/ledger.jsonl`, selects the next config, runs it, analyzes artifacts, and writes a proposal.
   - Keep the implementation deterministic and inspectable.

3. Report should be reorganized before final submission.
   - Remove working-draft language.
   - Put evidence in the same order as the official guide: agent, framework, data, training loop, checkpoint/resume, speed, debugging, reflection.
   - Include at least one compact table from real `events.jsonl` data.

4. TensorBoard evidence can be a scalar export if screenshots are inconvenient.
   - The guide asks for logs/plots/evidence, not necessarily a fancy figure.
   - A CSV or markdown table exported from JSONL/TensorBoard scalars is acceptable and easier to reproduce.

## Switching And Extensibility Decisions

These decisions refine the next layer of the scaffold: model/data switching,
profile composition, preflight checks, and optimization matrices.

1. Optimization features are framework capabilities, not guaranteed speedups.
   - Cache, mixed precision, gradient accumulation, gradient clipping, batch-size sweeps, and dataloader tuning should be exposed through config.
   - The report should say these are measurable knobs, not magic optimizations.
   - Neutral or negative results remain useful evidence when explained through timing breakdowns.

2. "Comfortable switching" means config-level switching within the causal-LM training problem.
   - The immediate target is GPT-style `AutoModelForCausalLM` training on text datasets.
   - The framework should switch among model/data profiles without rewriting the training loop.
   - Fully seamless switching for any HuggingFace architecture is not the current claim.
   - "Any architecture" would mean handling decoder-only LM, encoder-only MLM/classification, encoder-decoder seq2seq, vision, multimodal, or remote-code models with different collators/losses/tasks without code changes. That requires adapter interfaces beyond this phase's core scope.

3. Extensibility ambition is staged.
   - Stage 1: practical, well-tested profiles for tiny-gpt2, distilgpt2, gpt2, local fixture, TinyStories, and WikiText-2.
   - Stage 2: broader profiles with preflight warnings, starting with Qwen small smoke and throughput probes.
   - Stage 3: DeepSeek/LLaMA-family models as safety-gated stretch targets, not the main submission path.

4. Final report positioning should be precise.
   - Call it a causal-LM mini training framework with extensible profiles.
   - Do not call it a broad general-purpose training framework.
   - The ultra goal is a larger model training framework, but the current deliverable should emphasize model/data profile switching and capability limits.

5. Model profile support should be added.
   - Add `configs/model_profiles/*.yaml`.
   - First-class path: GPT-style models, then Qwen small, then DeepSeek-family safety gates.
   - Add a `gpt2` stress profile, Qwen smoke/throughput profiles, and DeepSeek preflight-only safety probes.
   - Treat full DeepSeek training as pending until a low-memory optimizer/offload path exists.

6. Model profiles should include operational caveats.
   - Include model name, tokenizer name, `trust_remote_code`, `from_pretrained`, gradient checkpointing, recommended batch/sequence length, expected memory class, and known caveats.
   - Pretrained handling should support both config-from-scratch and pretrained loading.
   - Default remains config-from-scratch unless the agent/preflight has a reason to choose pretrained for a specific model-size experiment.

7. Dataset profile support should be added.
   - Add `configs/data_profiles/*.yaml`.
   - First-class datasets: local fixture, TinyStories, WikiText-2, and possibly OpenWebText subset.
   - Keep arbitrary HuggingFace dataset loading configurable where possible, but first-class reliability should be claimed only for tested profiles.
   - Data profiles should include dataset name/config/split/text field/max samples/validation split/cache policy/dataloader worker policy/expected token-count notes.

8. Data path should stay simple and reliable.
   - Fixed token blocks remain the default packing strategy.
   - Preflight should check whether the requested `text_field` exists and show available columns when it does not.
   - Cache keys should include profile/cache version and environment-sensitive information such as Python or transformers version if practical.
   - Full git/environment hashing can be future work if reproducibility needs become stricter.

9. Config composition should be implemented without adopting a heavy config framework.
   - Add a simple merge path such as `--base --model-profile --data-profile --override`.
   - Keep `python train.py --config configs/debug.yaml` as the default simple path.
   - Always save the fully resolved config into the run directory.
   - Unknown keys and invalid core values should continue to fail loudly.
   - Add optional `tags` or `notes` metadata for ledger/report grouping.

10. Preflight should become a lightweight first-class check.
    - Print and log imports, device, tokenizer pad/eos, dataset field availability, sequence length, batch settings, GPU memory, and download/cache availability.
    - Estimate parameter count and write it to the preflight report.
    - The framework may recommend smaller batch/sequence length on CUDA OOM, but should not silently retry or mutate configs yet.

11. Environment bootstrap should be stronger than documentation only.
    - Keep `selfcmd install-deps` for the course server.
    - Move toward a full scoped environment bootstrap for remote Phase 4 work.
    - Avoid heavy local virtualenv setup unless absolutely needed because the laptop has limited space.

12. Final workspace cleaning should remain conservative for now.
    - For the next submission, ensure `/workspace/report4.md` and Phase 4 files are correct even if older phase files remain.
    - A Phase4-only clean deploy mode can be future work if the endpoint requires a cleaner workspace.

13. Agent matrix support should be added.
    - Add a small YAML-driven experiment matrix.
    - First matrix priority: batch size / gradient accumulation sweep.
    - Cache on/off remains a secondary matrix because it is already implemented and useful for evidence.
    - The agent should continue to write patch proposals with risk and rollback, not automatically edit source code.

14. Agent comparison should be cautious.
    - The agent may name a "best" config based on tokens/sec plus validation-loss sanity.
    - In future work, consider a self-identified weighted benchmark score that balances speed, validation loss, stability, and complexity.
    - The current report should avoid over-optimizing a single scalar.

15. Evidence for switching should be staged.
    - First prove two successful switches through profiles: one GPT-style model profile and one dataset profile.
    - Next stage can attempt three model families including Qwen/DeepSeek if time permits.
    - Failed or neutral optimization results should be included briefly and honestly.

16. Most important next deliverable.
    - Add profile composition, preflight, and a batch/gradient-accumulation sweep.
    - Also test at least one additional model or dataset profile if remote time allows.

## Switching Implementation Status

Added locally on 2026-06-16:

- `training_framework/config_merge.py` for base/model-profile/data-profile/override composition.
- `training_framework/preflight.py` for per-run profile, model, data, tokenizer, package, CUDA, and token-budget checks.
- `agent/matrix_runner.py` for YAML-defined experiment sweeps.
- Model profiles for `tiny_gpt2`, `distilgpt2`, `gpt2`, Qwen-small placeholder, and DeepSeek placeholder.
- Data profiles for local fixture, TinyStories, WikiText-2, and optional OpenWebText subset.
- Matrix configs for cache on/off and batch-size/gradient-accumulation sweeps.
- `selfcmd` shortcuts for profile-smoke, matrix dry-run, matrix run, remote test, bootstrap, and DeepSeek probe.
- `selfcmd` shortcuts for auto-probe and recommendation-driven stability runs.

Validation status:

- Local syntax checks passed.
- Local manual unit harness passed because the laptop Python lacks `pytest`.
- `train.py --base ... --model-profile ... --data-profile ... --override ... --print-config` produced the expected resolved config.
- `agent.matrix_runner --dry-run` generated resolved matrix configs.
- Fresh remote validation later succeeded on the course GPU server.
- Remote `py_compile + pytest` passed with `24 passed in 2.55s` after matrix reasoning and preflight enhancements.
- A later remote validation after DeepSeek failure classification, preflight-only execution, Adafactor support, checkpoint-disable support, CUDA memory logging, and the token-budget probes passed with `37 passed in 3.03s`.
- The latest remote validation after adding `agent/auto_probe.py`, `agent/stability_runner.py`, and stability-runner regressions passed with `44 passed in 2.92s`.
- The latest remote validation after adding the WikiText real-data auto-probe regression passed with `45 passed in 3.10s`.
- The latest remote validation after adding calibrated memory prediction and recommendation-report generation passed with `48 passed in 3.00s`.
- The workflow-level `./selfcmd test` command saved its log under `/workspace/selfcmd-workflow/logs/`.
- Remote `profile-smoke` trained through the composed config path and wrote `preflight.json` / `preflight.md`.
- Remote `batch_grad_sweep` matrix dry-run expanded four resolved configs.
- The preflight pass exposed and helped fix a PyYAML YAML-1.1 parsing issue where unquoted `off` became boolean `False`; `TrainerConfig.mixed_precision` now normalizes booleans back to string enum values.
- Remote WikiText-2 data-profile smoke completed successfully, proving the data profile path can switch from a local fixture to a HuggingFace dataset.
- Full remote `batch_grad_sweep` completed successfully.
- Matrix result: `bs8_ga1` produced the best throughput, while `bs4_ga2` was slower despite the same effective batch size because gradient accumulation repeats forward/backward work.
- Matrix analysis exposed a train-loss logging bug for gradient accumulation; the trainer now logs average accumulated loss instead of summed microbatch loss.
- Matrix runner now writes best-config reasoning based on speed, validation-loss sanity, and complexity.
- Preflight now records inspectable dataset columns and OOM fallback recommendations.
- Qwen stretch smoke completed successfully with `Qwen/Qwen2.5-0.5B`, local fixture, `seq_len=64`, `batch_size=1`, and `max_steps=2`.
- Smoke results are compatibility evidence, not quality evidence. They prove construction, data flow, train/validation/checkpoint/preflight behavior, but do not replace longer training evidence.
- A controlled full `gpt2` TinyStories sanity run completed for 60 steps with 124.4M parameters, repeated validation, checkpointing, final validation loss `5.9795`, and mean throughput `5460.35` tokens/sec.
- A controlled Qwen TinyStories sanity run completed for 60 steps with 494.0M parameters, repeated validation, checkpointing, final validation loss `7.2141`, and mean throughput `348.71` tokens/sec.
- The Qwen longer run is real training-loop evidence beyond smoke, but it is still not a final quality or convergence claim.
- A targeted Qwen throughput probe showed that the original Qwen path was slow partly because it used too few tokens per optimizer step and had gradient checkpointing enabled.
- Qwen `seq_len=64,batch=1,gradient_checkpointing=true` reached only `350.41` tokens/sec and produced NaN loss in the short random-init run.
- Qwen `seq_len=64,batch=2,gradient_checkpointing=false` reached `897.67` tokens/sec, and `seq_len=128,batch=1,gradient_checkpointing=false` reached `897.69` tokens/sec.
- Matrix selection now ignores NaN validation losses, after the Qwen probe exposed that non-finite losses could pollute best-config reasoning.
- DeepSeek config/tokenizer loading succeeded for `deepseek-ai/deepseek-coder-1.3b-base`, identifying a LLaMA-style 1.346B-parameter random-init model path.
- A direct DeepSeek `seq_len=16,batch_size=1,max_steps=1` AdamW smoke failed inside `optimizer.step()` with CUDA OOM on the 24GB RTX 3090, confirming that optimizer state and temporary AdamW buffers are the current blocker.
- Added `train.py --preflight-only` so large stretch models can build tokenizer/model, write preflight evidence, and exit before optimizer/training.
- Added `deepseek_safety_probe`: `preflight_s16_b1` succeeds, while `adamw_s16_b1` is correctly classified as `cuda_oom`.
- Extended `deepseek_safety_probe` with `adafactor_s16_b1_no_ckpt`; this low-memory optimizer path completed one real DeepSeek training step at about `18.52` tokens/sec with train loss `10.6670`.
- Added DeepSeek auto-probe configs and a stability-runner path; the 2048-token fixture recommendation now has a 50-step pass.
- Fixed run discovery so `runs/matrix_logs` and other auxiliary directories are not mistaken for training run directories.

Post-matrix optimization decision:

1. Use `batch_size=8, grad_accum_steps=1` as the preferred current TinyStories/DistilGPT2 throughput config when it fits on the 24GB GPU.
2. Treat gradient accumulation as a memory fallback, not as the default speed path.
3. Keep validation-loss sanity in matrix selection; do not pick a faster variant if validation loss is clearly degraded.
4. Keep best-config reasoning in the matrix runner so the agent artifact explains speed, validation sanity, and complexity.

Next remote validation order:

1. Keep the DistilGPT2/GPT2-style path as the default experiment path because it is fast enough for repeated controlled sweeps.
2. Treat Qwen as the validated cross-family stretch profile. Use `seq_len=128,batch_size=1,grad_accum_steps=1,gradient_checkpointing=false` as its first tested recommendation, not the older conservative checkpointing-on path.
3. Treat DeepSeek as a memory-aware stretch path. Preflight-only is validated, full AdamW still OOMs, the first Adafactor/no-checkpoint smoke succeeded, and the auto-probe path now trains and validates through `2048` tokens/step.
4. The current DeepSeek stretch recommendation is `Adafactor + checkpoint.disabled + gradient_checkpointing=false`, with `2048` tokens/step as the best validated real-data feasibility profile so far: `3100.38` avg tokens/sec in the short WikiText probe and a 100-step WikiText stability pass at `3650.34` avg tokens/sec, with peak memory around `14.31GB allocated / 18.83GB reserved`.
5. Do not promote DeepSeek to a convergence-quality or final-model-quality claim yet. The next decision gate should be a longer 200-500 step real-data run or a second real-data profile, because the current result proves feasibility, not final training quality.

## Current Direction And v0.4 Alignment

The clearer current direction is:

```text
agentic training feasibility and optimization harness for causal-LM experiments
```

This keeps the spirit of the earlier v0.4 target, but makes the claim sharper.
The project is not trying to compete with DeepSpeed, FSDP, Megatron, PyTorch
Lightning, or HuggingFace Trainer. It is a smaller control plane that answers:

- Can this model/data/config fit on the available GPU?
- Which short-run knobs are safe enough to test next?
- What did speed, validation loss, memory, and failure logs say?
- Should this path graduate to a longer run or to a heavier training system?

Overlap with the v0.4 goals:

1. Config-level model/data switching remains central.
   - The `--base --model-profile --data-profile --override` path is exactly the comfortable switching layer we wanted.
   - The current claim is still causal-LM profile switching, not arbitrary task switching.

2. Cache, mixed precision, gradient accumulation, checkpointing, and memory-aware optimizer choices remain base framework capabilities.
   - They are exposed as measurable knobs.
   - They are not promised speedups; the report should keep neutral or negative results as evidence.

3. The agentic loop remains the project identity.
   - `selfcmd`, preflight, matrix runner, analyzer, ledger, and fetched artifacts together form the run -> inspect -> decide loop.
   - The agent should remain deterministic and auditable before it becomes more autonomous.

4. Stretch profiles remain useful.
   - GPT2 gives a practical stress profile.
   - Qwen proves cross-family profile switching.
   - DeepSeek proves memory-aware feasibility probing and low-memory optimizer exploration.

Revisions to the v0.4 target:

1. Replace "seamless switching for any architecture" with "bounded, reliable causal-LM switching."
   - Any-architecture support would require task adapters, collators, losses, metrics, and remote-code safety policies.
   - That is a future direction, not the current claim.

2. Replace "larger model training framework" with "larger model feasibility gate."
   - DeepSeek is valuable because it exposed AdamW optimizer-state OOM and validated an Adafactor path.
   - It should not become the main experiment until the 2048-token Adafactor path is repeated on a real dataset profile, not only on a small fixture.

3. Treat DeepSpeed/FSDP/Megatron as downstream export targets.
   - The scaffold can later emit recommended configs for those systems.
   - It should not try to reproduce distributed sharding, pipeline parallelism, or communication overlap inside Phase 4.

4. Add a stronger next-layer goal: automatic recommendation artifacts.
   - Each matrix should write a concise recommendation based on speed, validation-loss sanity, CUDA memory, failure class, and complexity.
   - Future work can add a self-identified weighted score, but the score should remain explainable.

5. Add a stronger next-layer goal: calibrated preflight.
   - Preflight should keep improving from static parameter counts toward calibrated memory estimates using observed CUDA peaks.
   - This is more useful for early-stage training decisions than chasing one more short-run speed number.

Implementation note on 2026-06-21:

- Added `agent/auto_probe.py` and `configs/auto_probes/deepseek_adafactor_token_budget.yaml`.
- The first auto-probe expands DeepSeek Adafactor token budgets through `64 -> 128 -> 256 -> 512`, stops on failure, and writes both timestamped summaries and `runs/recommendations/*-latest.md/json`.
- This turns the new direction into a concrete loop: start from a known-safe point, expand only inside a bounded ladder, classify failures, and recommend whether to try a larger budget, promote the last safe config, or run a longer stability check.
- Remote validation then followed the tool's recommendations through `1024` and `2048` tokens/step. `2048` succeeded at `3092.42` avg tokens/sec with about `14.31GB allocated / 18.58GB reserved`; because that is near the memory-headroom threshold, the tool correctly recommended a stability run rather than further token-budget expansion.
- Added `agent/stability_runner.py` and `./selfcmd deepseek-stability-run`, which materialize a longer config from the selected recommendation, run training, and write `runs/stability_summaries/*` plus `runs/recommendations/*-stability-latest.*`.
- The first stability-runner implementation exposed a harness bug: it counted train log events instead of optimizer steps, so `log_every=5` made a completed 50-step run look like 10 steps. The runner now reads `summary.json.global_step` first and records `logged_train_events` separately.
- After that fix, the 2048-token DeepSeek fixture path passed a 50-step run: `50/50` steps, `3680.77` avg tokens/sec, last train loss `0.0651`, last validation loss `0.0608`, and peak memory around `14.31GB allocated / 18.58GB reserved`.
- Decision at that point: record this as strong systems evidence, but do not call it quality evidence. The next useful development step was real-data probing and a calibrated memory predictor, not blindly pushing token budget higher.
- Added `configs/auto_probes/deepseek_adafactor_wikitext_realdata.yaml` plus `./selfcmd deepseek-realdata-probe` and `./selfcmd deepseek-realdata-stability-run`.
- After the course server recovered, the WikiText real-data probe succeeded at `512`, `1024`, and `2048` tokens/step. The selected `2048` probe reached `3100.38` avg tokens/sec, val loss `10.0320`, and peak memory around `14.31GB allocated / 18.83GB reserved`.
- The follow-up WikiText 50-step stability run passed: `50/50` steps, `3655.35` avg tokens/sec, train loss `8.9343 -> 6.4882`, last validation loss `6.9378`, and the same `14.31GB / 18.83GB` peak memory envelope.
- Updated decision: the 2048-token DeepSeek Adafactor path is now validated as real-data feasibility evidence, not just fixture evidence. It is still not a convergence or final-quality claim.
- Added a calibrated memory predictor to preflight. For the 100-step WikiText stability run, predicted reserved peak was `18814 MiB` versus observed `18828 MiB` (`-0.1%` error), and predicted allocated peak was `15051 MiB` versus observed `14305 MiB` (`+5.2%` error).
- Extended the WikiText stability gate to 100 steps: `100/100` steps, `3650.34` avg tokens/sec, train loss `9.5252 -> 6.3307`, last validation loss `6.6539`, and peak memory around `14.31GB allocated / 18.83GB reserved`.
- Added `agent/recommendation_report.py` and `./selfcmd recommendation-report`, which merge auto-probe, stability, preflight, and summary artifacts into a compact recommendation markdown/json.
- Updated decision: the tool should promote the DeepSeek `2048`-token Adafactor/WikiText profile only for longer probing, not as a final model-quality result.

## Non-Negotiables

- Keep previous Phase 1-3 files unchanged unless explicitly asked.
- Make checkpoint/resume real, not cosmetic.
- Make timing evidence real, not just total wall time.
- Keep the agent loop deterministic and auditable first.
- Prefer a small complete framework over a large fragile one.
