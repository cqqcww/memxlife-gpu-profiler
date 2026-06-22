# Phase 4 Honor Requirements Questions

Source guide: [PHASE4_HONOR_GUIDE.md](/Users/amanda/Desktop/School/mlsys/phase4-honor/PHASE4_HONOR_GUIDE.md)

Goal: build an agentic mini training framework that can construct, run, inspect, optimize, checkpoint, resume, log, and report a small language-model training workflow.

Reply format suggestion:

```text
1C 2A 3B 4B ...
```

If one question needs explanation, answer it separately after the option list.

---

## A. Project Scope

1. What should the Phase 4 agent primarily do?
   - A. Generate the whole framework once from templates
   - B. Maintain a fixed framework and only generate configs
   - C. Generate, run, inspect logs, then modify configs/code iteratively (recommended)
   - D. Mostly document manual engineering work as an "agent" narrative

2. What should be the first milestone?
   - A. Minimal end-to-end training loop with one batch, one checkpoint, one resume (recommended)
   - B. Full feature-complete framework before any run
   - C. Start from speed optimizations immediately
   - D. Start from the final report outline

3. How ambitious should the framework be?
   - A. Small, robust, easy to explain (recommended)
   - B. Medium complexity with several optional optimizations
   - C. Aggressive training-speed engineering
   - D. Large model focused, even if iteration is slower

4. How should we position the final result?
   - A. A complete mini framework with clear logs and reflection (recommended)
   - B. A speed-optimization project first
   - C. A model-quality fine-tuning project first
   - D. A framework-generation agent demo first

---

## B. Model And Dataset

5. Which model scale should we target first?
   - A. Tiny GPT-style config from scratch for fastest debugging
   - B. Small HuggingFace causal LM, around GPT-2 small or smaller (recommended)
   - C. Qwen-family small model if hardware allows
   - D. Qwen3-0.6B target from the start

6. Should we load pretrained weights?
   - A. No, construct from config first for speed and reproducibility
   - B. Yes, fine-tune pretrained weights if download and GPU memory allow
   - C. Support both through config, default to config-from-scratch first (recommended)

7. Which dataset direction do you prefer?
   - A. Tiny local text fixture first, then HuggingFace dataset
   - B. WikiText-2 / WikiText subset
   - C. TinyStories or small story/text subset
   - D. Custom course/report text corpus
   - E. Support A plus one real HF dataset path (recommended)

8. Sequence length for the first real experiment?
   - A. 128 for fastest iteration
   - B. 256 as default balance (recommended)
   - C. 512 for more realistic throughput
   - D. Configurable sweep over 128/256/512

9. Data preprocessing strategy?
   - A. Tokenize on the fly inside Dataset
   - B. Pre-tokenize once and cache token blocks (recommended)
   - C. Dynamic packing with variable-length samples from the start
   - D. Implement both simple and packed modes

10. Validation split style?
   - A. Fixed percentage split from the dataset (recommended)
   - B. Separate named validation dataset
   - C. No validation in early version
   - D. Configurable split plus optional fixed seed

---

## C. Framework Architecture

11. Should we follow the guide's exact module structure?
   - A. Yes: `data.py`, `model.py`, `optim.py`, `scheduler.py`, `trainer.py`, `checkpoint.py`, `logger.py` (recommended)
   - B. Fewer files for simplicity
   - C. More files, including profiler and agent modules
   - D. Start exact, then add `agent/` and `experiments/`

12. Config system?
   - A. YAML config loaded into dataclasses (recommended)
   - B. Pure Python config files
   - C. argparse only
   - D. Hydra/OmegaConf-style config

13. Training entrypoint shape?
   - A. `python train.py --config configs/debug.yaml` (recommended)
   - B. `bash run.sh`
   - C. Both `train.py` and `run.sh`
   - D. Agent command only, e.g. `python agent.py run`

14. Optimizer support?
   - A. AdamW only
   - B. AdamW with weight-decay/no-decay parameter groups (recommended)
   - C. AdamW plus SGD/Adafactor
   - D. Include custom optimizer experiment

15. Scheduler support?
   - A. Constant learning rate
   - B. Linear warmup plus cosine decay (recommended)
   - C. Linear warmup plus linear decay
   - D. Configurable scheduler registry

16. Core trainer features for v1?
   - A. Train, validate, log, checkpoint, resume (minimum)
   - B. A plus gradient clipping and gradient accumulation (recommended)
   - C. B plus mixed precision
   - D. C plus `torch.compile`

---

## D. Speed And Optimization

17. What should be the primary speed metric?
   - A. Step time
   - B. Samples/sec
   - C. Tokens/sec (recommended)
   - D. Tokens/sec plus timing breakdown

18. What timing breakdown do we need?
   - A. Overall step time only
   - B. Data time vs compute time
   - C. Data, forward, backward, optimizer, logging/checkpoint overhead (recommended)
   - D. Full `torch.profiler` trace from the start

19. First optimization direction?
   - A. Pre-tokenization and token-block caching (recommended)
   - B. Mixed precision
   - C. Dataloader workers/pin memory/persistent workers
   - D. `torch.compile`
   - E. Compare A/B/C in a small sweep

20. Mixed precision policy?
   - A. Disable initially for simplicity
   - B. Enable bf16/fp16 automatically when GPU supports it (recommended after v1)
   - C. Always use fp16
   - D. Make it config-only, off by default

21. Should we include activation checkpointing?
   - A. No for v1
   - B. Optional config toggle, off by default (recommended)
   - C. On by default for larger models
   - D. Only if we target Qwen3-0.6B

22. Should we reuse earlier phase operators/kernels?
   - A. No, keep Phase 4 focused on framework engineering first (recommended)
   - B. Try to integrate Phase 2 LoRA ideas as optional module
   - C. Try custom kernels only after baseline framework works
   - D. Make operator reuse a report-only discussion

---

## E. Agentic Workflow

23. Agent implementation style?
   - A. Rule-based planner/runner/analyzer scripts, no external LLM dependency (recommended)
   - B. Optional LLM suggestions with deterministic fallback
   - C. LLM-first agent that edits code automatically
   - D. Manual workflow documented as agentic development

24. What should the agent loop automate?
   - A. Generate config and run training
   - B. Run training, parse logs, summarize metrics
   - C. B plus propose next config optimization (recommended)
   - D. C plus modify source code automatically

25. What artifacts should the agent write after each run?
   - A. Plain log only
   - B. JSON summary only
   - C. JSON summary, markdown summary, TensorBoard logs, copied config (recommended)
   - D. C plus plots generated automatically

26. How much autonomy should code modification have?
   - A. None; code changes are manual
   - B. Agent proposes diffs, we apply manually (recommended)
   - C. Agent writes patches automatically inside allowed modules
   - D. Agent can rewrite framework modules freely

---

## F. Testing And Debugging

27. Required smoke tests?
   - A. Import modules only
   - B. One tiny train step
   - C. Tiny train + validation + checkpoint save + resume (recommended)
   - D. Full public-style run only

28. Resume correctness check?
   - A. Only check checkpoint loads
   - B. Check global step and learning rate continue correctly (recommended)
   - C. B plus compare resumed loss continuity
   - D. B plus RNG-state restoration

29. Data tests?
   - A. Batch shape and dtype only
   - B. Batch shape, label shift, attention mask, split size (recommended)
   - C. B plus deterministic token cache hash
   - D. B plus packing efficiency report

30. Debug logging level?
   - A. Minimal console logs
   - B. Console + TensorBoard
   - C. Console + TensorBoard + JSONL events (recommended)
   - D. C plus detailed per-step trace files

31. Failure cases we should intentionally test?
   - A. Bad config
   - B. Resume from missing checkpoint
   - C. Dataset/tokenizer cache invalidation
   - D. All of the above (recommended)

---

## G. Report And Evidence

32. Report emphasis?
   - A. Framework architecture
   - B. Agent workflow and decisions
   - C. Bugs/debugging/reflection
   - D. Balanced across architecture, agent loop, experiments, reflection (recommended)

33. Evidence to collect for final report?
   - A. Config snippets and module diagram
   - B. TensorBoard screenshots/loss curves
   - C. Checkpoint/resume logs
   - D. Throughput tables and timing breakdown
   - E. All of the above (recommended)

34. How much experiment comparison?
   - A. Baseline only
   - B. Baseline vs one optimization
   - C. Baseline vs 2-3 targeted optimizations (recommended)
   - D. Larger sweep across model/data/trainer settings

35. Final report file name?
   - A. `report4.md` (recommended)
   - B. `phase4_report.md`
   - C. `honor_report.md`
   - D. Decide later based on submission instructions

36. What should be our "if redesigning" reflection angle?
   - A. More robust agent autonomy
   - B. Better profiling and run ledger from day one
   - C. Cleaner config/data cache abstractions
   - D. All of the above (recommended)
