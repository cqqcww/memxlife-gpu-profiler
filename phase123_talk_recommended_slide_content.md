# Phase 1-2-3 Recommended Slide Content

这份文档只服务一件事：  
**每页 PPT 上到底建议放什么字。**

原则：

- 每页只放一个主观点
- 文字只做路标，不做讲稿
- 重心放在实现思路、问题、调试、反思
- 尽量少报排名和结果

---

## Slide 1. Title

### Title

`From GPU Profiling to Runtime Specialization`

### Subtitle

`Measure -> Model -> Specialize -> Validate`

### Small line

`One project, three layers of system understanding`

---

## Slide 2. Unified Narrative

### Title

`One Project, Three Increasingly Realistic Questions`

### Table

| Phase | Core question | What changed |
|---|---|---|
| 1 | What is the GPU really doing? | API values were not enough |
| 2 | What is the operator really repeating? | Single-call speed was not enough |
| 3 | What is the runtime really spending time on? | Local fast paths were not enough |

### Bottom line

`From local signals to system structure`

---

## Slide 3. Phase 1 Problem And System

### Title

`Phase 1: Hardware Profiling Under Untrusted Signals`

### Left-side text

Problem:

- `APIs may be incomplete or untrustworthy`
- `We needed active probing, not passive reading`

### Right-side text

System idea:

- `Build a minimal closed-loop profiling agent`
- `Generate -> compile -> run -> validate`

---

## Slide 4. Phase 1 Turning Point

### Title

`Phase 1 Turning Point: Evidence Matters More Than A Single Number`

### Left column

Initially:

- `Read metrics`
- `Run probes`
- `Collect outputs`

### Right column

What we learned:

- `Build evidence`
- `Check consistency`
- `Repair failing probes`

### Bottom line

`Trust the explanation, not just the number`

---

## Slide 5. Phase 2 Problem

### Title

`Phase 2: The First Optimization Target Was Not The Best One`

### Formula

`Y = W X + A(B^T X)`

### Left column

Initial intuition:

- `Stabilize the baseline`
- `Improve B^T layout`
- `Start from the low-rank branch`

### Right column

What changed:

- `W @ X stayed dominant`
- `The hidden workload repeated`
- `Reuse mattered more than local fusion`

---

## Slide 6. Phase 2 Core Insight

### Title

`Phase 2 Core Insight: Optimize The Reuse Pattern, Not Just The Formula`

### Formula

`W X + A(B^T X) = (W + A B^T) X`

### Dispatch policy

- `Exact repeat -> reuse output`
- `Same weights -> reuse W_eff`
- `Cold case -> safe fallback`

### Bottom line

`Optimize repeated structure, not one isolated call`

---

## Slide 7. Phase 2 Debug Reality

### Title

`Phase 2: Debug Evidence Was As Important As The Idea`

### Main bullets

- `The fast path still did extra work`
- `Local intuition and remote behavior diverged`
- `Submission workflow also mattered`

### Bottom line

`When theory and performance disagree, inspect the evidence chain`

---

## Slide 8. Phase 3 Structural Upgrade

### Title

`Phase 3: Throughput Became A Runtime Structure Problem`

### Left column

Baseline:

- `Too much repeated decode work`
- `Limited runtime specialization`

### Right column

Our redesign:

- `Per-layer KV cache`
- `Request-state tracking`
- `Prefill / decode split`

### Bottom line

`Throughput is a runtime-structure problem`

---

## Slide 9. Phase 3 Experiment Logic

### Title

`Phase 3: We Kept The Most Robust Runtime, Not The Most Exciting Local Trick`

### Left column

Retained:

- `Batched index_put_`
- `F.rms_norm`
- `Conservative promotion`

### Right column

Rejected:

- `Generalized shared-row path`
- `Over-aggressive promotion`
- `Fragile manual rewrites`

### Bottom line

`Local fastest did not mean overall best`

---

## Slide 10. Final Takeaways

### Title

`What We Actually Built Across The Three Phases`

### Main bullets

- `Structure-first optimization`
- `Evidence-first debugging`
- `Fast path + safe fallback`

### Reflection block

If redesigning the whole project:

- `Standardize instrumentation earlier`
- `Track why directions changed`
- `Optimize for understanding, not just score`

---

## Optional Backup A. Repository Structure

### Title

`Repository Structure`

### Bullets

- `memxlife-project`
- `phase2_agent`
- `phase3_engine_sources`
- `workspace`
- `evaluator`

### Bottom line

`Each phase closed the loop independently before we connected them narratively`

---

## Optional Backup B. Key Numbers

### Title

`Representative Result Snapshots`

### Bullets

- `Use only 2-3 evidence points per phase`
- `Explain what each number means`
- `Do not let this slide become the talk`

### Bottom line

`Numbers support the reasoning; they are not the reasoning`
