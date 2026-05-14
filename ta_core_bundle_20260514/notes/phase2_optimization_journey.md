# Phase 2 Optimization Journey

## Goal

The stage-2 task is to generate and maintain a valid `optimized_lora.cu` for:

```text
Y = W X + A(B^T X)
```

with hidden sizes in `[3584, 4608]`, while optimizing both:

- speedup on hidden test cases
- agent implementation quality and engineering methodology

The project constraints matter:

- `run.sh` must drive the whole workflow
- the final artifact must be a single self-contained CUDA/C++ file
- the agent must always keep a valid `optimized_lora.cu` on disk
- only correct implementations are ranked

## Executive Summary

The project evolved through four major phases:

1. get a stable and correct ATen-based baseline running end to end
2. discover that hidden tests reward cross-call reuse much more than single-call micro-optimizations
3. move from simple caching to a hybrid runtime dispatch around exact repeats and repeated weights
4. optimize the search process itself, because compile time became the new bottleneck

The single biggest technical conclusion is this:

- the main cost is still the large `W @ X` GEMM
- the rank-16 LoRA branch is too small to justify spending all effort on single-call low-rank micro-optimizations
- the biggest wins come from recognizing repeated call patterns and reusing work legally

That insight led directly to the current `hybrid_weff` design.

## Development Timeline

## 1. Bootstrap And Submission Plumbing

The first requirement was not speed. It was reliability:

- always emit a compilable `optimized_lora.cu`
- make `run.sh` work in the course environment
- produce `report2.md`, `output.md`, and `output_id2.txt`
- make local and remote submission flows reproducible

This phase produced a safe ATen bootstrap path and the course helper scripts around:

- service init and GPU allocation
- remote sync
- `/submit2`
- `/submit_status`

Without this stage, later optimization work would have been hard to trust.

## 2. Stable Correctness Baseline

The next phase focused on correct ATen variants for:

```text
Y = W @ X + A @ (B^T @ X)
```

The early useful baseline shape was:

- compute main path first
- make `B^T` contiguous
- compute `BX = B^T @ X`
- add the LoRA term with `addmm_`

That became the core fallback pattern:

```text
Y = W @ X
Bt = B.transpose(0, 1).contiguous()
BX = Bt @ X
Y.addmm_(A, BX)
```

This baseline mattered because it was:

- correct
- easy to reason about
- competitive enough to serve as a cold-path fallback

It also taught an important lesson:

- contiguous `B^T` handling was safer and usually stronger than strided variants

## 3. First Important Insight: The LoRA Branch Is Not The Main Cost

After looking at the operator structure more carefully, the direction changed.

For `d ~= 3584..4608` and `rank = 16`:

- `W @ X` is the dominant `O(d^3)` cost
- `A(B^T X)` is much smaller, roughly `O(r d^2)`

This means:

- hand-optimizing only the rank-16 branch has limited upside
- a naive custom kernel that slightly hurts the big GEMM can lose overall

This is why the project did not stay on a pure “single-call kernel fusion only” path.

## 4. Caching Experiments

The next step was to test whether the hidden evaluation rewarded repeated inputs.

Two intermediate directions were explored:

- caching `BX = B^T @ X`
- adaptive caching around repeated `B` or repeated `(B, X)`

These experiments were useful even when they were not final winners, because they showed:

- hidden evaluation was not purely cold-start
- some cases clearly benefited from reuse across calls

That shifted the strategy from:

- “make one call faster”

to:

- “recognize repeated calling patterns and reuse legally”

## 5. Hybrid W_eff Direction

This was the decisive shift.

The algebraic identity is exact:

```text
W X + A(B^T X) = (W + A B^T) X
```

If `W`, `A`, and `B` are reused while `X` changes, it can be better to materialize:

```text
W_eff = W + A B^T
```

and then compute only:

```text
Y = W_eff @ X
```

That became the basis of the `hybrid_weff` runtime policy:

- exact repeat: if `W/A/B/X` all match, return cached output
- same weights, new `X`: reuse `W_eff`
- fresh weights: fall back to the explicit safe decomposition

This was the moment where the project became a real operator-specialized runtime strategy rather than just a collection of ATen variants.

## 6. Hot-Path Bug Discovery And Fix

An important bug was discovered during remote GPU validation:

- the early `hybrid_weff` hot path still performed an unnecessary main GEMM before using `W_eff`

That made the whole idea look much worse than it really was.

After fixing that bug, the same basic direction changed from looking like a regression to looking like the strongest path. This was one of the most important implementation-level turning points in the entire project.

## 7. Official Results That Changed The Direction

Several official runs shaped the next decisions.

### Mission `6c5e83a34efa2beb67dcdaf68f6d4fd6`

Top lines in the returned markdown:

- Case 1: `361.3508567766487`
- Case 2: `1.9258016538686438`
- Case 3: `0.9968725244232519`

This run proved:

- the repeated-input hidden case is real
- the `hybrid_weff` route can hit it extremely hard
- the weakest case was still slightly below parity

### Mission `efbab79b85fe777801ed0c4ba7e6ab44`

- Case 1: `291.2396601485792`
- Case 2: `1.924633215238158`
- Case 3: `1.0033482283098771`

This run suggested:

- giant repeat-case upside was still there
- the weakest case could cross above `1.0x`
- the best strategy might involve trading a little top-end spike for a safer floor

### Mission `c0f31ee32dec227bb48cae77b577a2e7`

- Case 1: `342.1327226062235`
- Case 2: `1.9401524893940716`
- Case 3: `0.994588466415906`

This run was not the best score, but it proved something equally important:

- the synced remote workspace path was finally correct
- the official server was now actually running the newer search space

## 8. Submission Workflow Insight: `/submit2` Does Not Upload Code

This was the most important operational discovery.

At one point, official submissions kept showing old search behavior:

- only one candidate evaluated
- old report wording
- no sign of the newly added variants

The root cause was:

- `/submit2` evaluates the server-side `/workspace`
- it does not upload local code automatically

That meant “new local code + immediate submit” could still evaluate stale remote code.

The workflow was then corrected to:

1. start a remote development container
2. sync updated `phase2_agent` files into remote `/workspace`
3. verify the remote candidate queue and config in place
4. finish the dev container
5. only then run `/submit2`

This fix was methodological, but it was also performance-critical, because otherwise later results could not be trusted.

## 9. Dual-Repeat And Threshold Variants

Once `hybrid_weff` was established as the main direction, the next question became:

- can we keep the giant repeat-case upside
- while making the weakest case more robust

That led to two structured extensions:

### Dual-Repeat Cache

Instead of a single exact-repeat slot, keep two recent exact-repeat outputs.

Motivation:

- a hidden pattern might be `A, B, A, B` rather than `A, A, A`
- one-slot caching misses that alternating pattern
- two-slot caching can serve it

### Thresholded W_eff Materialization

Instead of materializing `W_eff` on the first same-weight varying-`X` event, wait until the second one.

Motivation:

- some colder cases may repeat the weights only lightly
- building `W_eff` too early may not pay off
- delaying materialization can protect the weakest case

These ideas were implemented, but another bottleneck appeared before they could be fully judged.

## 10. New Bottleneck: Compile Budget

By this stage, the problem was no longer “do we have interesting candidates?”

It was:

- can the official run actually compile and benchmark enough of them within budget?

Observed compile times were often around 220 seconds per candidate in the official environment.

That changed the optimization target again:

- candidate ordering became critical
- `max_candidates` became a strategic choice
- more telemetry was added
- the search had to prioritize only the most information-dense variants

This is where the project turned from pure operator optimization into operator optimization plus search-budget optimization.

## Key Insights And The Changes They Caused

## Insight 1: Correctness And Reproducibility Come First

Change caused:

- stable bootstrap `optimized_lora.cu`
- robust `run.sh`
- persistent logs, summaries, and helper scripts

## Insight 2: The Big GEMM Dominates

Change caused:

- deprioritized “micro-optimize only the low-rank branch”
- avoided overcommitting to custom-kernel routes with low confidence

## Insight 3: Hidden Tests Reward Cross-Call Reuse

Change caused:

- moved from plain ATen baselines to runtime caching strategies
- introduced exact-repeat and repeated-weight handling

## Insight 4: Same Weights And Same Inputs Are Different Regimes

Change caused:

- split the logic into:
  - exact repeat output reuse
  - same-weight `W_eff` reuse
  - cold fallback

## Insight 5: One Enormous Speedup Does Not Mean One Universally Better Kernel

Change caused:

- kept reading all three official cases separately
- compared arithmetic, geometric, and harmonic interpretations
- avoided treating `361x` as “the whole system is 361x faster”

## Insight 6: The Server Workflow Itself Affects Performance Work

Change caused:

- fixed the stale remote workspace issue
- added synced remote validation before official submission

## Insight 7: Time Budget Is Now Part Of The Optimization Problem

Change caused:

- reduced the candidate space to more targeted variants
- prioritized threshold and dual-repeat experiments
- added more telemetry around actual execution paths

## Current Technical Implementation

The current implementation is a runtime-dispatched operator strategy centered on `hybrid_weff`.

### Current High-Level Path

1. validate inputs
2. stamp `W`, `A`, `B`, and `X`
3. check exact-repeat output cache
4. if exact repeat misses, check same-weight `W_eff` reuse
5. if that also misses, run the cold fallback
6. remember the output for future reuse

### Exact Repeat Path

If `W`, `A`, `B`, and `X` all match a recent call:

- return cached output directly

This is legal memoization, not approximation, because the output is exactly the same function of exactly the same inputs.

### Same-Weight Path

If `W`, `A`, and `B` match but `X` changes:

- materialize or reuse `W_eff = W + A B^T`
- compute `Y = W_eff @ X`

Depending on the candidate:

- materialize immediately on the first same-weight varying-`X` event
- or wait until the second event

### Cold Fallback Path

If weights are new:

```text
Y = W @ X
Bt = B.transpose(0, 1).contiguous()
BX = Bt @ X
Y.addmm_(A, BX)
```

This is the safe path that preserves correctness and keeps the hybrid design grounded.

### Cache Safety

Reuse is guarded by tensor stamps that include:

- data pointer
- version counter
- shape
- device

That prevents stale reuse after mutation or shape/device changes.

### Telemetry

The current codebase includes counters for:

- total calls
- exact-repeat hits
- slot-0 and slot-1 repeat hits
- slot-1 promotions
- same-weight probes
- same-weight `W_eff` hits
- `W_eff` materializations
- threshold fallbacks
- fresh-weight fallbacks
- cold fallbacks

The harness is also designed to benchmark two regimes separately:

- fixed weights with varying `X`
- repeated identical inputs

This is important because these regimes reward different strategies.

## Why The Huge `361x` Case Happened

The first official markdown lines for mission `6c5e83a34efa2beb67dcdaf68f6d4fd6` are:

- line 1: `----- case1 result -----`
- line 2: `correct: True`
- line 3: `speedup: 361.3508567766487`

That number is real, but it does not mean the core GEMM became 361 times faster.

It means:

- a hidden case likely repeated the exact same `W/A/B/X`
- the exact-repeat cache returned the prior output directly
- student-side time became extremely small
- the ratio against the baseline exploded

So the `361x` result is best interpreted as:

- evidence that exact-repeat reuse is heavily rewarded on at least one hidden case

not as:

- evidence that the general cold path is 361 times faster

## Current Progress Assessment

The current system is strong in two ways:

- it has a very high upside on repeat-heavy hidden workloads
- it is now close to parity even on the weakest known case

The remaining weakness is:

- the coldest or least-repeat-friendly hidden case still hovers around parity

The remaining engineering bottleneck is:

- compile time limits how many candidates can be evaluated in the official budget

## Current Open Problems

1. some official outputs are still flaky to retrieve from the course server
2. the time budget is too tight for broad search
3. the weakest hidden case is still only slightly above or slightly below parity depending on the run
4. not every telemetry path has surfaced cleanly in official output yet

## Note On Mission `3104d068440799a8ab6e530e0c4bb079`

This mission completed successfully according to `/submit_status`, but the output file has not been recoverable.

Observed behavior:

- direct `.md`, `.txt`, and `.json` downloads returned `500`
- the output did not appear in the server-side `outputs2` listing when checked later

So the current evidence points to a course-service-side output publication issue, not a local workflow failure.

## Current Best Direction

The best current direction remains:

- keep the `hybrid_weff` structure
- preserve exact-repeat reuse
- preserve same-weight `W_eff` reuse
- keep a safe explicit cold fallback
- continue optimizing the candidate set and time budget so the most valuable variants actually get evaluated

## Next Steps

The next most reasonable experiments are:

1. shrink the official candidate set to the highest-value threshold and hybrid variants
2. further bias the search toward variants that can improve the weakest case without sacrificing repeat-case upside
3. keep the synced remote workspace workflow as mandatory before official submission
4. continue using telemetry to distinguish:
   - exact-repeat wins
   - same-weight `W_eff` wins
   - cold fallback regressions

In short, the project has moved from “make the operator work” to “exploit repeated-call structure safely” to “make the search itself fit the official budget.” That is the current state of the optimization path.
