# Stage 2 Output Summary

- Mission id: `c0f31ee32dec227bb48cae77b577a2e7`
- Status: `completed`
- Saved markdown output: [c0f31ee32dec227bb48cae77b577a2e7.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/c0f31ee32dec227bb48cae77b577a2e7.md)

## Top-level Evaluation Results

- Case 1: `correct = True`, `speedup = 342.1327226062235`
- Case 2: `correct = True`, `speedup = 1.9401524893940716`
- Case 3: `correct = True`, `speedup = 0.994588466415906`

## What This Run Confirms

- This submission definitely ran the newer synced agent rather than the stale server workspace.
- The official report now includes the new search-strategy wording about dual-slot exact-repeat caching and delayed `W_eff` thresholds.
- The official run evaluated `3` candidates instead of the old stale `1` candidate behavior.
- The third evaluated candidate was `aten_addmm_inplace_btcontig_mainfirst_hybridweff_dualrepeat`, so the new search space was at least partially exercised.

## What It Did Not Reach

- The run did not get through all `6` local candidates within the current time budget.
- The threshold-based variants were not reached before the run stopped.
- `debug_stats` still came back as `{}` in the official report for every evaluated candidate, so the new telemetry fields did not yet surface useful path counters remotely.

## Comparison Against Prior Reference Runs

- Versus [6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/6c5e83a34efa2beb67dcdaf68f6d4fd6_summary.md):
  - Case 1 is lower: `342.13x` vs `361.35x`
  - Case 2 is higher: `1.9402x` vs `1.9258x`
  - Case 3 is lower: `0.9946x` vs `0.9969x`
- Versus [efbab79b85fe777801ed0c4ba7e6ab44_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/efbab79b85fe777801ed0c4ba7e6ab44_summary.md):
  - Case 1 is higher: `342.13x` vs `291.24x`
  - Case 2 is higher: `1.9402x` vs `1.9246x`
  - Case 3 is lower: `0.9946x` vs `1.0033x`
- Versus [e8b863df4dbd37afa379c4944216229a_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage2_outputs/e8b863df4dbd37afa379c4944216229a_summary.md):
  - Case 1 is lower: `342.13x` vs `353.98x`
  - Case 2 is higher: `1.9402x` vs `1.9356x`
  - Case 3 is lower: `0.9946x` vs `0.9958x`

## Aggregation Notes

- Arithmetic mean: lower than `6c5e...` and `e8...`, higher than `efbab...`
- Geometric mean: lower than `6c5e...` and `e8...`, higher than `efbab...`
- Harmonic mean: slightly above `6c5e...`, slightly below `efbab...`, and essentially tied with `e8...`

## Interpretation

- This is not the new best official result.
- The main value of this run is methodological: it proves the synced workspace path is now correct and that the official evaluator can see the new candidate space.
- Performance-wise, the old single-slot `hybrid_weff` path still won among the candidates that actually got evaluated.
- The next bottleneck is now clear: candidate compile time is so expensive that we are only getting through three candidates in the official budget, which means the threshold variants never get a chance.
