# Remote Decode Experiments (2026-06-01)

Baseline source: [remote_breakdown_20260601_summary.md](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_20260601_summary.md)

## Baseline

```json
{
  "same_prefill_tokens_per_second": 84328.92071175003,
  "same_decode_tokens_per_second": 5317.799929663348,
  "mixed_prefill_tokens_per_second": 48622.72615789409,
  "mixed_decode_tokens_per_second": 3764.238118498255
}
```

## Experiment A: bool SDPA mask for varlen paths

Result from remote breakdown:

```json
{
  "same_prefill_tokens_per_second": 89400.23829859201,
  "same_decode_tokens_per_second": 5227.9535791826165,
  "mixed_prefill_tokens_per_second": 48535.73876888359,
  "mixed_decode_tokens_per_second": 3674.287369228541
}
```

Read:

- `mixed_decode` regressed from `3764.24` to `3674.29`.
- `same_decode` also regressed slightly.
- This variant was rejected.

## Experiment B: shared-varlen decode K/V writes via batched `index_put_`

Result from remote breakdown:

```json
{
  "same_prefill_tokens_per_second": 87157.86463000828,
  "same_decode_tokens_per_second": 5205.420369755823,
  "mixed_prefill_tokens_per_second": 48648.086887617246,
  "mixed_decode_tokens_per_second": 4500.557491304203
}
```

Read:

- `mixed_decode` improved materially from `3764.24` to `4500.56` (`+19.6%`).
- `same_decode` dipped slightly from `5317.80` to `5205.42` (`-2.1%`).
- `mixed_prefill` stayed effectively flat.
- This is the best decode-focused change so far and is worth keeping.

## Experiment C: add shared `batch_positions` tensor cache on top of Experiment B

Result from remote breakdown:

```json
{
  "same_prefill_tokens_per_second": 86349.8153669632,
  "same_decode_tokens_per_second": 5178.550617317211,
  "mixed_prefill_tokens_per_second": 47534.44987447637,
  "mixed_decode_tokens_per_second": 4528.28853928011
}
```

Read:

- `mixed_decode` moved only marginally beyond Experiment B.
- `same_decode` and `mixed_prefill` both slipped slightly.
- This looked like noise rather than a robust win, so the extra `batch_positions` machinery was removed.

## Current takeaway

- Keep Experiment B: shared-varlen decode K/V writes should stay batched.
- Drop Experiment A and Experiment C.
- Next optimization work should focus on decode again, but with minimal semantic surface area.

## Experiment D: shared GQA cache for same/shared decode

Result from remote breakdown:

```json
{
  "same_prefill_tokens_per_second": 86326.45956721812,
  "same_decode_tokens_per_second": 4870.5244255955495,
  "mixed_prefill_tokens_per_second": 47687.56868606594,
  "mixed_decode_tokens_per_second": 4240.904191387085
}
```

Read:

- Both `same_decode` and `mixed_decode` regressed.
- Lazily storing an expanded shared GQA cache did not pay for its extra writes and memory traffic on the course GPU.
- This variant was rejected.

## Experiment E: use `F.rms_norm` fast path when available

Result from remote breakdown:

```json
{
  "same_prefill_tokens_per_second": 88704.2147711649,
  "same_decode_tokens_per_second": 5248.052372784535,
  "mixed_prefill_tokens_per_second": 49020.24475307869,
  "mixed_decode_tokens_per_second": 4534.947992743533
}
```

Read:

- `same_decode` improved modestly over the current kept baseline.
- `mixed_decode` also ticked upward, and prefill did not regress.
- This is a small but real win and is worth keeping.

## Experiment F: generalized shared-row decode path for subset shared batches

Result from remote public eval and breakdown:

```json
{
  "same_prefill_tokens_per_second": 88311.85326717069,
  "same_decode_tokens_per_second": 4761.586126661191,
  "mixed_prefill_tokens_per_second": 48756.88550992319,
  "mixed_decode_tokens_per_second": 4357.750963170688
}
```

Public eval also showed:

```json
{
  "throughput_tokens_per_second": 4270.149984940647,
  "mixed_tokens_per_second": 5418.1847037215075
}
```

Read:

- The fully generalized row-subset path helped the mixed public test, which suggests the serving trace does contain shrinking shared batches.
- But it hurt `same_decode` badly relative to the kept baseline, which means paying `index_put_` plus `index_select` on every full shared batch is too expensive.
- This variant was rejected as-is, but it provided the key insight for the next step: keep the generalized path only for true subsets and preserve the original fast path for full shared batches.

## Experiment G: dense/full shared fast path restoration

Status:

- Implemented locally after Experiment F.
- The change restores the original `k_buffer[:, position]` / slice fast path for full shared batches and only uses the generalized row-subset update when the batch has actually shrunk.
- This version was not fully benchmarked yet because the course service became unstable again before a clean fallback run could finish.

## Experiment H: proactive promotion of non-shared decode groups into shared batches

Status:

- Implemented locally after Experiment G.
- Idea: if decode receives multiple requests that are currently separate but now travel together, repack them into a shared batch so subsequent decode steps can use shared-cache paths instead of repeating non-shared batching overhead.
- The first run surfaced a correctness bug in varlen token-buffer updates caused by writing through advanced indexing plus `.copy_`, which does not mutate the original tensor.
- That bug has been fixed by switching the varlen shared token update back to `index_put_`.
- This version still needs a clean remote fallback run once the course service recovers.

## Experiment I: fixed dense/shared fast path + proactive shared-batch promotion

Result from remote fallback (2026-06-02):

```json
{
  "same_prefill_tokens_per_second": 88756.68001185993,
  "same_decode_tokens_per_second": 4594.4265059343925,
  "mixed_prefill_tokens_per_second": 48802.21409333266,
  "mixed_decode_tokens_per_second": 4192.540006193073
}
```

Public eval for the same run:

```json
{
  "throughput_tokens_per_second": 4290.140586190027,
  "mixed_tokens_per_second": 5315.109120833072
}
```

Read:

- Correctness recovered; both public smoke tests passed again after fixing the token-buffer write bug.
- The public mixed benchmark improved materially relative to the earlier public fallback, which suggests promotion can help some serving traces.
- But both `same_decode` and `mixed_decode` in the more diagnostic breakdown regressed versus the kept `Experiment E` baseline.
- Current conclusion: unconditional promotion is too aggressive to keep as the default path. The next version should keep the idea but gate it behind a tighter heuristic.

## Experiment J: tighter gated promotion + subset repack

Public fallback result (2026-06-02):

```json
{
  "throughput_tokens_per_second": 4038.104805850672,
  "mixed_tokens_per_second": 5055.415794062275
}
```

Read:

- This variant tightened non-shared promotion thresholds and added repack for clearly shrunken shared subsets.
- It still passed public correctness, but both public throughput and public mixed throughput were worse than the immediately preceding official-local fallback (`4317.53 / 5227.28`).
- That is enough to reject it without waiting for a full breakdown: the heuristic surface is still too aggressive for the dominant serving paths.

## Experiment K: flatten layer dicts into fixed-weight objects

Public fallback result (2026-06-08):

```json
{
  "throughput_tokens_per_second": 4278.129343452676,
  "mixed_tokens_per_second": 5249.819460713653
}
```

Read:

- This variant replaced per-layer string-keyed dict lookups with a lightweight fixed-weight object and hoisted a few bound-method lookups out of the decode/prefill loops.
- It remained correct, but both public throughput and public mixed throughput slipped versus the kept same-length shared-promotion baseline (`4341.23 / 5298.91`).
- Conclusion: the Python dict lookup cost is not the dominant bottleneck here; flattening alone is not worth carrying forward.

## Experiment L: force tiny decode paths onto manual attention instead of SDPA

Public fallback result (2026-06-08):

```json
{
  "throughput_tokens_per_second": 3780.9997660825584,
  "mixed_tokens_per_second": 4524.279681459482
}
```

Read:

- This variant kept the stable serving heuristics but disabled SDPA for decode paths on the tiny course model, forcing the eager matmul-softmax-matmul branch instead.
- It still passed public correctness, but the regression was large enough to reject immediately.
- Conclusion: despite the small model size, the SDPA kernels are still decisively better than the hand-written decode attention path on the course GPU.
