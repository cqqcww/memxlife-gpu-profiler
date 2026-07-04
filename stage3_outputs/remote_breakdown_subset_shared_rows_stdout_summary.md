# Remote Breakdown Summary

Source log: [remote_breakdown_subset_shared_rows_stdout.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_subset_shared_rows_stdout.txt)

## Breakdown

```json
{
  "same_prefill_tokens_per_second": 88311.85326717069,
  "same_decode_tokens_per_second": 4761.586126661191,
  "mixed_prefill_tokens_per_second": 48756.88550992319,
  "mixed_decode_tokens_per_second": 4357.750963170688
}
```

## Read

- `same_prefill` reflects the uniform prefill path.
- `same_decode` reflects the uniform decode path.
- `mixed_prefill` and `mixed_decode` are the most useful local fallback signals for serving-like traces.
