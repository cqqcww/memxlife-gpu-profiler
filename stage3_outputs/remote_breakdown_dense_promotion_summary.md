# Remote Breakdown Summary

Source log: [remote_breakdown_dense_promotion.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_dense_promotion.txt)

## Breakdown

```json
{
  "same_prefill_tokens_per_second": 85311.69497112469,
  "same_decode_tokens_per_second": 4633.5848827467835,
  "mixed_prefill_tokens_per_second": 48137.56160347993,
  "mixed_decode_tokens_per_second": 4079.481369334548
}
```

## Read

- `same_prefill` reflects the uniform prefill path.
- `same_decode` reflects the uniform decode path.
- `mixed_prefill` and `mixed_decode` are the most useful local fallback signals for serving-like traces.
