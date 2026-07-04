# Remote Breakdown Summary

Source log: [remote_breakdown_dense_promotion_fix_20260602.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_dense_promotion_fix_20260602.txt)

## Breakdown

```json
{
  "same_prefill_tokens_per_second": 88756.68001185993,
  "same_decode_tokens_per_second": 4594.4265059343925,
  "mixed_prefill_tokens_per_second": 48802.21409333266,
  "mixed_decode_tokens_per_second": 4192.540006193073
}
```

## Read

- `same_prefill` reflects the uniform prefill path.
- `same_decode` reflects the uniform decode path.
- `mixed_prefill` and `mixed_decode` are the most useful local fallback signals for serving-like traces.
