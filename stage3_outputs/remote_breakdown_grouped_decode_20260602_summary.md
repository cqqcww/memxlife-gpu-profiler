# Remote Breakdown Summary

Source log: [remote_breakdown_grouped_decode_20260602.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_grouped_decode_20260602.txt)

## Breakdown

```json
{
  "same_prefill_tokens_per_second": 80988.74520819818,
  "same_decode_tokens_per_second": 3995.3740118487585,
  "mixed_prefill_tokens_per_second": 43677.67053361909,
  "mixed_decode_tokens_per_second": 3623.1935071002736
}
```

## Read

- `same_prefill` reflects the uniform prefill path.
- `same_decode` reflects the uniform decode path.
- `mixed_prefill` and `mixed_decode` are the most useful local fallback signals for serving-like traces.
