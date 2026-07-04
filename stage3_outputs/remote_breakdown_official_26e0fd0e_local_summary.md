# Remote Breakdown Summary

Source log: [remote_breakdown_official_26e0fd0e_local.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_official_26e0fd0e_local.txt)

## Breakdown

```json
{
  "same_prefill_tokens_per_second": 88886.26468101406,
  "same_decode_tokens_per_second": 4596.577395720418,
  "mixed_prefill_tokens_per_second": 48390.73757425263,
  "mixed_decode_tokens_per_second": 4213.134506510093
}
```

## Read

- `same_prefill` reflects the uniform prefill path.
- `same_decode` reflects the uniform decode path.
- `mixed_prefill` and `mixed_decode` are the most useful local fallback signals for serving-like traces.
