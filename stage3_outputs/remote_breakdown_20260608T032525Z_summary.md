# Remote Breakdown Summary

Source log: [remote_breakdown_20260608T032525Z.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_20260608T032525Z.txt)

## Breakdown

```json
{
  "same_prefill_tokens_per_second": 88277.23195804728,
  "same_decode_tokens_per_second": 4674.025219417824,
  "mixed_prefill_tokens_per_second": 49465.599469957124,
  "mixed_decode_tokens_per_second": 4254.620488919216
}
```

## Read

- `same_prefill` reflects the uniform prefill path.
- `same_decode` reflects the uniform decode path.
- `mixed_prefill` and `mixed_decode` are the most useful local fallback signals for serving-like traces.
