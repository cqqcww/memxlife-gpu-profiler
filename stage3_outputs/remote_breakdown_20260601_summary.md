# Remote Breakdown Summary

Source log: [remote_breakdown_20260601.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_breakdown_20260601.txt)

## Breakdown

```json
{
  "same_prefill_tokens_per_second": 84328.92071175003,
  "same_decode_tokens_per_second": 5317.799929663348,
  "mixed_prefill_tokens_per_second": 48622.72615789409,
  "mixed_decode_tokens_per_second": 3764.238118498255
}
```

## Read

- `same_prefill` is strong and lines up with the higher public-like prefill numbers.
- `same_decode` and `mixed_decode` are much lower than the strongest public website entries, so decode is still the main gap.
- Relative to the public outputs we sampled from `outputs3`, this looks competitive on prefill but not on decode-heavy cases.
