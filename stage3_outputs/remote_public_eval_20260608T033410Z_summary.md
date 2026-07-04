# Remote Public Eval Summary

Source log: [remote_public_eval_20260608T033410Z.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_20260608T033410Z.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.033853480010293424,
  "tokens": 128,
  "tokens_per_second": 3780.9997660825584
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.14499545699800365,
  "tokens": 656,
  "tokens_per_second": 4524.279681459482
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
