# Remote Public Eval Summary

Source log: [remote_public_eval_20260608T032525Z.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_20260608T032525Z.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.029919619002612308,
  "tokens": 128,
  "tokens_per_second": 4278.129343452676
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.12495667801704258,
  "tokens": 656,
  "tokens_per_second": 5249.819460713653
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
