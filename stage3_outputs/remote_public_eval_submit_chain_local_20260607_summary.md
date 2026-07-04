# Remote Public Eval Summary

Source log: [remote_public_eval_submit_chain_local_20260607.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_submit_chain_local_20260607.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.030025882995687425,
  "tokens": 128,
  "tokens_per_second": 4262.988702726393
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.12527132099785376,
  "tokens": 656,
  "tokens_per_second": 5236.63353091997
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
