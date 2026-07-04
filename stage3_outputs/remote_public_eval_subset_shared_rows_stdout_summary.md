# Remote Public Eval Summary

Source log: [remote_public_eval_subset_shared_rows_stdout.txt](/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/stage3_outputs/remote_public_eval_subset_shared_rows_stdout.txt)

## Status

- correctness smoke passed
- stress correctness passed

## Throughput

### Benchmark throughput

```json
{
  "elapsed_seconds": 0.02997552789747715,
  "tokens": 128,
  "tokens_per_second": 4270.149984940647
}
```

### Benchmark mixed

```json
{
  "elapsed_seconds": 0.12107376102358103,
  "tokens": 656,
  "tokens_per_second": 5418.1847037215075
}
```

## Notes

- These results were produced inside a course GPU dev container by running `bash scripts/run_public_tests.sh`.
- This is the local fallback path when the official `outputs3` publishing chain is unstable.
