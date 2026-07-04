# Matrix Summary: deepseek_safety_probe

Stage DeepSeek through a safe preflight gate before the risky AdamW smoke step.

| Variant | Execution | Status | Avg tokens/sec | Last val loss | Failure | Complexity | Run dir |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `preflight_s16_b1` | preflight_only | `0` | n/a | n/a |  | batch=1, grad_accum=1 (direct batch), seq_len=16, gradient_checkpointing=True | `/workspace/runs/deepseek-matrix-preflight-s16-b1-20260619T070401Z` |
| `adamw_s16_b1` | train | `1` | n/a | n/a | cuda_oom | batch=1, grad_accum=1 (direct batch), seq_len=16, gradient_checkpointing=True | `/workspace/runs/deepseek-matrix-adamw-s16-b1-20260619T070428Z` |
