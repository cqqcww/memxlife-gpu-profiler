# Matrix Summary: qwen_throughput_probe

Probe Qwen throughput bottlenecks by separating gradient checkpointing, batch size, and sequence length.

| Variant | Status | Avg tokens/sec | Last val loss | Complexity | Run dir |
| --- | ---: | ---: | ---: | --- | --- |
| `s64_b1_gc_on` | `0` | 350.41 | nan | batch=1, grad_accum=1 (direct batch), seq_len=64, gradient_checkpointing=True | `/workspace/runs/qwen-probe-s64-b1-gc-on-20260619T032706Z` |
| `s64_b1_gc_off` | `0` | 447.92 | 9.2218 | batch=1, grad_accum=1 (direct batch), seq_len=64, gradient_checkpointing=False | `/workspace/runs/qwen-probe-s64-b1-gc-off-20260619T032851Z` |
| `s64_b2_gc_off` | `0` | 897.67 | 8.9437 | batch=2, grad_accum=1 (direct batch), seq_len=64, gradient_checkpointing=False | `/workspace/runs/qwen-probe-s64-b2-gc-off-20260619T032947Z` |
| `s128_b1_gc_off` | `0` | 897.69 | 8.9839 | batch=1, grad_accum=1 (direct batch), seq_len=128, gradient_checkpointing=False | `/workspace/runs/qwen-probe-s128-b1-gc-off-20260619T033035Z` |

## Best Config Reasoning

- Selected variant: `s128_b1_gc_off`
- Average tokens/sec: `897.69`
- Last validation loss: `8.9839`
- Reason: s128_b1_gc_off is selected because it has the highest average tokens/sec among validation-sane candidates; validation loss 8.9839 is within 0.4% of the best observed 8.9437; complexity=batch=1, grad_accum=1 (direct batch), seq_len=128, gradient_checkpointing=False.
