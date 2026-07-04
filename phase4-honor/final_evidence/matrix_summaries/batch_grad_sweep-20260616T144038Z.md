# Matrix Summary: batch_grad_sweep

Probe whether larger effective batches amortize per-step overhead.

| Variant | Status | Avg tokens/sec | Last val loss | Complexity | Run dir |
| --- | ---: | ---: | ---: | --- | --- |
| `bs2_ga1` | `0` | 15114.61 | 5.7362 | batch=2, grad_accum=1 (direct batch) | `/workspace/runs/sweep-bs2-ga1-20260616T143901Z` |
| `bs4_ga1` | `0` | 20284.70 | 5.3667 | batch=4, grad_accum=1 (direct batch) | `/workspace/runs/sweep-bs4-ga1-20260616T143923Z` |
| `bs8_ga1` | `0` | 23949.54 | 5.1880 | batch=8, grad_accum=1 (direct batch) | `/workspace/runs/sweep-bs8-ga1-20260616T143947Z` |
| `bs4_ga2` | `0` | 22139.84 | 5.1907 | batch=4, grad_accum=2 (extra forward/backward passes) | `/workspace/runs/sweep-bs4-ga2-20260616T144012Z` |

## Best Config Reasoning

- Selected variant: `bs8_ga1`
- Average tokens/sec: `23949.54`
- Last validation loss: `5.1880`
- Reason: bs8_ga1 is selected because it has the highest average tokens/sec among validation-sane candidates; validation loss 5.1880 is within 0.0% of the best observed 5.1880; complexity=batch=8, grad_accum=1 (direct batch).
