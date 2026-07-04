# Current Phase 4 Recommendation

## Recommended Configuration

- Model: `deepseek-ai/deepseek-coder-1.3b-base`
- Data profile: `wikitext2`
- Optimizer: `adafactor`
- Tokens/step: `2048`
- Shape: `seq_len=2048, batch=1, grad_accum=1`
- Mixed precision: `auto`
- Gradient checkpointing: `False`

## Evidence

- Stability status: `pass`
- Completed steps: `100/100`
- Average tokens/sec: `3650.34`
- Train loss: `9.5252 -> 6.3307`
- Last validation loss: `6.6539`
- Peak CUDA allocated/reserved: `14305 / 18828 MiB`

## Memory Calibration

- Predicted allocated peak: `15051 MiB`
- Actual allocated peak: `14305 MiB`
- Allocated prediction error: `5.2%`
- Predicted reserved peak: `18814 MiB`
- Actual reserved peak: `18828 MiB`
- Reserved prediction error: `-0.1%`

## Decision

- Promote for longer probe: `True`
- Reason: 50-step real-data stability passed without OOM or NaN
- Risk: 50 steps is feasibility evidence, not convergence evidence.
- Risk: WikiText-2 subset is still small; longer runs are needed for stability confidence.
- Risk: Reserved memory is high enough that larger token budgets should be gated by probes.
- Next: Run 100-200 step real-data stability before calling this a durable training recipe.
- Next: Keep AdamW classified as unsafe on this GPU unless offload/sharding is introduced.
- Next: Use the calibrated memory estimate as the first preflight gate for new model profiles.
