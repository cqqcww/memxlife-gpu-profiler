# Phase 4 Final Evidence

This directory contains a compact copy of the most important logs and generated
evidence used by `report4-final.md` and `report4-final-zh.md`.

The original fetched remote snapshot is:

`selfcmd-workflow/artifacts/remote-20260621T095347Z/`

The submit-friendly archived snapshot copy is:

`final_evidence/phase4-artifacts-20260621T095347Z.tar.gz`

It was copied from:

`selfcmd-workflow/artifacts/phase4-artifacts-20260621T095347Z.tar.gz`

## Evidence Map

| Evidence file | Original source | Report claim |
| --- | --- | --- |
| `tests/tests-20260621T095143Z.log` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/selfcmd-workflow/logs/tests-20260621T095143Z.log` | Final remote test result: `48 passed in 3.00s`. |
| `resume/debug-resume-rng-summary.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/debug-resume-rng-20260616T090218Z/summary.md` | Resume path generated remote evidence. |
| `resume/debug-resume-rng-events.jsonl` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/debug-resume-rng-20260616T090218Z/events.jsonl` | Raw JSONL events for the resume/RNG run. |
| `matrix_summaries/batch_grad_sweep-20260616T144038Z.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/matrix_summaries/batch_grad_sweep-20260616T144038Z.md` | `bs8_ga1` was the strongest TinyStories batch candidate. |
| `matrix_summaries/batch_grad_sweep-20260616T144038Z.json` | Same remote matrix summary, JSON form. | Machine-readable batch sweep evidence. |
| `matrix_summaries/qwen_throughput_probe-20260619T033125Z.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/matrix_summaries/qwen_throughput_probe-20260619T033125Z.md` | Qwen throughput comparison and checkpointing observation. |
| `matrix_summaries/qwen_throughput_probe-20260619T033125Z.json` | Same remote matrix summary, JSON form. | Machine-readable Qwen matrix evidence. |
| `qwen/qwen-long-tinystories-summary.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/qwen-long-tinystories-20260616T145919Z/summary.md` | Qwen compatibility run summary. |
| `qwen/qwen-long-tinystories-preflight.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/qwen-long-tinystories-20260616T145919Z/preflight.md` | Qwen model/data/device preflight evidence. |
| `qwen/qwen-long-tinystories-events.jsonl` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/qwen-long-tinystories-20260616T145919Z/events.jsonl` | Raw Qwen run events. |
| `deepseek/deepseek_safety_probe-20260619T070456Z.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/matrix_summaries/deepseek_safety_probe-20260619T070456Z.md` | DeepSeek AdamW classified as `cuda_oom`. |
| `deepseek/deepseek_safety_probe-20260619T070456Z.json` | Same remote matrix summary, JSON form. | Machine-readable DeepSeek safety matrix. |
| `deepseek/deepseek-adamw-oom-agent-summary.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/deepseek-matrix-adamw-s16-b1-20260619T070428Z/agent_summary.md` | AdamW run failed before train metrics, matching OOM classification. |
| `deepseek/deepseek_adafactor_wikitext_realdata-20260621T072201Z.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/auto_probe_summaries/deepseek_adafactor_wikitext_realdata-20260621T072201Z.md` | WikiText token-budget auto-probe selected `2048`. |
| `deepseek/deepseek_adafactor_wikitext_realdata-20260621T072201Z.json` | Same auto-probe summary, JSON form. | Machine-readable auto-probe evidence. |
| `deepseek/deepseek_adafactor_wikitext_realdata-stability-20260621T095331Z.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/stability_summaries/deepseek_adafactor_wikitext_realdata-stability-20260621T095331Z.md` | DeepSeek/WikiText 100-step stability passed. |
| `deepseek/deepseek_adafactor_wikitext_realdata-stability-20260621T095331Z.json` | Same stability summary, JSON form. | Machine-readable stability evidence. |
| `deepseek/deepseek-wikitext-2048-100step-events.jsonl` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/deepseek-realdata-wikitext-adafactor-tok2048-stability-100step-20260621T095155Z/events.jsonl` | Raw events for the final 100-step DeepSeek run. |
| `recommendations/phase4-current-recommendation.md` | `selfcmd-workflow/artifacts/remote-20260621T095347Z/runs/recommendations/phase4-current-recommendation.md` | Final recommendation and memory calibration. |
| `recommendations/phase4-current-recommendation.json` | Same recommendation, JSON form. | Machine-readable final recommendation. |
