# Auto Probe Recommendation: deepseek_adafactor_wikitext_realdata

Real-data DeepSeek Adafactor probe on WikiText-2; checks whether the fixture-selected token budget transfers to a HuggingFace dataset profile.

| Variant | Status | Tokens/step | Avg tokens/sec | Last val loss | Peak CUDA MB | Failure |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `tok512` | `0` | 512 | 1341.83 | 9.8190 | 11173/13666 |  |
| `tok1024` | `0` | 1024 | 2137.52 | 10.7984 | 11899/15140 |  |
| `tok2048` | `0` | 2048 | 3100.38 | 10.0320 | 14305/18828 |  |

## Recommendation

- Action: `stability_run`
- Selected token budget: `2048`
- Next token budget: `n/a`
- Reason: `tok2048` is the best bounded safe point. Prefer a longer 50-100 step run before expanding further.
