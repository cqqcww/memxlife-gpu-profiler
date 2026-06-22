# Phase 4 Self-Command Workflow

This folder keeps the Phase 4 remote-development loop short and repeatable.

Principle:

- Local `phase4-honor/` is the source of truth for code and report text.
- Remote `/workspace` is the GPU execution and validation area.
- Heavy artifacts stay remote: token caches, checkpoints, large run folders.
- Local fetches are intentionally small: summaries, JSONL logs, markdown reports, and selected text logs.

## First-Time Setup

```bash
cd /Users/amanda/Desktop/School/mlsys/phase4-honor/selfcmd-workflow
cp config.env.example config.env
./selfcmd doctor
```

Usually the defaults are enough:

- course server: `10.176.37.31`
- remote directory: `/workspace`
- final report path on server: `/workspace/report4.md`

## Normal Development Loop

Start a GPU dev container and remember its ssh port:

```bash
./selfcmd start
```

Deploy local source over the remote workspace while preserving remote heavy artifacts:

```bash
./selfcmd deploy-clean
```

Install or repair remote Python dependencies when a fresh course container is missing HuggingFace packages:

```bash
./selfcmd install-deps
```

Deploy, repair dependencies, and run the remote test suite in one bootstrap pass:

```bash
./selfcmd bootstrap
```

Run only the remote compile/test check:

```bash
./selfcmd test
```

Run the smoke test remotely:

```bash
./selfcmd smoke
```

Run the profile-composed smoke path remotely:

```bash
./selfcmd profile-smoke
```

Run a specific training config remotely:

```bash
./selfcmd train configs/debug.yaml
```

Check or run an experiment matrix:

```bash
./selfcmd matrix-dry configs/matrices/cache_on_off.yaml
./selfcmd matrix-run configs/matrices/batch_grad_sweep.yaml
```

Run a bounded auto-probe that expands token budget and writes a recommendation:

```bash
./selfcmd auto-probe configs/auto_probes/deepseek_adafactor_token_budget.yaml
```

Run the staged DeepSeek safety probe:

```bash
./selfcmd deepseek-probe
```

Run the longer low-memory DeepSeek Adafactor probe:

```bash
./selfcmd deepseek-adafactor-probe
```

Run the DeepSeek scale probe for `seq_len=64` and checkpointing on/off:

```bash
./selfcmd deepseek-scale-probe
```

Run the DeepSeek token-budget probe comparing `seq_len=128,batch=1` against
`seq_len=64,batch=2` on a medium local fixture:

```bash
./selfcmd deepseek-budget-probe
```

Run the DeepSeek 256-token shape probe:

```bash
./selfcmd deepseek-256-probe
```

Run the DeepSeek bounded token-budget auto-probe:

```bash
./selfcmd deepseek-auto-probe
```

Fetch only lightweight evidence back to this folder:

```bash
./selfcmd fetch
```

When done with the dev container:

```bash
./selfcmd finish
```

## One-Command Iteration

For quick code-test-evidence loops:

```bash
./selfcmd cycle configs/debug.yaml
```

This runs:

```text
deploy-clean -> train config -> fetch lightweight artifacts
```

For model/data switching evidence, prefer this explicit loop:

```text
deploy-clean -> profile-smoke -> matrix-dry -> fetch lightweight artifacts
```

For a fresh GPU container, prefer:

```text
bootstrap -> deepseek-probe or deepseek-adafactor-probe or deepseek-scale-probe or deepseek-budget-probe or deepseek-256-probe or deepseek-auto-probe -> fetch
```

## Remote Commands

For short checks:

```bash
./selfcmd exec 'cd /workspace && python3 -V && ls -la'
```

For commands that may print a lot, use logged mode. It stores the full output on
the server and prints only the tail locally:

```bash
./selfcmd run 'python3 train.py --config configs/baseline_tinystories.yaml' baseline
```

Remote logs land under:

```text
/workspace/selfcmd-workflow/logs/
```

## Final Submission

The Phase 4 guide says the final submission is a report. This workflow assumes
the report is visible as:

```text
/workspace/report4.md
```

If the course service exposes a Phase 4 submit endpoint, set it in `config.env`:

```bash
COURSE_SUBMIT_PATH=/submit4
```

Then:

```bash
./selfcmd deploy-clean
./selfcmd exec 'test -f /workspace/report4.md && ls -lh /workspace/report4.md'
./selfcmd submit
./selfcmd status
```

If the course service uses a different Phase 4 endpoint, only `COURSE_SUBMIT_PATH`
needs to change.

## What `deploy-clean` Deletes Remotely

`deploy-clean` removes known Phase 4 source paths before syncing:

```text
training_framework/
agent/
configs/
tests/
scripts/
selfcmd-workflow/
train.py
report4.md
README.md
requirements.txt
pyproject.toml
setup.cfg
setup.py
```

It intentionally preserves:

```text
runs/
data_cache/
checkpoints/
large model or tokenizer caches
```

This gives us overwrite semantics for code without repeatedly pulling or
destroying expensive remote experiment data.

It also removes known stale Phase 1-3 source/report files and macOS AppleDouble
metadata files from the remote workspace. The local `phase4-honor/` folder stays
the source of truth; remote cleanup is only workspace hygiene for repeatable
Phase 4 runs and final report submission.

## Local Space Policy

The fetch command skips files larger than `PHASE4_FETCH_MAX_BYTES`.

Default:

```text
25 MB
```

If we need a larger log later, temporarily raise the limit in `config.env`, fetch
once, then lower it again.
