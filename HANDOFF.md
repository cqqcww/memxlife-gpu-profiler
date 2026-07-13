# HANDOFF

Last updated: 2026-07-13  
Workspace root: `/Users/amanda/Desktop/School/mlsys`

This document is for a fresh Codex session with no prior context. It summarizes
what we were doing, what is already finished, where the important files are,
what is still pending or uncertain, and what mistakes to avoid.

## 1. Big Picture

We have been working on Amanda's MLSys course project across multiple phases.
The main repository is:

`/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler`

The active Git branch is:

`pj3_dev`

Remote:

`https://github.com/cqqcww/memxlife-gpu-profiler.git`

The broad work completed in this thread includes:

- Phase 1/2/3 performance work and reports/talk materials.
- Phase 3 local evaluation harness, engine sources, and output logs.
- Phase 4 Honor project: an agentic mini training framework / feasibility
  harness for early training experiments.
- Final Phase 4 reports, evidence appendix, evidence bundle, and submission
  archives.
- A separate exploratory CANN/Ascend deployment scaffold branch/worktree.

The current immediate state is mostly "submission packaging complete"; no active
coding blocker is open unless the user asks for more cleanup, another submission,
or CANN follow-up work.

## 2. Current GitHub Repo State

Repo:

`/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler`

Current branch:

`pj3_dev`

As of the last check, Git status was clean and aligned with remote:

```bash
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler status --short --branch
# ## pj3_dev...origin/pj3_dev
```

Latest commits:

```text
ed77cc8 (HEAD -> pj3_dev, origin/pj3_dev) Add phase presentation deck
8fe8e6f Add phase3 local artifacts and reports
e11f70f Add phase4 final evidence appendix
77f4f99 Expand final phase4 report and add Chinese version
f40086d Polish final phase4 report evidence
d358590 Add final phase4 honor report
```

Important note: we had HTTPS push failures with HTTP 408 when trying to push a
larger commit containing an 11MB PPTX. The fix was to split the commit:

- `8fe8e6f`: pushed all Phase 3 local artifacts/reports/logs/code except PPTX.
- `ed77cc8`: pushed the PPTX separately.

Both commits eventually pushed successfully. Do not assume the earlier failed
commit `c02bce6` is on GitHub; it was replaced by the split commits above. A
local backup branch may exist:

`backup/all-dirty-c02bce6`

It was created only as a safety branch before splitting the failed push.

## 3. Submission Archives

There are two important local archives:

### Full archive including the previously dirty repo files

Use this if the user wants the broad repository snapshot including Phase 3
artifacts, PPTX, workspace engine, reports, and Phase 4 materials:

`/Users/amanda/Desktop/School/mlsys/submission-20260704.tar.gz`

Size at creation: about `12MB`.

It was created from committed Git contents using:

```bash
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler archive \
  --format=tar.gz \
  --prefix=memxlife-gpu-profiler/ \
  -o /Users/amanda/Desktop/School/mlsys/submission-20260704.tar.gz \
  HEAD
```

It was checked to include:

- `20260616submit-temp/王丰淼-OptimazationJourney-20260616Presentation.pptx`
- `workspace/engine.py`
- `stage3_outputs/...`
- `total-report.md`
- `phase4-honor/report4-final.md`
- `phase4-honor/final_evidence/README.md`

It was checked to exclude ignored/private/cache files such as:

- `__pycache__`
- `.DS_Store`
- `.pj3_work`
- `.phase2_work`
- `memxlife-project/api_config.py`

### Phase4-only archive

Use this only if the user specifically wants the Phase 4 Honor folder without
the Phase 3 dirty artifacts:

`/Users/amanda/Desktop/School/mlsys/phase4-honor-final-submission-20260704.tar.gz`

Size at creation: about `284KB`.

This archive contains the repo's `phase4-honor/` directory with final reports,
code, configs, tests, fixtures, selfcmd workflow scripts, and `final_evidence/`.
It intentionally excludes cache/runs/remote artifacts.

Do not confuse these two archives. If the user says "the one with the dirty
files", they mean:

`submission-20260704.tar.gz`

## 4. Phase 4 Honor Project

There are two important Phase 4 directories:

### Main standalone Phase4 working directory

`/Users/amanda/Desktop/School/mlsys/phase4-honor`

This contains the richer local working copy, including heavier experiment runs
and fetched remote artifacts. It is useful for investigation and local evidence
lookup.

### Repo Phase4 directory

`/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase4-honor`

This is the GitHub-tracked copy. It has the final report, framework code,
configs, tests, and curated compact evidence.

Important final report files:

- English final:
  `/Users/amanda/Desktop/School/mlsys/phase4-honor/report4-final.md`
- Chinese final:
  `/Users/amanda/Desktop/School/mlsys/phase4-honor/report4-final-zh.md`
- Repo copies:
  `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase4-honor/report4-final.md`
  `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase4-honor/report4-final-zh.md`

The final reports include:

- Full technical explanation of the Phase 4 scaffold.
- Development process and debugging history.
- Agentic coding / harness engineering discussion.
- Concrete evidence index.
- Section 19 appendix with selected raw outputs embedded directly in the report.

Curated evidence folder:

`/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase4-honor/final_evidence`

It contains 22 files, about 244KB, including:

- remote pytest log: `48 passed in 3.00s`
- TinyStories batch sweep
- Qwen throughput and compatibility run evidence
- DeepSeek AdamW OOM classification
- DeepSeek Adafactor WikiText auto-probe
- DeepSeek 100-step stability summary
- final recommendation and memory calibration
- a small artifact tar copy

Important Phase4 result summary:

- The scaffold is positioned as an "agentic training feasibility and optimization
  harness", not a replacement for DeepSpeed/FSDP/Megatron.
- It answers early training questions like:
  - Can this model run on this GPU?
  - Which optimizer is safe?
  - What token/batch shape is worthwhile?
  - What failed and why?
  - What should the next probe be?
- DeepSeek AdamW caused CUDA OOM in the safety probe.
- DeepSeek + Adafactor + WikiText reached a 2048-token/step path and passed a
  100-step stability run.
- Final recommendation uses:
  - model: `deepseek-ai/deepseek-coder-1.3b-base`
  - data profile: `wikitext2`
  - optimizer: `adafactor`
  - tokens/step: `2048`
  - shape: `seq_len=2048, batch=1, grad_accum=1`
  - gradient checkpointing: `False`
  - mixed precision: `auto`

## 5. Phase 3 / Report / Talk Artifacts

The following were committed to GitHub in `8fe8e6f` and `ed77cc8`:

- `workspace/engine.py`
- `workspace/output3.log`
- `workspace/report3.md`
- `target/model_config.json`
- evaluator scripts under `evaluator/`
- helper scripts under `scripts/`
- `phase3_engine_sources/current_best_engine.py`
- many logs and summaries under `stage3_outputs/`
- talk/report materials:
  - `phase123_talk_oral_script.md`
  - `phase123_talk_outline.md`
  - `phase123_talk_recommended_slide_content.md`
  - `phase123_talk_script.md`
  - `total-report.md`
  - `20260616submit-temp/...Presentation.pptx`
  - `20260616submit-temp/...Presentation_oral_script.md`

The user explicitly requested that these previously dirty files be included in
the repo and in the full submission archive. They are now included.

## 6. CANN / Ascend Worktree

There is a separate exploratory worktree/branch for CANN/Ascend deployment:

`/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler-cann-deploy`

Branch:

`cann-deploy-harness`

Known commit:

`c8feada Add CANN LLM deploy harness scaffold`

This contains a `cann-llm-deploy/` scaffold for Qwen30B / DeepSeek-V4-Flash
deployment planning on unknown Ascend hardware, including 910A/910B/A2/A3 risk
profiles. It is separate from the main Phase 4 submission and should not be
mixed into `pj3_dev` unless the user explicitly asks.

## 7. What Is Completed

Completed:

- Phase4 final English and Chinese reports.
- Section 19 appendix in both final reports with embedded key outputs.
- Compact `final_evidence/` evidence bundle.
- Phase4-only archive.
- Full repository submission archive including previously dirty files.
- GitHub push of Phase4 final evidence appendix.
- GitHub push of Phase3 local artifacts/reports.
- GitHub push of the presentation PPTX.
- Main repo is clean and synced with `origin/pj3_dev`.

## 8. Current Blockers / Open Questions

No active technical blocker is known.

Possible remaining user-facing tasks:

- Actually upload/submit `submission-20260704.tar.gz` if the course portal
  requires a file upload.
- If only a report is required, submit `phase4-honor/report4-final.md` or the
  full archive depending on the instructions.
- If the user wants cleanup, decide whether to keep or remove local ignored
  caches and temporary files.
- If the user wants to continue the CANN/Ascend branch, resume in the separate
  `memxlife-gpu-profiler-cann-deploy` worktree.

## 9. Local Files Not In Git

After the final commit/push, normal `git status --short --branch` was clean.

Ignored files still exist locally but are intentionally not in Git or archive:

- `.DS_Store`
- `.phase2_work/`
- `.pj3_work/`
- `memxlife-project/__pycache__/`
- `memxlife-project/api_config.py`
- `phase2_agent/__pycache__/`
- `phase3_engine_sources/__pycache__/`
- `phase4-honor/__pycache__/`
- `phase4-honor/agent/__pycache__/`
- `phase4-honor/tests/__pycache__/`
- `phase4-honor/training_framework/__pycache__/`
- `scripts/__pycache__/`
- `ta_core_bundle_20260514.tar.gz`
- `workspace/__pycache__/`
- `workspace/results.log`

Do not force-add these unless the user explicitly asks and understands the risk.
Especially avoid committing `memxlife-project/api_config.py`; it sounds like a
private config file and may contain credentials.

## 10. Pitfalls To Avoid

### Do not confuse the two archives

- Full dirty-file-inclusive archive:
  `/Users/amanda/Desktop/School/mlsys/submission-20260704.tar.gz`
- Phase4-only archive:
  `/Users/amanda/Desktop/School/mlsys/phase4-honor-final-submission-20260704.tar.gz`

If the user asks for "包含那些脏文件", use the full archive.

### Do not recommit ignored caches/private files

Ignored files were intentionally excluded. Avoid:

```bash
git add -f memxlife-project/api_config.py
git add -f '**/__pycache__'
git add -f .pj3_work .phase2_work
```

### Be careful with Git push failures

We hit repeated HTTPS `HTTP 408` failures when pushing a single commit that
included the 11MB PPTX. Splitting the commit solved it:

- first push text/code/log artifacts;
- then push the PPTX separately.

If this happens again, do not assume "Everything up-to-date" means success.
Always verify with:

```bash
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler status --short --branch
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler rev-parse HEAD
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler rev-parse origin/pj3_dev
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler ls-remote origin refs/heads/pj3_dev
```

### The standalone Phase4 directory is richer than the repo copy

`/Users/amanda/Desktop/School/mlsys/phase4-honor` contains heavier experiment
runs and remote artifacts. The repo copy contains curated final materials.

If editing final reports, keep these in sync:

```bash
cp /Users/amanda/Desktop/School/mlsys/phase4-honor/report4-final.md \
   /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase4-honor/report4-final.md
cp /Users/amanda/Desktop/School/mlsys/phase4-honor/report4-final-zh.md \
   /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase4-honor/report4-final-zh.md
```

### Do not mix CANN branch into main submission accidentally

The CANN/Ascend scaffold is in a separate worktree:

`/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler-cann-deploy`

Keep it separate from `memxlife-gpu-profiler` unless the user explicitly asks to
merge or copy it.

## 11. Recommended Next Steps For A New Session

If the user asks "what is the current state?", run:

```bash
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler status --short --branch
git -C /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler log -4 --oneline --decorate
ls -lh /Users/amanda/Desktop/School/mlsys/submission-20260704.tar.gz
```

If the user wants to submit:

1. Prefer `submission-20260704.tar.gz` for a complete repo snapshot.
2. Prefer `phase4-honor/report4-final.md` if the platform only wants one report
   file.
3. Mention that `report4-final-zh.md` is for Chinese reference/review.

If the user wants to continue Phase4 development:

1. Work in `/Users/amanda/Desktop/School/mlsys/phase4-honor` for richer local
   context.
2. Sync important final changes back into
   `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler/phase4-honor`.
3. Keep evidence compact through `final_evidence/`; do not dump all remote runs
   into Git unless explicitly requested.

If the user wants CANN/Ascend deployment work:

1. Switch to `/Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler-cann-deploy`.
2. Read `cann-llm-deploy/docs/` first.
3. Keep this separate from the main `pj3_dev` branch.

## 12. Useful Commands

Check main repo:

```bash
cd /Users/amanda/Desktop/School/mlsys/memxlife-gpu-profiler
git status --short --branch
git log -6 --oneline --decorate
```

Recreate the full archive from the latest committed repo state:

```bash
git archive --format=tar.gz \
  --prefix=memxlife-gpu-profiler/ \
  -o /Users/amanda/Desktop/School/mlsys/submission-20260704.tar.gz \
  HEAD
```

Verify archive contents:

```bash
tar -tzf /Users/amanda/Desktop/School/mlsys/submission-20260704.tar.gz | \
  grep -E 'workspace/engine.py|stage3_outputs|phase4-honor/report4-final.md|final_evidence|total-report.md'
```

Check ignored files:

```bash
git status --ignored --short
```

## 13. Tone / User Preference Notes

The user prefers Chinese explanations and likes concrete, implementation-level
status updates. They often ask for:

- "现在是什么状态"
- "能不能提交"
- "和 repo/local 文件有什么差别"
- "有没有证据/log"
- "下一步怎么做"

Be direct, specific, and path-heavy. Avoid vague assurances. If there are two
similar artifacts, explicitly say which one to use.
