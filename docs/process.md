# Training and Loop Process

BaT uses one bootstrap phase followed by repeated RL rounds. Cold-start SFT is
not repeated after benchmark feedback; all later learning is online GRPO over
held-out-safe reset states.

## Phase 1: cold-start SFT

The one-time SFT phase establishes response structure, structured tool intent,
and minimum S1-S5 behavior.

Reference 9B recipe:

| Setting | Value |
| --- | ---: |
| Update | Full parameter |
| Precision | BF16 |
| Maximum sequence length | 8,192 tokens |
| Per-device batch | 1 |
| Gradient accumulation | 4 |
| Global batch | 32 |
| Learning rate | `2e-5` |
| Training steps | 200 |
| Validation fraction | 5% |
| Maximum validation examples | 512 |
| Checkpoint interval | 10 steps |
| Retained restart checkpoints | 3 |

The source example is model-neutral. Tool actions are rendered into the target
model's interface only during tokenization, and only the final teacher action
is supervised.

## Phase 2: initial evaluation

The SFT checkpoint is evaluated over the full held-out matrix. The evaluator
must provide valid paired per-cell task/agentic scores and aggregate S1-S5
scores. Incomplete or infrastructure-invalid cells stop the loop until they are
repaired; they are not silently assigned zero.

The weakest valid stage becomes the first S-target. Only aggregate stage values
cross from evaluation into the data router.

## Phase 3: build a three-pool round

The safe 126-state source grid is projected into a deterministic 3,024-row
round view. Each eight-row optimizer batch has fixed composition:

```text
S-target: 4 rows from the selected stage
S-mix:    2 rows spanning the other stages
E2E:      2 rows covering the complete workflow
```

S-target supplies concentrated improvement pressure. S-mix limits cross-stage
forgetting. E2E preserves long-horizon completion and final task quality.

Before admission, verify row schema, source digests, exact batch composition,
stage/track/outcome balance, absence of assistant targets, absence of
model-specific tool syntax, and absence of held-out evidence.

## Phase 4: online grouped rollouts

For each prompt group, sample four continuations from the current policy:

```text
one state_id -> four continuations -> four rewards -> one GRPO group
```

Never normalize advantages across different state identifiers, stages, tracks,
or tasks. The grouping key is the reset state, not the optimizer batch.

Reference multi-turn limits:

| Limit | Value |
| --- | ---: |
| Assistant turns | 24 |
| Tool-response turns | 23 |
| Tool calls | 23 |
| Cumulative response tokens | 16,384 |
| Tool-observation characters per response | 2,048 |

The runtime binds the model-appropriate tool serializer/parser. Every rollout
uses an isolated resettable workspace and the same allowed tool capability.
Model-specific syntax and generated transcripts remain runtime artifacts and
never flow back into the shared RL rows.

## Phase 5: reward and GRPO update

Stage-local rows use benchmark-aligned checklist rewards. S3-S5 require
observable execution evidence; prose cannot substitute for a validation run,
complete inference, or a submission artifact. E2E reward combines workflow
completion with the final task outcome.

Reference RL optimizer settings:

| Setting | Value |
| --- | ---: |
| Algorithm | GRPO |
| Prompt groups per batch | 8 |
| Continuations per group | 4 |
| Learning rate | `2.5e-7` |
| Actor KL-loss coefficient | `0.006` |
| Advantage standard-deviation normalization | Disabled |
| Loss aggregation | Sequence mean, then token mean |
| Sampling temperature | `0.7` |
| Steps per round | 50 |
| Checkpoint interval | 5 steps |
| Retained restart checkpoints | 3 |

The reference eight-GPU topology uses rollout tensor parallelism 1, producing
one rollout replica per GPU. This is a deployment setting rather than a BaT
algorithm requirement.

Although the deterministic round file contains 3,024 prompt groups, a 50-step
reference round consumes 400 groups at batch size 8. Data order is fixed and
recorded so a restart cannot change which groups belong to the round boundary.

## Phase 6: evaluate and gate

After step 50:

1. Export the actor checkpoint into an evaluation-ready model.
2. Evaluate the same paired held-out cells as the baseline.
3. Validate task, stage, and infrastructure evidence.
4. Check aggregate, per-track, and late-stage regressions.
5. Promote the candidate only when every gate passes; otherwise retain the
   previous eligible checkpoint.
6. Select the next S-target from aggregate stage evidence.
7. Materialize a fresh safe round view and repeat.

A reference campaign uses up to ten RL rounds, but the stop condition should be
policy-based: stop when the evaluation objective stabilizes, the improvement
budget is exhausted, or a safety gate cannot be satisfied.

## Evaluator boundary

The held-out evaluator and its task data are not part of this branch. The
included `run_bat_core_eval.py` adapter gives that evaluator a small contract:

```text
input:  checkpoint identity + fixed evaluation protocol
output: paired per-cell scores + aggregate S1-S5 scores + validity status
```

BaT consumes only those validated aggregates. Raw evaluation content remains
on the evaluation side of the firewall.

`run_bat_core_loop.py` composes this adapter with the safe data router, an
explicit external trainer argv template, the paired gate, and checkpoint
inheritance. Neither command uses a shell template. The next-round builder
receives only the selected stage and fixed aggregate routing values; it never
receives raw evaluator cells or transcripts.

The loop accepts each command template as a JSON array of argv strings. The
evaluator template must contain the exact placeholder elements
`{checkpoint}`, `{request}`, and `{output}` and must write the required raw
evaluation JSON to `{output}`. The trainer template must contain
`{checkpoint}`, `{dataset}`, `{output}`, `{health}`, and
`{runtime_tool_format}`. It must leave an evaluation-ready Hugging Face model
under `{output}` and write `{"status":"passed"}` to `{health}` only after the
GRPO round and checkpoint export both succeed. A deployment can call
`launch_core_grpo.sh` followed by `export_checkpoint.py` from that explicit
trainer wrapper. The command files contain no credentials; authentication and
cluster configuration remain environment-owned.
