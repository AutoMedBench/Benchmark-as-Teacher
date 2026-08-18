# Data Contract

BaT separates semantic source data, model-scoped derived views, RL round views,
and held-out evaluation evidence. Only the first three may participate in
training, and evaluation evidence influences them only through aggregate
selection values.

## Logical layout

```text
safe_source/
  sft/                 semantic cold-start examples
  rl/                  prompt-only abstract workflow states

derived/
  sft/<model-family>/  tokenized model-scoped views
  rl/<round>/          deterministic S-target, S-mix, and E2E views

evaluation_boundary/  held-out evaluator inputs and outputs; never training data
```

These names describe roles rather than required filesystem locations. This
branch includes only fabricated examples, not the underlying datasets.

## Cold-start SFT source

Cold-start SFT teaches initial response structure, tool intent, and basic stage
behavior. A semantic source row contains:

- prior system/user/assistant/tool context;
- exactly one target assistant decision;
- a structured action such as tool name plus typed arguments;
- stage and provenance metadata;
- an explicit loss-mask policy.

Only the target assistant decision receives loss. System and user messages,
prior assistant messages, tool observations, and padding are masked.

Tool syntax is a model-interface concern, not a source-data concern. Shared
rows therefore store an action object. A model adapter renders the action with
that model's tokenizer/chat template while building a derived view. Unsupported
models fail preparation instead of silently inheriting another model's syntax.

Derived SFT views must record:

- source row digest;
- tokenizer and chat-template identity;
- selected runtime protocol identifier;
- target-action digest before and after rendering;
- supervised-token digest;
- rejected-row counts and reasons.

Target actions are never character-clipped. If a complete rendered target does
not fit, the row is rejected. Train/evaluation SFT splits are grouped by source
trajectory so adjacent decisions cannot cross the split.

See [`examples/sft-semantic-row.json`](../examples/sft-semantic-row.json).

## RL source rows

An RL row is a reset point, not a demonstration. It contains:

```text
prompt               system/user messages only
state_id             grouping key for sampled continuations
target_stage         S1, S2, S3, S4, S5, or E2E
source_pool          s_target, s_mix, or e2e
track                abstract benchmark track
outcome_bucket       success, recovery, or failure
reward_contract      stage or E2E rubric definition
provenance           safe-source and transformation digests
```

It must not contain an assistant completion, answer, teacher trajectory,
rollout transcript, model-specific tool serialization, or held-out identifier.
The current model generates every scored continuation online.

See [`examples/grpo-prompt-row.json`](../examples/grpo-prompt-row.json).

## Clean source grid

The compact reference bank is balanced over three dimensions:

| Dimension | Values | Count |
| --- | --- | ---: |
| Focus | E2E, S1, S2, S3, S4, S5 | 6 |
| Track | classification, synthesis, detection, segmentation, VQA, report, enhancement | 7 |
| Outcome | success, recovery, failure | 3 |

One abstract source state per cell gives `6 x 7 x 3 = 126` states. The states
describe observable workflow conditions without copying evaluation questions,
answers, dataset names, file paths, reports, or fixes.

## Round materialization

The selected stage changes which safe states are viewed and which reward rubric
is attached; it does not rewrite source semantics. The reference materializer
creates 378 complete eight-row batches, or 3,024 prompt groups:

```text
4 S-target rows
2 S-mix rows distributed over the other four stages
2 E2E rows
```

Sampling remains balanced over tracks and the success/recovery/failure outcome
strata. Each materialized row receives a fresh round-scoped `state_id` derived
from source identity, target stage, pool, and deterministic view index.

The round view carries no generated assistant response. Four online rollouts
are associated with the same `state_id` only after policy sampling begins.

## Held-out firewall

The following are forbidden in SFT and RL source or derived views:

- held-out task identifiers, questions, answers, labels, or expected outputs;
- evaluator reports, conversations, traces, tool logs, or workspace paths;
- correction text copied from a scored evaluation;
- model-generated evaluation transcripts reused as RL targets;
- credentials, authenticated endpoints, or machine-specific absolute paths.

Evaluation may contribute only finite aggregates such as S1-S5 scores, task
means, per-track score changes, completion counts, and failure categories. Those
aggregates select a stage or safe-data slice; they are not inserted into model
prompts or targets.

Every train and validation artifact should be immutable after admission and
bound to its source and transformation digests. A failed leakage, schema,
protocol, or provenance check stops the launch.
