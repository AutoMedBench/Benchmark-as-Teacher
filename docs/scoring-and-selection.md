# Scoring and Target Selection

Evaluation serves two roles: measure the complete agent and choose the focus of
the next safe RL round. It does not create demonstrations.

## Stage scores

| Stage | Evidence expected | Weight |
| --- | --- | ---: |
| S1 Plan | Correct task framing, approach, resources, and output requirements | 0.25 |
| S2 Setup | Dependencies, resource preparation, and successful model setup | 0.15 |
| S3 Validate | A real small-case execution, sanity checks, and recovery | 0.35 |
| S4 Infer | Complete inference with adequate coverage and persisted outputs | 0.15 |
| S5 Submit | Schema, completeness, validation, and final handoff | 0.10 |

For applicable stages:

```text
agentic = 0.25*S1 + 0.15*S2 + 0.35*S3 + 0.15*S4 + 0.10*S5
overall = 0.50*task + 0.50*agentic
```

If a stage is not applicable, remove its weight and renormalize over the
remaining applicable stages. Missing judge or infrastructure evidence is null,
not zero. A null cell cannot enter strict aggregation or target selection.

## Evidence rules

- S1-S3 require an explicit stage verdict based on retained action evidence.
- S3 requires a real small-case execution; a written validation plan is not
  sufficient.
- S4 requires recognized inference output, coverage evidence, and a valid
  non-degeneracy signal.
- S5 requires a task-native artifact, completeness checks, and a valid handoff.
- Invalid infrastructure, parser, or harness outcomes are repaired before
  aggregation rather than scored as model failure.
- A scored model cutoff may keep observed partial stage credit, but task and
  output-dependent stages follow the evaluator's eligibility rules.

## Paired evaluation gate

Baseline and candidate must contain the same valid evaluation cells. The
reference promotion checks are:

| Check | Requirement |
| --- | --- |
| Overall non-inferiority | candidate mean is no more than `0.01` below baseline |
| Task non-inferiority | candidate task mean is no more than `0.03` below baseline |
| Per-track regression | no track drops by more than `0.05` |
| Late-stage regression | none of S3, S4, or S5 drops by more than `0.03` |
| Catastrophic forgetting | overall drop is less than `0.20` |
| Training health | finite metrics, bounded aborts, and no sustained collapse |

All checks must pass to promote the candidate. A held candidate is not used as
the next round's starting checkpoint.

## S-target selection

Target routing is deterministic and separate from checkpoint promotion.

Normal case:

```text
target = stage with the lowest candidate score
```

Held-gate corrective case:

```text
regression[stage] = baseline[stage] - candidate[stage]

if any regression is positive:
    target = stage with the largest positive regression
else:
    target = stage with the lowest candidate score
```

Ties use the earliest stage in the fixed order `S1, S2, S3, S4, S5`. This
prevents non-deterministic routing when scores are equal.

Examples:

| Candidate S1-S5 | Gate | Selected target | Reason |
| --- | --- | --- | --- |
| `0.70, 0.55, 0.20, 0.45, 0.60` | pass | S3 | lowest candidate stage |
| `0.40, 0.40, 0.65, 0.70, 0.75` | pass | S1 | S1/S2 tie; earlier stage wins |
| candidate S4 fell `0.12`, S3 fell `0.04` | hold | S4 | largest positive regression |

## What the prescription may contain

The next-round prescription may contain only:

- selected target stage;
- fixed pool counts;
- aggregate outcome/track weights;
- optimizer settings;
- checkpoint promotion decision and content digest;
- aggregate rationale using finite stage/score changes.

It must not contain held-out questions, answers, task identifiers, paths,
reports, traces, tool output, or correction text. The prescription selects
among already prepared safe rows and never generates a new training target.
