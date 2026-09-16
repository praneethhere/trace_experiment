# TRACE v2 — ToolBench-X Gate 1

Status: CONDITIONAL PASS

Pinned upstream commit:

    a0948f0eb34f6028ebc7af0a2d83fce94bbfc41a

Inspection date:

    September 2026

This document records implementation evidence only.
It does not claim research novelty.

## 1. Upstream competition check

At the Gate-1 live check:

- ToolBench-X public repository remained small;
- no open competing pull request implementing TRACE's proposed evaluation
  protocol was identified;
- no upstream source change invalidated the inspected commit.

The experiment must re-check upstream HEAD, issues, pull requests and
maintainer comments again before pilot execution and again before any OSS PR.

## 2. Oracle timing

ToolBench-X stores the oracle hazard label in task metadata but initializes:

    oracle_label_visible = False

The policy therefore cannot see the oracle label before a manifested hazard.

After either:

- a semantic tool-result issue; or
- a hard tool execution failure,

the runtime sets:

    oracle_label_visible = True

The next policy decision may then receive the oracle hazard label.

Therefore:

    PROVEN:
    oracle diagnosis exposure is post-hazard rather than pre-hazard.

## 3. Oracle information content

The oracle block communicates:

- hazard category;
- category definition.

The oracle block explicitly states that it does not provide:

- a recovery procedure;
- final-answer information.

Therefore the oracle pathway is suitable in principle as a
diagnosis-information upper-bound condition.

Any TRACE adapter must still normalize it into the frozen common diagnosis
interface.

## 4. Prescriptive hint condition

ToolBench-X `with_hint` is NOT a diagnosis-only treatment.

Its hint machinery may contain:

- retry strategy;
- mandatory tool sequence;
- required inputs;
- verification checks;
- finish blockers;
- canonical-answer rules;
- prescriptive continuation guidance.

Therefore:

    PROVEN:
    `with_hint` confounds diagnosis with recovery prescription.

TRACE v2 must not use `with_hint` as a pure diagnosis arm.

## 5. Native checkpointing

No native mid-trajectory checkpoint/resume API was identified in the pinned
ToolBench-X runtime.

The main execution path retains continuation state as in-process Python
objects and local variables.

The ToolBench-X test-time-scaling script performs a fresh rerun/replay-style
execution rather than restoring an exact first-failure interpreter snapshot.

Therefore:

    PROVEN:
    native exact first-hazard checkpoint restoration is unavailable.

## 6. Agent-side continuation state

At the first-hazard boundary, relevant agent-side state includes at least:

- task identity;
- user prompt;
- tool specifications;
- tool runtime catalog;
- tool results;
- tool execution log;
- policy/action history;
- current round;
- last tool;
- last error;
- deferred hint state;
- oracle visibility state;
- model-visible trajectory;
- frozen hazard configuration and failure seed.

This state is tractable but is not sufficient by itself.

## 7. Hidden tool-module state

ToolBench-X hazard-generation requirements explicitly call for monotonic
per-failpoint call counters persisted in module runtime state when max_times
is positive.

Therefore tool execution may mutate hidden state that is not represented by
the agent's local variables.

Consequently:

    snapshot(agent_state) alone

does NOT prove:

    identical branch environment state.

## 8. Proposed experimental instrument

Candidate approach:

    Verified Prefix Replay

For each eligible task:

1. execute one natural pre-hazard trajectory;
2. record every pre-branch policy action;
3. record every exact tool name and argument payload;
4. record every exact tool result;
5. identify the first manifested hazard;
6. freeze a canonical prefix record;
7. start a fresh isolated runtime for each counterfactual branch;
8. do not regenerate pre-branch LLM decisions;
9. replay the exact recorded tool calls in the exact original order;
10. reconstruct the first-hazard state;
11. compute branch-state evidence hashes;
12. permit diagnosis/recovery divergence only if replay equivalence passes.

## 9. Required replay-equivalence evidence

Before a task can enter the causal pilot, independently replayed branches must
match on all objectively reproducible pre-branch evidence, including:

- canonical task hash;
- upstream ToolBench-X commit;
- clean/exception tool-file hash;
- failure seed;
- ordered tool-name sequence;
- ordered tool-argument hashes;
- ordered tool-result hashes;
- first-hazard type;
- first-hazard location;
- first-hazard observation hash;
- model-visible prefix hash;
- action-history hash;
- branch round;
- last-tool value;
- last-error value;
- any observable injection-event state.

If actual released ToolBench-X modules expose additional mutable state, the
equivalence contract must expand before pilot execution.

## 10. Hard refusal rule

A task is ineligible if replay does not reconstruct an equivalent branch
state.

Do not:

- approximate the prefix;
- regenerate pre-branch LLM decisions independently;
- silently ignore tool-state mismatch;
- reset counters differently across branches;
- condition eligibility on downstream recovery success.

## 11. Synthetic replay gate

Before interacting with released ToolBench-X benchmark data, implement an
offline synthetic contract containing:

- deterministic task identity;
- deterministic failure seed;
- hidden monotonic failpoint counter;
- multiple pre-hazard tool calls;
- first manifested failure;
- recorded prefix;
- two independent fresh-process replays.

The synthetic gate passes only if both fresh replays reconstruct identical
branch-state evidence from the frozen prefix.

If this fails:

    STOP Verified Prefix Replay design.

## 12. Released-data validation

Synthetic success is necessary but not sufficient.

Before real pilot execution:

1. resolve ToolBench-X dataset-use permissions;
2. inspect released exception-tool modules without modifying them;
3. identify actual mutable injection-state implementations;
4. validate replay equivalence across all five hazard families;
5. freeze eligibility and failure-handling rules;
6. re-check upstream live state.

Until then:

    exact ToolBench-X branch replay remains INFERRED, not PROVEN.
