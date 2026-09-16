# TRACE v2 Falsification Pilot Protocol

Status: PRE-IMPLEMENTATION
Purpose: falsify the publication thesis before full experiment investment

This protocol does not claim novelty and does not authorize publication claims.

## 1. Candidate research question

Does conventional failure-attribution accuracy rank attribution systems by
their actual usefulness to downstream recovery?

More strongly:

Is diagnosis utility dependent on the recovery consumer that receives it?

The target phenomenon is an interaction between:

    attribution source
        x
    recovery consumer

rather than merely a main effect of diagnosis correctness.

## 2. Core empirical object

For failed checkpoint s, attribution source A_i, recovery consumer R_j,
and task-success outcome Y:

    U_ij(s) =
        E[Y | s, A_i, R_j]
        -
        E[Y | s, no_diagnosis, R_j]

Every consumer therefore has its own matched no-diagnosis fallback.

Do not compare consumers solely by raw success rate.

## 3. Attributor Selection Regret

Let:

    A_acc

be the attribution system selected using conventional attribution accuracy.

Let:

    A*_R

be the attribution system with the highest downstream recovery utility for
recovery consumer R.

Define:

    SelectionRegret(R)
        =
        U(A*_R, R)
        -
        U(A_acc, R)

This quantity measures the operational cost of selecting an attributor using
the conventional benchmark metric.

The generic concept of downstream decision regret is not claimed as novel.

## 3A. Cross-fitted Attributor Selection Regret

Selection regret must be estimated out of sample.

It is not valid to identify the best recovery-utility attributor and report
its advantage on the same tasks used to select it.

The 30-task pilot therefore uses six deterministic folds.

Because task selection yields six tasks from each of the five hazard
families, assign:

    hash-rank 1 from every hazard family -> fold 1
    hash-rank 2 from every hazard family -> fold 2
    ...
    hash-rank 6 from every hazard family -> fold 6

Each held-out fold therefore contains:

    5 tasks
    =
    1 task from each hazard family.

For fold k:

1. use the other 25 tasks to rank the deployable real attributors by
   conventional attribution score;
2. use those same 25 tasks to identify the highest-recovery-utility
   real attributor for each recovery consumer;
3. freeze both selections;
4. evaluate their utility difference only on the five held-out tasks.

Aggregate only the held-out regret contributions across all six folds.

The selection-regret comparison is restricted to real deployable
attributors:

    A2 -- TRACE
    A3 -- full-prefix one-shot attribution
    A4 -- AgentDebugX / DeepDebug attribution

A0 is a fallback control.

A1 is an oracle upper bound.

Neither A0 nor A1 participates in the attributor-selection-regret ranking.

Ties in training-fold selection are resolved by a deterministic,
predeclared lexical attributor identifier.

No held-out recovery outcome may influence the selected attributor.

## 4. Primary environment

Use ToolBench-X for the falsification pilot.

Current upstream properties verified before this protocol:

- executable multi-step tasks;
- deterministic tool implementations;
- five recoverable hazard families;
- deferred-on-first-error intervention support;
- explicit failure seed support;
- no-hint evaluation;
- oracle-label evaluation;
- prescriptive recovery-hint evaluation.

The five hazard families are:

- Specification Drift;
- Invocation Error;
- Execution Failure;
- Output Drift;
- Cross-source Conflict.

## 5. Critical ToolBench-X treatment distinction

DO NOT use ToolBench-X `with_hint` as a pure diagnosis condition.

The upstream recovery hint contains prescriptive information including:

- retry strategy;
- mandatory tool sequence;
- required inputs;
- verification checks;
- finish blockers;
- canonical answer rules.

Using that condition would confound diagnosis information with recovery
instructions.

The diagnosis-only upper-bound condition must instead be derived from the
upstream `oracle_label` pathway or an equivalently non-prescriptive wrapper.

## 6. Data handling

ToolBench-X remains an external benchmark.

Do not vendor, redistribute, republish, or modify its dataset in this
repository without explicit permission from the benchmark owners.

The TRACE repository may contain:

- task identifiers;
- task hashes;
- upstream commit identifiers;
- experiment manifests;
- our adapter code;
- prompts;
- configs;
- generated run evidence;
- analysis outputs that do not redistribute restricted benchmark contents.

## 7. Unit of intervention

The intervention point is the first manifested tool-environment hazard.

The branch occurs:

    after the first failing or semantically invalid tool observation
    and
    before diagnosis information influences the next policy decision.

No task may enter the experiment unless this branch point can be identified
and reproduced.

## 8. Same-prefix requirement

All treatment branches for one task must share the exact pre-intervention
history.

Before paid pilot execution, the adapter must prove:

- identical model-visible message prefix;
- identical prior tool calls;
- identical prior tool outputs;
- identical injected-hazard state;
- identical source task;
- identical failure seed;
- identical branch checkpoint hash.

If exact/equivalent branch-state replay cannot be demonstrated, stop the
ToolBench-X experiment rather than approximating the causal comparison.

## 9. Pilot population

Target:

    30 eligible ToolBench-X instances

with:

    6 instances from each of the five hazard families.

Eligibility requires:

1. the clean task is successfully solvable under the frozen baseline;
2. the injected hazard actually manifests;
3. the first-hazard checkpoint can be captured and replayed;
4. required task/tool material is locally available under the benchmark
   license.

Do NOT condition eligibility on whether the no-diagnosis recovery branch
eventually succeeds or fails.

That would select on a post-branch outcome.

## 10. Deterministic task selection

Within each hazard family, rank eligible tasks by:

    SHA256(
        canonical_task_id
        || "|"
        || hazard_family
        || "|trace-v2-pilot-v1"
    )

Take the first six.

If a hazard family has fewer than six eligible instances:

- take all eligible instances;
- record the shortage;
- do not silently replace them with another hazard family.

The selection manifest must be frozen before recovery outcomes are observed.

## 11. Diagnosis interface

All diagnosis sources must be normalized into the same source-blind schema:

    failure_step
    tool_or_component
    hazard_type
    mechanism
    supporting_evidence_refs

The recovery consumer must not receive:

- the attributor identity;
- labels such as TRACE, Oracle, or AgentDebugX;
- hidden recovery instructions;
- benchmark answers;
- mandatory tool sequences;
- source-specific formatting.

The purpose is to evaluate diagnosis content rather than source reputation or
prompt-format differences.

## 12. Core diagnosis sources

The core pilot uses:

### A0 -- No diagnosis

A neutral message indicating that no verified diagnosis is available.

This is the matched fallback.

### A1 -- Oracle diagnostic label

Derived from ToolBench-X ground-truth hazard metadata.

It may communicate the diagnosis category and non-prescriptive definition.

It must not expose recovery procedure or answer information.

### A2 -- Frozen TRACE attribution

Use the current TRACE attribution behavior frozen before pilot outcomes.

Only normalization into the common diagnosis interface is allowed.

### A3 -- Full-prefix one-shot attribution

A simple LLM baseline that sees the complete pre-branch trajectory and emits
the common diagnosis schema in one pass.

The prompt is frozen before outcome collection.

### A4 -- AgentDebugX / DeepDebug attribution

Use AgentDebugX as an external attribution baseline if and only if a
pre-outcome adapter can convert its diagnostic output into the common schema
without changing its diagnosis algorithm.

If this adapter cannot be validated before any pilot recovery outcomes are
observed, the pilot does not substitute another attributor post hoc.

Instead, stop and revise the protocol openly.

## 12A. Diagnosis realization pairing and provenance

Receiver-dependence requires the recovery consumers to receive the same
diagnostic information.

For every:

    task
    x
    natural attribution source
    x
    diagnosis replicate

generate the diagnosis once.

Normalize it once.

Serialize it once.

Hash it once.

The exact same normalized diagnosis bytes and diagnosis hash must then be
supplied to both R1 and R2.

Do NOT regenerate an attributor output separately for each recovery consumer.

Otherwise an apparent attributor-by-consumer interaction could be caused by
different stochastic diagnoses rather than different receivers.

Every normalized diagnosis artifact must record:

- source attributor identifier in provenance metadata that is NOT exposed to
  the recovery consumer;
- source trajectory/checkpoint hash;
- source model and model snapshot;
- attribution prompt hash where applicable;
- raw attribution-output hash;
- normalized diagnosis hash;
- normalization version;
- generation status;
- provider/runtime failure if generation failed.

A failed diagnosis generation is retained as experimental evidence.

Do not selectively regenerate an undesirable diagnosis.

For AgentDebugX / DeepDebug:

- use attribution evidence only;
- remove recovery proposals, retry directives, suggestions, mandatory
  actions, or other prescriptive correction content before normalization;
- do not expose AgentDebugX source identity to the receiver.

If an external attributor cannot be separated from prescriptive recovery
content without materially changing its diagnosis algorithm, it is not a
valid A4 adapter for this pilot.

## 13. Secondary mechanistic controls

These are not the primary evidence for the thesis.

On a separately frozen mechanistic subset, construct deterministic controls:

    correct target / wrong mechanism
    correct mechanism / wrong target

They exist to interpret causal mechanisms.

The central publication thesis must survive using natural attributor outputs.

## 14. Recovery consumers

Use two source-blind, fixed downstream consumers.

### R1 -- Continue

The diagnosis is added as bounded diagnostic context.

The existing tool-orchestration policy then continues without a separate
replanning stage.

### R2 -- Replan

The same diagnosis schema is consumed.

The first post-branch model decision must explicitly revise the recovery plan
before selecting the next tool action.

After that first decision, execution follows the same tool environment.

The two consumers must be frozen before pilot outcomes.

They must not contain logic specific to TRACE, AgentDebugX, or the oracle.

## 15. Compute discipline

Within a given recovery consumer:

- every diagnosis source gets the same maximum post-branch model-call budget;
- every diagnosis source gets the same maximum tool-call budget;
- every diagnosis source gets the same maximum generated-token budget;
- provider retry policy is identical;
- model snapshot is identical;
- reasoning configuration is identical.

Across different recovery consumers, report raw compute separately.

Receiver-dependence analysis uses utility relative to each consumer's own
matched no-diagnosis fallback.

No hidden extra diagnosis-derived tool calls are allowed.

## 15A. Primary estimand versus end-to-end compute

The falsification pilot has one primary causal estimand:

    conditional diagnostic utility.

Interpretation:

    given that diagnosis d has already been produced,
    what causal value does d have for recovery consumer R?

Therefore attribution-generation compute is NOT charged against the
post-branch recovery budget in the primary pilot estimand.

However, attribution-generation cost must still be measured and preserved:

- model calls;
- prompt tokens;
- completion tokens;
- reasoning tokens when exposed;
- wall-clock latency;
- provider/runtime failures.

This separation is intentional.

It isolates the diagnosis-as-interface question from the separate engineering
question of whether generating that diagnosis is worth its cost.

The pilot MUST NOT claim end-to-end cost effectiveness.

If the falsification pilot survives, the full confirmatory study must add a
second system-level estimand in which attribution generation and downstream
recovery compete under the same total compute budget.

## 16. Model gate

Do not run the paid pilot until the model configuration is separately frozen.

The model must use an immutable provider snapshot.

Current OpenAI candidates are evaluated separately from this protocol.

The pilot model is not selected by observing pilot outcomes.

## 17. Primary outcomes

For every attribution-source x recovery-consumer cell record:

- task success;
- exact final-answer success where defined;
- model calls;
- prompt tokens;
- completion tokens;
- reasoning tokens when exposed;
- tool calls;
- recovery rounds;
- wall-clock latency;
- provider/runtime failures.

## 18. Attribution metrics

Record conventional diagnosis quality independently of recovery:

- failure-step correctness;
- tool/component correctness;
- hazard-type correctness;
- joint correctness;
- evidence validity where objectively checkable.

Do not use downstream recovery outcomes when computing these attribution
scores.

## 18A. ToolBench-X attribution-target discipline

The first manifested hazard is also the branch checkpoint.

Therefore simple identification of the branch step may be partly trivial and
must not, by itself, determine A_acc.

Before any recovery outcome is generated, freeze the conventional
ToolBench-X attribution score from objectively available benchmark-native
targets.

Candidate target fields include only fields that can be established from the
pinned upstream hazard manifest and tool metadata, such as:

- hazard type;
- affected tool or component;
- failure manifestation event;
- mechanism, only when the benchmark metadata objectively provides one.

Do not invent mechanism ground truth when upstream metadata does not support
it.

The branch checkpoint location itself does not count as successful root-cause
localization merely because every condition is injected at that checkpoint.

The exact conventional score used to select A_acc must be frozen before
pilot recovery outcomes are observed.

Failure-step correctness may still be reported as a secondary descriptive
metric when it is nontrivial.

## 19. Recovery-grounded metrics

Required:

### Recovery utility

    success_with_diagnosis
    -
    success_no_diagnosis

paired within task and consumer.

### Help rate

Probability that diagnosis succeeds when the matched fallback fails.

### Harm rate

Probability that diagnosis fails when the matched fallback succeeds.

### Selection regret

Utility lost by selecting an attributor using conventional attribution
accuracy instead of held-out recovery utility.

### Rank agreement

Compare attribution-metric ranking with downstream-utility ranking.

Use both rank direction and effect magnitude.

### Consumer interaction

For attributors A and B and consumers R1 and R2:

    I =
        [U(A,R1) - U(B,R1)]
        -
        [U(A,R2) - U(B,R2)]

A sign reversal is especially informative but is not assumed.

## 20. Statistical discipline

The pilot is a falsification study, not the final confirmatory study.

Use task-clustered bootstrap intervals for paired utility quantities.

Do not promote a result solely because p < 0.05.

Report effect sizes and uncertainty.

No metric, threshold, subgroup, or hazard family may be introduced after
outcomes are inspected and then presented as preregistered.

## 20A. Replicates, pairing, randomization, and unit of inference

The task is the primary unit of inference.

Repeated API calls are not independent tasks.

For the falsification pilot, use three post-branch recovery replicates for
each:

    task
    x
    diagnosis source
    x
    recovery consumer

For stochastic natural attributors, generate three diagnosis realizations per
task and attribution source.

Diagnosis replicate r must be reused unchanged across:

    R1 at replicate r
    and
    R2 at replicate r.

When the provider exposes a reproducibility seed that is supported by the
frozen model API:

- use matched replicate seeds across paired treatment branches;
- record the seed;
- record provider model identity and system fingerprint when available.

When deterministic provider seeding is unavailable:

- do not pretend the run is deterministic;
- retain all runs;
- use the predeclared replicate index;
- record provider request identifiers and fingerprints;
- deterministically interleave execution order across treatments.

Execution order must be fixed before outcomes using a hash such as:

    SHA256(
        task_id
        || "|"
        || diagnosis_source
        || "|"
        || recovery_consumer
        || "|"
        || replicate
        || "|trace-v2-pilot-order-v1"
    )

Do not rerun a branch merely because its result is surprising.

Infrastructure failures follow the separately frozen inclusion/exclusion
policy and remain visible in the raw evidence.

Statistical uncertainty is clustered by task.

Recovery replicates estimate within-task stochastic variation and do not
increase the nominal task sample size from 30.


## 20B. Numeric continuation thresholds must be frozen before outcomes

The phrases:

    meaningful selection regret
    comparably large interaction
    meaningful label/utility disagreement

are not allowed to remain subjective once pilot execution begins.

Before the first recovery outcome is generated, freeze a versioned decision
table containing numeric thresholds for Signals A, B, and C.

Those thresholds must be chosen using:

- scientific relevance;
- the 30-task clustered design;
- hypothetical/simulated effect sizes;
- sensitivity or power calculations where appropriate;
- compute budget constraints.

They must NOT be selected using observed pilot outcomes.

After the first pilot recovery outcome exists, those thresholds are
immutable.

If the study later reports exploratory thresholds, they must be explicitly
labeled post hoc and may not determine the preregistered continuation
decision.

## 21. Pilot continuation criteria

Advance to a full preregistered study only if at least TWO of the following
three signals survive:

### Signal A -- Metric-selection regret

The attribution system preferred by conventional attribution accuracy is
meaningfully worse in downstream recovery utility than another real
attributor under at least one consumer, and the effect is not confined to a
single hazard family.

### Signal B -- Receiver dependence

At least one pair of real attribution systems changes ordering across the two
recovery consumers, or exhibits a comparably large pre-specified
attributor-by-consumer interaction.

### Signal C -- Label/utility disagreement

Natural attributor outputs show a meaningful population of either:

- label-incorrect diagnoses with positive recovery utility; or
- label-correct diagnoses with negative recovery utility.

Synthetic corrupted diagnoses do not count toward this continuation signal.

## 22. Decision after pilot

If zero signals survive:

    STOP this publication thesis.

If exactly one signal survives:

    permit one bounded replication/extension only;
    do not start a new recovery method paper.

If at least two signals survive:

    perform a renewed live literature search;
    freeze the full preregistration;
    validate in an independent environment such as SREGym;
    only then consider a utility-aware diagnosis-selection method.

## 23. Claims forbidden from pilot alone

The pilot cannot establish that:

- attribution accuracy is generally invalid;
- receiver dependence universally exists;
- TRACE is superior;
- AgentDebugX is inferior;
- ToolBench-X represents all agent failures;
- recovery utility should universally replace attribution accuracy;
- the proposed evaluation framework is novel.

The pilot exists to decide whether a full study is justified.

## 24. Next implementation gate

Before writing the ToolBench-X adapter:

1. record exact upstream ToolBench-X commit;
2. re-check upstream issues and pull requests;
3. inspect the exact first-failure/hint/oracle implementation;
4. verify licensing constraints;
5. prove same-prefix branch replay in an offline fixture;
6. freeze the model snapshot and reasoning settings;
7. freeze exact R1/R2 prompts and post-branch budgets;
8. freeze the deterministic 30-task selection manifest.

Only then may pilot outcomes be generated.
