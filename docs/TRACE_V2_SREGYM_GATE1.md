# TRACE v2 — SREGym Gate 1 Decision

Date: 2026-09-21

## Status

**CONDITIONAL PASS — MATCHED-RESTART EXTERNAL VALIDATION ONLY**

SREGym is not approved as a same-prefix or identical-branch causal
environment.

No SREGym benchmark outcomes have been generated at this gate.

## Frozen upstream identity

Audited SREGym commit:

`c0d57d13d25231a9a6f68390afe460cbfda4d77e`

The source tree was audited read-only.

## Research question

TRACE v2 studies whether the downstream utility and preferred selection of a
failure diagnosis depend on the recovery policy consuming that diagnosis.

SREGym is being considered as an independent live-system validation
environment for that question.

## Gate 1 findings

### PROVEN — diagnosis checkpoint is not a system-state checkpoint

After fault injection, SREGym calls
`diagnosis_oracle.load_diagnosis_checkpoint()`.

The diagnosis oracle stores the result of `expect()` as its checkpoint and
later recomputes `expect()` to test whether the diagnosis truth remained
stable.

This is an oracle-truth checkpoint. It does not snapshot or restore the
Kubernetes incident state.

The term "diagnosis checkpoint" must therefore never be described by TRACE v2
as a cluster checkpoint, execution checkpoint, or branch checkpoint.

### PROVEN — healthy mitigation baseline precedes fault injection

The conductor captures the mitigation oracle baseline before calling the
problem's fault injector.

The observed order is:

1. capture healthy mitigation baseline;
2. inject the fault;
3. mark the fault injected;
4. capture the diagnosis-oracle expectation;
5. begin agent stages.

This baseline supports grading and cleanup. It does not establish a
post-fault branch clone.

### PROVEN — no generic post-fault full-state branch snapshot was found

The Gate 1C source audit found problem-specific snapshots and restoration
mechanisms, but did not identify a general mechanism that captures the entire
already-faulted system state and later restores two independent consumers to
that exact state.

The global cluster-state machinery is oriented toward baseline capture and
reconciliation for benchmark isolation.

Therefore TRACE v2 must not claim that two independently started SREGym runs
share an identical pre-recovery state.

### PROVEN — live SREGym execution contains temporal and mutable state

Examples identified during the source audit include:

- time-dependent workload collection and accumulated workload history;
- live Kubernetes object state;
- fault-specific persisted snapshots;
- asynchronous workloads;
- a singleton noise manager;
- wall-clock-dependent noise scheduling;
- random noise-experiment selection and random name suffixes.

A repeated problem identifier is therefore insufficient evidence that two
runs are equivalent.

### PROVEN — diagnosis evaluation observes the live environment

Diagnosis evaluation recomputes the oracle expectation and verifies it
against the stored diagnosis checkpoint before scoring the submitted
solution.

TRACE v2 must not assume that native diagnosis evaluation is a neutral
treatment-delivery primitive.

The receiver experiment will keep diagnosis treatment bytes external to the
SREGym diagnosis-submission mechanism unless a later gate proves use of that
mechanism cannot contaminate the treatment comparison.

## Experimental role

SREGym is assigned the following candidate role:

**live-system matched-restart external validation**

It is explicitly not assigned the following role:

**verified same-prefix causal environment**

This distinction is part of the study design rather than an implementation
limitation to hide.

## Matched-restart requirements

Before any substantive SREGym outcomes are admitted into TRACE v2, a later
gate must establish all of the following:

1. Each recovery arm starts from a fresh recreation of the same frozen
   problem and fault specification.

2. The exact normalized diagnosis bytes used for a diagnosis realization are
   reused unchanged across recovery consumers.

3. Receiver identity cannot influence fault generation, diagnosis
   generation, task eligibility, or pre-recovery state measurement.

4. Recovery-arm order is deterministically randomized or counterbalanced
   within matched problem/fault/replicate blocks.

5. A pre-recovery state fingerprint is recorded before the recovery consumer
   acts.

6. The state fingerprint excludes inherently ephemeral identifiers only when
   exclusion is scientifically justified and frozen in advance.

7. Fault target, fault manifestation, relevant resource specifications,
   health state, and benchmark-visible evidence must be represented strongly
   enough to detect materially different recreations.

8. Failed or materially unmatched recreations are retained as infrastructure
   evidence and handled by a frozen inclusion/exclusion rule. They may not be
   silently rerun until a favorable branch appears.

9. Noise, asynchronous workloads, random selection, and other uncontrolled
   state must either be disabled by a frozen benchmark-supported setting or
   explicitly incorporated into the matched-restart design.

10. TRACE v2 must report restart variability rather than implying that fresh
    SREGym recreations are identical.

## Statistical consequence

SREGym observations must be analyzed as matched-restart live-system trials,
not as exact paired branches.

Problem/task remains the primary inferential unit.

Restart pair/block and replicate structure must be represented in the
analysis, and repeated runs must not be counted as independent tasks.

## Relationship to ToolBench-X

The two environments are intended to provide different forms of evidence.

ToolBench-X remains a candidate for a verified same-prefix causal experiment,
subject to author permission and successful released-module replay
validation.

SREGym is a candidate for live-system matched-restart external validation,
subject to an empirical incident-recreation and state-fingerprint gate.

Neither environment is approved for substantive TRACE v2 claims merely by
this document.

## Hard refusal conditions

TRACE v2 will not use SREGym to support receiver-conditional diagnosis utility
claims if the next validation gate shows that:

- the same problem/fault cannot be recreated with sufficiently controlled
  pre-recovery state;
- receiver assignment affects incident construction;
- state mismatch cannot be detected before recovery;
- diagnosis bytes cannot be held identical across receiver arms;
- infrastructure failures would require outcome-dependent rerunning;
- benchmark noise dominates or confounds the receiver comparison.

If those conditions cannot be solved without materially changing the
benchmark, SREGym will be removed from the confirmatory design.

## Claims not permitted after Gate 1

Gate 1 does not establish that:

- SREGym branches are identical;
- SREGym restarts are deterministic;
- SREGym recovery-consumer effects are causal under exact branch equality;
- TRACE improves SREGym recovery;
- receiver dependence exists in SREGym;
- conventional attribution accuracy fails in SREGym;
- the TRACE v2 scientific thesis is novel or confirmed.

## Next gate

The next SREGym gate is an **incident-recreation feasibility study**.

It must first select a small, frozen set of representative problems using
source-only criteria.

Then, without model calls or agent recovery, it must recreate each selected
incident multiple times and determine whether a pre-recovery state
fingerprint can reliably distinguish equivalent-enough recreations from
materially different ones.

No recovery outcomes may be observed before that contract is frozen.
