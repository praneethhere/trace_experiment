# TRACE v2 Publication Novelty Gate

Status: CONDITIONAL GO
Frozen after live literature review: September 2026

## Research standard

This document does not claim novelty.

Its purpose is to define the narrow research hypothesis that survived the
current literature falsification pass and to prevent the implementation from
drifting toward claims already occupied by prior work.

## Claims we will NOT make

TRACE v2 will not claim novelty for:

1. automated failure attribution itself;
2. locating the responsible agent or failure step;
3. using interventions to validate attribution;
4. showing that better diagnosis can improve recovery;
5. showing that misleading feedback can hurt an agent;
6. evidence-gated recovery in general;
7. outcome-oriented debugging instead of attribution accuracy;
8. counterfactual replay or same-prefix branching in general;
9. utility-aware decision making in general;
10. showing that one root-cause label can be insufficient.

These areas are substantially occupied by prior work.

## Closest work that constrains the claim

- DoVer, ICLR 2026:
  intervention-driven debugging; evaluates resolution/progress rather than
  attribution accuracy; demonstrates multiple repairing interventions.

- Who&When / Who&When Pro:
  agent/step/error-mode failure attribution with controlled labels.

- TraceElephant, ACL 2026:
  full-observability failure-attribution benchmark.

- RAFFLES, EACL 2026:
  reasoning-based iterative fault attribution.

- AgentRx, 2026:
  evidence-backed agent failure diagnosis.

- StepFinder, 2026:
  efficient temporal-semantic failure attribution.

- OAT, 2026:
  unsupervised failure attribution learned from successful trajectories.

- PROBE, arXiv:2605.08717:
  telemetry -> diagnosis -> bounded recovery guidance with a Guidance Gate.

- Causal Agent Replay, arXiv:2606.08275:
  counterfactual causal attribution through replay.

- CausalFlow, arXiv:2605.25338:
  causal attribution plus counterfactual repair.

- Calibration Is Not Control, arXiv:2606.21399:
  intervention advantage and same-prefix branching for action-conditioned
  runtime control.

- Don't Blindly Trust It, arXiv:2606.21409:
  matched-loop faithful/misleading/no-feedback evaluation.

- Diagnosis Is Not Prescription, arXiv:2605.21958:
  the diagnosed bottleneck is not necessarily the best intervention target.

- Localizing Emergent Failures / MRFR, arXiv:2608.29228:
  multiple and jointly necessary repair families.

- FaulT-Bench, arXiv:2608.27021:
  unreliable troubleshooting tickets including incorrect root-cause claims.

- ToolBench-X, arXiv:2606.25819:
  recoverable tool hazards, diagnosis hints and test-time scaling.

- SREGym, arXiv:2605.07161:
  live high-fidelity SRE failure benchmark.

- Utility-Directed Conformal Prediction, ICLR 2025:
  downstream decision loss as an evaluation/training objective.

Therefore TRACE v2 must not present generic downstream utility or
decision-focused evaluation as a new idea.

## September 2026 live-check addendum

AgentDebugX (arXiv:2607.18754) further constrains the claim boundary.

It already implements a closed debugging loop of:

    Detect -> Attribute -> Recover -> Rerun

including checkpoint replay and downstream repair evaluation.

Therefore TRACE v2 will also NOT claim novelty for connecting failure
attribution to recovery/rerun as a generic closed-loop debugging system.

Current candidate distinction:

    evaluate whether conventional attribution metrics select diagnoses
    that maximize downstream recovery utility, and whether that selection
    is stable across different recovery consumers.

This remains a hypothesis, not a novelty claim.

## Candidate surviving question

Does conventional failure-attribution accuracy rank attribution systems by
their actual value to downstream recovery policies?

More strongly:

Is the operational utility of a failure diagnosis receiver-dependent?

A diagnosis may be useful, useless, or harmful depending on the recovery
policy that consumes it.

## Receiver-conditional diagnostic utility

For failed state s, attributor A_i, recovery policy R_j, and downstream
outcome Y:

    U_ij(s) =
        E[Y | s, d ~ A_i(s), R_j(d)]

The empirical object of interest is the Recovery Utility Matrix whose rows
are attribution systems and whose columns are downstream recovery policies.

The generic decision-theoretic concept is not claimed as novel.

The possible contribution is the application-grounded evaluation of failure
attribution as an information interface between diagnosis and recovery.

## Primary falsification hypothesis

Conventional attribution accuracy may fail to preserve downstream recovery
rankings.

The strongest evidence would be an attributor ranking reversal:

    Accuracy(A) > Accuracy(B)

while:

    Utility_R(A) < Utility_R(B)

A stronger receiver-dependence result is:

    Utility_R1(A) > Utility_R1(B)

but:

    Utility_R2(A) < Utility_R2(B)

## Secondary hypothesis

Exact-label disagreement is not identical to operational invalidity.

Some diagnoses that disagree with one annotated root-cause label may still
enable a valid repair, while a label-correct diagnosis may induce a harmful
intervention.

This hypothesis must be tested rather than assumed.

## Pilot environment

Primary falsification environment:

    ToolBench-X

Reasons:

- executable tasks;
- deterministic tools;
- five structured recoverable hazard classes;
- canonical answers;
- existing diagnosis-hint condition;
- existing no-hint and test-time-scaling controls.

ToolBench-X data restrictions must be respected.
Do not redistribute or modify its dataset without permission.
Release only our code, task identifiers/hashes, transformations, configs and
our own generated evidence where legally permitted.

Independent validation environment if the pilot survives:

    SREGym / SREGym-Lite

## Pilot design

Use a small balanced ToolBench-X subset spanning all five hazard families.

Branch from the same failure checkpoint.

Hold downstream recovery compute constant.

Candidate diagnostic conditions:

- no diagnosis;
- oracle diagnosis;
- TRACE predicted diagnosis;
- independent LLM diagnosis;
- correct target / wrong mechanism;
- correct mechanism / wrong target.

Use at least two meaningfully different fixed recovery consumers.

The purpose is not to prove TRACE wins.

The purpose is to determine whether diagnosis quality is operationally
receiver-dependent.

## Pilot metrics

Required:

- conventional attribution correctness;
- downstream task success;
- paired recovery gain versus no-diagnosis fallback;
- harm probability versus no-diagnosis fallback;
- token cost;
- tool-call cost;
- recovery rounds;
- diagnostic regret relative to the best available branch;
- rank agreement between attribution accuracy and recovery utility;
- rank stability across recovery consumers.

## Kill criteria

Do not proceed to a full paper if the pilot indicates that:

- attribution accuracy essentially preserves recovery rankings;
- rankings remain stable across distinct recovery consumers;
- label-incorrect diagnoses rarely support valid alternative repairs;
- label-correct diagnoses rarely induce harmful recovery;
- no-diagnosis fallback is never competitive;
- effects exist only for artificial corrupted diagnoses and disappear for
  real attribution systems.

If the central phenomenon fails, stop rather than redesigning metrics around
the observed results.

## Potential later contribution -- NOT YET APPROVED

If the pilot survives, investigate Recovery-Equivalent Diagnosis Sets:

diagnoses that differ semantically or in benchmark labels but are equivalent
with respect to successful admissible recovery families.

This extension is motivated by work showing that failures can admit multiple
or jointly necessary repair interventions.

Do not implement this before the pilot establishes that conventional labels
and downstream utility meaningfully diverge.

## Publication claim discipline

Until the pilot and renewed literature search are complete:

- do not call the idea novel;
- do not call TRACE state of the art;
- do not claim attribution accuracy is invalid in general;
- do not claim receiver dependence exists;
- do not claim ranking reversal exists;
- do not claim recovery-equivalent diagnosis sets are needed.

All are hypotheses to be falsified.

## Next gate

Before full implementation:

1. freeze the pilot task-selection rule;
2. freeze diagnostic perturbation definitions;
3. freeze the two recovery-consumer contracts;
4. freeze compute matching;
5. freeze statistical analysis;
6. run a small falsification pilot;
7. perform another live novelty search before scaling.

Only a surviving pilot advances to full preregistration.
