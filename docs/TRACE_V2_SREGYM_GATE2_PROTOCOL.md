# TRACE v2 — SREGym Gate 2 Incident-Recreation Feasibility Protocol

Original draft date: 2026-09-25
Pre-outcome amendment date: 2026-10-15

## Status

**PREREGISTERED FEASIBILITY PROTOCOL — NO RECOVERY OUTCOMES PERMITTED**

Protocol freeze date: 2026-10-15

This gate tests whether a small source-selected set of SREGym incidents can be
recreated with sufficiently controlled pre-recovery state to support a later
matched-restart receiver experiment.

It does not test diagnosis quality, mitigation quality, receiver dependence,
TRACE utility, agent performance, or publication hypotheses.

No recovery-agent action, diagnosis submission, mitigation submission,
LLM-as-a-judge evaluation, or model call is permitted during Gate 2.

## Relationship to Gate 1

Gate 1 classified SREGym as:

**CONDITIONAL PASS — MATCHED-RESTART EXTERNAL VALIDATION ONLY**

SREGym is not treated as a same-prefix or identical-branch environment.

The original Gate 1 source audit remains:

`c0d57d13d25231a9a6f68390afe460cbfda4d77e`

The post-Gate-1 steady-state-baseline recheck remains:

`0cdc976d4ba4fa6c11a982121a32bf6e391a3742`

Original Gate 2 candidate selection was audited against:

`3f6bd041231715184f962cedb33cfbcefa9b5b3d`

A subsequent, still-pre-outcome source review identified:

- PR #1046: incomplete Kubernetes cluster-baseline capture could
  compromise cleanup correctness;
- PR #1077: Valkey authentication injection and internal-traffic-policy
  fault manifestation required correctness/race fixes;
- PR #999: the original P1 was retired from the current Lite suite;
- PR #1105: the benchmark cleanup watchdog required a longer default.

The original P1 was replaced before any Gate 2 recreation outcomes.

The amended execution revision is:

`9093167a7082402bf553c6efcda1c9dd48dd5138`

Its applications submodule revision is:

`b2006cb6458785c1b353d0fd1753db7fe3a30795`

This amendment changes source provenance and the candidate-specific
measurement contract before any incident-recreation results were observed.

The September source-selection history is retained.

The amendment is not evidence that SREGym supports exact replay or
identical causal branches.

## Gate 2 research question

Can independently recreated instances of a frozen SREGym problem and fault
reach a pre-recovery state that is equivalent under a predeclared semantic
and structural fingerprint?

The word "equivalent" in this protocol means equivalent under the frozen
fingerprint defined below.

It does not mean byte-identical Kubernetes state.

## Source-selection rule

The original candidates were selected before observing any TRACE Gate 2
recreation outcomes, using SREGym-Lite membership and diversity of
fault mechanism.

On October 15, 2026, the previously selected service-selector candidate
was found to have been retired from the current Lite suite.

It is replaced before execution by a service-routing fault that keeps the
frontend Service populated with both correct and incorrect endpoints.

The amended sample remains mechanism-stratified:

- mixed-endpoint service routing;
- topology-sensitive networking;
- application authentication state;
- namespace resource admission.

No TRACE Gate 2 recreation outcomes have been used for candidate selection.

Selection bias disclosure:

SREGym maintainers curate Lite partly using benchmark difficulty and
observed agent behavior. Therefore membership in the suite is not
independent of upstream performance evidence.

We do not claim that the sample is randomly selected, performance-blind
at the upstream level, or representative of all SREGym problems.

The purpose is controlled feasibility of matched-restart recreation,
not estimation of benchmark-wide success rates.

## Frozen candidate set

The amended four-problem set is exactly:

P1:
`service_wrong_pod_selection_hotel_reservation`

Failure stratum:
mixed intended/unintended Service endpoint selection.

P2:
`internal_traffic_policy_local_astronomy_shop`

Failure stratum:
topology-sensitive service-networking failure.

P3:
`valkey_auth_disruption`

Failure stratum:
application-state / authentication mutation.

P4:
`namespace_memory_limit`

Failure stratum:
namespace ResourceQuota / admission and replacement failure.

Original P1, retained in the audit history but not executed:

`wrong_service_selector_social_network`

No candidate may be replaced after any Gate 2 incident-recreation
outcome has been observed.

Any further pre-outcome change requires a new dated amendment and
explicit source-based justification.

## Frozen SREGym source

All Gate 2 executions must use:

`9093167a7082402bf553c6efcda1c9dd48dd5138`

The frozen applications submodule must resolve to:

`b2006cb6458785c1b353d0fd1753db7fe3a30795`

The execution harness must reject any mismatch in either revision.

The source-selection SHA `3f6bd041...` is preserved as historical
provenance and is not the execution SHA.

All four amended candidates must be present in the SREGym-Lite suite
defined by the frozen execution revision.

The experiment must not silently adopt newer upstream implementations
or application-chart updates during execution.

No local modifications to SREGym or its submodule are authorized
without a versioned, pre-outcome protocol amendment.

## Upstream recheck before execution

Immediately before the first Gate 2 execution:

1. fetch current upstream `main`;
2. record current upstream HEAD;
3. inspect commits, PRs, issues, and maintainer comments affecting the four
   candidate problems, conductor lifecycle, KIND setup, cluster reset,
   baseline-window behavior, or fault injection;
4. verify that no known upstream evidence invalidates the frozen design.

If upstream has advanced, Gate 2 still executes against the frozen SHA only
after the delta is reviewed.

If new evidence invalidates a candidate or this protocol, execution stops and
the protocol is revised before outcomes.

## Cluster topology

Use SREGym's frozen standard KIND setup.

Required topology:

- one control-plane node;
- three worker nodes;
- all four nodes Ready before any candidate begins.

This topology is part of the experiment contract.

The topology must not be reduced for resource convenience.

In particular,
`internal_traffic_policy_local_astronomy_shop`
requires at least two worker nodes and deliberately separates the backend and
caller across workers.

## Deployment profile

Freeze:

`profile = full`

The deployment profile must remain identical across all Gate 2 recreations.

Gate 2 must not silently fall back to `svelte`.

If the machine cannot execute the full profile reliably, Gate 2 stops as an
environment-capacity problem before scientific interpretation.

## SREGym-Lite scope

The amended P1, P2, P3 and P4 must all be registered in the frozen
execution revision's `sregym-lite` problem set.

The retired original P1 is not part of this experiment.

The number and contents of the historical Lite suite may differ from
those in the current upstream `main`. Suite membership for this protocol
is evaluated at the pinned execution SHA.

No other SREGym or SREGym-Lite problem is executed.

Lite membership supports feasibility-oriented sampling; it does not
prove repeatability, independence, or exact state equivalence.

## Pre-fault steady-state window

Freeze for Gate 2 only:

`baseline_duration_s = 0`

and no non-zero `--baseline` override.

Reason:

Gate 2 tests structural incident recreation. It does not test whether a
particular amount of healthy telemetry history is sufficient for an agent.

Zero is the upstream default and avoids inventing an unvalidated duration
after the addition of SREGym's optional baseline-window mechanism.

A Gate 2 pass at zero seconds does not establish recreation equivalence under
a later non-zero baseline window.

If the later SREGym recovery experiment uses a non-zero baseline duration,
that duration must be frozen before recovery outcomes and the incident-
recreation feasibility check must be repeated under that temporal policy.

## Post-fault propagation policy

Use the frozen SREGym problem-specific propagation durations.

The exact effective durations are:

- P1: 0 seconds;
- P2: 65 seconds;
- P3: 0 seconds;
- P4: 0 seconds.

The harness must preserve the upstream per-problem setting rather than
globally overriding propagation to zero.

For a Conductor-based implementation, use:

`propagation_override_s = None`

and verify the resulting effective duration for every candidate.

For a direct-injection implementation, explicitly reproduce the same
effective elapsed post-fault interval before fingerprint collection.

Do not apply the interval twice.

For P2, the full 65-second propagation boundary must complete before
the additional Gate 2 manifestation/quiescence observations begin.

P2's 65-second duration is upstream's declared interval for exposing
the injected network fault in timeout and telemetry signals.

The propagation interval is distinct from:

- the zero-second pre-fault steady-state baseline;
- the harness's three-observation quiescence rule;
- the 120-second maximum harness-side quiescence wait.

Raw injection completion, propagation start/completion, quiescence start,
and fingerprint capture times must be recorded.

Gate 2 must not silently shorten propagation to improve repeatability
or execution speed.

## Noise policy

SREGym optional benchmark noise is disabled for Gate 2.

Freeze:

`enable_noise = false`

No external workload or perturbation may be intentionally added beyond
workloads required by the frozen problem implementation.

Unexpected background activity is recorded rather than silently suppressed
after results are seen.

## Model and agent prohibition

Gate 2 must make:

`0 model calls`

It must launch:

`0 recovery agents`

It must perform:

`0 LLM-as-a-judge evaluations`

It must perform:

`0 diagnosis submissions`

and:

`0 mitigation submissions`.

Fault construction may use SREGym's benchmark implementation.

The harness may inspect Kubernetes state and execute deterministic,
predeclared non-mutating probes required by this protocol.

A probe must not repair, mitigate, or alter the incident.

## Cluster lifetime

Create one clean frozen KIND cluster for the Gate 2 campaign.

Each recreation is a fresh benchmark problem recreation on that cluster:

1. clean/reconcile according to frozen SREGym lifecycle;
2. undeploy leftovers;
3. deploy the required application;
4. use the frozen zero-second steady-state-window policy;
5. inject the candidate fault;
6. wait only for the frozen manifestation/quiescence predicate;
7. capture the pre-recovery fingerprint;
8. perform no recovery-agent action;
9. clean the problem using the frozen benchmark cleanup path;
10. verify cleanup before the next scheduled recreation.

The KIND cluster is not recreated between ordinary Gate 2 episodes.

Persistent cross-episode drift is intentionally visible to this feasibility
test rather than eliminated by rebuilding the cluster after every episode.

A campaign-level infrastructure failure does not authorize silently replacing
the cluster and continuing under the same campaign identity.

## Between-episode cleanup and carryover barrier

The persistent KIND campaign is permitted only with a successful cleanup
barrier between every pair of scheduled episodes.

SREGym's cleanup implementation is allowed to perform its frozen benchmark
mutations, including fault recovery, application teardown, and cluster-state
reconciliation.

The TRACE Gate 2 barrier itself is read-only.

After SREGym cleanup returns and before the next episode begins, require all
of:

1. SREGym reports no cleanup failure or cleanup exception;
2. cluster-state reconciliation, when enabled by the frozen SREGym lifecycle,
   completed without exception;
3. the completed problem's application namespace is fully absent;
4. the namespace is not merely `Terminating`;
5. no PersistentVolume has `spec.claimRef.namespace` equal to the completed
   application namespace;
6. the frozen KIND node-name set is unchanged;
7. the frozen KIND node-UID set is unchanged;
8. node role assignments are unchanged;
9. all four KIND nodes are `Ready`.

Namespace/PV/node observations are sampled every two seconds for at most:

`180 seconds`

The barrier passes only when all nine requirements hold in the same
observation.

No next scheduled episode may start while the preceding cleanup barrier is
unresolved.

A cleanup command returning successfully is not by itself evidence that the
barrier passed.

If the infrastructure remains usable but SREGym cleanup leaves candidate or
application residue that violates this barrier, the completed episode is
classified `RECREATION_FAILURE`.

If an external Docker/KIND/host failure prevents the barrier from being
evaluated fairly, classify according to the frozen
`INFRASTRUCTURE_FAILURE` rule.

This barrier intentionally makes cross-episode carryover observable rather
than hiding it by rebuilding the cluster after every incident.

## Number of recreations

Freeze:

`8 recreations per candidate`

for:

`4 candidates`

giving:

`32 scheduled incident recreations`.

This is an engineering feasibility stress test.

Eight was not selected from observed Gate 2 variance and does not constitute
a statistical power claim.

## Frozen execution order

Each problem appears exactly twice in each ordinal position across the eight
rounds.

The order is:

Round 1:
P1, P2, P3, P4

Round 2:
P2, P3, P4, P1

Round 3:
P3, P4, P1, P2

Round 4:
P4, P1, P2, P3

Round 5:
P4, P3, P2, P1

Round 6:
P3, P2, P1, P4

Round 7:
P2, P1, P4, P3

Round 8:
P1, P4, P3, P2

No successful or failed recreation may be reordered, replaced, or repeated
because of its observed fingerprint.

## Episode identity

Every scheduled recreation receives an immutable identity:

`G2-R<round>-P<candidate>`

Examples:

`G2-R1-P1`
`G2-R1-P2`
...
`G2-R8-P4`

An episode identity is allocated from the frozen schedule before execution.

Failed episodes retain their identities.

## Provenance recorded for every episode

Before fault injection, record at minimum:

- TRACE repository commit;
- SREGym repository commit;
- SREGym applications submodule commit;
- Gate 2 protocol SHA-256;
- operating system;
- architecture;
- Docker version;
- KIND version;
- Kubernetes server version;
- kubectl version;
- Helm version;
- Python version;
- uv version;
- deployment profile;
- baseline duration;
- noise policy;
- cluster node identities and roles;
- problem identifier;
- round number;
- ordinal position;
- UTC start time.

Tool versions are provenance.

They are not silently normalized away.

## Fingerprint timing boundary

The fingerprint represents:

**post-fault, pre-recovery state**

No recovery consumer may act before fingerprint capture completes.

Fingerprint capture begins only after the candidate's frozen manifestation
predicate is satisfied.

No arbitrary post-hoc settling delay may be introduced after observing
candidate behavior.

Where a candidate implementation already waits for its fault state, that wait
is part of the frozen fault implementation.

Additional harness-side waiting may only implement the frozen quiescence rule
defined below.

## Generic quiescence rule

After `inject_fault()` returns, evaluate the candidate-specific manifestation
predicate.

A candidate is quiescent when its complete predicate is true on three
consecutive observations separated by two seconds.

Maximum harness-side quiescence wait:

`120 seconds`

If the complete predicate does not remain true for three consecutive
observations within 120 seconds, classify:

`RECREATION_FAILURE`

Do not rerun that scheduled episode.

## P1 manifestation predicate

For:

`service_wrong_pod_selection_hotel_reservation`

require all of:

1. Service `frontend` exists in the problem namespace.
2. Its complete selector is exactly:

   `{"service-route": "frontend"}`

3. Deployments `frontend` and `search` exist.
4. Both Deployment pod templates contain:

   `service-route = frontend`

5. Both Deployments have positive desired replica counts.
6. At least one Ready, non-terminating `frontend` pod exists.
7. At least one Ready, non-terminating `search` pod exists.
8. Service `frontend` has at least one ready endpoint whose target pod
   belongs to Deployment `frontend`.
9. The same Service has at least one ready endpoint whose target pod
   belongs to Deployment `search`.
10. No ready Service endpoint has an unidentified workload owner.
11. The Service port and target-port configuration is captured, including
    the frontend traffic path using port 5000.

Determine endpoint ownership through Kubernetes owner references:

Pod -> ReplicaSet -> Deployment

Do not infer ownership from pod-name substrings.

Endpoint attribution may use the Kubernetes Endpoints API; retain
EndpointSlice evidence as well where available.

The condition is not satisfied by a Service with zero endpoints or by a
Service containing exclusively frontend endpoints.

It is also not satisfied merely because the wrong label was applied;
the unintended search pod must actually be a ready Service endpoint.

Retain raw pod identities, owner references, labels, endpoint addresses,
readiness conditions and rollout evidence.

The fault's intermittent HTTP impact is a plausible consequence of
polluted routing, but Gate 2 does not grade recovery or require a
particular number of failed HTTP requests.

## P2 manifestation predicate

For:

`internal_traffic_policy_local_astronomy_shop`

require all of:

1. Service `recommendation` exists.
2. `spec.internalTrafficPolicy == Local`.
3. `recommendation` Deployment is pinned using
   `kubernetes.io/hostname`.
4. `frontend` Deployment is pinned using
   `kubernetes.io/hostname`.
5. The two pinned worker nodes are different.
6. A Running `recommendation` pod exists on the backend worker.
7. A Running `frontend` pod exists on the caller worker.
8. `recommendation` has a populated endpoint set.

Absolute node names are recorded.

For semantic comparison they are additionally represented as stable worker
ranks derived by sorting worker-node names.

The required topology relation is:

`backend_worker_rank != caller_worker_rank`

## P3 manifestation predicate

For:

`valkey_auth_disruption`

immediately before fault injection, capture as raw evidence the UID set of
cart-service pods selected by:

`app.kubernetes.io/name=cart`

The pod UIDs are restart witnesses only. They are not semantic fingerprint
identifiers.

After fault injection require all of:

1. at least one `valkey-cart` pod exists and is `Running`;
2. an unauthenticated read-only Valkey `PING` reports authentication required;
3. an authenticated read-only Valkey `PING` using the frozen injected
   credential `invalid_pass` returns `PONG`;
4. the pre-fault cart pod UID set was non-empty;
5. the current cart pod UID set contains at least one UID that was not in the
   pre-fault set;
6. at least one replacement cart pod exists and is `Running`;
7. the Valkey pod remains available, so disappearance of Valkey itself is not
   substituted for the authentication incident.

The P3 probe commands are frozen as the semantic equivalent of:

    valkey-cli --raw PING

with no credential, and:

    VALKEYCLI_AUTH=invalid_pass valkey-cli --no-auth-warning --raw PING

inside the selected `valkey-cart` pod.

For the first probe, normalized combined output must contain:

`NOAUTH Authentication required`

For the second probe, normalized stdout must contain a line exactly:

`PONG`

These probes are read-only.

The Gate 2 collector must not use `CONFIG SET`, `SET`, `DEL`, `FLUSHDB`,
`FLUSHALL`, or another command that changes Valkey state while establishing
the P3 manifestation predicate.

Raw command return codes and outputs are preserved.

### P3 upstream injection verification

At the frozen execution SHA, SREGym itself checks that the Valkey pod
did not restart during fault injection and verifies that `requirepass`
remains set to the injected value.

The Gate 2 collector's read-only PING probes supplement that upstream
injection verification rather than replacing it.

Record the upstream injection verification outcome and the pod/container
restart count before and after the Gate 2 probes.

A probe-induced restart, loss of `requirepass`, or other disappearance
of the injected fault is a recreation failure, not a successful
authentication-state match.

The frozen applications submodule is required because its Valkey
resource configuration is relevant to the upstream OOM fix.

## P4 manifestation predicate

For:

`namespace_memory_limit`

require all of:

1. ResourceQuota `memory-limit-quota` exists;
2. its `spec.hard.memory` value is exactly `1Gi`;
3. Deployment `search` has a positive desired replica count;
4. its Ready replica count is lower than desired;
5. the active ReplicaSet set owned by Deployment `search` is captured;
6. deterministic evidence establishes that replacement creation was rejected
   by the injected memory ResourceQuota.

Requirement 6 passes only through one of these predeclared evidence paths.

### P4 evidence path A — ReplicaSet condition

At least one active ReplicaSet owned by Deployment `search` has a status
condition satisfying all of:

- condition type: `ReplicaFailure`;
- condition status: `True`;
- reason: `FailedCreate`;
- condition message contains `memory-limit-quota`;
- condition message contains `must specify`;
- condition message contains `memory`.

### P4 evidence path B — Kubernetes Event

At least one Warning Event satisfies all of:

- reason: `FailedCreate`;
- involved-object kind: `ReplicaSet`;
- involved-object name identifies an active ReplicaSet owned by Deployment
  `search`;
- message contains `memory-limit-quota`;
- message contains `must specify`;
- message contains `memory`.

The comparison of required message tokens is case-insensitive.

The complete raw condition and Event messages are preserved unchanged.

The semantic fingerprint records:

`quota_admission_failure = true`

only when evidence path A or evidence path B passes.

Generated ReplicaSet names may be canonicalized only according to the frozen
ephemeral-identity rules already defined in this protocol.

An unrelated `Pending`, scheduling-capacity, image-pull, node, or admission
failure does not satisfy the P4 predicate.

There is no free-form human classification of the failure message during
Gate 2.

## Fingerprint layers

Each episode produces three representations:

### Layer A — raw evidence

Raw Kubernetes/API/probe evidence exactly as observed.

Raw evidence is immutable.

### Layer B — canonical structural state

Structured state after removing only the frozen ephemeral fields listed in
this protocol.

### Layer C — semantic fingerprint

A compact candidate-specific object containing the variables required to
establish that the intended fault and materially relevant pre-recovery state
are equivalent.

All three layers are retained.

No raw field may be discarded merely because it differed between runs.

## Required generic structural fields

Capture at minimum for the problem namespace:

- Deployment names;
- desired replicas;
- ready replicas;
- available replicas;
- unavailable replicas;
- Deployment selectors;
- pod-template labels;
- container images;
- container resources;
- environment-variable names and values except secrets requiring redaction;
- pod phases;
- pod readiness;
- pod restart counts;
- pod-to-node placement;
- Service names;
- Service selectors;
- Service ports;
- Service type;
- Service traffic-policy fields;
- endpoint counts;
- endpoint target relationships;
- ResourceQuota specifications;
- relevant ConfigMap specifications;
- relevant Secret metadata;
- warning Event reason;
- warning Event involved-object kind/name;
- normalized warning Event message;
- node role/topology information required by the candidate.

The implementation may collect additional fields.

Additional fields cannot replace the frozen minimum set.

## Candidate-specific semantic fields

P1 semantic fingerprint includes:

- exact faulty Service selector;
- intended `frontend` Deployment desired/ready state;
- unintended `search` Deployment desired/ready state;
- frontend-owned ready endpoint count;
- search-owned ready endpoint count;
- unidentified ready endpoint count, required to be zero;
- frontend Service port and target-port mapping;
- `mixed_endpoint_contamination = true`.

Generated Pod/ReplicaSet identities are excluded only according
to the frozen ephemeral-identity canonicalization rules.

P2 semantic fingerprint includes:

- Service internalTrafficPolicy;
- backend worker rank;
- caller worker rank;
- backend/caller distinctness;
- relevant nodeSelectors;
- backend Running/Ready state;
- caller Running/Ready state;
- endpoint count.

P3 semantic fingerprint includes:

- Valkey availability state;
- `unauthenticated_ping_requires_auth`;
- `injected_password_ping_returns_pong`;
- cart desired/ready state;
- `cart_replacement_observed`;
- endpoint state relevant to the cart/Valkey dependency.

P4 semantic fingerprint includes:

- exact quota hard-memory specification;
- search desired/ready/unavailable state;
- `quota_admission_failure`;
- active quota identity.

The particular raw P4 evidence source (ReplicaSet condition, Event, or both)
is retained for audit but is not itself treated as a different failure
mechanism when the frozen semantic predicate is satisfied.

## Frozen ephemeral-field exclusions

The following may be excluded from the canonical structural digest while
remaining present in raw evidence:

- `metadata.uid`;
- `metadata.resourceVersion`;
- `metadata.managedFields`;
- creation timestamps;
- deletion timestamps;
- event timestamps;
- pod-generated name suffixes;
- ReplicaSet hash suffixes;
- automatically assigned pod IP addresses;
- automatically assigned Service ClusterIP values;
- automatically generated container IDs;
- automatically generated image runtime IDs.

These fields are excluded because they are runtime identities rather than the
fault semantics under study.

They must remain available in raw evidence for audit.

`metadata.generation` is recorded and is not silently discarded from raw
evidence.

## Node identity normalization

Absolute node names are retained in raw evidence.

For semantic comparison, worker nodes are additionally assigned deterministic
ranks from lexicographically sorted worker-node names:

`worker-0`
`worker-1`
`worker-2`

Topology-sensitive comparisons use both:

- raw node identity;
- normalized worker rank.

This prevents generated node naming from being mistaken for a scientific
difference while preserving the actual placement evidence.

## No fuzzy post-hoc similarity score

Gate 2 does not use an outcome-tuned weighted similarity score.

There is no learned distance threshold.

There is no manually adjusted "close enough" percentage after fingerprints
are observed.

Equivalence is determined by the frozen hard rules below.

## Episode classification

Every scheduled episode receives exactly one terminal classification:

`VALID_RECREATION`

`RECREATION_FAILURE`

`INFRASTRUCTURE_FAILURE`

`PROTOCOL_VIOLATION`

### VALID_RECREATION

The problem deploys, the frozen fault injector completes, the manifestation
predicate reaches quiescence, all required fingerprint evidence is captured,
and the post-episode cleanup/carryover barrier passes.

The terminal classification is not assigned until the cleanup barrier has
been evaluated.

### RECREATION_FAILURE

The infrastructure remains usable but any of the following occurs:

- the intended problem/fault cannot reach or maintain the frozen
  manifestation predicate;
- required semantic state is materially different;
- required fingerprint evidence cannot be captured for benchmark-state
  reasons;
- frozen benchmark cleanup completes or remains callable but leaves
  candidate/application residue that violates the cleanup/carryover barrier.

### INFRASTRUCTURE_FAILURE

Execution cannot fairly reach or complete the incident test because of an
external campaign-level failure such as Docker/KIND failure, host exhaustion,
corrupted cluster infrastructure, or equivalent external failure not caused
by the candidate.

An external failure occurring during the cleanup barrier may also receive
this classification when the barrier cannot be evaluated fairly.

Infrastructure failures are retained.

They are not replaced with a new episode under the same frozen schedule.

### PROTOCOL_VIOLATION

The frozen source, cluster contract, order, profile, baseline policy, noise
policy, evidence boundary, or model/agent prohibition is violated.

A protocol violation invalidates interpretation of the campaign until a
versioned decision is made.

## Pairwise equivalence rule

For each candidate, compare every pair of `VALID_RECREATION` semantic
fingerprints.

With eight valid episodes this produces:

`C(8,2) = 28`

pairwise comparisons per candidate.

A pair is semantically equivalent only when every frozen candidate-specific
semantic field satisfies its exact rule.

No majority vote is used.

## Candidate Gate 2 decision

A candidate receives:

`PASS`

only if all eight scheduled episodes are `VALID_RECREATION` and all 28
pairwise semantic comparisons pass.

A candidate receives:

`FAIL_RECREATION`

if any scheduled episode reaches the candidate test but is classified
`RECREATION_FAILURE`, or any required pairwise semantic comparison fails.

A candidate receives:

`INCONCLUSIVE_INFRASTRUCTURE`

if an infrastructure failure prevents all eight scheduled incident
recreations from being evaluated and no recreation failure has already
established failure.

Infrastructure-failed episodes are not silently replaced.

A candidate receives:

`INVALID_PROTOCOL`

if a protocol violation affects its interpretation.

## Overall Gate 2 decision

SREGym receives:

`GATE2_PASS_FOR_FROZEN_CANDIDATE_SET`

only if P1, P2, P3, and P4 all receive `PASS`.

If any candidate receives `FAIL_RECREATION`:

`GATE2_FAIL_MATCHED_RESTART_FEASIBILITY`

for the frozen four-problem design.

If no candidate fails recreation but one or more are
`INCONCLUSIVE_INFRASTRUCTURE`:

`GATE2_INCONCLUSIVE_INFRASTRUCTURE`

If a protocol violation prevents valid interpretation:

`GATE2_INVALID_PROTOCOL`

No candidate may be dropped after failure merely to obtain an overall pass.

A future narrower or revised design requires a new versioned gate with the
failed evidence retained.

## No rerun-to-success rule

No scheduled episode may be silently rerun because:

- the fingerprint differed;
- the fault did not manifest;
- the incident looked unusual;
- later episodes were cleaner;
- a preferred result would change;
- the candidate would otherwise fail.

Any repeat performed for infrastructure debugging must receive a separate
debug identity and is excluded from the frozen Gate 2 decision unless a new
protocol version explicitly preregisters its use before outcomes.

## Evidence preservation

For every episode preserve:

- stdout/stderr;
- execution timestamps;
- provenance;
- raw fingerprint evidence;
- canonical fingerprint;
- semantic fingerprint;
- fingerprint SHA-256;
- episode terminal classification;
- failure details;
- cleanup result;
- post-cleanup/carryover barrier evidence.

At the campaign level preserve:

- frozen schedule;
- protocol file and SHA-256;
- source SHAs;
- environment versions;
- all episode manifests;
- an artifact SHA-256 manifest.

Failed and invalid episodes are evidence.

They are never deleted from the research record.

## What Gate 2 may establish

A successful Gate 2 may establish only that, for the frozen SREGym revision,
cluster configuration, temporal policy, and four selected incidents, repeated
fresh recreations satisfied this preregistered matched-restart equivalence
test.

## What Gate 2 cannot establish

Gate 2 cannot establish that:

- SREGym branches are identical;
- SREGym is deterministic;
- SREGym provides exact causal branch replay;
- all SREGym problems recreate equivalently;
- a non-zero baseline-window experiment will recreate equivalently;
- recovery consumers receive identical hidden state;
- receiver dependence exists;
- TRACE improves recovery;
- TRACE is a better attributor;
- conventional attribution accuracy is insufficient;
- any model or agent performs better;
- the TRACE v2 novelty thesis is confirmed.

## Gate after Gate 2

Only after Gate 2 passes may TRACE v2 design the SREGym matched-restart
receiver experiment.

Before any recovery outcome is observed, that later gate must freeze:

- recovery consumers;
- diagnosis treatment construction;
- exact diagnosis-byte reuse;
- receiver assignment;
- recovery-arm ordering;
- model and snapshot provenance;
- token/time budgets;
- baseline-window policy for the outcome study;
- outcome metrics;
- exclusion policy;
- statistical analysis;
- multiplicity handling;
- stopping rule.

If that later experiment uses a non-zero SREGym steady-state window, incident
recreation must first be revalidated under that frozen duration.

## October 15 source amendment and validity limits

This amendment was made before any Gate 2 recreation outcomes.

It replaces an obsolete candidate and incorporates upstream fault-
validity, lifecycle, runtime, and application changes.

The selection was not random and its scope remains four
mechanism-stratified incidents.

Passing eight recreations for a candidate will support only the
preregistered semantic-state equivalence claim.

It will not prove equivalence of unobserved process memory, request
history, network timing, random state, or all information available
to a subsequent recovery consumer.

Before implementation, confirm that the collector's permitted
read-only probes do not themselves alter the fault's effective state.

## Implementation gate before execution

This document freezes the scientific contract.

It does not authorize immediate execution.

Before creating the Gate 2 campaign cluster, implementation must provide:

1. a read-only fingerprint collector;
2. deterministic canonicalization;
3. candidate-specific manifestation predicates;
4. schema validation;
5. unit tests for ephemeral-field removal;
6. unit tests proving material semantic changes alter the fingerprint;
7. unit tests proving permitted ephemeral identity changes do not alter the
   semantic fingerprint;
8. a dry-run provenance check;
9. a hard guard preventing model/judge/agent invocation;
10. unit tests for the frozen P3 two-PING read-only authentication probe;
11. unit tests for both frozen P4 quota-admission evidence paths;
12. unit tests for the between-episode cleanup/carryover barrier;
13. unit tests requiring both frontend-owned and search-owned ready P1
    endpoints, including negative tests for empty or unmapped endpoints;
14. unit tests proving P2's 65-second effective propagation interval is
    applied exactly once before fingerprint capture;
15. checks for the frozen SREGym applications-submodule revision;
16. a final targeted upstream delta and maintainer-issue recheck.

No incident recreation may execute until that implementation gate passes.
