# TRACE v2 — Gate 2 Measurement Addendum

Date: 2026-10-18

Status: DRAFT — NOT FROZEN — NOT AUTHORIZED FOR EXECUTION

## Relationship to preregistration

Parent protocol:
`docs/TRACE_V2_SREGYM_GATE2_PROTOCOL.md`

Parent commit: `f039049`

Parent SHA-256:
`a5e77dbbc08869f07fdd1f2e5c2b4a7e343e44fe970913ae6e908611c3b0d562`

Evidence-layer checkpoint: `9134519`.
P3 comparator proposal commit: `2736527`.

This is a prospective measurement clarification following
independent adversarial review.

No Gate-2 incident-recreation or recovery outcomes have been observed.

This draft does not change the original frozen document.

## 1. Semantic equivalence — proposed common rule

Compare only fields explicitly included in each candidate's
preregistered Layer-C semantic fingerprint.

Proposed default: exact equality of typed canonical values.

- Booleans compare as booleans, never as integers.
- Counts compare as nonnegative integers.
- String-valued configuration fields compare exactly.
- Dictionaries compare by keys and values, independent of key order.
- Sets compare after deterministic ordering.
- Lists compare in canonical order only when order is not scientifically meaningful.
- Missing fields, null values and unrecognized states do not silently
  acquire defaults.
- No floating-point tolerance or fuzzy threshold is permitted.
- Runtime-generated identities excluded by the parent protocol remain
  available in raw evidence.

A fingerprint hash is an implementation of these rules, not a
substitute for defining them.

## 2. Candidate-specific rules requiring final review

P1:
- Exact frontend Service selector.
- Exact frontend and search desired/ready counts.
- Exact frontend-owned, search-owned and unidentified ready endpoint counts.
- Exact normalized Service port/target-port/protocol mapping.
- Exact mixed-endpoint contamination boolean.

P2:
- Exact internalTrafficPolicy.
- Exact normalized backend and caller worker ranks.
- Exact backend/caller distinctness.
- Exact normalized workload nodeSelector mappings.
- Exact backend and caller Running/Ready states.
- Exact relevant endpoint count.

P3:
- Exact Valkey availability state.
- Exact unauthenticated/authenticated PING predicates.
- Exact cart desired/ready counts.
- Exact cart-replacement-observed predicate.
- Dependency endpoint-state schema: PROPOSED — NOT YET FROZEN.

  Source-verified identities:
  - SREGym execution: 9093167a7082402bf553c6efcda1c9dd48dd5138
  - Applications: b2006cb6458785c1b353d0fd1753db7fe3a30795
  - Astronomy chart: 7d7b074714345a0c282b0be75af7a2c504b44c95
  - cart connects to valkey-cart:6379.
  - cart Service uses port 8080.
  - valkey-cart Service uses port 6379.
  - The cart UID replacement witness uses app.kubernetes.io/name=cart.
  - Helm Service endpoint selection uses opentelemetry.io/name.

  Proposed Layer-C endpoint-state representation:
  STATUS: PROSPECTIVE — NOT FROZEN.

  Scope:
  - Obtain a complete, unfiltered namespace EndpointSlice LIST.
  - Identify cart and valkey-cart independently by the
    kubernetes.io/service-name label.
  - Preserve the associated Service objects and their
    publishNotReadyAddresses settings.
  - Retain all raw EndpointSlice metadata, addresses, ports,
    target references and condition fields in Layer A.

  Condition interpretation under Kubernetes v1.32:
  - ready omitted/null: interpret as true.
  - serving omitted/null: defer to the effective ready value (Kubernetes v1.32.1 EndpointConditions).
  - terminating omitted/null: interpret as false.
  - Explicit booleans must remain booleans.
  - Retain original missing/null representation in raw evidence.
  - Do not infer serving from ready: publishNotReadyAddresses
    can make ready=true despite different serving state.

  Deduplication proposal:
  - Aggregate all EndpointSlices for each relevant Service.
  - Before counting, identify network endpoints using raw
    namespace, Service, address family, address, port and protocol.
  - Deduplicate network endpoints repeated across slices.
  - Conflicting states or target references for an identical
    network endpoint require an explicit inconsistency result.
  - Preserve Pod target UID relationships for raw validation.
  - Never include generated IPs or Pod UIDs in Layer C.

  Semantic representation proposal:
  - For each Service, retain publishNotReadyAddresses as a
    typed boolean.
  - Produce a typed histogram keyed by each observed
    (ready, serving, terminating) boolean triple.
  - Preserve Service-port/protocol and address-family
    distinctions in the semantic representation.
  - Compare typed canonical values exactly across episodes.
  - Do not collapse the eight possible condition triples into
    only ready/unready or terminating/not-terminating counts.
  - Keep a complete, valid, zero-endpoint observation distinct
    from an incomplete or failed EndpointSlice LIST.
  - Authentication failure must not be equated with missing
    or unready network endpoints.

  Outstanding implementation validation:
  - Test duplicate endpoints across slices.
  - Test conflicting duplicate states and target references.
  - Test all three omitted-condition defaults.
  - Test publishNotReadyAddresses behavior.
  - Test endpoint reordering and generated IP changes.
  - Test changes to ports, condition triples and address family.
  - Verify applicability to the effective pinned Helm render.
  - Review whether per-network-endpoint counts sufficiently
    represent the cart/Valkey dependency or whether unique
    backing-Pod counts must also be reported.

  Kubernetes v1.32 references:
  https://v1-32.docs.kubernetes.io/docs/concepts/services-networking/endpoint-slices/
  https://kubernetes.io/docs/reference/kubernetes-api/discovery/endpoint-slice-v1/

  Source limitations:
  - The exact SREGym Helm render has not yet been independently validated.
  - EndpointSlice aggregation and Kubernetes condition normalization
    require adversarial tests before this proposal can be frozen.
  - The source-verified identity does not establish runtime availability.

  Fault-specific evidence:
  - Preserve pre/post cart UID sets to establish replacement.
  - Preserve Valkey restart counts before and after read-only probes.
  - Preserve unauthenticated NOAUTH and authenticated PONG evidence.
  - Preserve the upstream injected-password verification outcome.
  - A Valkey restart that erases the injected fault cannot count as
    successful authentication-state manifestation.


P4:
- Exact ResourceQuota hard-memory specification.
- Exact search desired/ready/unavailable counts.
- Exact quota-admission-failure boolean.
- Quota identity compares stable namespace/name, not generated UID.
- ReplicaSet condition versus Event evidence path remains recorded
  in raw evidence but does not distinguish semantic mechanisms.

Before freezing this addendum, verify P2/P3/P4 value schemas and
normalizations directly against the pinned SREGym implementations.

## 3. Quiescence and capture consistency — OPEN DECISION

The frozen protocol requires three consecutive true manifestation
observations separated by two seconds, within 120 seconds.

Open question:
Should those observations also have equal candidate-specific
semantic fingerprints, and should the final raw capture be required
to satisfy the manifestation predicate again?

Potential benefit:
Avoid a PASS built from an unstable or outdated pre-recovery state.

Potential cost:
A stricter rule changes which recreations can pass.

Do not implement this additional requirement until a versioned
pre-outcome decision is explicitly documented.

## 4. Measurement failure classification — OPEN DECISION

The parent protocol has exactly four terminal episode classes:

- VALID_RECREATION
- RECREATION_FAILURE
- INFRASTRUCTURE_FAILURE
- PROTOCOL_VIOLATION

Collector failure, incomplete LIST capture, watch-continuity failure,
and genuine benchmark fault non-manifestation must not be conflated.

Before executing incidents, determine whether measurement failures
use an existing parent class or require a prospectively defined fifth
class with explicit candidate and overall decision precedence.

Any newly introduced terminal class changes the frozen decision
contract and requires an explicit dated amendment.

Failed measurements must remain in the evidence record.

No repeat-to-success, unscheduled replacement, or outcome-dependent
extension of the observation budget is permitted.

## 5. Layer-B differences

Preserve all required raw and canonical structural fields.

Proposed reporting:
- Enumerate differences by candidate and episode pair.
- Identify changes in placement, rollouts, restart history, images,
  unrelated workload health, traffic policy and EndpointSlices.
- Distinguish excluded ephemeral identities from material state.
- Do not silently add Layer-B equality to the frozen Layer-C PASS rule.

This reporting cannot justify claims of complete recovery-starting-
state equivalence without a separately designed evaluation.

## 6. Observation-consistency requirements

The collector must preserve raw request and response evidence,
resource versions, relevant timestamps and provenance.

Do not assume an atomic Kubernetes snapshot across resource kinds.

A successful two-capture UID/resourceVersion comparison alone does
not prove that no intervening resource was created and deleted.

Watch continuity, read failures and retry limits require a
pre-outcome operational specification and synthetic fault tests.

The currently committed offline evidence module deliberately
does not authorize experimental execution.

## 7. Research claims

Passing Gate 2 supports only the approved, narrowly defined
feasibility/equivalence test at the frozen source revision.

It does not prove deterministic replay, identical hidden state,
causal attribution accuracy, improved agent recovery, or novelty.

Eight recreations represent eight observations, not 28 independent
samples merely because 28 pairwise comparisons are performed.

## 8. Freeze blockers

This document must remain a draft until:

1. P1-P4 comparator schemas are complete and source-verified.
2. Quiescence/final-capture consistency policy is decided.
3. Measurement-failure classification and precedence are decided.
4. Retry and watch-continuity rules are unambiguous.
5. Tests demonstrate comparator and classification behavior.
6. The original protocol remains intact and deviations are recorded.
7. The independent methodological review is reconciled.
8. No Gate-2 outcome has been observed before amendment freeze.

## 9. Prospective decision-resolution review

STATUS: PROPOSED — NOT FROZEN — NOT EXECUTION-AUTHORIZING.

These recommendations require an explicit pre-outcome
methodological decision. They do not silently amend the parent.

### R1. Primary equivalence claim

Propose defining a Gate-2 PASS as equivalence under the
predeclared candidate-specific Layer-C fields only.

Layer-B structural differences must be reported independently.

The parent research question mentions structural equivalence,
but the formal pairwise rule only compares Layer C.
This discrepancy must be explicitly resolved before execution.

No claim of complete recovery-starting-state equivalence
is authorized merely because Layer C matches.

### R2. Candidate comparators

Propose exact, typed canonical equality for every required
field enumerated in the parent protocol.

P1: retain the existing frozen semantic fields.
P2: worker ranks and other required topology relations
must use the frozen rank-normalization rules.
P3: core authentication and cart replacement fields remain
mandatory. Decide explicitly which proposed EndpointSlice
histogram fields belong in Layer C versus Layer B.
P4: retain exact quota and admission-state requirements.

Absent or malformed evidence cannot silently receive defaults.

### R3. Quiescence

Retain the parent requirement of three consecutive
true manifestation observations, two seconds apart,
within the 120-second harness-side boundary.

Equal semantic hashes across those observations are NOT
currently required by the parent. Adding that requirement
would be a prospective protocol change.

Propose checking that the final captured evidence still
supports the manifestation predicate, without introducing
an additional settling interval.

### R4. Measurement-system failures

Retain the parent's four terminal episode classes.

Propose distinguishing measurement-system failure using
a reason code under PROTOCOL_VIOLATION when an inadmissible
capture prevents trustworthy interpretation.

Genuine benchmark non-manifestation remains
RECREATION_FAILURE.

External Docker, KIND, or host failure remains
INFRASTRUCTURE_FAILURE.

The exact precedence for multiple observed failure causes
must be specified and tested before freezing.

No affected scheduled episode may be silently replaced.

### R5. Collection and retry budget

No retry-to-success is permitted.

Before freezing, specify an exact collector read sequence,
watch-continuity requirement, timeout and fixed retry budget.

A matching two-capture resourceVersion bracket alone
cannot establish atomic state across Kubernetes kinds.

Test the proposed method on synthetic observations and
a separate non-candidate environment before any Gate-2 fault.

### R6. Required adversarial decisions

Before freezing, test:

- exact typed equality and missing semantic fields;
- P3 endpoint conditions, duplication and excluded configuration;
- persistent predicate truth with changing semantic values;
- loss of predicate truth at final capture;
- failed LIST, stale data and watch-continuity failure;
- measurement-system versus recreation versus infrastructure failure;
- competing terminal failure causes and campaign precedence;
- Layer-B divergence despite equal Layer-C fingerprints.

No P1-P4 incident execution is authorized by these tests.
