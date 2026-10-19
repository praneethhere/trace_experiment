# TRACE v2 Gate-2 measurement-decision prototype (2026-10-18)

**Status: PROPOSED; NOT FROZEN; NO EXPERIMENTS AUTHORIZED.**

Parent protocol: `docs/TRACE_V2_SREGYM_GATE2_PROTOCOL.md`, committed at `f039049`, SHA-256 `a5e77dbbc08869f07fdd1f2e5c2b4a7e343e44fe970913ae6e908611c3b0d562`.

Implementation checkpoint before this prototype: `2736527`.

The original protocol already has exactly four terminal episode classes. This proposal adds a separate reason-code vocabulary, **not a fifth terminal class**. `terminal=None` means not enough evidence has been recorded to classify the episode; it is never promoted to `VALID_RECREATION`. It is not a campaign decision.

## Proposed precedence, not yet approved

1. A protocol/evidence-contract breach, collector failure, or unclosed WATCH gap yields `PROTOCOL_VIOLATION` and retains all other observed failures.
2. Otherwise, a verified benchmark fault-manifestation failure or residue yields `RECREATION_FAILURE`.
3. Otherwise, an external Docker/KIND/host failure yields `INFRASTRUCTURE_FAILURE`.
4. `VALID_RECREATION` is proposed only when injection completes, three observations satisfy the predicate, the final capture independently satisfies the predicate, raw evidence is admissible, and the cleanup barrier passes. An unknown outcome is **pending**, not automatically successful.

Unresolved scientific choice: whether collector-origin failure should invalidate a previously verified recreation failure (this prototype takes the conservative approach). A watch loss caused *solely* by a proven external cluster outage might be infrastructure rather than protocol; the root-cause/precedence policy must be explicitly decided before freezing. The required final-capture manifestation recheck is also a **prospective strengthening** not required verbatim by the parent, so it must be documented as an amendment.

## Typed comparison

The standalone function compares *caller-declared* complete Layer-C field sets; it does not infer P1-P4 field lists. It preserves booleans vs integers, rejects missing/null/float/unexpected fields, and reports exact mismatch paths. Dictionaries ignore key order. Lists remain order-sensitive; unordered structures **must** be explicitly canonicalized by a candidate-specific rule before calling this function. This is NOT an experiment-facing validator and does not prove raw capture admissibility.

## Scientific blockers after this offline review

- Freeze exact P1-P4 Layer-C schema and make explicit whether P3 proposed port/address-family/condition histogram entries extend Layer C or belong only to structural Layer B.
- Decide quiescence semantic stability, final capture recheck and exact observation/retry/watch budget.
- Resolve failure precedence from both a scientific and operational standpoint.
- Independently adversarially review the finished addendum and source-specific tests.
- Implement read-only collection, provenance, no-agent guard, the remaining candidate validators, and full 16-item implementation gate.

No files from this package should be wired into an episode runner until the addendum is frozen and independently reviewed. The test count is engineering evidence only, not experimental replication.

## Independent review follow-up — 2026-10-18

Two additional limitations were reproduced offline.

### Nested-schema completeness

The generic comparison function accepts:

    left={"deployment": {}}
    right={"deployment": {}}
    required_fields=("deployment",)

as admissible and equivalent.

This is not sufficient for an experiment-facing comparator.
P1-P4 must each have a frozen complete nested schema, with
explicit required types and fields. Identical missing evidence
cannot be accepted as equivalence.

### Confounded competing failures

When both EXTERNAL_DOCKER_KIND_HOST_FAILURE and
MANIFESTATION_TIMEOUT are present, the prototype selects
RECREATION_FAILURE.

That decision is not scientifically justified if the timeout
was caused by the infrastructure outage.

Before freezing, the classification contract must distinguish
independently established benchmark failures from symptoms
caused by infrastructure or measurement failure. Event presence
alone does not establish causality.

### Status

Both findings are prospective methodological blockers.

They do not invalidate the offline tests, authorize changing
the frozen parent protocol, or establish a Gate-2 result.

The proposal remains disconnected from live classification.
