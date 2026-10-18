# TRACE Gate-2 — P3 endpoint comparator proposal (offline only)

Status: **NOT FROZEN. NOT AN EXPERIMENTAL DECISION RULE.**

Parent frozen protocol: `docs/TRACE_V2_SREGYM_GATE2_PROTOCOL.md`, SHA-256
`a5e77dbbc08869f07fdd1f2e5c2b4a7e343e44fe970913ae6e908611c3b0d562`.

Current implementation starting point: TRACE commit `9134519`.

Pinned sources: SREGym `9093167a7082402bf553c6efcda1c9dd48dd5138`,
applications `b2006cb6458785c1b353d0fd1753db7fe3a30795`,
OpenTelemetry Helm chart `7d7b074714345a0c282b0be75af7a2c504b44c95`,
Kubernetes KIND base `v1.32.1`.

## What this slice tests

- Pure-function cart/valkey-cart EndpointSlice aggregation and deduplication by
  network endpoint (Service, address family, IP, port, protocol).
- Fail-closed handling of conflicting duplicate endpoint conditions or Pod targets.
- Exact `ready`, `serving`, `terminating` boolean triples; no collapse to only
  ready/not-ready counters.
- Specifically for **Kubernetes v1.32.1**, nil `serving` falls back to the
  effective `ready` value. Nil `ready` is interpreted as true and nil
  `terminating` as false. This differs from the current online API reference's
  nil-`serving` phrasing; for our pinned environment use the pinned code.
- Per-Service `publishNotReadyAddresses`, ports, and address-family distinctions.
- UID/IP invariance of the resulting histogram while preserving raw identities
  as prerequisites for independent Layer-A validation.
- Strict recursive typed equality; booleans are not integers.

Pinned official source:
https://github.com/kubernetes/kubernetes/blob/v1.32.1/staging/src/k8s.io/api/discovery/v1/types.go

Kubernetes 1.32 EndpointSlice deduplication guidance:
https://v1-32.docs.kubernetes.io/docs/concepts/services-networking/endpoint-slices/

## Boundaries / deliberately unresolved

The normalizer takes in-memory dictionaries. It does **not** authenticate
captures, enforce LIST completeness, prove watch continuity, determine an
observation bracket, check rollout generation, validate total application
health, or classify an episode. A legitimate empty result is not proof that a
network backend has no endpoints without a separately validated complete LIST.

No P1–P4 incident data was inspected. No model, cluster, fault injection, helm
install, kubectl, or network connection is involved in the test suite.

This is an implementation proposal for independent review and falsification.
The following decisions remain open in the measurement addendum: complete
P1–P4 field schemas, observation/quiescence stability, final capture rules,
measurement failure precedence, watch continuity, retry budget and claim
limitations. Do not freeze or classify Gate-2 PASS based on this module alone.

The candidate's pre-existing Layer-C field set must not be silently expanded
by this prototype. `publishNotReadyAddresses`, port, and address-family fields
are proposals, and their inclusion in any PASS comparator requires a dated,
prospective rule confirmation.

## Supplemental adversarial review — 2026-10-18

Two additional synthetic cases were tested independently:

1. Changing cart Service `spec.selector` to a nonmatching
   workload was accepted and preserved the same proposed semantic
   result. Service-selector configuration is not represented in
   this proposed endpoint histogram.
2. Changing an EndpointSlice port name to disagree with the
   corresponding Service port name was accepted and produced
   a different proposed semantic result.

These characterize the current normalizer's boundary. They do
not establish an experimental failure or justify silently changing
the preregistered candidate-specific semantic fields.

Before experimental execution, the independent evidence
admissibility and structural-validation path must address
cross-resource Service/EndpointSlice consistency or explicitly
document any excluded differences.

This module remains a prospective proposal, not a frozen
Gate-2 decision rule.
