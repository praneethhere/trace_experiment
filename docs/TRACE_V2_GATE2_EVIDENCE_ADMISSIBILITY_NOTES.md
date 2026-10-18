# TRACE v2 Gate 2 — Offline evidence-contract slice

**Status:** implementation proposal for review, **not** an approved experimental collector.

**Intended base commit:** `90c2018` on `trace-v2-attribution-study`.

**Scientific contract:** `docs/TRACE_V2_SREGYM_GATE2_PROTOCOL.md`, frozen at `f039049`; verified SHA-256 `a5e77dbbc08869f07fdd1f2e5c2b4a7e343e44fe970913ae6e908611c3b0d562`.

## What is implemented

- Offline validator for recorded raw response JSON strings and SHA-256 digests, with explicit schema and provenance fields.
- Exact expected SREGym and applications revision checks; expected campaign/episode/cluster identity is passed separately by the caller.
- Namespaced Kubernetes API read contract for Namespace, Deployment, ReplicaSet, Pod, Service, Endpoints, and EndpointSlice.
- Read-only unfiltered, non-paginated "most recent" LIST **request description** checks; all submitted responses must be complete single-page lists.
- Kubernetes kind and API version checks, namespace/name/UID/RV checks, duplicate UID and Running pod-IP detection, absent `spec.replicas` detection. Duplicate Running pod IPs are permitted only for explicit `hostNetwork` Pods on the same named node, because those Pods legitimately share that node IP.
- Stale Deployment generation or suspicious readiness counts are recorded as **unstable observations**, not silently interpreted as benchmark recreation failures.
- A two-capture **observed-object UID/RV comparison**, reporting only whether the observed items match. It does **not** prove event-free intervals.
- 21 synthetic tests, including negative cases and a positive format example.

## Deliberate non-capabilities and validity boundaries

`CaptureAudit.schema_valid=True` means that a proposed **recording format** satisfied its offline checks. It does **not** prove the cluster really returned those bytes, or that a malicious recorder did not forge the entire JSON and recompute the hash. The capture contract does not attest to the runtime environment.

`CaptureAudit.execution_admissible=False` **always**. There is no trusted collector or watch-continuity proof. `BracketAudit.covers_intervening_events=False` and `.cross_kind_atomic=False` **always**. Two stable UID/RV sets cannot detect a resource created and deleted between the reads, nor can Kubernetes provide a single atomic read spanning different resource kinds.

The module does not alter the existing `evaluate_p1()` meaning or call it. It does not assign a terminal episode class, authenticate the recorder, observe or modify a Kubernetes cluster, invoke an agent/model, or run P1–P4.

This module intentionally does **not** make a rule on whether a Deployment-ready count must equal Pods Ready across separate API lists. It surfaces generation/readiness anomalies for a later preregistered observation-consistency policy.

The expected request contract is purposely narrow (`resourceVersion=""`, no pagination or filtering). A future collector must issue/read requests accordingly and preserve exact response body bytes and request provenance. This version uses raw JSON strings, not a wire-level HTTP capture; a later trusted collector must define its capture boundary.

**Important:** A later dated *pre-outcome* protocol addendum is required to resolve per-field equivalence comparators, instrument failure outcomes, observation retry budget, and the role of EndpointSlice agreement/rollout completion before any Gate 2 incident begins. Do not silently revise the frozen protocol.

## How to test locally

```bash
python3 -m unittest discover -s tests -p 'test_gate2_evidence.py' -v
python3 -m unittest discover -s tests -p 'test_gate2_p1_fingerprint.py' -v
```

Both sets must pass. Keep the new files **uncommitted** pending local review.
