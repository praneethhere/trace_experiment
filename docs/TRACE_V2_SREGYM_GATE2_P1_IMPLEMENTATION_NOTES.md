# TRACE v2 Gate 2 — P1 offline validator, first implementation slice

Frozen protocol: `docs/TRACE_V2_SREGYM_GATE2_PROTOCOL.md`, commit `f039049`.

This patch adds `trace_gate2/p1.py` and `tests/test_gate2_p1_fingerprint.py`.

## Purpose

Offline validation of the candidate-specific **P1 manifestation predicate and semantic digest** for `service_wrong_pod_selection_hotel_reservation`. Tests use synthetic Kubernetes JSON objects. Nothing connects to Kubernetes, SREGym, LLMs, or a network.

## Run

From your `trace_experiment` repository root:

```sh
python3 -m unittest discover -s tests -p 'test_gate2_p1_fingerprint.py' -v
```

## Strict limits

- This is **not** the full `Layer A` raw archive, `Layer B` structural canonicalizer, or the Gate-2 campaign harness.
- The caller supplies a snapshot with `namespace`, and complete `deployments`, `replicasets`, `pods`, `services`, and `endpoints` Kubernetes JSON lists, each object with `.metadata.namespace`.
- The owner-reference chain must be explicit (Pod -> ReplicaSet -> Deployment), with matching UIDs. Name/label pattern guessing is forbidden.
- The P1 protocol's mixed ready endpoint ownership must be demonstrable, not inferred from selectors alone.
- Endpoints must explicitly match Pod UIDs and IPs, and expose the frontend TCP target port. Missing or inconsistent evidence fails closed.
- `P1Result.valid_manifestation` is a *single-observation* predicate. Three observations 2s apart, the 120s bound, and the per-episode lifecycle are **not** implemented.
- The semantic hash only considers the fields in `P1Result.semantic`; it makes **no claim** that hidden application/cluster state is equivalent.
- The 0-model-call and no-mutation constraints are implemented by this module's complete lack of model/cluster imports or calls, **not** a campaign-wide runtime guard.
- We intentionally have not written a Kubernetes collector: building that before proving the offline schema can obscure what the validator is allowed to infer.

## Unresolved integration gates

1. Validate real capture schema completeness and resource versions. Snapshot must not silently convert API reads to empty lists.
2. Add raw-evidence preservation and general structural canonicalization.
3. Implement P2, P3, P4 predicates and negative controls; include non-mutating P3 probes and strict P4 event ownership checks.
4. Implement quiescence, SREGym cleanup lifecycle and 180s barrier; test that failures cannot be retried into success.
5. Add pinned-source/submodule, propagation and forbidden agent/model call guards, plus final pre-execution upstream recheck.

No incident recreation is authorized by this patch.
