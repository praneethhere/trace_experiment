"""Prospective, offline-only Kubernetes v1.32.1 Deployment status projection.

This is a narrow decoder for the P4 pre-recovery measurement design. It does
NOT validate capture provenance, authorize an episode, or define any Gate-2
PASS/FAIL rule. Only the explicitly enumerated apps/v1 status integer fields
with Go's `omitempty` are eligible for absent -> 0 decoding.

Source: kubernetes/kubernetes v1.32.1,
  staging/src/k8s.io/api/apps/v1/types.go, DeploymentStatus.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

STATUS_OMITEMPTY_COUNTERS = (
    "replicas",
    "updatedReplicas",
    "readyReplicas",
    "availableReplicas",
    "unavailableReplicas",
)


@dataclass(frozen=True)
class DeploymentStatusProjection:
    schema_valid: bool
    status_current: bool
    reasons: tuple[str, ...]
    name: str | None
    namespace: str | None
    desired: int | None
    counters: dict[str, int] | None
    omitted_zero_fields: tuple[str, ...]
    # This never attests to live evidence or makes an execution decision.
    execution_admissible: bool = False

    @property
    def p4_replica_fields(self) -> dict[str, int] | None:
        """Only return candidate-relevant integers for current, valid status."""
        if not self.schema_valid or not self.status_current:
            return None
        assert self.desired is not None and self.counters is not None
        return {
            "desired": self.desired,
            "ready": self.counters["readyReplicas"],
            "unavailable": self.counters["unavailableReplicas"],
        }


def _nonempty(v: Any) -> bool:
    return type(v) is str and bool(v.strip())


def _nonnegative_int(v: Any) -> bool:
    return type(v) is int and v >= 0


def decode_deployment_status_v132(resource: Any) -> DeploymentStatusProjection:
    """Decode one complete raw apps/v1 Deployment object without modifying it.

    A server-omitted numeric `status` counter is 0 under this pinned API's
    non-pointer integer `omitempty` rule. Explicit null is never treated as 0.
    A missing Deployment spec or absent/stale `observedGeneration` must not
    yield a usable semantic state.
    """
    problems: list[str] = []
    not_current: list[str] = []
    omitted: list[str] = []
    counts: dict[str, int] = {}
    name: str | None = None
    namespace: str | None = None
    desired: int | None = None

    if not isinstance(resource, dict):
        return DeploymentStatusProjection(False, False, ("resource: not an object",), None, None, None, None, ())

    if resource.get("kind") != "Deployment" or resource.get("apiVersion") != "apps/v1":
        problems.append("resource: expected apps/v1 Deployment")
    meta = resource.get("metadata")
    spec = resource.get("spec")
    status = resource.get("status")
    if not isinstance(meta, dict):
        problems.append("metadata: missing or invalid")
        meta = {}
    name = meta.get("name")
    namespace = meta.get("namespace")
    if not all(_nonempty(meta.get(k)) for k in ("uid", "name", "namespace")):
        problems.append("metadata: uid, name and namespace required")
    generation = meta.get("generation")
    if type(generation) is not int or generation < 1:
        problems.append("metadata.generation: positive integer required")
    if not isinstance(spec, dict):
        problems.append("spec: missing or invalid")
    else:
        v = spec.get("replicas")
        # Unlike optional *status* counters, missing spec.replicas is not
        # accepted for this candidate-specific raw-evidence contract.
        if not _nonnegative_int(v):
            problems.append("spec.replicas: explicit nonnegative integer required")
        else:
            desired = v
    if not isinstance(status, dict):
        problems.append("status: missing or invalid")
    else:
        observed = status.get("observedGeneration")
        if observed is None and "observedGeneration" not in status:
            not_current.append("status.observedGeneration: omitted / not yet observed")
        elif type(observed) is not int or observed < 0:
            problems.append("status.observedGeneration: invalid integer")
        elif type(generation) is int and generation > 0 and observed != generation:
            not_current.append("status.observedGeneration: differs from metadata.generation")
        elif type(generation) is not int or generation < 1:
            not_current.append("status: cannot verify generation")
        for field in STATUS_OMITEMPTY_COUNTERS:
            if field not in status:
                counts[field] = 0
                omitted.append(field)
            else:
                value = status[field]
                if not _nonnegative_int(value):
                    problems.append(f"status.{field}: invalid integer; not omitted")
                else:
                    counts[field] = value

    valid = not problems
    current = valid and not not_current
    # Information about stale status is retained; it does not silently become
    # a recreation failure or evidence of valid manifestation.
    return DeploymentStatusProjection(
        valid, current, tuple(problems + not_current),
        name if _nonempty(name) else None,
        namespace if _nonempty(namespace) else None,
        desired, counts if len(counts) == len(STATUS_OMITEMPTY_COUNTERS) else None,
        tuple(omitted),
    )
