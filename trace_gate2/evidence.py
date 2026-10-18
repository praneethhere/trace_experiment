"""Offline audit of recorded P1 Kubernetes reads, NOT proof of live authenticity.

This does not collect Kubernetes state, determine terminal episode outcomes, or
supply a cross-resource atomic snapshot. A future trusted collector/runner must
bind expected episode identity and verify watch continuity before using P1.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from typing import Any, Mapping

SCHEMA = "trace-gate2-offline-capture-v0"
PROTOCOL_SHA256 = "a5e77dbbc08869f07fdd1f2e5c2b4a7e343e44fe970913ae6e908611c3b0d562"
SREGYM_SHA = "9093167a7082402bf553c6efcda1c9dd48dd5138"
APPLICATIONS_SHA = "b2006cb6458785c1b353d0fd1753db7fe3a30795"
P1_ID = "service_wrong_pod_selection_hotel_reservation"

# Key -> expected Kubernetes LIST kind, apiVersion, resource kind, resource API version.
KINDS = {
    "deployments": ("DeploymentList", "apps/v1", "Deployment", "apps/v1"),
    "replicasets": ("ReplicaSetList", "apps/v1", "ReplicaSet", "apps/v1"),
    "pods": ("PodList", "v1", "Pod", "v1"),
    "services": ("ServiceList", "v1", "Service", "v1"),
    "endpoints": ("EndpointsList", "v1", "Endpoints", "v1"),
    "endpointslices": ("EndpointSliceList", "discovery.k8s.io/v1", "EndpointSlice", "discovery.k8s.io/v1"),
}

@dataclass(frozen=True)
class CaptureAudit:
    schema_valid: bool
    reasons: tuple[str, ...]
    unstable_reasons: tuple[str, ...]
    # Deliberately *never* true: no cryptographic attestation or WATCH gap audit.
    execution_admissible: bool = False

@dataclass(frozen=True)
class BracketAudit:
    same_observed_items: bool
    reasons: tuple[str, ...]
    covers_intervening_events: bool = False
    cross_kind_atomic: bool = False


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _raw_record(record: Any, label: str, errors: list[str]) -> dict | None:
    if not isinstance(record, dict):
        errors.append(f"{label}: missing record")
        return None
    raw = record.get("raw_json")
    sha = record.get("sha256")
    if not isinstance(raw, str) or not isinstance(sha, str):
        errors.append(f"{label}: raw_json and sha256 required")
        return None
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    if digest != sha:
        errors.append(f"{label}: raw response SHA-256 mismatch")
        return None
    if record.get("http_status") != 200:
        errors.append(f"{label}: HTTP status not 200")
    start, end = _timestamp(record.get("started_at_utc")), _timestamp(record.get("ended_at_utc"))
    if start is None or end is None or end < start:
        errors.append(f"{label}: invalid request timing")
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError):
        errors.append(f"{label}: invalid JSON body")
        return None
    if not isinstance(decoded, dict):
        errors.append(f"{label}: body must be object")
        return None
    return decoded


def audit_capture(capture: Any, *, expected_campaign_id: str,
                  expected_episode_id: str, expected_cluster_uid: str) -> CaptureAudit:
    """Fail-closed *format* audit. Expectation args must come from trusted runner.

    Not an authentication proof. Caller must not interpret schema_valid as a
    permission to classify a Gate-2 episode as VALID_RECREATION.
    """
    errors: list[str] = []
    unstable: list[str] = []
    if not isinstance(capture, dict):
        return CaptureAudit(False, ("capture must be object",), ())
    if capture.get("schema") != SCHEMA:
        errors.append("schema version mismatch")
    ident = capture.get("identity")
    if not isinstance(ident, dict):
        return CaptureAudit(False, tuple(errors + ["identity missing"]), ())
    mandatory = {
        "campaign_id": expected_campaign_id,
        "episode_id": expected_episode_id,
        "cluster_uid": expected_cluster_uid,
        "protocol_sha256": PROTOCOL_SHA256,
        "sregym_sha": SREGYM_SHA,
        "applications_sha": APPLICATIONS_SHA,
        "problem_id": P1_ID,
    }
    for key, expected in mandatory.items():
        if not _nonempty(expected) or ident.get(key) != expected:
            errors.append(f"identity.{key}: mismatch")
    for key in ("namespace", "namespace_uid", "trace_commit", "collector_version", "api_server"):
        if not _nonempty(ident.get(key)):
            errors.append(f"identity.{key}: required")
    ns = ident.get("namespace")
    records = capture.get("records")
    if not isinstance(records, dict):
        return CaptureAudit(False, tuple(errors + ["records missing"]), ())
    if set(records) != (set(KINDS) | {"namespace"}):
        errors.append("records: must contain exactly P1 required resource kinds and namespace")

    namespace_record = records.get("namespace")
    if isinstance(namespace_record, dict):
        request = namespace_record.get("request")
        if request != {"method": "GET", "path": f"/api/v1/namespaces/{ns}", "query": {}}:
            errors.append("namespace: request mismatch")
    namespace_body = _raw_record(namespace_record, "namespace", errors)
    if namespace_body is not None:
        meta = namespace_body.get("metadata")
        if (namespace_body.get("kind"), namespace_body.get("apiVersion")) != ("Namespace", "v1"):
            errors.append("namespace: wrong kind/apiVersion")
        if not isinstance(meta, dict) or meta.get("name") != ns or meta.get("uid") != ident.get("namespace_uid"):
            errors.append("namespace: UID/name does not bind identity")

    global_uids: set[str] = set()
    # Ordinary pod-network IPs should not collide. hostNetwork Pods on the
    # same node legitimately share its IP; preserve that narrow exception.
    running_ips: dict[str, list[tuple[bool, str | None]]] = {}
    for key, (list_kind, list_api, kind, api) in KINDS.items():
        record = records.get(key)
        if isinstance(record, dict):
            request = record.get("request")
            # Explicit no-pagination, unfiltered most-recent LIST. No resourceVersion=0.
            expected_request = {"method": "GET", "resource": key, "namespace": ns,
                                "query": {"resourceVersion": "", "limit": None,
                                          "continue": "", "labelSelector": "", "fieldSelector": ""}}
            if request != expected_request:
                errors.append(f"{key}: nonconforming LIST request (pagination/filter/stale read)")
        body = _raw_record(record, key, errors)
        if body is None:
            continue
        if (body.get("kind"), body.get("apiVersion")) != (list_kind, list_api):
            errors.append(f"{key}: incorrect LIST kind/apiVersion")
        lm = body.get("metadata")
        if not isinstance(lm, dict) or not _nonempty(lm.get("resourceVersion")) or lm.get("resourceVersion") == "0":
            errors.append(f"{key}: missing/invalid list resourceVersion")
        if not isinstance(lm, dict) or lm.get("continue", "") != "" or lm.get("remainingItemCount") not in (None, 0):
            errors.append(f"{key}: truncated or continued list")
        items = body.get("items")
        if not isinstance(items, list):
            errors.append(f"{key}: LIST items missing")
            continue
        names: set[str] = set()
        for idx, obj in enumerate(items):
            loc = f"{key}[{idx}]"
            if not isinstance(obj, dict):
                errors.append(f"{loc}: item not an object")
                continue
            if (obj.get("kind"), obj.get("apiVersion")) != (kind, api):
                errors.append(f"{loc}: wrong item kind/apiVersion")
            meta = obj.get("metadata")
            if not isinstance(meta, dict):
                errors.append(f"{loc}: missing metadata")
                continue
            if meta.get("namespace") != ns:
                errors.append(f"{loc}: wrong namespace")
            uid, name, rv = meta.get("uid"), meta.get("name"), meta.get("resourceVersion")
            if not all(_nonempty(x) for x in (uid, name, rv)):
                errors.append(f"{loc}: missing UID/name/resourceVersion")
            if _nonempty(uid):
                if uid in global_uids:
                    errors.append(f"{loc}: duplicate UID")
                global_uids.add(uid)
            if _nonempty(name):
                if name in names:
                    errors.append(f"{loc}: duplicate name")
                names.add(name)
            if key == "pods":
                status = obj.get("status")
                if isinstance(status, dict) and status.get("phase") == "Running":
                    ip = status.get("podIP")
                    if _nonempty(ip):
                        spec = obj.get("spec")
                        host_network = isinstance(spec, dict) and spec.get("hostNetwork") is True
                        node_name = spec.get("nodeName") if isinstance(spec, dict) else None
                        previous = running_ips.get(ip, [])
                        for prior_host_network, prior_node_name in previous:
                            if not (
                                host_network and prior_host_network
                                and _nonempty(node_name) and node_name == prior_node_name
                            ):
                                errors.append(f"{loc}: duplicate running pod IP")
                                break
                        running_ips.setdefault(ip, []).append((host_network, node_name))
            if key == "deployments":
                spec, status = obj.get("spec"), obj.get("status")
                replicas = spec.get("replicas") if isinstance(spec, dict) else None
                if type(replicas) is not int or replicas < 0:
                    errors.append(f"{loc}: spec.replicas absent or invalid; no defaulting allowed")
                generation = meta.get("generation")
                observed = status.get("observedGeneration") if isinstance(status, dict) else None
                if type(generation) is not int or type(observed) is not int:
                    unstable.append(f"{loc}: generation/observedGeneration missing")
                elif generation != observed:
                    unstable.append(f"{loc}: Deployment status has not observed current generation")
                ready = status.get("readyReplicas") if isinstance(status, dict) else None
                if ready is not None and (type(ready) is not int or ready < 0):
                    errors.append(f"{loc}: invalid readyReplicas")
                if type(replicas) is int and type(ready) is int and ready > replicas:
                    unstable.append(f"{loc}: readyReplicas exceeds desired; needs temporal review")
    return CaptureAudit(not errors, tuple(errors), tuple(unstable))


def check_observed_item_bracket(first: Mapping[str, Any], second: Mapping[str, Any]) -> BracketAudit:
    """Compare recorded UID/RV sets only. Not watch continuity or atomicity.

    Input captures must already have passed audit_capture(). A false stable
    result cannot make either capture admissible for experiment execution.
    """
    reasons: list[str] = []
    for key in KINDS:
        try:
            a = json.loads(first["records"][key]["raw_json"])["items"]
            b = json.loads(second["records"][key]["raw_json"])["items"]
            def ids(items: list[dict]) -> dict[str, str]:
                return {o["metadata"]["uid"]: o["metadata"]["resourceVersion"] for o in items}
            if ids(a) != ids(b):
                reasons.append(f"{key}: observed UID/resourceVersion set changed")
        except (KeyError, TypeError, ValueError, AttributeError):
            reasons.append(f"{key}: invalid comparison input")
    return BracketAudit(not reasons, tuple(reasons))
