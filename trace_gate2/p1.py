"""Offline P1 validation against the frozen SREGym Gate-2 protocol.

Input is a *previously captured*, namespace-scoped JSON Kubernetes snapshot.
This module neither collects evidence nor modifies Kubernetes state.

The output is only the P1 candidate-specific semantic layer. It is not the
complete Gate-2 structural canonicalization, raw-evidence archive, or runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping


PROBLEM_ID = "service_wrong_pod_selection_hotel_reservation"
EXPECTED_SELECTOR = {"service-route": "frontend"}


@dataclass(frozen=True)
class P1Result:
    valid_manifestation: bool
    errors: tuple[str, ...]
    semantic: dict[str, Any]
    semantic_sha256: str


def _json_sha256(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _resource_map(snapshot: Mapping[str, Any], key: str, namespace: str, errors: list[str]) -> dict[str, dict]:
    items = snapshot.get(key)
    if not isinstance(items, list):
        errors.append(f"{key}: expected list of complete Kubernetes objects")
        return {}
    result = {}
    for obj in items:
        if not isinstance(obj, dict):
            errors.append(f"{key}: non-object resource")
            continue
        meta = obj.get("metadata")
        if not isinstance(meta, dict):
            errors.append(f"{key}: resource metadata is not an object")
            continue
        name = meta.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{key}: resource has no name")
            continue
        if meta.get("namespace") != namespace:
            errors.append(f"{key}/{name}: missing or wrong namespace")
            continue
        if name in result:
            errors.append(f"{key}/{name}: duplicate name")
            continue
        uid = meta.get("uid")
        if not isinstance(uid, str) or not uid.strip():
            errors.append(f"{key}/{name}: missing or invalid resource UID")
            continue
        result[name] = obj
    return result


def _controller(obj: Mapping[str, Any], kind: str) -> tuple[str, str] | None:
    refs = obj.get("metadata", {}).get("ownerReferences") or []
    if not isinstance(refs, list) or any(not isinstance(r, dict) for r in refs):
        return None
    refs = [r for r in refs if r.get("kind") == kind and r.get("controller") is True]
    if len(refs) != 1:
        return None
    name = refs[0].get("name")
    uid = refs[0].get("uid")
    if (
        not isinstance(name, str) or not name
        or not isinstance(uid, str) or not uid.strip()
    ):
        return None
    return name, uid


def _pod_deployment(pod: Mapping[str, Any], replicasets: Mapping[str, dict], deployments: Mapping[str, dict]) -> str | None:
    rs_ref = _controller(pod, "ReplicaSet")
    if not rs_ref:
        return None
    rs = replicasets.get(rs_ref[0])
    if not rs or rs["metadata"].get("uid") != rs_ref[1]:
        return None
    dep_ref = _controller(rs, "Deployment")
    if not dep_ref:
        return None
    dep = deployments.get(dep_ref[0])
    if not dep or dep["metadata"].get("uid") != dep_ref[1]:
        return None
    return dep_ref[0]


def _ready_pod(pod: Mapping[str, Any]) -> bool:
    meta = pod.get("metadata", {})
    status = pod.get("status")
    if not isinstance(meta, dict) or not isinstance(status, dict):
        return False
    conditions = status.get("conditions")
    if not isinstance(conditions, list) or any(not isinstance(c, dict) for c in conditions):
        return False
    ready_conditions = [c for c in conditions if c.get("type") == "Ready"]
    return (
        not meta.get("deletionTimestamp")
        and status.get("phase") == "Running"
        and len(ready_conditions) == 1
        and ready_conditions[0].get("status") == "True"
    )


def evaluate_p1(snapshot: Mapping[str, Any]) -> P1Result:
    """Compute P1 semantic state and falsify the frozen manifestation predicate.

    Refuses incomplete snapshots; does not guess owners from names/labels.
    The semantic SHA is meaningful for *valid* recreations only, and depends
    solely on the included semantic fields, never ephemeral identities.
    """
    errors: list[str] = []
    namespace = snapshot.get("namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("snapshot.namespace must be a nonempty string")

    deps = _resource_map(snapshot, "deployments", namespace, errors)
    pods = _resource_map(snapshot, "pods", namespace, errors)
    rs = _resource_map(snapshot, "replicasets", namespace, errors)
    services = _resource_map(snapshot, "services", namespace, errors)
    endpoints = _resource_map(snapshot, "endpoints", namespace, errors)

    service = services.get("frontend")
    if service is None:
        errors.append("frontend Service missing")
    selector = service.get("spec", {}).get("selector") if service else None
    if selector != EXPECTED_SELECTOR:
        errors.append("frontend Service selector is not the exact frozen selector")

    state: dict[str, dict[str, int]] = {}
    ready_by_owner = {"frontend": 0, "search": 0}
    for name in ("frontend", "search"):
        dep = deps.get(name)
        if dep is None:
            errors.append(f"{name} Deployment missing")
            continue
        spec = dep.get("spec", {})
        status = dep.get("status")
        if not isinstance(status, dict):
            errors.append(f"{name}: Deployment status missing or malformed")
            status = {}
        desired = spec.get("replicas", 1)  # Kubernetes API defaults to one
        ready = status.get("readyReplicas")
        if type(desired) is not int or desired <= 0:
            errors.append(f"{name}: desired replicas not positive")
        if type(ready) is not int or ready <= 0:
            errors.append(f"{name}: readyReplicas missing, zero, or invalid")
        labels = spec.get("template", {}).get("metadata", {}).get("labels", {})
        deployment_selector = spec.get("selector")
        valid_selector = (
            isinstance(deployment_selector, dict)
            and (
                (
                    isinstance(deployment_selector.get("matchLabels"), dict)
                    and bool(deployment_selector["matchLabels"])
                )
                or (
                    isinstance(deployment_selector.get("matchExpressions"), list)
                    and bool(deployment_selector["matchExpressions"])
                )
            )
        )
        if not valid_selector:
            errors.append(f"{name}: Deployment selector missing or invalid")
        elif isinstance(deployment_selector.get("matchLabels"), dict):
            if not isinstance(labels, dict) or any(
                labels.get(k) != v
                for k, v in deployment_selector["matchLabels"].items()
            ):
                errors.append(f"{name}: Deployment selector/template mismatch")
        if labels.get("service-route") != "frontend":
            errors.append(f"{name}: pod template missing injected service-route label")
        state[name] = {"desired": desired, "ready": ready}

    for pod in pods.values():
        owner = _pod_deployment(pod, rs, deps)
        if owner in ready_by_owner and _ready_pod(pod):
            ready_by_owner[owner] += 1
    for owner, count in ready_by_owner.items():
        if count == 0:
            errors.append(f"{owner}: no Ready non-terminating owned pods")

    ep = endpoints.get("frontend")
    if ep is None:
        errors.append("frontend Endpoints object missing")
    counts = {"frontend": 0, "search": 0, "unidentified": 0}
    addresses_seen: set[tuple[str, str]] = set()
    if ep is not None:
        for subset in ep.get("subsets", []) or []:
            for address in subset.get("addresses", []) or []:
                if not isinstance(address, dict):
                    errors.append("frontend Endpoints contains invalid address evidence")
                    counts["unidentified"] += 1
                    continue
                target = address.get("targetRef") or {}
                if not isinstance(target, dict):
                    errors.append("frontend Endpoints targetRef is malformed")
                    counts["unidentified"] += 1
                    continue
                pod_name = target.get("name", "")
                pod = pods.get(pod_name) if target.get("kind") == "Pod" else None
                key = (pod_name, address.get("ip", ""))
                if key in addresses_seen:
                    errors.append("duplicate ready endpoint address")
                    continue
                addresses_seen.add(key)
                # Require explicit Pod identity and matching endpoint IP.
                pod_uid = pod.get("metadata", {}).get("uid") if pod else None
                target_uid = target.get("uid") if isinstance(target, dict) else None
                if (
                    not pod or not isinstance(pod_uid, str) or not pod_uid
                    or not isinstance(target_uid, str) or target_uid != pod_uid
                    or target.get("namespace", namespace) != namespace
                    or not isinstance(address.get("ip"), str)
                    or not address["ip"]
                    or address["ip"] != pod.get("status", {}).get("podIP")
                ):
                    counts["unidentified"] += 1
                    continue
                if not _ready_pod(pod):
                    counts["unidentified"] += 1
                    continue
                if any(pod.get("metadata", {}).get("labels", {}).get(k) != v for k, v in EXPECTED_SELECTOR.items()):
                    counts["unidentified"] += 1
                    continue
                owner = _pod_deployment(pod, rs, deps)
                if owner in ("frontend", "search"):
                    counts[owner] += 1
                else:
                    counts["unidentified"] += 1
    if counts["frontend"] < 1:
        errors.append("frontend Service has no ready endpoint owned by frontend Deployment")
    if counts["search"] < 1:
        errors.append("frontend Service has no ready endpoint owned by search Deployment")
    if counts["unidentified"]:
        errors.append("frontend Service has unidentified ready endpoints")

    ports = service.get("spec", {}).get("ports", []) if service else []
    if not isinstance(ports, list) or not ports:
        errors.append("frontend Service port mapping missing")
        ports = []
    port_mapping = sorted(
        (
            {"port": item.get("port"), "targetPort": item.get("targetPort", item.get("port")),
             "protocol": item.get("protocol", "TCP")}
            for item in ports
        ),
        key=lambda item: (str(item["port"]), str(item["targetPort"]), item["protocol"]),
    )
    frontend_target_ports = {
        entry["targetPort"] for entry in port_mapping
        if entry["port"] == 5000 and entry["protocol"] == "TCP"
    }
    if not frontend_target_ports:
        errors.append("frontend Service has no required TCP port 5000")
    if ep is not None:
        for subset in ep.get("subsets", []) or []:
            if not isinstance(subset, dict):
                errors.append("frontend Endpoints contains an invalid subset")
                continue
            subset_ports = subset.get("ports", [])
            if not isinstance(subset_ports, list) or not any(
                isinstance(port, dict)
                and port.get("protocol", "TCP") == "TCP"
                and port.get("port") in frontend_target_ports
                for port in subset_ports
            ):
                errors.append(
                    "frontend Endpoints subset lacks the frontend TCP target port"
                )

    semantic = {
        "problem_id": PROBLEM_ID,
        "service_selector": selector,
        "deployment_replicas": state,
        "frontend_ready_endpoints": counts["frontend"],
        "search_ready_endpoints": counts["search"],
        "unidentified_ready_endpoints": counts["unidentified"],
        "service_ports": port_mapping,
        "mixed_endpoint_contamination": counts["frontend"] >= 1 and counts["search"] >= 1 and counts["unidentified"] == 0,
    }
    return P1Result(
        valid_manifestation=not errors,
        errors=tuple(errors),
        semantic=semantic,
        semantic_sha256=_json_sha256(semantic),
    )
