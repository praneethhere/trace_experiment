"""Offline P3 EndpointSlice normalization PROPOSAL, not a Gate-2 decision rule.

Pinned target: Kubernetes v1.32.1, SREGym 9093167a, application b2006cb.
Inputs MUST come from a separately audited complete namespace collection.
This does not establish capture authenticity, watch continuity, fault
manifestation, observation stability or execution admissibility.

v1.32.1 serving:nil -> ready, NOT unconditional True. Verified in
kubernetes/kubernetes v1.32.1 staging/src/k8s.io/api/discovery/v1/types.go.
"""
from __future__ import annotations

from collections import Counter
import ipaddress
from typing import Any

P3_SERVICES = ("cart", "valkey-cart")


class EndpointNormalizationError(ValueError):
    """Insufficient or internally inconsistent endpoint evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EndpointNormalizationError(message)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _bool_default(value: Any, default: bool, path: str) -> bool:
    if value is None:
        return default
    _require(type(value) is bool, f"{path}: expected boolean or null")
    return value


def _metadata(obj: Any, namespace: str, expected_kind: str, api: str, path: str) -> dict:
    _require(isinstance(obj, dict), f"{path}: object required")
    _require(obj.get("kind") == expected_kind and obj.get("apiVersion") == api,
             f"{path}: kind/apiVersion mismatch")
    meta = obj.get("metadata")
    _require(isinstance(meta, dict), f"{path}: metadata required")
    _require(meta.get("namespace") == namespace, f"{path}: namespace mismatch")
    _require(_text(meta.get("name")) and _text(meta.get("uid")),
             f"{path}: name/UID required")
    return meta


def _port(item: Any, path: str) -> tuple[str, int, str, str]:
    _require(isinstance(item, dict), f"{path}: port entry not an object")
    number = item.get("port")
    _require(type(number) is int and 0 < number <= 65535,
             f"{path}: numeric port required for pinned P3 chart")
    name = item.get("name")
    if name is None:
        name = ""
    _require(isinstance(name, str), f"{path}: invalid port name")
    protocol = item.get("protocol", "TCP")
    if protocol is None:
        protocol = "TCP"
    _require(protocol in ("TCP", "UDP", "SCTP"), f"{path}: invalid protocol")
    app_protocol = item.get("appProtocol", "")
    if app_protocol is None:
        app_protocol = ""
    _require(isinstance(app_protocol, str), f"{path}: invalid appProtocol")
    return name, number, protocol, app_protocol


def normalize_p3_endpoints(*, services: list[dict], endpoint_slices: list[dict],
                           namespace: str) -> dict:
    """Return proposed typed canonical endpoint histogram for cart/valkey-cart.

    Each network endpoint is keyed by service, address family, address, port
    and protocol; generated IP/UID values do NOT enter the returned histogram.
    Conflicting copies of one network endpoint fail closed. A complete empty
    EndpointSlice list represents zero endpoints ONLY IF a separate collector
    already proved the LIST was complete and current.
    """
    _require(_text(namespace), "namespace required")
    _require(isinstance(services, list) and isinstance(endpoint_slices, list),
             "services/endpoint_slices must be lists")
    relevant_services: dict[str, dict] = {}
    for idx, service in enumerate(services):
        meta = _metadata(service, namespace, "Service", "v1", f"service[{idx}]")
        name = meta["name"]
        if name not in P3_SERVICES:
            continue
        _require(name not in relevant_services, f"duplicate Service {name}")
        spec = service.get("spec")
        _require(isinstance(spec, dict), f"{name}: spec missing")
        publish = _bool_default(spec.get("publishNotReadyAddresses"), False,
                                f"{name}.publishNotReadyAddresses")
        ports = spec.get("ports")
        _require(isinstance(ports, list) and ports, f"{name}: missing Service ports")
        normalized_ports = sorted(_port(p, f"{name}.ports[{i}]") for i, p in enumerate(ports))
        _require(len(normalized_ports) == len(set(normalized_ports)), f"{name}: repeated Service port")
        relevant_services[name] = {"publish_not_ready_addresses": publish,
                                   "ports": [list(p) for p in normalized_ports]}
    _require(set(relevant_services) == set(P3_SERVICES), "cart and valkey-cart Services required")

    # Full raw network identity -> conditions, pod target, and port name.
    seen: dict[tuple, tuple] = {}
    for idx, sl in enumerate(endpoint_slices):
        meta = _metadata(sl, namespace, "EndpointSlice", "discovery.k8s.io/v1",
                         f"slice[{idx}]")
        labels = meta.get("labels")
        _require(isinstance(labels, dict), f"slice[{idx}]: service-name label missing")
        svc = labels.get("kubernetes.io/service-name")
        _require(_text(svc), f"slice[{idx}]: service-name label missing")
        if svc not in relevant_services:
            continue
        family = sl.get("addressType")
        _require(family in ("IPv4", "IPv6", "FQDN"), f"slice[{idx}]: invalid addressType")
        ports = sl.get("ports")
        _require(isinstance(ports, list) and ports, f"slice[{idx}]: missing ports")
        normalized_ports = [_port(p, f"slice[{idx}].ports[{i}]") for i, p in enumerate(ports)]
        _require(len(set(normalized_ports)) == len(normalized_ports),
                 f"slice[{idx}]: duplicate port entry")
        endpoints = sl.get("endpoints")
        _require(isinstance(endpoints, list), f"slice[{idx}]: endpoints missing")
        for e_idx, ep in enumerate(endpoints):
            path = f"slice[{idx}].endpoints[{e_idx}]"
            _require(isinstance(ep, dict), f"{path}: invalid endpoint")
            addresses = ep.get("addresses")
            _require(isinstance(addresses, list) and bool(addresses), f"{path}: addresses required")
            conditions = ep.get("conditions")
            if conditions is None:
                conditions = {}
            _require(isinstance(conditions, dict), f"{path}: invalid conditions")
            ready = _bool_default(conditions.get("ready"), True, f"{path}.ready")
            # IMPORTANT: v1.32.1 fallback is to ready, not always True.
            serving = _bool_default(conditions.get("serving"), ready, f"{path}.serving")
            terminating = _bool_default(conditions.get("terminating"), False,
                                        f"{path}.terminating")
            triple = (ready, serving, terminating)
            ref = ep.get("targetRef")
            _require(isinstance(ref, dict) and ref.get("kind") == "Pod"
                     and _text(ref.get("uid")) and _text(ref.get("name")),
                     f"{path}: pod target reference required for pinned P3")
            target = (ref["kind"], ref["name"], ref["uid"])
            for addr in addresses:
                _require(_text(addr), f"{path}: invalid address")
                if family != "FQDN":
                    try:
                        parsed = ipaddress.ip_address(addr)
                    except ValueError as exc:
                        raise EndpointNormalizationError(f"{path}: invalid IP address") from exc
                    _require((family == "IPv4" and parsed.version == 4) or
                             (family == "IPv6" and parsed.version == 6),
                             f"{path}: address family mismatch")
                for name, port, protocol, app_protocol in normalized_ports:
                    network_key = (svc, family, addr, port, protocol)
                    value = (triple, target, name, app_protocol)
                    if network_key in seen:
                        _require(seen[network_key] == value,
                                 f"{path}: conflicting duplicate network endpoint")
                    else:
                        seen[network_key] = value

    counts: dict[str, Counter] = {svc: Counter() for svc in P3_SERVICES}
    for (svc, family, _addr, port, protocol), (triple, _target, name, app_protocol) in seen.items():
        # Identity intentionally excluded; these fields remain in Layer A.
        counts[svc][(family, name, port, protocol, app_protocol, *triple)] += 1
    semantic: dict[str, dict] = {}
    for svc in P3_SERVICES:
        histogram = []
        for (family, name, port, protocol, app_protocol, ready, serving, terminating), count in sorted(counts[svc].items()):
            histogram.append({"address_type": family,
                              "port": {"name": name, "number": port,
                                       "protocol": protocol, "app_protocol": app_protocol},
                              "ready": ready, "serving": serving,
                              "terminating": terminating, "count": count})
        semantic[svc] = {**relevant_services[svc], "endpoint_histogram": histogram}
    return semantic


def typed_exact_equal(a: Any, b: Any) -> bool:
    """Proposed strict comparison over *already canonicalized* semantic data."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(typed_exact_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(typed_exact_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        # No floating metrics in the currently proposed P3 schema; reject
        # even nan==nan surprises and unspecified tolerance policies.
        raise ValueError("floating-point comparator policy not frozen")
    if type(a) not in (str, int, bool, type(None)):
        raise ValueError("unsupported comparison type")
    return a == b
