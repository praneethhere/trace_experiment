"""Offline, prospective TRACE Gate-2 measurement decisions: NOT a frozen rule.

Purpose: make ambiguous classification and typed-comparison choices falsifiable
before Gate-2 outcomes. No import from a campaign runner, Kubernetes or models.
Event codes in this module are proposals and MUST NOT be used for live grading
until a reviewed, versioned, pre-outcome amendment explicitly authorizes them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


TERMINAL_CLASSES = frozenset({
    "VALID_RECREATION", "RECREATION_FAILURE",
    "INFRASTRUCTURE_FAILURE", "PROTOCOL_VIOLATION",
})

# Evidence-condition vocabulary. This does not replace raw failure evidence.
PROTOCOL_EVENTS = frozenset({
    "FROZEN_CONTRACT_BREACH", "COLLECTOR_FAILURE",
    "INCOMPLETE_CAPTURE", "WATCH_CONTINUITY_FAILURE",
})
RECREATION_EVENTS = frozenset({
    "INJECTOR_BENCHMARK_FAILURE", "MANIFESTATION_TIMEOUT",
    "BENCHMARK_STATE_EVIDENCE_MISSING", "CLEANUP_RESIDUE",
})
INFRASTRUCTURE_EVENTS = frozenset({"EXTERNAL_DOCKER_KIND_HOST_FAILURE"})
SUCCESS_WITNESSES = frozenset({
    "INJECTOR_COMPLETED", "THREE_TRUE_OBSERVATIONS",
    "FINAL_CAPTURE_MANIFESTATION_TRUE", "RAW_EVIDENCE_ADMISSIBLE",
    "CLEANUP_BARRIER_PASSED",
})
KNOWN_EVENTS = PROTOCOL_EVENTS | RECREATION_EVENTS | INFRASTRUCTURE_EVENTS | SUCCESS_WITNESSES


@dataclass(frozen=True)
class EpisodeProposal:
    terminal: str | None
    reason: str
    observed_events: tuple[str, ...]
    missing_witnesses: tuple[str, ...]

    @property
    def terminal_is_authorized_for_live_experiment(self) -> bool:
        """Always false: this module is a proposal, not an execution gate."""
        return False


def classify_episode_proposal(events: Iterable[str]) -> EpisodeProposal:
    """Adversarially test a *proposed* conservative precedence rule.

    A pending result (terminal=None) is NOT a fifth terminal class.
    Independently documented failure evidence is retained, never rerun away.
    Proposed precedence: protocol > verified recreation > external infra.
    """
    if isinstance(events, (str, bytes)):
        raise ValueError("events must be a collection, not text")
    try:
        seen = set(events)
    except (TypeError, ValueError) as exc:
        raise ValueError("events must be a collection of valid event strings") from exc
    if not all(type(x) is str for x in seen):
        raise ValueError("all events must be strings")
    unknown = seen - KNOWN_EVENTS
    if unknown:
        raise ValueError(f"unknown event(s): {sorted(unknown)}")

    ordered = tuple(sorted(seen))
    missing = tuple(sorted(SUCCESS_WITNESSES - seen))
    for group, terminal in (
        (PROTOCOL_EVENTS, "PROTOCOL_VIOLATION"),
        (RECREATION_EVENTS, "RECREATION_FAILURE"),
        (INFRASTRUCTURE_EVENTS, "INFRASTRUCTURE_FAILURE"),
    ):
        found = sorted(seen & group)
        if found:
            return EpisodeProposal(terminal, found[0], ordered, missing)
    if not missing:
        return EpisodeProposal("VALID_RECREATION", "ALL_REQUIRED_WITNESSES", ordered, ())
    return EpisodeProposal(None, "INCOMPLETE_EVIDENCE_PENDING", ordered, missing)


class FingerprintSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ComparisonProposal:
    admissible: bool
    equivalent: bool
    mismatch_paths: tuple[str, ...]
    error: str | None

    @property
    def authorized_for_live_experiment(self) -> bool:
        return False


def _validate_value(value: Any, path: str) -> None:
    if type(value) in (str, bool):
        return
    if type(value) is int:
        if value < 0:
            raise FingerprintSchemaError(f"{path}: negative integer")
        return
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise FingerprintSchemaError(f"{path}: non-string key")
        for k, v in value.items():
            _validate_value(v, f"{path}.{k}")
        return
    if type(value) is list:
        for i, v in enumerate(value):
            _validate_value(v, f"{path}[{i}]")
        return
    raise FingerprintSchemaError(f"{path}: unsupported or missing value type {type(value).__name__}")


def _mismatches(a: Any, b: Any, path: str, out: list[str]) -> None:
    if type(a) is not type(b):
        out.append(path)
    elif type(a) is dict:
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}")
            else:
                _mismatches(a[k], b[k], f"{path}.{k}", out)
    elif type(a) is list:
        if len(a) != len(b):
            out.append(f"{path}.length")
        for i, (x, y) in enumerate(zip(a, b)):
            _mismatches(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(path)


def compare_declared_semantics_proposal(
    left: Mapping[str, Any], right: Mapping[str, Any], *, required_fields: tuple[str, ...]
) -> ComparisonProposal:
    """Exact typed comparison. Caller must *later* supply frozen field schema.

    Lists remain order-sensitive pending candidate-specific ordering policy.
    No implicit cast, bool/int conflation, null default, fuzzy score or float.
    This cannot establish evidence provenance or authorize experimental PASS.
    """
    try:
        if type(left) is not dict or type(right) is not dict:
            raise FingerprintSchemaError("fingerprints must be dictionaries")
        if type(required_fields) is not tuple or not required_fields:
            raise FingerprintSchemaError("a nonempty tuple of required fields is needed")
        if not all(type(k) is str and k for k in required_fields):
            raise FingerprintSchemaError("required fields must be nonempty strings")
        if len(set(required_fields)) != len(required_fields):
            raise FingerprintSchemaError("duplicate required fields")
        expected = set(required_fields)
        if set(left) != expected or set(right) != expected:
            raise FingerprintSchemaError(
                f"top-level fields must equal declared schema: expected={sorted(expected)} "
                f"left={sorted(left)} right={sorted(right)}"
            )
        for key in required_fields:
            _validate_value(left[key], f"$.{key}")
            _validate_value(right[key], f"$.{key}")
    except FingerprintSchemaError as exc:
        return ComparisonProposal(False, False, (), str(exc))
    except (TypeError, ValueError) as exc:
        return ComparisonProposal(False, False, (), f"malformed schema: {exc}")
    differences: list[str] = []
    _mismatches(left, right, "$", differences)
    return ComparisonProposal(True, not differences, tuple(differences), None)
