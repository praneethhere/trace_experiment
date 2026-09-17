"""Verified Prefix Replay experimental contract.

This module is intentionally incomplete in the RED commit.

The implementation must eventually reconstruct a recorded pre-intervention
tool prefix inside a genuinely fresh Python process and emit evidence proving
that independently reconstructed branch states are equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


class ReplayMismatch(RuntimeError):
    """Raised when replay does not reproduce frozen prefix evidence."""


@dataclass(frozen=True)
class PrefixCall:
    """One frozen pre-branch tool invocation."""

    function_name: str
    arguments: Mapping[str, Any]
    expected_result: Any


@dataclass(frozen=True)
class PrefixRecord:
    """Canonical pre-intervention prefix."""

    task_id: str
    failure_seed: str
    calls: Sequence[PrefixCall]
    first_hazard_call_index: int


@dataclass(frozen=True)
class ReplayEvidence:
    """Evidence emitted by one fresh-process prefix replay."""

    process_id: int
    ordered_argument_hashes: Tuple[str, ...]
    ordered_result_hashes: Tuple[str, ...]
    first_hazard_call_index: int
    first_hazard_observation_hash: str
    hidden_state_hash: Optional[str]
    branch_state_hash: str


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 of a deterministic canonical JSON representation."""
    raise NotImplementedError(
        "RED CONTRACT: canonical evidence hashing is not implemented"
    )


def replay_prefix_fresh_process(
    *,
    module_path: Path,
    record: PrefixRecord,
    state_probe_name: Optional[str] = None,
) -> ReplayEvidence:
    """Replay a frozen prefix in an isolated fresh Python process.

    Required future behavior:

    - import the target module in a fresh interpreter;
    - set the frozen failure seed;
    - execute exactly the recorded calls in order;
    - compare every reproduced result with the frozen expected result;
    - refuse immediately on mismatch;
    - optionally obtain non-model-visible state evidence through a test probe;
    - hash the complete branch-equivalence evidence;
    - return the child process PID.
    """
    raise NotImplementedError(
        "RED CONTRACT: fresh-process verified prefix replay is not implemented"
    )
