"""Synthetic deterministic tool for Verified Prefix Replay tests.

This fixture intentionally models the ToolBench-X hazard property that
motivated the replay contract:

- hidden monotonic module-runtime state;
- deterministic failure keyed by a frozen failure seed;
- agent-visible results that do not expose the hidden counter.

It is test infrastructure only.
"""

from __future__ import annotations

import os
from typing import Any, Dict


_CALL_COUNT = 0

_REQUIRED_FAILURE_SEED = "trace-v2-synthetic-seed-v1"


def synthetic_lookup(*, item: str) -> Dict[str, Any]:
    """Execute one deterministic synthetic tool call."""
    global _CALL_COUNT

    _CALL_COUNT += 1

    seed = os.environ.get("FAIL_SEED", "")

    if (
        seed == _REQUIRED_FAILURE_SEED
        and _CALL_COUNT == 3
    ):
        return {
            "ok": False,
            "error": "synthetic_execution_failure",
            "item": item,
        }

    return {
        "ok": True,
        "item": item,
        "value": f"resolved:{item}",
    }


def replay_state_probe() -> Dict[str, Any]:
    """Experimental probe; never part of model-visible tool output."""
    return {
        "call_count": _CALL_COUNT,
        "failure_seed": os.environ.get("FAIL_SEED", ""),
    }
