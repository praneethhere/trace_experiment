"""Synthetic stateful tool whose first hazard is a raised exception.

This fixture models a hard tool failure while retaining hidden monotonic
module-runtime state. The hidden counter is deliberately not exposed in the
normal tool result.

Test infrastructure only.
"""

from __future__ import annotations

import os
from typing import Any, Dict


_CALL_COUNT = 0

_REQUIRED_FAILURE_SEED = (
    "trace-v2-synthetic-hard-failure-seed-v1"
)


def synthetic_lookup(
    *,
    item: str,
) -> Dict[str, Any]:
    global _CALL_COUNT

    _CALL_COUNT += 1

    seed = os.environ.get(
        "FAIL_SEED",
        "",
    )

    if (
        seed == _REQUIRED_FAILURE_SEED
        and _CALL_COUNT == 3
    ):
        raise RuntimeError(
            f"synthetic_hard_failure:{item}"
        )

    return {
        "ok": True,
        "item": item,
        "value": f"resolved:{item}",
    }


def replay_state_probe() -> Dict[str, Any]:
    return {
        "call_count": _CALL_COUNT,
        "failure_seed": os.environ.get(
            "FAIL_SEED",
            "",
        ),
    }
