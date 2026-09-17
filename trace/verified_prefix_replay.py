"""Verified Prefix Replay experimental instrument.

A frozen pre-intervention tool prefix is reconstructed inside a genuinely
fresh Python interpreter. Every replayed tool result must reproduce the
recorded prefix exactly before branch-equivalence evidence is accepted.

This module is an experimental instrument. Passing its synthetic contract
does not by itself prove replay equivalence for any external benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import hashlib
import importlib.util
import json
import os
import subprocess
import sys


_CHILD_MARKER = "TRACE_V2_REPLAY_RESULT="

_REPLAY_SCHEMA = "trace-v2-verified-prefix-replay/1"


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


def _canonical_bytes(value: Any) -> bytes:
    """Serialize evidence without lossy string coercion."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return SHA-256 of deterministic canonical JSON."""

    return hashlib.sha256(
        _canonical_bytes(value)
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


def _validate_record(record: PrefixRecord) -> None:
    if not isinstance(record.task_id, str):
        raise TypeError(
            "record.task_id must be a string"
        )

    if not record.task_id:
        raise ValueError(
            "record.task_id must not be empty"
        )

    if not isinstance(record.failure_seed, str):
        raise TypeError(
            "record.failure_seed must be a string"
        )

    calls = tuple(record.calls)

    if not calls:
        raise ValueError(
            "A replay prefix must contain at least one call"
        )

    if not isinstance(
        record.first_hazard_call_index,
        int,
    ):
        raise TypeError(
            "first_hazard_call_index must be an integer"
        )

    if not (
        0
        <= record.first_hazard_call_index
        < len(calls)
    ):
        raise ValueError(
            "first_hazard_call_index is outside the prefix"
        )

    for index, call in enumerate(calls):
        if not isinstance(call, PrefixCall):
            raise TypeError(
                f"calls[{index}] is not PrefixCall"
            )

        if (
            not isinstance(call.function_name, str)
            or not call.function_name
        ):
            raise ValueError(
                f"calls[{index}] has invalid function_name"
            )

        if not isinstance(call.arguments, Mapping):
            raise TypeError(
                f"calls[{index}].arguments must be a mapping"
            )

        _canonical_bytes(call.arguments)
        _canonical_bytes(call.expected_result)


def _payload_for_child(
    record: PrefixRecord,
    state_probe_name: Optional[str],
) -> dict[str, Any]:
    return {
        "task_id": record.task_id,
        "failure_seed": record.failure_seed,
        "first_hazard_call_index":
            record.first_hazard_call_index,
        "state_probe_name": state_probe_name,
        "calls": [
            {
                "function_name": call.function_name,
                "arguments": dict(call.arguments),
                "expected_result":
                    call.expected_result,
            }
            for call in record.calls
        ],
    }


def _extract_child_message(
    *,
    stdout: str,
    stderr: str,
    returncode: int,
) -> dict[str, Any]:
    marker_lines = [
        line[len(_CHILD_MARKER):]
        for line in stdout.splitlines()
        if line.startswith(_CHILD_MARKER)
    ]

    if not marker_lines:
        stderr_tail = stderr[-1000:]

        raise ReplayMismatch(
            "Fresh-process replay returned no "
            "structured evidence "
            f"(returncode={returncode}, "
            f"stderr_tail={stderr_tail!r})"
        )

    try:
        message = json.loads(marker_lines[-1])
    except json.JSONDecodeError as exc:
        raise ReplayMismatch(
            "Fresh-process replay emitted invalid "
            "structured evidence"
        ) from exc

    if not isinstance(message, dict):
        raise ReplayMismatch(
            "Fresh-process replay evidence "
            "must be a JSON object"
        )

    return message


def replay_prefix_fresh_process(
    *,
    module_path: Path,
    record: PrefixRecord,
    state_probe_name: Optional[str] = None,
) -> ReplayEvidence:
    """Replay a frozen prefix in an isolated fresh Python process.

    The child process imports the target module from its file path,
    reconstructs the recorded tool-call sequence, compares each actual
    result with its frozen expected result, optionally records hidden-state
    evidence through a non-model-visible experimental probe, and emits a
    branch-state evidence hash.

    The child PID is returned as evidence that replay occurred outside the
    calling interpreter.
    """

    _validate_record(record)

    module_path = Path(module_path).resolve()

    if not module_path.is_file():
        raise FileNotFoundError(
            f"Replay module does not exist: {module_path}"
        )

    if (
        state_probe_name is not None
        and (
            not isinstance(state_probe_name, str)
            or not state_probe_name
        )
    ):
        raise ValueError(
            "state_probe_name must be a non-empty "
            "string or None"
        )

    payload = _payload_for_child(
        record,
        state_probe_name,
    )

    environment = os.environ.copy()
    environment["FAIL_SEED"] = record.failure_seed

    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            str(module_path),
        ],
        input=json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    message = _extract_child_message(
        stdout=process.stdout,
        stderr=process.stderr,
        returncode=process.returncode,
    )

    status = message.get("status")

    if status == "mismatch":
        raise ReplayMismatch(
            "Replay result mismatch at call index "
            f"{message.get('call_index')}: "
            f"expected_sha256="
            f"{message.get('expected_sha256')} "
            f"actual_sha256="
            f"{message.get('actual_sha256')}"
        )

    if status != "ok":
        raise ReplayMismatch(
            "Fresh-process replay failed: "
            f"{message.get('error_type', 'unknown')}: "
            f"{message.get('error_message', '')}"
        )

    if process.returncode != 0:
        raise ReplayMismatch(
            "Fresh-process replay reported success "
            f"but exited with {process.returncode}"
        )

    try:
        return ReplayEvidence(
            process_id=int(
                message["process_id"]
            ),
            ordered_argument_hashes=tuple(
                message[
                    "ordered_argument_hashes"
                ]
            ),
            ordered_result_hashes=tuple(
                message[
                    "ordered_result_hashes"
                ]
            ),
            first_hazard_call_index=int(
                message[
                    "first_hazard_call_index"
                ]
            ),
            first_hazard_observation_hash=str(
                message[
                    "first_hazard_observation_hash"
                ]
            ),
            hidden_state_hash=(
                None
                if message.get(
                    "hidden_state_hash"
                ) is None
                else str(
                    message[
                        "hidden_state_hash"
                    ]
                )
            ),
            branch_state_hash=str(
                message["branch_state_hash"]
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ReplayMismatch(
            "Fresh-process replay evidence "
            "schema is incomplete"
        ) from exc


def _emit_child_message(
    value: Mapping[str, Any],
) -> None:
    print(
        _CHILD_MARKER
        + json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ),
        flush=True,
    )


def _load_module_from_path(
    module_path: Path,
):
    module_name = (
        "_trace_v2_replay_"
        f"{os.getpid()}_"
        f"{hashlib.sha256(str(module_path).encode()).hexdigest()[:12]}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load replay module: {module_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(module)

    return module


def _child_execute(
    *,
    module_path: Path,
    payload: Mapping[str, Any],
) -> int:
    try:
        module = _load_module_from_path(
            module_path
        )

        calls = payload["calls"]
        first_hazard_call_index = int(
            payload[
                "first_hazard_call_index"
            ]
        )

        argument_hashes: list[str] = []
        result_hashes: list[str] = []

        for index, call in enumerate(calls):
            function_name = call[
                "function_name"
            ]

            arguments = call["arguments"]
            expected_result = call[
                "expected_result"
            ]

            function = getattr(
                module,
                function_name,
                None,
            )

            if not callable(function):
                raise AttributeError(
                    "Replay target has no callable "
                    f"{function_name!r}"
                )

            argument_hash = canonical_sha256(
                arguments
            )

            actual_result = function(
                **arguments
            )

            actual_hash = canonical_sha256(
                actual_result
            )

            expected_hash = canonical_sha256(
                expected_result
            )

            if actual_result != expected_result:
                _emit_child_message({
                    "status": "mismatch",
                    "process_id":
                        os.getpid(),
                    "call_index": index,
                    "expected_sha256":
                        expected_hash,
                    "actual_sha256":
                        actual_hash,
                })
                return 3

            argument_hashes.append(
                argument_hash
            )
            result_hashes.append(
                actual_hash
            )

        state_probe_name = payload.get(
            "state_probe_name"
        )

        hidden_state_hash: Optional[str]

        if state_probe_name is None:
            hidden_state_hash = None
        else:
            state_probe = getattr(
                module,
                state_probe_name,
                None,
            )

            if not callable(state_probe):
                raise AttributeError(
                    "Replay target has no callable "
                    f"state probe "
                    f"{state_probe_name!r}"
                )

            hidden_state = state_probe()

            hidden_state_hash = (
                canonical_sha256(
                    hidden_state
                )
            )

        first_hazard_observation_hash = (
            result_hashes[
                first_hazard_call_index
            ]
        )

        branch_evidence = {
            "schema": _REPLAY_SCHEMA,
            "task_id":
                payload["task_id"],
            "failure_seed":
                payload["failure_seed"],
            "module_sha256":
                _file_sha256(module_path),
            "ordered_argument_hashes":
                argument_hashes,
            "ordered_result_hashes":
                result_hashes,
            "first_hazard_call_index":
                first_hazard_call_index,
            "first_hazard_observation_hash":
                first_hazard_observation_hash,
            "hidden_state_hash":
                hidden_state_hash,
        }

        branch_state_hash = (
            canonical_sha256(
                branch_evidence
            )
        )

        _emit_child_message({
            "status": "ok",
            "process_id": os.getpid(),
            "ordered_argument_hashes":
                argument_hashes,
            "ordered_result_hashes":
                result_hashes,
            "first_hazard_call_index":
                first_hazard_call_index,
            "first_hazard_observation_hash":
                first_hazard_observation_hash,
            "hidden_state_hash":
                hidden_state_hash,
            "branch_state_hash":
                branch_state_hash,
        })

        return 0

    except Exception as exc:
        _emit_child_message({
            "status": "error",
            "process_id": os.getpid(),
            "error_type":
                type(exc).__name__,
            "error_message":
                str(exc),
        })

        return 4


def _main() -> int:
    if (
        len(sys.argv) == 3
        and sys.argv[1] == "--child"
    ):
        module_path = Path(
            sys.argv[2]
        ).resolve()

        try:
            payload = json.loads(
                sys.stdin.read()
            )
        except Exception as exc:
            _emit_child_message({
                "status": "error",
                "process_id":
                    os.getpid(),
                "error_type":
                    type(exc).__name__,
                "error_message":
                    str(exc),
            })
            return 4

        return _child_execute(
            module_path=module_path,
            payload=payload,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
