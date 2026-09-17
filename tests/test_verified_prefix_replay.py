from __future__ import annotations

from pathlib import Path

import pytest

from trace.verified_prefix_replay import (
    PrefixCall,
    PrefixRecord,
    ReplayMismatch,
    canonical_sha256,
    replay_prefix_fresh_process,
)


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "stateful_replay_tool.py"
)

FAILURE_SEED = "trace-v2-synthetic-seed-v1"


def _valid_record() -> PrefixRecord:
    return PrefixRecord(
        task_id="synthetic-hidden-state-task",
        failure_seed=FAILURE_SEED,
        calls=(
            PrefixCall(
                function_name="synthetic_lookup",
                arguments={"item": "alpha"},
                expected_result={
                    "ok": True,
                    "item": "alpha",
                    "value": "resolved:alpha",
                },
            ),
            PrefixCall(
                function_name="synthetic_lookup",
                arguments={"item": "beta"},
                expected_result={
                    "ok": True,
                    "item": "beta",
                    "value": "resolved:beta",
                },
            ),
            PrefixCall(
                function_name="synthetic_lookup",
                arguments={"item": "gamma"},
                expected_result={
                    "ok": False,
                    "error": "synthetic_execution_failure",
                    "item": "gamma",
                },
            ),
        ),
        first_hazard_call_index=2,
    )


def test_canonical_sha256_is_independent_of_mapping_key_order() -> None:
    left = {
        "task": "t1",
        "nested": {
            "b": 2,
            "a": 1,
        },
        "values": [3, 2, 1],
    }

    right = {
        "values": [3, 2, 1],
        "nested": {
            "a": 1,
            "b": 2,
        },
        "task": "t1",
    }

    assert canonical_sha256(left) == canonical_sha256(right)


def test_two_fresh_process_replays_reconstruct_identical_branch_state() -> None:
    record = _valid_record()

    first = replay_prefix_fresh_process(
        module_path=FIXTURE,
        record=record,
        state_probe_name="replay_state_probe",
    )

    second = replay_prefix_fresh_process(
        module_path=FIXTURE,
        record=record,
        state_probe_name="replay_state_probe",
    )

    # A same-interpreter module reset is not sufficient evidence.
    assert first.process_id != second.process_id

    # Exact prefix evidence must reproduce.
    assert first.ordered_argument_hashes == second.ordered_argument_hashes
    assert first.ordered_result_hashes == second.ordered_result_hashes

    # Both replays must identify the exact same intervention boundary.
    assert first.first_hazard_call_index == 2
    assert second.first_hazard_call_index == 2

    assert (
        first.first_hazard_observation_hash
        == second.first_hazard_observation_hash
    )

    # Hidden monotonic tool state must also reconstruct.
    assert first.hidden_state_hash is not None
    assert first.hidden_state_hash == second.hidden_state_hash

    # Final equivalence evidence must match byte-for-byte semantically.
    assert first.branch_state_hash == second.branch_state_hash


def test_replay_refuses_if_recorded_prefix_result_does_not_reproduce() -> None:
    record = PrefixRecord(
        task_id="synthetic-tampered-prefix",
        failure_seed=FAILURE_SEED,
        calls=(
            PrefixCall(
                function_name="synthetic_lookup",
                arguments={"item": "alpha"},
                expected_result={
                    "ok": True,
                    "item": "alpha",
                    "value": "THIS_RESULT_WAS_TAMPERED",
                },
            ),
        ),
        first_hazard_call_index=0,
    )

    with pytest.raises(ReplayMismatch):
        replay_prefix_fresh_process(
            module_path=FIXTURE,
            record=record,
            state_probe_name="replay_state_probe",
        )
