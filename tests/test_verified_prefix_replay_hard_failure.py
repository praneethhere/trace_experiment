from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

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
    / "stateful_exception_replay_tool.py"
)

FAILURE_SEED = (
    "trace-v2-synthetic-hard-failure-seed-v1"
)

EXPECTED_EXCEPTION_OBSERVATION = {
    "__trace_v2_observation__": {
        "kind": "exception",
        "type": "RuntimeError",
        "message": "synthetic_hard_failure:gamma",
    }
}

EXPECTED_HIDDEN_BRANCH_STATE = {
    "call_count": 3,
    "failure_seed": FAILURE_SEED,
}


def _hard_failure_record() -> PrefixRecord:
    return PrefixRecord(
        task_id="synthetic-hard-failure-task",
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
                expected_result=(
                    EXPECTED_EXCEPTION_OBSERVATION
                ),
            ),
        ),
        first_hazard_call_index=2,
    )


def test_hard_exception_prefix_replays_as_branch_evidence() -> None:
    record = _hard_failure_record()

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

    assert first.process_id != second.process_id

    assert (
        first.ordered_argument_hashes
        == second.ordered_argument_hashes
    )

    assert (
        first.ordered_result_hashes
        == second.ordered_result_hashes
    )

    assert (
        first.first_hazard_call_index
        == 2
    )

    expected_hazard_hash = canonical_sha256(
        EXPECTED_EXCEPTION_OBSERVATION
    )

    assert (
        first.first_hazard_observation_hash
        == expected_hazard_hash
    )

    assert (
        second.first_hazard_observation_hash
        == expected_hazard_hash
    )

    expected_hidden_hash = canonical_sha256(
        EXPECTED_HIDDEN_BRANCH_STATE
    )

    assert (
        first.hidden_state_hash
        == expected_hidden_hash
    )

    assert (
        second.hidden_state_hash
        == expected_hidden_hash
    )

    assert (
        first.branch_state_hash
        == second.branch_state_hash
    )


def test_wrong_expected_exception_is_refused() -> None:
    wrong_observation = {
        "__trace_v2_observation__": {
            "kind": "exception",
            "type": "RuntimeError",
            "message": (
                "synthetic_hard_failure:"
                "WRONG"
            ),
        }
    }

    record = PrefixRecord(
        task_id="synthetic-hard-failure-mismatch",
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
                expected_result=wrong_observation,
            ),
        ),
        first_hazard_call_index=2,
    )

    with pytest.raises(ReplayMismatch):
        replay_prefix_fresh_process(
            module_path=FIXTURE,
            record=record,
            state_probe_name="replay_state_probe",
        )


def test_visible_snapshot_without_prefix_replay_is_not_branch_equivalent() -> None:
    visible_snapshot = {
        "task_id":
            "synthetic-hard-failure-task",
        "failure_step": 2,
        "last_tool":
            "synthetic_lookup",
        "last_error": {
            "type": "RuntimeError",
            "message":
                "synthetic_hard_failure:gamma",
        },
    }

    child_code = r'''
import importlib.util
import json
import os
from pathlib import Path
import sys

module_path = Path(sys.argv[1]).resolve()

spec = importlib.util.spec_from_file_location(
    "_trace_snapshot_only_negative_control",
    module_path,
)

if spec is None or spec.loader is None:
    raise RuntimeError("unable to load fixture")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

visible = json.loads(
    os.environ["AGENT_VISIBLE_STATE_JSON"]
)

print(
    json.dumps(
        {
            "process_id": os.getpid(),
            "visible_snapshot": visible,
            "hidden_state":
                module.replay_state_probe(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
'''

    environment = os.environ.copy()

    environment["FAIL_SEED"] = FAILURE_SEED

    environment[
        "AGENT_VISIBLE_STATE_JSON"
    ] = json.dumps(
        visible_snapshot,
        sort_keys=True,
        separators=(",", ":"),
    )

    process = subprocess.run(
        [
            sys.executable,
            "-c",
            child_code,
            str(FIXTURE),
        ],
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )

    observed = json.loads(
        process.stdout.strip()
    )

    assert (
        observed["visible_snapshot"]
        == visible_snapshot
    )

    assert (
        observed["hidden_state"][
            "failure_seed"
        ]
        == FAILURE_SEED
    )

    assert (
        observed["hidden_state"][
            "call_count"
        ]
        == 0
    )

    snapshot_only_hidden_hash = (
        canonical_sha256(
            observed["hidden_state"]
        )
    )

    true_branch_hidden_hash = (
        canonical_sha256(
            EXPECTED_HIDDEN_BRANCH_STATE
        )
    )

    assert (
        snapshot_only_hidden_hash
        != true_branch_hidden_hash
    )
