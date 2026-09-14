import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import MODEL
from trace.llm_gateway import MeteredLLMGateway
from trace.run_artifacts import verify_run_artifact

from tests.test_v2_execution_runner_contract import (
    FakeClient,
    clean_source_probe,
    synthetic_prompts,
    synthetic_task,
)


FAILURE_MESSAGE = (
    "synthetic provider failure after partial execution"
)


class FailingCompletions:
    """
    First provider request succeeds and causes a real tool execution.

    The second provider request fails. In the current TRACE path this is
    expected to be the detector_f1 call, which gives us genuine partial
    agent/tool/attribution evidence before the exception.
    """

    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)

        request_number = len(
            self.requests
        )

        if request_number == 1:
            return SimpleNamespace(
                id="partial-request-1",
                model=kwargs["model"],
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=(
                                "Reasoning: the root cause is "
                                "connection exhaustion.\n"
                                "Action: check_service_status"
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=4,
                    total_tokens=16,
                ),
            )

        if request_number == 2:
            raise RuntimeError(
                FAILURE_MESSAGE
            )

        raise AssertionError(
            "The failed run must stop after "
            "the injected second-call failure."
        )


class FailingClient:
    def __init__(self):
        self.completions = (
            FailingCompletions()
        )

        self.chat = SimpleNamespace(
            completions=self.completions
        )


def load_runner_api(testcase):
    module = importlib.import_module(
        "trace.v2_runner"
    )

    testcase.assertTrue(
        hasattr(
            module,
            "V2RunExecutionError",
        ),
        msg=(
            "The v2 runner needs a dedicated "
            "execution error that carries the "
            "persisted artifact path."
        ),
    )

    return module


def create_fixture_tree(root, task):
    responses_root = (
        root / "responses"
    )

    task_dir = (
        responses_root
        / task["task_id"]
    )

    task_dir.mkdir(
        parents=True
    )

    (
        task_dir
        / "check_service_status.json"
    ).write_text(
        json.dumps({
            "response": {
                "status": "success",
                "service": "api",
                "database_pool":
                    "exhausted",
            }
        })
    )

    return responses_root


class TestV2FailedRunArtifactContract(
    unittest.TestCase
):

    def test_partial_provider_failure_is_persisted_then_surfaced(self):
        """
        Once execution has begun, a provider/runtime failure is scientific
        evidence. The partial run must be written before the caller receives
        an exception.
        """

        module = load_runner_api(self)

        task = synthetic_task()
        prompts = synthetic_prompts()

        client = FailingClient()

        gateway = MeteredLLMGateway(
            client=client,
            model=MODEL,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            responses_root = (
                create_fixture_tree(
                    root,
                    task,
                )
            )

            with (
                patch(
                    "tools.tool_layer."
                    "RESPONSES_DIR",
                    str(responses_root),
                ),
                patch(
                    "trace.audit_layer."
                    "AuditLayer.save",
                    side_effect=AssertionError(
                        "legacy persistence must "
                        "not execute"
                    ),
                ),
            ):
                with self.assertRaises(
                    module.V2RunExecutionError
                ) as caught:
                    module.execute_trace_v2_run(
                        run_id=(
                            "failed-run-001"
                        ),
                        task=task,
                        treatment=(
                            "predicted_attribution"
                        ),
                        seed=7,
                        prompts=prompts,
                        root_dir=root,
                        llm_gateway=gateway,
                        source_probe=(
                            clean_source_probe
                        ),
                    )

            error = caught.exception

            expected_path = (
                root
                / "artifacts"
                / "v2"
                / "runs"
                / "failed-run-001"
                / "run.json"
            )

            self.assertEqual(
                Path(
                    error.artifact_path
                ).resolve(),
                expected_path.resolve(),
            )

            self.assertEqual(
                error.run_id,
                "failed-run-001",
            )

            self.assertIsInstance(
                error.__cause__,
                RuntimeError,
            )

            self.assertEqual(
                str(error.__cause__),
                FAILURE_MESSAGE,
            )

            self.assertTrue(
                expected_path.exists()
            )

            artifact = json.loads(
                expected_path.read_text()
            )

            self.assertTrue(
                verify_run_artifact(
                    artifact
                )
            )

            self.assertEqual(
                artifact["run_id"],
                "failed-run-001",
            )

            self.assertEqual(
                artifact["treatment"],
                "predicted_attribution",
            )

            self.assertEqual(
                artifact["seed"],
                7,
            )

            self.assertEqual(
                artifact["source"],
                clean_source_probe(None),
            )

            llm_calls = (
                artifact["llm"]["calls"]
            )

            self.assertEqual(
                [
                    call["call_id"]
                    for call in llm_calls
                ],
                [1, 2],
            )

            self.assertEqual(
                [
                    call["purpose"]
                    for call in llm_calls
                ],
                [
                    "agent",
                    "detector_f1",
                ],
            )

            self.assertEqual(
                [
                    call["status"]
                    for call in llm_calls
                ],
                [
                    "success",
                    "error",
                ],
            )

            failed_call = (
                llm_calls[1]
            )

            self.assertEqual(
                failed_call[
                    "error_type"
                ],
                "RuntimeError",
            )

            self.assertEqual(
                failed_call[
                    "error_message"
                ],
                FAILURE_MESSAGE,
            )

            self.assertEqual(
                artifact["llm"][
                    "usage_totals"
                ],
                {
                    "calls": 2,
                    "successful_calls": 1,
                    "failed_calls": 1,
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "total_tokens": 16,
                },
            )

            tool_calls = (
                artifact["tools"]["calls"]
            )

            self.assertEqual(
                len(tool_calls),
                1,
            )

            self.assertEqual(
                tool_calls[0][
                    "tool_name"
                ],
                "check_service_status",
            )

            execution = (
                artifact["execution"]
            )

            self.assertEqual(
                execution["status"],
                "error",
            )

            self.assertEqual(
                execution["error"],
                {
                    "phase": "agent_run",
                    "type": "RuntimeError",
                    "message":
                        FAILURE_MESSAGE,
                },
            )

            self.assertEqual(
                len(
                    execution[
                        "trajectory"
                    ]
                ),
                1,
            )

            self.assertEqual(
                execution[
                    "terminal_state"
                ],
                None,
            )

            self.assertFalse(
                execution[
                    "goal_satisfied"
                ]
            )

            self.assertIsNone(
                execution[
                    "final_response"
                ]
            )

            self.assertFalse(
                (root / "results").exists()
            )

    def test_completed_run_has_explicit_completed_status(self):
        """
        Successful and failed artifacts must use the same explicit execution
        status field so downstream analysis never infers completion merely
        from missing error metadata.
        """

        module = load_runner_api(self)

        task = synthetic_task()
        prompts = synthetic_prompts()

        client = FakeClient()

        gateway = MeteredLLMGateway(
            client=client,
            model=MODEL,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            responses_root = (
                create_fixture_tree(
                    root,
                    task,
                )
            )

            with (
                patch(
                    "tools.tool_layer."
                    "RESPONSES_DIR",
                    str(responses_root),
                ),
                patch(
                    "trace.audit_layer."
                    "AuditLayer.save",
                    side_effect=AssertionError(
                        "legacy persistence must "
                        "not execute"
                    ),
                ),
            ):
                path = (
                    module.execute_trace_v2_run(
                        run_id=(
                            "completed-run-001"
                        ),
                        task=task,
                        treatment=(
                            "predicted_attribution"
                        ),
                        seed=7,
                        prompts=prompts,
                        root_dir=root,
                        llm_gateway=gateway,
                        source_probe=(
                            clean_source_probe
                        ),
                    )
                )

            artifact = json.loads(
                path.read_text()
            )

            self.assertEqual(
                artifact["execution"][
                    "status"
                ],
                "completed",
            )

            self.assertIsNone(
                artifact["execution"][
                    "error"
                ]
            )

            self.assertTrue(
                verify_run_artifact(
                    artifact
                )
            )

    def test_failed_run_id_cannot_be_reused_for_retry(self):
        """
        A failed execution is still an immutable run. Retrying under the same
        run_id would permit silent replacement/selection of inconvenient
        failures, so a retry must use a new run identity.
        """

        module = load_runner_api(self)

        task = synthetic_task()
        prompts = synthetic_prompts()

        failing_client = (
            FailingClient()
        )

        failing_gateway = (
            MeteredLLMGateway(
                client=failing_client,
                model=MODEL,
            )
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            responses_root = (
                create_fixture_tree(
                    root,
                    task,
                )
            )

            with patch(
                "tools.tool_layer."
                "RESPONSES_DIR",
                str(responses_root),
            ):
                with self.assertRaises(
                    module.V2RunExecutionError
                ):
                    module.execute_trace_v2_run(
                        run_id=(
                            "failed-run-retry"
                        ),
                        task=task,
                        treatment=(
                            "predicted_attribution"
                        ),
                        seed=7,
                        prompts=prompts,
                        root_dir=root,
                        llm_gateway=(
                            failing_gateway
                        ),
                        source_probe=(
                            clean_source_probe
                        ),
                    )

            failed_path = (
                root
                / "artifacts"
                / "v2"
                / "runs"
                / "failed-run-retry"
                / "run.json"
            )

            self.assertTrue(
                failed_path.exists()
            )

            original_bytes = (
                failed_path.read_bytes()
            )

            retry_client = (
                FakeClient()
            )

            retry_gateway = (
                MeteredLLMGateway(
                    client=retry_client,
                    model=MODEL,
                )
            )

            with self.assertRaises(
                FileExistsError
            ):
                module.execute_trace_v2_run(
                    run_id=(
                        "failed-run-retry"
                    ),
                    task=task,
                    treatment=(
                        "predicted_attribution"
                    ),
                    seed=7,
                    prompts=prompts,
                    root_dir=root,
                    llm_gateway=(
                        retry_gateway
                    ),
                    source_probe=(
                        clean_source_probe
                    ),
                )

            self.assertEqual(
                retry_client
                .completions
                .requests,
                [],
                msg=(
                    "A retry collision must be "
                    "rejected before inference."
                ),
            )

            self.assertEqual(
                failed_path.read_bytes(),
                original_bytes,
                msg=(
                    "The failed artifact must "
                    "remain byte-for-byte immutable."
                ),
            )


if __name__ == "__main__":
    unittest.main()
