import importlib
import inspect
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from config import (
    MODEL,
    AGENT_TEMPERATURE,
    DETECTOR_TEMPERATURE,
    MAX_STEPS,
    N_MAX,
    K_WINDOW,
    THETA_H,
    THETA_GROUND,
    THETA_LOOP,
    N_LOOP,
    RHO_THRESHOLD,
)
from trace.llm_gateway import MeteredLLMGateway
from trace.run_artifacts import verify_run_artifact


def load_runner_api(testcase):
    try:
        module = importlib.import_module(
            "trace.v2_runner"
        )
    except ModuleNotFoundError:
        testcase.fail(
            "trace.v2_runner does not exist; TRACE v2 has no "
            "official execution boundary."
        )

    required = (
        "capture_git_source",
        "capture_config_snapshot",
        "execute_trace_v2_run",
    )

    for name in required:
        testcase.assertTrue(
            hasattr(module, name),
            msg=f"trace.v2_runner is missing {name}.",
        )

    return module


class FakeCompletions:
    """
    Provider-compatible fake that exercises the real MeteredLLMGateway.

    Sequence:
      1. agent proposes one tool call and makes a claim
      2. detector_f1 marks the claim SUPPORTED
      3. agent returns the correct terminal resolution
    """

    def __init__(self):
        self.requests = []
        self.agent_calls = 0

    def create(self, **kwargs):
        self.requests.append(kwargs)

        last_content = (
            kwargs["messages"][-1]["content"]
        )

        if last_content.startswith("GROUNDING"):
            content = "SUPPORTED"
        elif last_content.startswith("CONTRADICTION"):
            content = "CONSISTENT"
        else:
            self.agent_calls += 1

            if self.agent_calls == 1:
                content = (
                    "Reasoning: the root cause is "
                    "connection exhaustion.\n"
                    "Action: check_service_status"
                )
            else:
                content = (
                    "RESOLUTION: connection exhaustion"
                )

        request_number = len(self.requests)

        return SimpleNamespace(
            id=f"fake-request-{request_number}",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=content
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=12,
                completion_tokens=4,
                total_tokens=16,
            ),
        )


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(
            completions=self.completions
        )


def synthetic_task():
    return {
        "task_id": "synthetic_e2e",
        "difficulty_level": 1,
        "incident_description": (
            "The API is failing because the database "
            "connection pool is exhausted."
        ),
        "available_tools": [
            "check_service_status",
        ],
        "ground_truth": {
            "root_cause": "connection exhaustion",
            "correct_outcome": "resolve",
            "correct_tool_sequence": [
                "check_service_status",
            ],
            "escalation_required": False,
        },
        "injected_failures": [],
        "knowledge_environment": {},
        "loop_risk": "low",
        "noise_level": "none",
    }


def synthetic_prompts():
    return {
        "system_react": (
            "You are an engineering agent.\n"
            "Available tools: {tool_list}"
        ),
        "grounding_check": (
            "GROUNDING\n"
            "Claim: {claim}\n"
            "Evidence: {evidence}"
        ),
        "contradiction_check": (
            "CONTRADICTION\n"
            "New: {new_statement}\n"
            "Prior: {prior_statements}"
        ),
    }


def clean_source_probe(_repo_root):
    return {
        "git_commit": (
            "0123456789abcdef"
            "0123456789abcdef"
            "01234567"
        ),
        "git_dirty": False,
    }


class TestV2ExecutionRunnerContract(unittest.TestCase):

    def test_config_snapshot_is_exact_and_secret_free(self):
        module = load_runner_api(self)

        snapshot = (
            module.capture_config_snapshot()
        )

        expected = {
            "model": MODEL,
            "agent_temperature":
                AGENT_TEMPERATURE,
            "detector_temperature":
                DETECTOR_TEMPERATURE,
            "max_steps": MAX_STEPS,
            "n_max": N_MAX,
            "k_window": K_WINDOW,
            "theta_h": THETA_H,
            "theta_ground": THETA_GROUND,
            "theta_loop": THETA_LOOP,
            "n_loop": N_LOOP,
            "rho_threshold":
                RHO_THRESHOLD,
        }

        self.assertEqual(
            snapshot,
            expected,
        )

        serialized = json.dumps(
            snapshot
        ).lower()

        self.assertNotIn(
            "api_key",
            serialized,
        )

        self.assertNotIn(
            "openai_api_key",
            serialized,
        )

    def test_runner_has_no_legacy_scorer_or_results_dependency(self):
        module = load_runner_api(self)

        source = inspect.getsource(module)

        self.assertNotIn(
            "evaluation.scorer",
            source,
        )

        self.assertNotIn(
            "RESULTS_DIR",
            source,
        )

        self.assertNotIn(
            "results/main",
            source,
        )

    def test_dirty_source_is_rejected_before_inference(self):
        module = load_runner_api(self)

        client = FakeClient()

        gateway = MeteredLLMGateway(
            client=client,
            model=MODEL,
        )

        def dirty_source_probe(_repo_root):
            return {
                "git_commit": "deadbeef",
                "git_dirty": True,
            }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            with self.assertRaises(RuntimeError):
                module.execute_trace_v2_run(
                    run_id="dirty-run",
                    task=synthetic_task(),
                    treatment=(
                        "predicted_attribution"
                    ),
                    seed=7,
                    prompts=synthetic_prompts(),
                    root_dir=root,
                    llm_gateway=gateway,
                    source_probe=(
                        dirty_source_probe
                    ),
                )

            self.assertFalse(
                (
                    root
                    / "artifacts"
                    / "v2"
                ).exists()
            )

        self.assertEqual(
            client.completions.requests,
            [],
            msg=(
                "Dirty-source refusal must occur "
                "before any provider request."
            ),
        )

    def test_duplicate_run_id_is_refused_before_inference(self):
        module = load_runner_api(self)

        client = FakeClient()

        gateway = MeteredLLMGateway(
            client=client,
            model=MODEL,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            existing = (
                root
                / "artifacts"
                / "v2"
                / "runs"
                / "duplicate-run"
            )

            existing.mkdir(
                parents=True
            )

            (
                existing
                / "run.json"
            ).write_text(
                '{"existing": true}\n'
            )

            with self.assertRaises(
                FileExistsError
            ):
                module.execute_trace_v2_run(
                    run_id="duplicate-run",
                    task=synthetic_task(),
                    treatment=(
                        "predicted_attribution"
                    ),
                    seed=7,
                    prompts=synthetic_prompts(),
                    root_dir=root,
                    llm_gateway=gateway,
                    source_probe=(
                        clean_source_probe
                    ),
                )

        self.assertEqual(
            client.completions.requests,
            [],
            msg=(
                "Duplicate run IDs must be "
                "rejected before inference cost."
            ),
        )

    def test_offline_trace_run_produces_complete_v2_artifact(self):
        module = load_runner_api(self)

        client = FakeClient()

        gateway = MeteredLLMGateway(
            client=client,
            model=MODEL,
        )

        task = synthetic_task()
        prompts = synthetic_prompts()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            responses_root = (
                root
                / "responses"
            )

            task_response_dir = (
                responses_root
                / task["task_id"]
            )

            task_response_dir.mkdir(
                parents=True
            )

            (
                task_response_dir
                / "check_service_status.json"
            ).write_text(
                json.dumps({
                    "response": {
                        "status": "success",
                        "service": "api",
                        "database_pool": (
                            "exhausted"
                        ),
                    }
                })
            )

            # A v2 execution must never enter the
            # legacy AuditLayer persistence path.
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
                        "legacy trace persistence "
                        "must not execute"
                    ),
                ),
            ):
                path = (
                    module.execute_trace_v2_run(
                        run_id=(
                            "synthetic-run-001"
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

            expected_path = (
                root
                / "artifacts"
                / "v2"
                / "runs"
                / "synthetic-run-001"
                / "run.json"
            )

            self.assertEqual(
                path.resolve(),
                expected_path.resolve(),
            )

            self.assertTrue(
                path.exists()
            )

            artifact = json.loads(
                path.read_text()
            )

            self.assertTrue(
                verify_run_artifact(
                    artifact
                )
            )

            self.assertEqual(
                artifact["run_id"],
                "synthetic-run-001",
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

            self.assertEqual(
                artifact["task"][
                    "snapshot"
                ],
                task,
            )

            expected_config = (
                module
                .capture_config_snapshot()
            )

            self.assertEqual(
                artifact["config"][
                    "snapshot"
                ],
                expected_config,
            )

            self.assertEqual(
                set(
                    artifact[
                        "prompts"
                    ].keys()
                ),
                {
                    "system_react_template",
                    "system_react_rendered",
                    "grounding_check",
                    "contradiction_check",
                },
            )

            self.assertEqual(
                artifact["prompts"][
                    "system_react_template"
                ]["text"],
                prompts["system_react"],
            )

            self.assertEqual(
                artifact["prompts"][
                    "system_react_rendered"
                ]["text"],
                (
                    "You are an engineering agent.\n"
                    "Available tools: "
                    "check_service_status"
                ),
            )

            llm_calls = (
                artifact["llm"]["calls"]
            )

            self.assertEqual(
                [
                    call["call_id"]
                    for call in llm_calls
                ],
                [1, 2, 3],
            )

            self.assertEqual(
                [
                    call["purpose"]
                    for call in llm_calls
                ],
                [
                    "agent",
                    "detector_f1",
                    "agent",
                ],
            )

            self.assertEqual(
                artifact["llm"][
                    "usage_totals"
                ],
                {
                    "calls": 3,
                    "successful_calls": 3,
                    "failed_calls": 0,
                    "prompt_tokens": 36,
                    "completion_tokens": 12,
                    "total_tokens": 48,
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

            self.assertEqual(
                tool_calls[0][
                    "fixture_path"
                ],
                (
                    "synthetic_e2e/"
                    "check_service_status.json"
                ),
            )

            execution = (
                artifact["execution"]
            )

            self.assertEqual(
                len(
                    execution["trajectory"]
                ),
                1,
            )

            self.assertEqual(
                execution[
                    "failure_events"
                ],
                [],
            )

            self.assertEqual(
                execution[
                    "recovery_events"
                ],
                [],
            )

            self.assertEqual(
                execution[
                    "terminal_state"
                ],
                "s_OK",
            )

            self.assertTrue(
                execution[
                    "goal_satisfied"
                ]
            )

            self.assertEqual(
                execution[
                    "final_response"
                ],
                (
                    "RESOLUTION: "
                    "connection exhaustion"
                ),
            )

            # The temp execution namespace must
            # contain only v2 artifacts, not a
            # legacy results tree.
            self.assertFalse(
                (root / "results").exists()
            )

    def test_capture_git_source_reports_exact_commit_and_dirty_state(self):
        """
        The official source probe must record the exact Git commit and must
        distinguish a clean checkout from modified source.
        """
        module = load_runner_api(self)

        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)

            subprocess.run(
                ["git", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )

            subprocess.run(
                [
                    "git",
                    "config",
                    "user.email",
                    "trace-test@example.invalid",
                ],
                cwd=repo,
                check=True,
            )

            subprocess.run(
                [
                    "git",
                    "config",
                    "user.name",
                    "TRACE Test",
                ],
                cwd=repo,
                check=True,
            )

            tracked = repo / "tracked.txt"
            tracked.write_text("version-1\n")

            subprocess.run(
                ["git", "add", "tracked.txt"],
                cwd=repo,
                check=True,
            )

            subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "fixture commit",
                ],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )

            expected_commit = subprocess.run(
                [
                    "git",
                    "rev-parse",
                    "HEAD",
                ],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            clean = module.capture_git_source(
                repo
            )

            self.assertEqual(
                clean,
                {
                    "git_commit": expected_commit,
                    "git_dirty": False,
                },
            )

            self.assertEqual(
                len(clean["git_commit"]),
                40,
            )

            tracked.write_text("version-2\n")

            dirty = module.capture_git_source(
                repo
            )

            self.assertEqual(
                dirty["git_commit"],
                expected_commit,
            )

            self.assertTrue(
                dirty["git_dirty"],
            )

    def test_unimplemented_treatment_is_rejected_before_inference(self):
        """
        Artifact treatment labels must describe behavior that actually
        exists. Unsupported causal arms must fail before provider use.
        """
        module = load_runner_api(self)

        client = FakeClient()

        gateway = MeteredLLMGateway(
            client=client,
            model=MODEL,
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            with self.assertRaises(ValueError):
                module.execute_trace_v2_run(
                    run_id=(
                        "unsupported-treatment"
                    ),
                    task=synthetic_task(),
                    treatment=(
                        "oracle_attribution"
                    ),
                    seed=7,
                    prompts=synthetic_prompts(),
                    root_dir=root,
                    llm_gateway=gateway,
                    source_probe=(
                        clean_source_probe
                    ),
                )

            self.assertFalse(
                (
                    root
                    / "artifacts"
                    / "v2"
                ).exists()
            )

        self.assertEqual(
            client.completions.requests,
            [],
            msg=(
                "An unimplemented treatment must "
                "be rejected before inference."
            ),
        )



if __name__ == "__main__":
    unittest.main()
