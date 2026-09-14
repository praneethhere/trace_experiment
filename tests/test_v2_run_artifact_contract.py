import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def load_artifact_api(testcase):
    try:
        module = importlib.import_module(
            "trace.run_artifacts"
        )
    except ModuleNotFoundError:
        testcase.fail(
            "trace.run_artifacts does not exist; TRACE v2 has no "
            "immutable raw-run artifact boundary."
        )

    testcase.assertTrue(
        hasattr(module, "build_run_artifact"),
        msg="build_run_artifact is missing.",
    )

    testcase.assertTrue(
        hasattr(module, "write_run_artifact"),
        msg="write_run_artifact is missing.",
    )

    return (
        module.build_run_artifact,
        module.write_run_artifact,
    )


class TestV2RunArtifactContract(unittest.TestCase):

    def _inputs(self):
        task = {
            "task_id": "task_001",
            "incident_description": "API timeout",
            "available_tools": [
                "check_status",
            ],
            "ground_truth": {
                "root_cause": "connection exhaustion",
            },
        }

        config_snapshot = {
            "model": "offline-model",
            "agent_temperature": 0.2,
            "detector_temperature": 0.0,
            "max_steps": 20,
            "n_max": 3,
            "k_window": 5,
        }

        prompts = {
            "system_react": "system prompt",
            "grounding_check": "grounding prompt",
        }

        llm_calls = [
            {
                "call_id": 1,
                "purpose": "agent",
                "model": "offline-model",
                "temperature": 0.2,
                "status": "success",
                "messages": [
                    {
                        "role": "user",
                        "content": "diagnose",
                    }
                ],
                "messages_sha256": "a" * 64,
                "provider_request_id": "req-1",
                "provider_model": "offline-model",
                "response_text": "Action: check_status",
                "response_sha256": "b" * 64,
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "elapsed_ms": 2.5,
            }
        ]

        tool_calls = [
            {
                "call_id": 1,
                "tool_call_index": 1,
                "tool_name": "check_status",
                "tool_attempt": 1,
                "args": None,
                "args_sha256": "c" * 64,
                "fixture_path": (
                    "tools/responses/task_001/check_status.json"
                ),
                "fixture_sha256": "d" * 64,
                "response": {
                    "status": "success",
                },
                "response_sha256": "e" * 64,
            }
        ]

        execution = {
            "trajectory": [
                {
                    "step": 0,
                    "reasoning": "inspect status",
                    "action": "check_status",
                    "observation": {
                        "status": "success",
                    },
                }
            ],
            "failure_events": [],
            "recovery_events": [],
            "terminal_state": "s_OK",
            "goal_satisfied": True,
            "final_response": (
                "RESOLUTION: connection exhaustion"
            ),
        }

        source = {
            "git_commit": "0123456789abcdef",
            "git_dirty": False,
        }

        return {
            "run_id": "run-000001",
            "task": task,
            "treatment": "predicted_attribution",
            "seed": 7,
            "source": source,
            "config_snapshot": config_snapshot,
            "prompts": prompts,
            "llm_calls": llm_calls,
            "llm_usage_totals": {
                "calls": 1,
                "successful_calls": 1,
                "failed_calls": 0,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "tool_calls": tool_calls,
            "execution": execution,
        }

    def test_bundle_contains_complete_reproducibility_inputs(self):
        build_run_artifact, _ = load_artifact_api(self)

        inputs = self._inputs()

        artifact = build_run_artifact(**inputs)

        self.assertEqual(
            artifact["schema_version"],
            "trace-v2-run-artifact/1",
        )

        self.assertEqual(
            artifact["run_id"],
            "run-000001",
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
            inputs["source"],
        )

        self.assertEqual(
            artifact["task"]["snapshot"],
            inputs["task"],
        )
        self.assertEqual(
            artifact["task"]["sha256"],
            canonical_sha256(inputs["task"]),
        )

        self.assertEqual(
            artifact["config"]["snapshot"],
            inputs["config_snapshot"],
        )
        self.assertEqual(
            artifact["config"]["sha256"],
            canonical_sha256(
                inputs["config_snapshot"]
            ),
        )

        for name, text in inputs["prompts"].items():
            self.assertEqual(
                artifact["prompts"][name]["text"],
                text,
            )
            self.assertEqual(
                len(
                    artifact["prompts"][name][
                        "sha256"
                    ]
                ),
                64,
            )

        self.assertEqual(
            artifact["llm"]["calls"],
            inputs["llm_calls"],
        )
        self.assertEqual(
            artifact["llm"]["usage_totals"],
            inputs["llm_usage_totals"],
        )

        self.assertEqual(
            artifact["tools"]["calls"],
            inputs["tool_calls"],
        )

        self.assertEqual(
            artifact["execution"],
            inputs["execution"],
        )

        self.assertEqual(
            len(artifact["artifact_sha256"]),
            64,
        )

    def test_builder_defensively_copies_inputs(self):
        build_run_artifact, _ = load_artifact_api(self)

        inputs = self._inputs()

        artifact = build_run_artifact(**inputs)

        inputs["task"]["incident_description"] = "MUTATED"
        inputs["llm_calls"][0]["purpose"] = "MUTATED"
        inputs["tool_calls"][0]["tool_name"] = "MUTATED"
        inputs["execution"]["terminal_state"] = "MUTATED"

        self.assertEqual(
            artifact["task"]["snapshot"][
                "incident_description"
            ],
            "API timeout",
        )
        self.assertEqual(
            artifact["llm"]["calls"][0]["purpose"],
            "agent",
        )
        self.assertEqual(
            artifact["tools"]["calls"][0][
                "tool_name"
            ],
            "check_status",
        )
        self.assertEqual(
            artifact["execution"]["terminal_state"],
            "s_OK",
        )

    def test_artifact_hash_detects_content_change(self):
        build_run_artifact, _ = load_artifact_api(self)

        first = build_run_artifact(**self._inputs())

        changed = self._inputs()
        changed["seed"] = 8

        second = build_run_artifact(**changed)

        self.assertNotEqual(
            first["artifact_sha256"],
            second["artifact_sha256"],
        )

    def test_writer_uses_v2_namespace_and_refuses_overwrite(self):
        build_run_artifact, write_run_artifact = (
            load_artifact_api(self)
        )

        artifact = build_run_artifact(
            **self._inputs()
        )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            path = write_run_artifact(
                artifact,
                root_dir=root,
            )

            expected = (
                root
                / "artifacts"
                / "v2"
                / "runs"
                / "run-000001"
                / "run.json"
            )

            self.assertEqual(
                path.resolve(),
                expected.resolve(),
            )
            self.assertTrue(path.exists())

            persisted = json.loads(
                path.read_text()
            )

            self.assertEqual(
                persisted,
                artifact,
            )

            with self.assertRaises(FileExistsError):
                write_run_artifact(
                    artifact,
                    root_dir=root,
                )

            self.assertEqual(
                json.loads(path.read_text()),
                artifact,
            )

    def test_dirty_source_is_preserved_not_hidden(self):
        build_run_artifact, _ = load_artifact_api(self)

        inputs = self._inputs()
        inputs["source"]["git_dirty"] = True

        artifact = build_run_artifact(**inputs)

        self.assertTrue(
            artifact["source"]["git_dirty"],
            msg=(
                "Run provenance must never silently convert a "
                "dirty source tree into a clean one."
            ),
        )


if __name__ == "__main__":
    unittest.main()
