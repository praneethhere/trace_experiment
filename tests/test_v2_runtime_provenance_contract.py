import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import (
    MODEL,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
)
from trace.llm_gateway import MeteredLLMGateway
from trace.run_artifacts import verify_run_artifact

from tests.test_v2_execution_runner_contract import (
    FakeClient,
    clean_source_probe,
    synthetic_prompts,
    synthetic_task,
)
from tests.test_v2_failed_run_artifact_contract import (
    create_fixture_tree,
)


def canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def load_runner():
    return importlib.import_module(
        "trace.v2_runner"
    )


class TestV2RuntimeProvenanceContract(
    unittest.TestCase
):

    def test_runtime_snapshot_is_explicit_complete_and_secret_free(
        self,
    ):
        module = load_runner()

        self.assertTrue(
            hasattr(
                module,
                "capture_runtime_snapshot",
            ),
            msg=(
                "Official v2 evidence has no "
                "runtime provenance capture."
            ),
        )

        snapshot = (
            module.capture_runtime_snapshot()
        )

        self.assertEqual(
            set(snapshot),
            {
                "python",
                "platform",
                "packages",
                "provider_transport",
            },
        )

        self.assertEqual(
            set(snapshot["python"]),
            {
                "version",
                "implementation",
            },
        )

        self.assertIsInstance(
            snapshot["python"]["version"],
            str,
        )

        self.assertTrue(
            snapshot["python"]["version"]
        )

        self.assertIsInstance(
            snapshot["python"][
                "implementation"
            ],
            str,
        )

        self.assertTrue(
            snapshot["python"][
                "implementation"
            ]
        )

        self.assertEqual(
            set(snapshot["platform"]),
            {
                "system",
                "release",
                "machine",
            },
        )

        for value in (
            snapshot["platform"].values()
        ):
            self.assertIsInstance(
                value,
                str,
            )

        packages = snapshot["packages"]

        self.assertIsInstance(
            packages,
            dict,
        )

        self.assertEqual(
            packages.get("openai"),
            "3.13.0",
        )

        for package in (
            "httpx2",
            "httpcore2",
            "pydantic",
            "pydantic-core",
            "anyio",
            "jiter",
            "typing_extensions",
        ):
            self.assertIn(
                package,
                packages,
            )

            self.assertIsInstance(
                packages[package],
                str,
            )

            self.assertTrue(
                packages[package]
            )

        self.assertEqual(
            snapshot[
                "provider_transport"
            ],
            {
                "max_retries":
                    OPENAI_MAX_RETRIES,
                "timeout_seconds":
                    OPENAI_TIMEOUT_SECONDS,
            },
        )

        encoded = json.dumps(
            snapshot,
            sort_keys=True,
        )

        forbidden = (
            "OPENAI_API_KEY",
            "api_key",
            "/Users/",
            "/home/",
        )

        for token in forbidden:
            self.assertNotIn(
                token,
                encoded,
            )

    def test_offline_run_persists_hashed_runtime_snapshot(
        self,
    ):
        module = load_runner()

        task = synthetic_task()
        prompts = synthetic_prompts()

        gateway = MeteredLLMGateway(
            client=FakeClient(),
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
                            "runtime-provenance-001"
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
                artifact[
                    "schema_version"
                ],
                "trace-v2-run-artifact/2",
                msg=(
                    "Adding required runtime "
                    "provenance changes the raw "
                    "artifact schema."
                ),
            )

            self.assertIn(
                "runtime",
                artifact,
            )

            runtime = artifact["runtime"]

            self.assertEqual(
                set(runtime),
                {
                    "snapshot",
                    "sha256",
                },
            )

            snapshot = runtime[
                "snapshot"
            ]

            self.assertEqual(
                runtime["sha256"],
                canonical_sha256(
                    snapshot
                ),
            )

            self.assertEqual(
                snapshot["packages"][
                    "openai"
                ],
                "3.13.0",
            )

            self.assertEqual(
                snapshot[
                    "provider_transport"
                ],
                {
                    "max_retries": 0,
                    "timeout_seconds": 120.0,
                },
            )

            self.assertTrue(
                verify_run_artifact(
                    artifact
                )
            )

    def test_official_default_provider_path_rejects_moving_alias_before_provider_construction(
        self,
    ):
        module = load_runner()

        task = synthetic_task()
        prompts = synthetic_prompts()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            with (
                patch.object(
                    module,
                    "MODEL",
                    "gpt-4o",
                ),
                patch.object(
                    module,
                    "MeteredLLMGateway",
                    side_effect=AssertionError(
                        "provider gateway was "
                        "constructed before model "
                        "snapshot validation"
                    ),
                ) as gateway_factory,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "snapshot",
                ):
                    module.execute_trace_v2_run(
                        run_id=(
                            "unpinned-model-001"
                        ),
                        task=task,
                        treatment=(
                            "predicted_attribution"
                        ),
                        seed=7,
                        prompts=prompts,
                        root_dir=root,
                        llm_gateway=None,
                        source_probe=(
                            clean_source_probe
                        ),
                    )

            gateway_factory.assert_not_called()

            self.assertFalse(
                (
                    root
                    / "artifacts"
                    / "v2"
                ).exists()
            )

    def test_dated_snapshot_validation_is_explicit(
        self,
    ):
        module = load_runner()

        self.assertTrue(
            hasattr(
                module,
                "validate_official_model_snapshot",
            ),
            msg=(
                "Official model pinning has no "
                "explicit validation boundary."
            ),
        )

        validator = (
            module
            .validate_official_model_snapshot
        )

        # Current GPT-4o documentation lists
        # this as an immutable snapshot.
        validator(
            "gpt-4o-2024-11-20"
        )

        # The validator is structural rather
        # than hard-coded to GPT-4o, because
        # final model selection is a later
        # scientific-design decision.
        validator(
            "research-model-2026-09-14"
        )

        for moving_alias in (
            "gpt-4o",
            "gpt-6-astra",
            "model-latest",
            "",
        ):
            with self.subTest(
                model=moving_alias
            ):
                with self.assertRaises(
                    ValueError
                ):
                    validator(
                        moving_alias
                    )


if __name__ == "__main__":
    unittest.main()
