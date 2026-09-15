import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import MODEL
from trace.llm_gateway import MeteredLLMGateway

from tests.test_v2_execution_runner_contract import (
    FakeClient,
    clean_source_probe,
    synthetic_prompts,
    synthetic_task,
)


def load_runner():
    return importlib.import_module(
        "trace.v2_runner"
    )


def source_repo_root(module):
    return (
        Path(module.__file__)
        .resolve()
        .parents[1]
    )


class TestV2ArtifactOutputPolicy(
    unittest.TestCase
):

    def test_explicit_artifact_root_validation_boundary_exists_and_accepts_external_root(
        self,
    ):
        module = load_runner()

        self.assertTrue(
            hasattr(
                module,
                "validate_artifact_root",
            ),
            msg=(
                "Official v2 execution has no "
                "artifact-output isolation boundary."
            ),
        )

        validator = (
            module.validate_artifact_root
        )

        repo_root = source_repo_root(
            module
        )

        with tempfile.TemporaryDirectory() as temp:
            external_root = (
                Path(temp)
                / "trace-runs"
            )

            resolved = validator(
                external_root,
                source_root=repo_root,
            )

            self.assertEqual(
                resolved,
                external_root.resolve(),
            )

            self.assertFalse(
                resolved.is_relative_to(
                    repo_root
                )
            )

    def test_source_repository_descendants_and_symlink_aliases_are_rejected(
        self,
    ):
        module = load_runner()

        self.assertTrue(
            hasattr(
                module,
                "validate_artifact_root",
            )
        )

        validator = (
            module.validate_artifact_root
        )

        repo_root = source_repo_root(
            module
        )

        invalid_roots = (
            repo_root,
            repo_root / "artifacts",
            repo_root / "artifacts" / "v2",
            repo_root / ".ignored-runs",
        )

        for candidate in invalid_roots:
            with self.subTest(
                candidate=str(candidate)
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "outside",
                ):
                    validator(
                        candidate,
                        source_root=repo_root,
                    )

        # A path outside the repo that resolves
        # back into it must also fail closed.
        with tempfile.TemporaryDirectory() as temp:
            alias = (
                Path(temp)
                / "repo-alias"
            )

            alias.symlink_to(
                repo_root,
                target_is_directory=True,
            )

            with self.assertRaisesRegex(
                ValueError,
                "outside",
            ):
                validator(
                    alias,
                    source_root=repo_root,
                )

    def test_missing_artifact_root_is_rejected_before_execution_objects_are_created(
        self,
    ):
        module = load_runner()

        task = synthetic_task()
        prompts = synthetic_prompts()

        gateway = MeteredLLMGateway(
            client=FakeClient(),
            model=MODEL,
        )

        with patch.object(
            module,
            "ToolLayer",
            side_effect=AssertionError(
                "ToolLayer constructed before "
                "artifact-root validation"
            ),
        ) as tool_layer:
            with self.assertRaisesRegex(
                ValueError,
                "artifact.*root",
            ):
                module.execute_trace_v2_run(
                    run_id="missing-root-001",
                    task=task,
                    treatment=(
                        "predicted_attribution"
                    ),
                    seed=7,
                    prompts=prompts,
                    root_dir=None,
                    llm_gateway=gateway,
                    source_probe=(
                        clean_source_probe
                    ),
                )

        tool_layer.assert_not_called()

        self.assertEqual(
            gateway.client
            .completions.requests,
            [],
        )

    def test_source_tree_artifact_root_is_rejected_before_execution_objects_are_created(
        self,
    ):
        module = load_runner()

        repo_root = source_repo_root(
            module
        )

        task = synthetic_task()
        prompts = synthetic_prompts()

        client = FakeClient()

        gateway = MeteredLLMGateway(
            client=client,
            model=MODEL,
        )

        run_id = "inside-repo-root-001"

        forbidden_run_dir = (
            repo_root
            / "artifacts"
            / "v2"
            / "runs"
            / run_id
        )

        self.assertFalse(
            forbidden_run_dir.exists()
        )

        with patch.object(
            module,
            "ToolLayer",
            side_effect=AssertionError(
                "ToolLayer constructed before "
                "artifact-root isolation check"
            ),
        ) as tool_layer:
            with self.assertRaisesRegex(
                ValueError,
                "outside",
            ):
                module.execute_trace_v2_run(
                    run_id=run_id,
                    task=task,
                    treatment=(
                        "predicted_attribution"
                    ),
                    seed=7,
                    prompts=prompts,
                    root_dir=repo_root,
                    llm_gateway=gateway,
                    source_probe=(
                        clean_source_probe
                    ),
                )

        tool_layer.assert_not_called()

        self.assertEqual(
            client.completions.requests,
            [],
        )

        self.assertFalse(
            forbidden_run_dir.exists()
        )


if __name__ == "__main__":
    unittest.main()
