import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from agents.trace_agent import TRACEAgent


class TerminalGateway:
    """
    Offline gateway that immediately returns a correct terminal answer.

    No provider or tool execution should occur in these persistence tests.
    """

    def __init__(self):
        self.calls = []

    def complete(
        self,
        messages,
        purpose,
        temperature,
    ):
        self.calls.append({
            "messages": messages,
            "purpose": purpose,
            "temperature": temperature,
        })

        return "RESOLUTION: connection exhaustion"

    def get_call_records(self):
        return list(self.calls)

    def get_usage_totals(self):
        return {
            "calls": len(self.calls),
            "successful_calls": len(self.calls),
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class NoToolLayer:
    """
    A terminal first model response must not invoke a tool.
    """

    def call(self, tool_name, args=None):
        raise AssertionError(
            f"Unexpected tool execution: {tool_name}"
        )

    def get_call_records(self):
        return []


def make_task():
    return {
        "task_id": "persistence_boundary_task",
        "incident_description": (
            "The API is failing under database connection pressure."
        ),
        "available_tools": [
            "check_db_connections",
        ],
        "ground_truth": {
            "root_cause": "connection exhaustion",
        },
    }


def make_agent(
    *,
    results_dir,
    persist_legacy_trace,
):
    return TRACEAgent(
        task=make_task(),
        tool_layer=NoToolLayer(),
        system_prompt="system",
        grounding_prompt=(
            "Claim: {claim}\n"
            "Evidence: {evidence}"
        ),
        contradiction_prompt=(
            "New: {new_statement}\n"
            "Prior: {prior_statements}"
        ),
        results_dir=results_dir,
        persist_legacy_trace=persist_legacy_trace,
        llm_gateway=TerminalGateway(),
    )


class TestTracePersistenceBoundary(unittest.TestCase):

    def test_v2_mode_does_not_call_legacy_audit_save(self):
        """
        TRACE execution must be usable as an in-memory computation.

        A v2 run must be able to obtain the execution trace without writing
        through the legacy AuditLayer persistence path.
        """
        agent = make_agent(
            results_dir=None,
            persist_legacy_trace=False,
        )

        agent.audit.save = Mock(
            side_effect=AssertionError(
                "legacy AuditLayer.save must not run in v2 mode"
            )
        )

        final_response, trajectory, trace = agent.run()

        self.assertEqual(
            final_response,
            "RESOLUTION: connection exhaustion",
        )

        self.assertEqual(
            trajectory,
            [],
        )

        self.assertEqual(
            trace["terminal_state"],
            "s_OK",
        )

        self.assertTrue(
            trace["goal_satisfied"],
        )

        agent.audit.save.assert_not_called()

    def test_legacy_mode_still_persists_when_explicitly_enabled(self):
        """
        Existing KBS-era runners must retain their legacy behavior until they
        are intentionally retired.
        """
        with tempfile.TemporaryDirectory() as temp:
            agent = make_agent(
                results_dir=temp,
                persist_legacy_trace=True,
            )

            final_response, trajectory, trace = agent.run()

            expected = (
                Path(temp)
                / (
                    "persistence_boundary_task_"
                    "TRACE_trace.json"
                )
            )

            self.assertTrue(
                expected.exists(),
                msg=(
                    "Explicit legacy persistence must continue "
                    "to write the historical trace file."
                ),
            )

            persisted = json.loads(
                expected.read_text()
            )

        self.assertEqual(
            final_response,
            "RESOLUTION: connection exhaustion",
        )

        self.assertEqual(
            trajectory,
            [],
        )

        self.assertEqual(
            persisted,
            trace,
        )

    def test_legacy_persistence_requires_results_directory(self):
        """
        Enabling legacy persistence without a destination must fail early
        rather than producing an ambiguous late filesystem error.
        """
        with self.assertRaises(ValueError):
            make_agent(
                results_dir=None,
                persist_legacy_trace=True,
            )


if __name__ == "__main__":
    unittest.main()
