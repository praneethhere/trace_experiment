import unittest

from trace.recovery_controller import RecoveryController


class RecordingToolLayer:
    def __init__(self):
        self.calls = []

    def call(self, tool_name, args=None):
        self.calls.append({
            "tool_name": tool_name,
            "args": args,
        })
        return {
            "status": "success",
            "source": tool_name,
            "args": args,
            "evidence": "recovery evidence",
        }


class FakeAgent:
    def __init__(
        self,
        trajectory=None,
        llm_response="Revised recovery plan",
    ):
        # Match BaseReActAgent's task contract.
        self.task = {
            "task_id": "offline_contract_test",
            "incident_description": (
                "Authentication incident under investigation."
            ),
            "available_tools": [
                "search_logs",
                "search_knowledge_base",
                "identity_check",
            ],
        }

        self.tool_layer = RecordingToolLayer()
        self.llm_calls = []
        self.llm_response = llm_response

        self.trajectory = trajectory or [
            {
                "step": 0,
                "reasoning": "inspect authentication logs",
                "action": "search_logs",
                "observation": {"status": "success"},
            },
            {
                "step": 1,
                "reasoning": "retry failed authentication lookup",
                "action": "search_logs",
                "observation": {"status": "fail"},
            },
        ]

        self.step = len(self.trajectory)

    def get_llm_response(self, messages):
        self.llm_calls.append(messages)
        return self.llm_response


class TestRecoveryPolicyContracts(unittest.TestCase):

    def test_retrieve_queries_the_knowledge_environment(self):
        agent = FakeAgent()
        controller = RecoveryController()

        outcome = controller.execute(
            "retrieve",
            agent,
            failure_state="s_UR",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "attempted")
        self.assertGreater(
            len(agent.tool_layer.calls),
            0,
            msg=(
                "retrieve did not execute a retrieval-capable tool."
            ),
        )

        self.assertEqual(
            agent.tool_layer.calls[-1]["tool_name"],
            "search_knowledge_base",
        )

    def test_retrieve_passes_a_nonempty_query_argument(self):
        """
        The intervention must carry failure context into the retrieval call,
        even though the legacy fixture layer itself is not query-sensitive.
        """
        agent = FakeAgent()
        controller = RecoveryController()

        outcome = controller.execute(
            "retrieve",
            agent,
            failure_state="s_UR",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "attempted")

        call = agent.tool_layer.calls[-1]

        self.assertEqual(
            call["tool_name"],
            "search_knowledge_base",
        )

        args = call["args"]

        self.assertIsInstance(args, dict)
        self.assertTrue(
            args.get("query", "").strip(),
            msg="retrieve executed without a meaningful query argument.",
        )

    def test_replan_generates_a_revised_plan(self):
        agent = FakeAgent()
        controller = RecoveryController()

        outcome = controller.execute(
            "replan",
            agent,
            failure_state="s_CD",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "attempted")

        self.assertGreater(
            len(agent.llm_calls),
            0,
            msg=(
                "replan did not invoke the agent/model planning path."
            ),
        )

    def test_switch_executes_an_alternative_tool(self):
        """
        A valid selector response must cause execution of a different,
        task-declared tool.
        """
        agent = FakeAgent(
            llm_response="Action: identity_check"
        )

        failing_tool = agent.trajectory[-1]["action"]
        controller = RecoveryController()

        outcome = controller.execute(
            "switch",
            agent,
            failure_state="s_TA",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "attempted")
        self.assertGreater(len(agent.tool_layer.calls), 0)

        selected = agent.tool_layer.calls[-1]["tool_name"]

        self.assertNotEqual(
            selected,
            failing_tool,
            msg=(
                "switch retried the same failing tool instead "
                "of replacing it."
            ),
        )

        self.assertIn(
            selected,
            agent.task["available_tools"],
        )

    def test_switch_executes_exact_valid_selected_tool(self):
        agent = FakeAgent(
            llm_response="Action: identity_check"
        )
        controller = RecoveryController()

        outcome = controller.execute(
            "switch",
            agent,
            failure_state="s_TA",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "attempted")

        self.assertEqual(
            agent.tool_layer.calls[-1]["tool_name"],
            "identity_check",
        )

    def test_switch_invalid_selection_fails_closed(self):
        """
        Invalid selector output must not silently receive a deterministic
        fallback that looks like successful recovery.
        """
        agent = FakeAgent(
            llm_response="Revised recovery plan"
        )
        controller = RecoveryController()

        outcome = controller.execute(
            "switch",
            agent,
            failure_state="s_TA",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "failed")
        self.assertEqual(agent.tool_layer.calls, [])

    def test_compact_performs_semantic_compaction(self):
        trajectory = [
            {
                "step": i,
                "reasoning": f"reasoning step {i}",
                "action": f"tool_{i}",
                "observation": {
                    "status": "success",
                    "value": i,
                },
            }
            for i in range(8)
        ]

        agent = FakeAgent(
            trajectory=trajectory,
            llm_response=(
                "Verified facts and unresolved state summary."
            ),
        )
        controller = RecoveryController()

        outcome = controller.execute(
            "compact",
            agent,
            failure_state="s_RL",
            last_verified_step=5,
        )

        self.assertEqual(outcome, "attempted")

        self.assertGreater(
            len(agent.llm_calls),
            0,
            msg=(
                "compact did not invoke semantic summarization."
            ),
        )

        self.assertEqual(
            agent.trajectory[0]["action"],
            "context_compaction",
        )

    def test_backtrack_restores_last_verified_step(self):
        agent = FakeAgent()
        controller = RecoveryController()

        outcome = controller.execute(
            "backtrack",
            agent,
            failure_state="s_CD",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "attempted")
        self.assertEqual(len(agent.trajectory), 1)
        self.assertEqual(agent.trajectory[-1]["step"], 0)
        self.assertEqual(agent.step, 1)

    def test_halt_returns_escalated(self):
        agent = FakeAgent()
        controller = RecoveryController()

        outcome = controller.execute(
            "halt",
            agent,
            failure_state="s_TA",
            last_verified_step=0,
        )

        self.assertEqual(outcome, "escalated")


if __name__ == "__main__":
    unittest.main()
