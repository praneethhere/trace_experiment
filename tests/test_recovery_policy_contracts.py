import sys
import types
import unittest


# recovery_controller.py creates an OpenAI client at import time even though
# none of the recovery-policy methods under test use that client. Stub the
# dependency so these contract tests remain fully offline and cannot make
# network/API calls.
class _OfflineOpenAI:
    def __init__(self, *args, **kwargs):
        pass


sys.modules["openai"] = types.SimpleNamespace(OpenAI=_OfflineOpenAI)

from trace.recovery_controller import RecoveryController


class RecordingToolLayer:
    def __init__(self):
        self.calls = []

    def call(self, tool_name):
        self.calls.append(tool_name)
        return {
            "status": "success",
            "source": tool_name,
            "evidence": "recovery evidence",
        }


class FakeAgent:
    def __init__(self, trajectory=None):
        self.tool_layer = RecordingToolLayer()
        self.llm_calls = []
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
        return "Revised recovery plan"


class TestRecoveryPolicyContracts(unittest.TestCase):

    def test_retrieve_queries_the_knowledge_environment(self):
        """
        pi_retrieve must perform an actual retrieval operation rather than
        merely append a recovery label to the trajectory.
        """
        agent = FakeAgent()
        controller = RecoveryController()

        controller.execute(
            "retrieve",
            agent,
            failure_state="s_UR",
            last_verified_step=0,
        )

        self.assertGreater(
            len(agent.tool_layer.calls),
            0,
            msg=(
                "retrieve did not query the tool/knowledge environment; "
                "a placeholder trajectory entry is not evidence retrieval."
            ),
        )

    def test_replan_generates_a_revised_plan(self):
        """
        pi_replan is described as prompting for a revised action plan, so
        executing it must invoke the agent/model planning path.
        """
        agent = FakeAgent()
        controller = RecoveryController()

        controller.execute(
            "replan",
            agent,
            failure_state="s_CD",
            last_verified_step=0,
        )

        self.assertGreater(
            len(agent.llm_calls),
            0,
            msg=(
                "replan did not invoke the model/planning path; "
                "it only recorded a recovery marker."
            ),
        )

    def test_switch_executes_an_alternative_tool(self):
        """
        pi_switch must replace the failing tool with an actual alternative,
        not record the synthetic action name 'tool_switch'.
        """
        agent = FakeAgent()
        failing_tool = agent.trajectory[-1]["action"]
        controller = RecoveryController()

        controller.execute(
            "switch",
            agent,
            failure_state="s_TA",
            last_verified_step=0,
        )

        self.assertGreater(
            len(agent.tool_layer.calls),
            0,
            msg="switch did not invoke any alternative tool.",
        )

        self.assertNotEqual(
            agent.tool_layer.calls[-1],
            failing_tool,
            msg="switch retried the same failing tool instead of replacing it.",
        )

    def test_compact_performs_semantic_compaction(self):
        """
        The manuscript describes pi_compact as summarizing/compressing
        trajectory context and counts such recovery as an inference-bearing
        operation. It therefore must invoke the model rather than only slice
        the Python list.
        """
        trajectory = [
            {
                "step": i,
                "reasoning": f"reasoning step {i}",
                "action": f"tool_{i}",
                "observation": {"status": "success", "value": i},
            }
            for i in range(8)
        ]

        agent = FakeAgent(trajectory=trajectory)
        controller = RecoveryController()

        controller.execute(
            "compact",
            agent,
            failure_state="s_RL",
            last_verified_step=5,
        )

        self.assertGreater(
            len(agent.llm_calls),
            0,
            msg=(
                "compact did not summarize/compress context semantically; "
                "it only truncated the trajectory."
            ),
        )

    def test_backtrack_restores_last_verified_step(self):
        """
        Existing backtracking should retain the trajectory only through the
        last verified step and reset the next execution index accordingly.
        """
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
