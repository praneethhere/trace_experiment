import unittest

from agents.base_react import BaseReActAgent
from agents.self_reflection import SelfReflectionAgent
from agents.trace_agent import TRACEAgent
from trace.failure_attribution import FailureAttributionModule
from trace.recovery_controller import RecoveryController


class RecordingGateway:
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

        responses = {
            "agent": "Action: check_service_status",
            "self_reflection": "Inspect alternate evidence.",
            "detector_f1": "SUPPORTED",
            "detector_f2": "CONSISTENT",
        }

        return responses.get(
            purpose,
            "gateway-response",
        )


class DummyToolLayer:
    def __init__(self):
        self.calls = []

    def call(self, tool_name, args=None):
        self.calls.append({
            "tool_name": tool_name,
            "args": args,
        })

        return {
            "status": "success",
            "tool": tool_name,
        }


class RecoveryAgent:
    def __init__(self):
        self.task = {
            "incident_description": "test incident",
            "available_tools": [
                "search_logs",
                "search_knowledge_base",
                "identity_check",
            ],
        }

        self.trajectory = [
            {
                "step": 0,
                "reasoning": "initial evidence",
                "action": "search_logs",
                "observation": {
                    "status": "fail",
                },
            }
        ]

        self.step = 1
        self.tool_layer = DummyToolLayer()
        self.llm_calls = []

    def get_llm_response(
        self,
        messages,
        purpose="agent",
        temperature=None,
    ):
        self.llm_calls.append({
            "purpose": purpose,
            "messages": messages,
            "temperature": temperature,
        })

        if purpose == "recovery_switch":
            return "Action: identity_check"

        if purpose == "recovery_compact":
            return "semantic compact summary"

        return "revised recovery plan"


class TestLLMIntegrationContract(unittest.TestCase):

    def test_agent_and_reflection_calls_are_tagged(self):
        gateway = RecordingGateway()

        base = BaseReActAgent(
            task={
                "incident_description": "incident",
            },
            tool_layer=DummyToolLayer(),
            system_prompt="system",
            llm_gateway=gateway,
        )

        base.get_llm_response(
            [{"role": "user", "content": "diagnose"}]
        )

        reflection = SelfReflectionAgent(
            task={
                "incident_description": "incident",
            },
            tool_layer=DummyToolLayer(),
            system_prompt="system",
            reflection_prompt="reflect",
            llm_gateway=gateway,
        )

        reflection._inject_reflection()

        self.assertEqual(
            [
                call["purpose"]
                for call in gateway.calls
            ],
            [
                "agent",
                "self_reflection",
            ],
        )

    def test_detector_calls_are_tagged(self):
        gateway = RecordingGateway()

        attributor = FailureAttributionModule(
            grounding_prompt=(
                "Claim: {claim}\n"
                "Evidence: {evidence}"
            ),
            contradiction_prompt=(
                "New: {new_statement}\n"
                "Prior: {prior_statements}"
            ),
            llm_gateway=gateway,
        )

        attributor._llm_grounding_check(
            "claim",
            "evidence",
        )

        attributor.detect_F2(
            "new statement",
            [
                {
                    "reasoning": "prior statement",
                },
                {
                    "reasoning": "current statement",
                },
            ],
        )

        self.assertEqual(
            [
                call["purpose"]
                for call in gateway.calls
            ],
            [
                "detector_f1",
                "detector_f2",
            ],
        )

    def test_recovery_llm_calls_are_tagged(self):
        controller = RecoveryController()

        replan_agent = RecoveryAgent()

        controller.execute(
            "replan",
            replan_agent,
            failure_state="s_CD",
            last_verified_step=0,
        )

        self.assertEqual(
            replan_agent.llm_calls[-1]["purpose"],
            "recovery_replan",
        )

        controller.reset_event()

        switch_agent = RecoveryAgent()

        controller.execute(
            "switch",
            switch_agent,
            failure_state="s_TA",
            last_verified_step=0,
        )

        self.assertEqual(
            switch_agent.llm_calls[-1]["purpose"],
            "recovery_switch",
        )

        controller.reset_event()

        compact_agent = RecoveryAgent()

        controller.execute(
            "compact",
            compact_agent,
            failure_state="s_RL",
            last_verified_step=0,
        )

        self.assertEqual(
            compact_agent.llm_calls[-1]["purpose"],
            "recovery_compact",
        )

    def test_trace_shares_one_gateway_with_attributor(self):
        gateway = RecordingGateway()

        agent = TRACEAgent(
            task={
                "task_id": "integration_test",
                "incident_description": "incident",
                "ground_truth": {
                    "root_cause": "cause",
                },
            },
            tool_layer=DummyToolLayer(),
            system_prompt="system",
            grounding_prompt=(
                "Claim: {claim}\n"
                "Evidence: {evidence}"
            ),
            contradiction_prompt=(
                "New: {new_statement}\n"
                "Prior: {prior_statements}"
            ),
            results_dir="unused",
            llm_gateway=gateway,
        )

        self.assertIs(
            agent.llm_gateway,
            gateway,
        )

        self.assertIs(
            agent.attributor.llm_gateway,
            gateway,
        )


if __name__ == "__main__":
    unittest.main()
