import sys
import types
import unittest
from unittest.mock import patch


# TRACE imports OpenAI-backed modules at module import time. This test exercises
# only event lifecycle behavior and must remain fully offline.
class _OfflineOpenAI:
    def __init__(self, *args, **kwargs):
        pass


sys.modules["openai"] = types.SimpleNamespace(OpenAI=_OfflineOpenAI)

from agents.base_react import BaseReActAgent
from agents.trace_agent import TRACEAgent
from trace.trajectory_monitor import TrajectoryMonitor


class HealthyAttributor:
    def detect_F1(self, reasoning, window):
        return False, 0.0, None

    def detect_F2(self, reasoning, window):
        return False, 0.0

    def detect_F3(self, reasoning, action):
        return False, 0.0

    def detect_F4(self, tool_status, rho, observation):
        return False, 0.0


class RecordingRecovery:
    def __init__(self):
        self.reset_calls = 0

    def reset_event(self):
        self.reset_calls += 1

    def should_escalate(self):
        return False


class RecordingAudit:
    def __init__(self):
        self.steps = []

    def log_step(self, event):
        self.steps.append(event)

    def log_failure(self, event):
        raise AssertionError("healthy test step must not log a failure")

    def log_recovery(self, event):
        raise AssertionError("healthy test step must not log recovery")

    def log_terminal(self, state, goal):
        pass


class TestTraceEventLifecycle(unittest.TestCase):

    def test_healthy_step_closes_previous_failure_event(self):
        """
        A failure-free step represents return to normal progress and must
        reset event-local recovery budget/policy state.
        """
        agent = TRACEAgent.__new__(TRACEAgent)

        agent.trajectory = [
            {
                "step": 0,
                "reasoning": "service status is healthy",
                "action": "check_service_status",
                "observation": {"status": "success"},
            }
        ]

        agent.step = 1
        agent.monitor = TrajectoryMonitor()
        agent.audit = RecordingAudit()
        agent.attributor = HealthyAttributor()
        agent.recovery = RecordingRecovery()

        agent.current_state = "s_NP"
        agent.lc_consecutive = 0
        agent.last_verified_step = None

        with patch.object(
            BaseReActAgent,
            "step_once",
            return_value=("continue", False),
        ):
            response, terminal = TRACEAgent.step_once(agent)

        self.assertEqual(response, "continue")
        self.assertFalse(terminal)
        self.assertEqual(agent.last_verified_step, 0)

        self.assertEqual(
            agent.recovery.reset_calls,
            1,
            msg=(
                "Returning to normal progress did not close/reset the "
                "previous recovery event."
            ),
        )


if __name__ == "__main__":
    unittest.main()
