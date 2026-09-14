import unittest

from config import N_MAX
from trace.recovery_controller import RecoveryController


class MinimalAgent:
    def __init__(self):
        self.trajectory = [
            {
                "step": 0,
                "reasoning": "verified starting state",
                "action": "check_service_status",
                "observation": {"status": "success"},
            },
            {
                "step": 1,
                "reasoning": "failure detected",
                "action": "check_service_status",
                "observation": {"status": "fail"},
            },
        ]
        self.step = 2


class TestRecoveryEventBudget(unittest.TestCase):

    def test_same_event_reaches_limit_after_n_max_attempts(self):
        """
        N_MAX applies within one active failure event.
        """
        controller = RecoveryController()
        agent = MinimalAgent()

        for _ in range(N_MAX):
            controller.execute(
                "backtrack",
                agent,
                failure_state="s_CD",
                last_verified_step=0,
            )

        self.assertTrue(
            controller.should_escalate(),
            msg=(
                "A single failure event did not reach the recovery "
                "limit after N_MAX attempts."
            ),
        )

    def test_new_event_gets_fresh_attempt_budget(self):
        """
        Returning to normal progress ends the previous failure event.
        A later independent failure must start with a fresh N_MAX budget.
        """
        controller = RecoveryController()
        agent = MinimalAgent()

        for _ in range(N_MAX - 1):
            controller.execute(
                "backtrack",
                agent,
                failure_state="s_CD",
                last_verified_step=0,
            )

        self.assertFalse(controller.should_escalate())

        self.assertTrue(
            hasattr(controller, "reset_event"),
            msg=(
                "RecoveryController has no per-event reset mechanism; "
                "attempts therefore accumulate across independent failures."
            ),
        )

        controller.reset_event()

        self.assertFalse(
            controller.should_escalate(),
            msg=(
                "A new failure event inherited the previous event's "
                "recovery-attempt budget."
            ),
        )

        # The new event should receive all N_MAX attempts.
        for _ in range(N_MAX - 1):
            controller.execute(
                "backtrack",
                agent,
                failure_state="s_TA",
                last_verified_step=0,
            )

        self.assertFalse(
            controller.should_escalate(),
            msg=(
                "The new failure event exhausted its budget too early."
            ),
        )

        controller.execute(
            "backtrack",
            agent,
            failure_state="s_TA",
            last_verified_step=0,
        )

        self.assertTrue(controller.should_escalate())

    def test_new_event_does_not_inherit_policy_history(self):
        """
        Policy selection must be scoped to the active failure event.

        If event 1 uses 'retrieve', a later independent s_UR event should
        again begin with 'retrieve', rather than incorrectly skipping to
        'replan' because of global recovery history.
        """
        controller = RecoveryController()
        agent = MinimalAgent()

        first_policy = controller.select_policy("s_UR")

        self.assertEqual(first_policy, "retrieve")

        controller.execute(
            first_policy,
            agent,
            failure_state="s_UR",
            last_verified_step=0,
        )

        self.assertTrue(
            hasattr(controller, "reset_event"),
            msg=(
                "RecoveryController cannot clear policy history between "
                "independent failure events."
            ),
        )

        controller.reset_event()

        second_policy = controller.select_policy("s_UR")

        self.assertEqual(
            second_policy,
            "retrieve",
            msg=(
                "A new s_UR event inherited policy history from the "
                "previous failure event."
            ),
        )


if __name__ == "__main__":
    unittest.main()
