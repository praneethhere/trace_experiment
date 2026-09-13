import unittest

from trace.trajectory_monitor import TrajectoryMonitor


class TestReasoningNoveltyScore(unittest.TestCase):

    def test_current_reasoning_must_not_match_itself(self):
        """
        H_t must compare the current reasoning only against PRIOR reasoning.

        These two reasoning steps share no unigrams or bigrams, so the
        current reasoning should have novelty H_t = 1.0.
        """
        monitor = TrajectoryMonitor()

        monitor.record({
            "step": 1,
            "reasoning": "database connection refused",
            "tool_id": "log_search",
        })

        # This mirrors TRACEAgent's current execution order:
        # current event is recorded before compute_H() is called.
        current = "authentication token expired"

        monitor.record({
            "step": 2,
            "reasoning": current,
            "tool_id": "identity_check",
        })

        h = monitor.compute_H(current)

        self.assertAlmostEqual(
            h,
            1.0,
            places=7,
            msg=(
                "Current reasoning appears to be included in its own "
                "comparison window; H_t should compare only with prior steps."
            ),
        )

    def test_true_repetition_should_have_zero_novelty(self):
        """
        A genuine repetition of the previous reasoning should still
        produce H_t = 0.0.
        """
        monitor = TrajectoryMonitor()

        reasoning = "database connection refused"

        monitor.record({
            "step": 1,
            "reasoning": reasoning,
            "tool_id": "log_search",
        })

        monitor.record({
            "step": 2,
            "reasoning": reasoning,
            "tool_id": "log_search",
        })

        h = monitor.compute_H(reasoning)

        self.assertAlmostEqual(h, 0.0, places=7)
    def test_oldest_of_five_prior_events_is_still_in_novelty_window(self):
        """
        With K_WINDOW=5, H_t must compare against all five prior events.

        The current reasoning repeats only the oldest of those five prior
        events. If the implementation accidentally keeps only four prior
        events after recording the current step, this test would return 1.0
        instead of 0.0.
        """
        monitor = TrajectoryMonitor()

        monitor.record({
            "step": 1,
            "reasoning": "critical authentication failure",
            "tool_id": "tool_1",
        })

        for step, reasoning in enumerate(
            [
                "database latency increased",
                "network route verified",
                "cache service healthy",
                "deployment status stable",
            ],
            start=2,
        ):
            monitor.record({
                "step": step,
                "reasoning": reasoning,
                "tool_id": f"tool_{step}",
            })

        current = "critical authentication failure"

        monitor.record({
            "step": 6,
            "reasoning": current,
            "tool_id": "tool_6",
        })

        h = monitor.compute_H(current)

        self.assertAlmostEqual(
            h,
            0.0,
            places=7,
            msg="H_t did not retain all K_WINDOW=5 prior reasoning events.",
        )


if __name__ == "__main__":
    unittest.main()
