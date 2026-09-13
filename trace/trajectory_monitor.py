from collections import deque
from config import K_WINDOW, THETA_H
import re

class TrajectoryMonitor:
    def __init__(self):
        self.window = deque(maxlen=K_WINDOW)
        self.full_log = []
        self.retry_counts = {}   # tool_name -> consecutive retry count
        self.last_tool = None

    def record(self, event: dict):
        self.window.append(event)
        self.full_log.append(event)

        # Update retry count
        tool = event.get("tool_id")
        if tool and tool == self.last_tool:
            self.retry_counts[tool] = self.retry_counts.get(tool, 0) + 1
        else:
            if self.last_tool:
                self.retry_counts[self.last_tool] = 0
            self.last_tool = tool

    def get_rho(self, tool_name):
        return self.retry_counts.get(tool_name, 0)

    def compute_H(self, current_reasoning):
        """
        Compute novelty of the current reasoning against up to K_WINDOW
        PRIOR reasoning events.

        TRACE records the current event before this method is called, so the
        final event in full_log is the current event and must not participate
        in its own novelty comparison.
        """
        current_ngrams = self._ngrams(current_reasoning)
        if not current_ngrams:
            return 0.0

        # full_log already contains the current event. Exclude it, then retain
        # at most K_WINDOW prior events so H_t compares r_t with
        # {r_(t-k), ..., r_(t-1)} rather than with itself.
        prior_events = self.full_log[-(K_WINDOW + 1):-1]

        prior_ngrams = set()
        for event in prior_events:
            prior_ngrams |= self._ngrams(event.get("reasoning", ""))

        overlap = len(current_ngrams & prior_ngrams)
        return 1.0 - overlap / len(current_ngrams)

    def _ngrams(self, text, n_values=(1, 2)):
        tokens = re.findall(r'\w+', text.lower())
        result = set()
        for n in n_values:
            for i in range(len(tokens) - n + 1):
                result.add(tuple(tokens[i:i+n]))
        return result

    def get_window(self):
        return list(self.window)

    def get_full_log(self):
        return self.full_log
