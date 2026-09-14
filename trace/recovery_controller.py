from config import N_MAX


POLICY_PRIORITY = {
    "s_UR": ["retrieve", "replan", "halt"],
    "s_CD": ["replan", "backtrack", "halt"],
    "s_RL": ["replan", "compact", "halt"],
    "s_TA": ["switch", "backtrack", "halt"],
    "s_UR+s_TA": ["retrieve", "switch", "halt"],
}


class RecoveryController:
    def __init__(self):
        # Full history remains append-only for audit/replay.
        self.recovery_log = []

        # Recovery budget and tried-policy state are scoped only to the
        # currently active failure event.
        self.event_attempt_count = 0
        self.event_policies = []

    def select_policy(self, failure_state):
        policies = POLICY_PRIORITY.get(failure_state, ["halt"])
        tried = set(self.event_policies)

        for policy in policies:
            if policy not in tried:
                return policy

        return "halt"

    @staticmethod
    def _available_tools(agent):
        task = getattr(agent, "task", {}) or {}
        tools = task.get("available_tools", [])

        if not isinstance(tools, list):
            return []

        return list(tools)

    @staticmethod
    def _build_context(agent):
        if hasattr(agent, "build_context"):
            return list(agent.build_context())

        # Lightweight fallback used by fully offline unit tests.
        messages = []
        for event in getattr(agent, "trajectory", []):
            messages.append({
                "role": "assistant",
                "content": (
                    f"Reasoning: {event.get('reasoning', '')}\n"
                    f"Action: {event.get('action', '')}"
                ),
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {event.get('observation', {})}",
            })
        return messages

    @staticmethod
    def _last_executed_tool(agent):
        recovery_actions = {
            "replan",
            "context_compaction",
        }

        for event in reversed(getattr(agent, "trajectory", [])):
            action = event.get("action")

            if action and action not in recovery_actions:
                return action

        return None

    @staticmethod
    def _append_event(agent, reasoning, action, observation):
        agent.trajectory.append({
            "step": agent.step,
            "reasoning": reasoning,
            "action": action,
            "observation": observation,
        })
        agent.step += 1

    @staticmethod
    def _parse_selected_tool(response, allowed_tools):
        for line in (response or "").splitlines():
            stripped = line.strip()

            if stripped.startswith("Action:"):
                candidate = stripped.split("Action:", 1)[1].strip().split()[0]

                if candidate in allowed_tools:
                    return candidate

        return None

    def _execute_retrieve(self, agent, failure_state):
        available_tools = self._available_tools(agent)

        if "search_knowledge_base" not in available_tools:
            return "failed"

        prior_reasoning = ""
        if agent.trajectory:
            prior_reasoning = agent.trajectory[-1].get("reasoning", "")

        query = (
            f"Failure state {failure_state}. Retrieve evidence relevant to: "
            f"{prior_reasoning}"
        )

        observation = agent.tool_layer.call(
            "search_knowledge_base",
            args={"query": query},
        )

        self._append_event(
            agent,
            reasoning=f"[RECOVERY: targeted evidence retrieval] {query}",
            action="search_knowledge_base",
            observation=observation,
        )

        return "attempted"

    def _execute_replan(self, agent, failure_state):
        messages = self._build_context(agent)
        messages.append({
            "role": "user",
            "content": (
                "A runtime failure has been detected with state "
                f"{failure_state}. Replan from the currently verified evidence. "
                "Do not repeat the failed approach. Produce a concise revised "
                "diagnostic plan for the next action."
            ),
        })

        plan = agent.get_llm_response(
            messages,
            purpose="recovery_replan",
        )

        self._append_event(
            agent,
            reasoning="[RECOVERY: replanning from current verified state]",
            action="replan",
            observation={
                "status": "success",
                "plan": plan,
            },
        )

        return "attempted"

    def _execute_switch(self, agent, failure_state):
        available_tools = self._available_tools(agent)
        failing_tool = self._last_executed_tool(agent)

        alternatives = [
            tool for tool in available_tools
            if tool != failing_tool
        ]

        if not alternatives:
            return "failed"

        messages = self._build_context(agent)
        messages.append({
            "role": "user",
            "content": (
                f"The previous tool '{failing_tool}' is unsuitable or failed "
                f"under runtime state {failure_state}. Select one alternative "
                "tool from the following list and respond with exactly "
                "'Action: <tool_name>': "
                + ", ".join(alternatives)
            ),
        })

        selection_response = agent.get_llm_response(
            messages,
            purpose="recovery_switch",
        )

        selected_tool = self._parse_selected_tool(
            selection_response,
            alternatives,
        )

        # Fail closed when the selector does not produce a valid,
        # task-declared alternative. Do not silently convert selector failure
        # into an apparently successful recovery action.
        if selected_tool is None:
            return "failed"

        observation = agent.tool_layer.call(selected_tool)

        self._append_event(
            agent,
            reasoning=(
                "[RECOVERY: tool switch] "
                f"{failing_tool} -> {selected_tool}"
            ),
            action=selected_tool,
            observation=observation,
        )

        return "attempted"

    def _execute_compact(self, agent, failure_state):
        trajectory = getattr(agent, "trajectory", [])

        if not trajectory:
            return "failed"

        messages = self._build_context(agent)
        messages.append({
            "role": "user",
            "content": (
                "Compact the trajectory into a concise execution-state "
                "summary. Preserve verified facts, relevant tool observations, "
                "unresolved uncertainties, failed approaches, and the current "
                f"runtime failure state ({failure_state}). Do not invent facts."
            ),
        })

        summary = agent.get_llm_response(
            messages,
            purpose="recovery_compact",
        )

        # Keep four most recent concrete events plus one semantic summary of
        # the older prefix. This bounds active context while preserving recent
        # execution detail.
        recent = trajectory[-4:]
        removed_count = max(0, len(trajectory) - len(recent))

        if removed_count > 0:
            first_recent_step = recent[0].get("step", 1)

            summary_event = {
                "step": max(0, first_recent_step - 1),
                "reasoning": summary,
                "action": "context_compaction",
                "observation": {
                    "status": "success",
                    "compacted_prefix_events": removed_count,
                },
            }

            agent.trajectory = [summary_event] + recent
        else:
            # Even when there is nothing worth truncating, the semantic
            # summary was genuinely generated; preserve it for auditability.
            self._append_event(
                agent,
                reasoning=summary,
                action="context_compaction",
                observation={
                    "status": "success",
                    "compacted_prefix_events": 0,
                },
            )

        return "attempted"

    def execute(self, policy, agent, failure_state, last_verified_step):
        self.event_attempt_count += 1
        self.event_policies.append(policy)
        outcome = "failed"

        if policy == "retrieve":
            outcome = self._execute_retrieve(agent, failure_state)

        elif policy == "replan":
            outcome = self._execute_replan(agent, failure_state)

        elif policy == "backtrack":
            if last_verified_step is not None:
                agent.trajectory = agent.trajectory[:last_verified_step + 1]
                agent.step = last_verified_step + 1
                outcome = "attempted"

        elif policy == "switch":
            outcome = self._execute_switch(agent, failure_state)

        elif policy == "compact":
            outcome = self._execute_compact(agent, failure_state)

        elif policy == "halt":
            outcome = "escalated"

        self.recovery_log.append({
            "failure_state": failure_state,
            "policy": policy,
            "outcome": outcome,
        })

        return outcome

    def should_escalate(self):
        return self.event_attempt_count >= N_MAX

    def reset_event(self):
        """
        Close the active failure event while preserving append-only
        recovery history.

        A later independent failure starts with a fresh attempt budget
        and fresh policy-selection state.
        """
        self.event_attempt_count = 0
        self.event_policies = []

    def get_log(self):
        return self.recovery_log
