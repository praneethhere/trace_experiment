from config import N_MAX
from openai import OpenAI
from config import MODEL, AGENT_TEMPERATURE

client = OpenAI()

POLICY_PRIORITY = {
    "s_UR": ["retrieve", "replan", "halt"],
    "s_CD": ["replan", "backtrack", "halt"],
    "s_RL": ["replan", "compact", "halt"],
    "s_TA": ["switch", "backtrack", "halt"],
    "s_UR+s_TA": ["retrieve", "switch", "halt"],
}

class RecoveryController:
    def __init__(self):
        self.recovery_log = []
        self.attempt_count = 0

    def select_policy(self, failure_state):
        policies = POLICY_PRIORITY.get(failure_state, ["halt"])
        # Use first policy that hasn't been tried for this failure event
        tried = [r["policy"] for r in self.recovery_log[-3:]]
        for policy in policies:
            if policy not in tried:
                return policy
        return "halt"

    def execute(self, policy, agent, failure_state, last_verified_step):
        self.attempt_count += 1
        outcome = "failed"

        if policy == "retrieve":
            # Reissue a retrieval query — append to agent context
            agent.trajectory.append({
                "step": agent.step, "reasoning": "[RECOVERY: evidence retrieval]",
                "action": "search_knowledge_base", "observation": {}})
            agent.step += 1
            outcome = "attempted"

        elif policy == "replan":
            agent.trajectory.append({
                "step": agent.step, "reasoning": "[RECOVERY: replanning from current state]",
                "action": "replan", "observation": {}})
            agent.step += 1
            outcome = "attempted"

        elif policy == "backtrack":
            # Roll trajectory back to last verified step
            if last_verified_step is not None:
                agent.trajectory = agent.trajectory[:last_verified_step + 1]
                agent.step = last_verified_step + 1
            outcome = "attempted"

        elif policy == "switch":
            agent.trajectory.append({
                "step": agent.step, "reasoning": "[RECOVERY: tool switch]",
                "action": "tool_switch", "observation": {}})
            agent.step += 1
            outcome = "attempted"

        elif policy == "compact":
            # Summarise and truncate trajectory context
            if len(agent.trajectory) > 5:
                agent.trajectory = agent.trajectory[-5:]
            outcome = "attempted"

        elif policy == "halt":
            outcome = "escalated"

        self.recovery_log.append({
            "failure_state": failure_state, "policy": policy, "outcome": outcome})
        return outcome

    def should_escalate(self):
        return self.attempt_count >= N_MAX

    def get_log(self):
        return self.recovery_log
