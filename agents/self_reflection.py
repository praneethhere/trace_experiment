from agents.base_react import BaseReActAgent

class SelfReflectionAgent(BaseReActAgent):
    def __init__(
        self,
        task,
        tool_layer,
        system_prompt,
        reflection_prompt,
        llm_gateway=None,
    ):
        super().__init__(
            task,
            tool_layer,
            system_prompt,
            llm_gateway=llm_gateway,
        )
        self.reflection_prompt = reflection_prompt

    def step_once(self):
        response, terminal = super().step_once()
        if not terminal:
            # Check if step was a failed tool call (observation has error/empty status)
            if self.trajectory:
                last_obs = self.trajectory[-1]["observation"]
                if last_obs.get("status") in ("error", "timeout", "empty"):
                    self._inject_reflection()
        return response, terminal

    def _inject_reflection(self):
        # Reflection is internal — no tool use, just appended to context
        messages = self.build_context()
        messages.append({"role": "user", "content": self.reflection_prompt})
        reflection_text = self.get_llm_response(
            messages,
            purpose="self_reflection",
        )
        # Add reflection as a special trajectory entry (no tool call, no observation)
        self.trajectory.append({
            "step": self.step,
            "reasoning": f"[REFLECTION] {reflection_text}",
            "action": "reflect",
            "observation": {}
        })
        self.step += 1
