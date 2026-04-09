from openai import OpenAI
from config import MODEL, AGENT_TEMPERATURE, MAX_STEPS
import json

client = OpenAI()

class BaseReActAgent:
    def __init__(self, task, tool_layer, system_prompt):
        self.task = task
        self.tool_layer = tool_layer
        self.system_prompt = system_prompt
        self.trajectory = []   # list of {step, reasoning, action, tool, observation}
        self.step = 0

    def build_context(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.append({"role": "user", "content": self.task["incident_description"]})
        for t in self.trajectory:
            messages.append({"role": "assistant", "content": f"Reasoning: {t['reasoning']}\nAction: {t['action']}"})
            messages.append({"role": "user", "content": f"Observation: {json.dumps(t['observation'])}"})
        return messages

    def get_llm_response(self, messages):
        resp = client.chat.completions.create(
            model=MODEL, temperature=AGENT_TEMPERATURE, messages=messages)
        return resp.choices[0].message.content

    def parse_action(self, response):
        # Extract tool name and check for terminal outputs
        if "RESOLUTION:" in response or "ESCALATION:" in response:
            return None, response   # terminal
        # Simple parse: look for "Action: tool_name"
        for line in response.split("\n"):
            if line.strip().startswith("Action:"):
                tool = line.split("Action:")[-1].strip().split()[0]
                return tool, response
        return None, response

    def step_once(self):
        messages = self.build_context()
        response = self.get_llm_response(messages)
        tool_name, reasoning = self.parse_action(response)
        if tool_name is None:
            return response, True   # terminal
        observation = self.tool_layer.call(tool_name)
        self.trajectory.append({
            "step": self.step,
            "reasoning": reasoning,
            "action": tool_name,
            "observation": observation
        })
        self.step += 1
        return response, False

    def run(self):
        terminal = False
        final_response = None
        while self.step < MAX_STEPS and not terminal:
            final_response, terminal = self.step_once()
        return final_response, self.trajectory
