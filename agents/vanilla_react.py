from agents.base_react import BaseReActAgent

class VanillaReActAgent(BaseReActAgent):
    def __init__(self, task, tool_layer, system_prompt):
        super().__init__(task, tool_layer, system_prompt)
    # No modification — pure base ReAct
