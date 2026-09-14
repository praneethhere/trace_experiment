from agents.base_react import BaseReActAgent

class VanillaReActAgent(BaseReActAgent):
    def __init__(
        self,
        task,
        tool_layer,
        system_prompt,
        llm_gateway=None,
    ):
        super().__init__(
            task,
            tool_layer,
            system_prompt,
            llm_gateway=llm_gateway,
        )
    # No modification — pure base ReAct
