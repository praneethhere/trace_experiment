import json, os
from config import RESPONSES_DIR

class ToolLayer:
    def __init__(self, task_id):
        self.task_id = task_id
        self.call_counts = {}   # track retry counts per tool

    def call(self, tool_name, args=None):
        self.call_counts[tool_name] = self.call_counts.get(tool_name, 0) + 1
        count = self.call_counts[tool_name]

        # Try call-specific response first (for injected tool failure)
        path_specific = os.path.join(RESPONSES_DIR, self.task_id, f"{tool_name}_call{count}.json")
        path_default  = os.path.join(RESPONSES_DIR, self.task_id, f"{tool_name}.json")

        if os.path.exists(path_specific):
            with open(path_specific) as f:
                return json.load(f)["response"]
        elif os.path.exists(path_default):
            with open(path_default) as f:
                return json.load(f)["response"]
        else:
            return {"status": "error", "message": f"Tool {tool_name} not available for this task"}
