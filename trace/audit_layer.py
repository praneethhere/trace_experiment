import json, os
from datetime import datetime

class AuditLayer:
    def __init__(self, task_id, system_name):
        self.task_id = task_id
        self.system_name = system_name
        self.trace_record = {
            "task_id": task_id,
            "system": system_name,
            "start_time": datetime.utcnow().isoformat(),
            "steps": [],
            "failure_events": [],
            "recovery_events": [],
            "terminal_state": None,
            "goal_satisfied": False
        }

    def log_step(self, step_event):
        self.trace_record["steps"].append(step_event)

    def log_failure(self, failure_event):
        self.trace_record["failure_events"].append(failure_event)

    def log_recovery(self, recovery_event):
        self.trace_record["recovery_events"].append(recovery_event)

    def log_terminal(self, state, goal_satisfied):
        self.trace_record["terminal_state"] = state
        self.trace_record["goal_satisfied"] = goal_satisfied

    def get_trace(self):
        return self.trace_record

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.task_id}_{self.system_name}_trace.json")
        with open(path, "w") as f:
            json.dump(self.trace_record, f, indent=2)
