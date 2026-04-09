from agents.base_react import BaseReActAgent
from trace.trajectory_monitor import TrajectoryMonitor
from trace.failure_attribution import FailureAttributionModule
from trace.recovery_controller import RecoveryController
from trace.audit_layer import AuditLayer
from config import THETA_H

class TRACEAgent(BaseReActAgent):
    def __init__(self, task, tool_layer, system_prompt,
                 grounding_prompt, contradiction_prompt, results_dir):
        super().__init__(task, tool_layer, system_prompt)
        self.monitor   = TrajectoryMonitor()
        self.attributor = FailureAttributionModule(grounding_prompt, contradiction_prompt)
        self.recovery   = RecoveryController()
        self.audit      = AuditLayer(task["task_id"], "TRACE")
        self.results_dir = results_dir
        self.current_state = "s_NP"
        self.lc_consecutive = 0
        self.last_verified_step = None
        self.recovery_attempts_this_event = 0

    def step_once(self):
        response, terminal = super().step_once()
        if terminal:
            return response, terminal

        last = self.trajectory[-1]
        event = {
            "step": last["step"],
            "reasoning": last["reasoning"],
            "tool_id": last["action"],
            "tool_status": last["observation"].get("status", "success"),
            "observation": last["observation"],
        }
        self.monitor.record(event)
        self.audit.log_step(event)

        # Compute soft signals
        H = self.monitor.compute_H(last["reasoning"])
        rho = self.monitor.get_rho(last["action"])
        window = self.monitor.get_window()

        # s_LC detection
        if H < THETA_H:
            self.lc_consecutive += 1
        else:
            self.lc_consecutive = 0

        if self.lc_consecutive >= 2:
            self.current_state = "s_LC"
        else:
            self.current_state = "s_NP"

        # Run failure detectors
        f1, _, _ = self.attributor.detect_F1(last["reasoning"], window)
        f2, _    = self.attributor.detect_F2(last["reasoning"], window)
        f3, _    = self.attributor.detect_F3(last["reasoning"], last["action"])
        f4, _    = self.attributor.detect_F4(event["tool_status"], rho, event["observation"])

        failure_state = None
        if f1 and f4: failure_state = "s_UR+s_TA"
        elif f1:      failure_state = "s_UR"
        elif f2:      failure_state = "s_CD"
        elif f3:      failure_state = "s_RL"
        elif f4:      failure_state = "s_TA"

        if failure_state:
            self.current_state = failure_state
            self.audit.log_failure({"step": last["step"], "state": failure_state})

            if self.recovery.should_escalate():
                self.current_state = "s_HE"
                self.audit.log_terminal("s_HE", False)
                return "ESCALATION: recovery limit reached | LAST_VERIFIED: step {}".format(
                    self.last_verified_step), True

            policy = self.recovery.select_policy(failure_state)
            outcome = self.recovery.execute(policy, self, failure_state, self.last_verified_step)
            self.audit.log_recovery({"step": last["step"], "policy": policy, "outcome": outcome})

            if outcome == "escalated":
                self.current_state = "s_HE"
                self.audit.log_terminal("s_HE", False)
                return "ESCALATION: halt policy | LAST_VERIFIED: step {}".format(
                    self.last_verified_step), True
        else:
            self.last_verified_step = last["step"]
            self.recovery_attempts_this_event = 0

        return response, False

    def run(self):
        import re as _re
        final_response, trajectory = super().run()
        goal = False
        if "RESOLUTION:" in (final_response or ""):
            stated = _re.search(r"RESOLUTION:\s*(.+?)(\||$)", final_response)
            stated_cause = stated.group(1).strip().lower() if stated else ""
            gt_cause = self.task.get("ground_truth", {}).get("root_cause", "").lower()
            goal = gt_cause in stated_cause
        self.audit.log_terminal("s_OK" if goal else "s_HE", goal)
        self.audit.save(self.results_dir)
        return final_response, trajectory, self.audit.get_trace()
