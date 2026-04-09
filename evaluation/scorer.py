import json, re
from config import RESULTS_DIR

def score_task(task, final_response, trajectory, system_name, trace=None):
    record = {
        "task_id": task["task_id"],
        "system": system_name,
        "difficulty_level": task["difficulty_level"],
    }

    # 1. Task success
    gt = task["ground_truth"]
    if "RESOLUTION:" in (final_response or ""):
        stated_cause = re.search(r"RESOLUTION:\s*(.+?)(\||$)", final_response)
        stated_cause = stated_cause.group(1).strip() if stated_cause else ""
        record["task_success"] = int(gt["root_cause"].lower() in stated_cause.lower())
    else:
        record["task_success"] = 0

    # 2. Tool-call efficiency
    tool_calls = [t for t in trajectory if t["action"] not in ("reflect","replan","tool_switch")]
    record["total_tool_calls"] = len(tool_calls)
    record["informative_tool_calls"] = None
    record["tool_call_efficiency"] = None

    # 3. Hallucination incidence — filled after annotation
    record["steps_total"] = len(trajectory)
    record["hallucination_steps_detector"] = None
    record["hallucination_incidence_detector"] = None


    # 4. Loop frequency — from Loop Detector or 0
    record["loop_steps"] = None
    record["loop_frequency"] = None

    # 5. Correction overhead
    record["correction_overhead"] = 0   # filled from trace.detector_calls for TRACE

    # TRACE-specific
    if trace:
        record["recovery_events"] = trace["recovery_events"]
        record["n_recovery_attempts"] = len(trace["recovery_events"])
        record["recovery_successes"] = sum(
            1 for r in trace["recovery_events"] if r["outcome"] == "attempted")
        record["recovery_success_rate"] = (
            record["recovery_successes"] / record["n_recovery_attempts"]
            if record["n_recovery_attempts"] > 0 else None)
        record["time_to_recovery"] = None   # computed separately from step deltas
        record["escalated"] = trace["terminal_state"] == "s_HE"
    else:
        record["recovery_events"] = None
        record["recovery_success_rate"] = None
        record["time_to_recovery"] = None
        record["escalated"] = "ESCALATION:" in (final_response or "")

    return record

def save_record(record, output_path):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(record) + "\n")
