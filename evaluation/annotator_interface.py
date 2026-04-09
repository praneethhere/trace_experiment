import csv
import json
import os
from glob import glob

def export_annotation_templates(results_dir="results", output_dir="annotation"):
    os.makedirs(output_dir, exist_ok=True)
    headers = {
        "hallucination_labels.csv": ["task_id","system","step_number","reasoning_trace","detector_flagged_F1","annotator_A_label","annotator_B_label","adjudicated_label"],
        "contradiction_labels.csv": ["task_id","system","step_number","new_statement","prior_statements_cited","detector_flagged_F2","annotator_A_label","annotator_B_label","adjudicated_label"],
        "escalation_labels.csv": ["task_id","system","failure_type_in_handoff","last_verified_state_in_handoff","annotator_A_correct","annotator_B_correct","adjudicated_correct"],
    }
    for filename, header in headers.items():
        with open(os.path.join(output_dir, filename), "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

def export_prefilled_from_trace(trace_dir="results/traces", output_dir="annotation"):
    export_annotation_templates(output_dir=output_dir)

    hall_path = os.path.join(output_dir, "hallucination_labels.csv")
    contr_path = os.path.join(output_dir, "contradiction_labels.csv")
    esc_path = os.path.join(output_dir, "escalation_labels.csv")

    with open(hall_path, "a", newline="", encoding="utf-8") as hall_f,          open(contr_path, "a", newline="", encoding="utf-8") as contr_f,          open(esc_path, "a", newline="", encoding="utf-8") as esc_f:
        hall_w = csv.writer(hall_f)
        contr_w = csv.writer(contr_f)
        esc_w = csv.writer(esc_f)

        for path in glob(os.path.join(trace_dir, "*.json")):
            with open(path, encoding="utf-8") as f:
                trace = json.load(f)
            task_id = trace["task_id"]
            system = trace["system"]

            for event in trace.get("failure_events", []):
                state = event.get("state")
                step_number = event.get("step")
                if state in {"s_UR", "s_UR+s_TA"}:
                    hall_w.writerow([task_id, system, step_number, "prefilled from trace", "TRUE", "", "", ""])
                elif state == "s_CD":
                    contr_w.writerow([task_id, system, step_number, "prefilled from trace", "prior confirmed reasoning", "TRUE", "", "", ""])

            if trace.get("terminal_state") == "s_HE":
                esc_w.writerow([task_id, system, "s_HE", "prefilled from trace", "", "", ""])

if __name__ == "__main__":
    export_prefilled_from_trace()
