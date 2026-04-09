import argparse
import glob
import json
import os
from contextlib import contextmanager

import config as cfg
from tools.tool_layer import ToolLayer
from agents.trace_agent import TRACEAgent
from evaluation.scorer import score_task, save_record


def load_prompt(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class NullMonitor:
    def record(self, event):
        return None
    def get_rho(self, tool_name):
        return 0
    def compute_H(self, current_reasoning):
        return 1.0
    def get_window(self):
        return []
    def get_full_log(self):
        return []


class NullAttributor:
    def detect_F1(self, reasoning, window):
        return False, 0.0, None
    def detect_F2(self, reasoning, window):
        return False, 0.0
    def detect_F3(self, reasoning, action):
        return False, 0.0
    def detect_F4(self, tool_status, rho, tool_output):
        return False, 0.0
    def get_detector_calls(self):
        return 0


class NullRecovery:
    def __init__(self):
        self.recovery_log = []
    def select_policy(self, failure_state):
        return "halt"
    def execute(self, policy, agent, failure_state, last_verified_step):
        self.recovery_log.append({
            "failure_state": failure_state,
            "policy": policy,
            "outcome": "escalated",
        })
        return "escalated"
    def should_escalate(self):
        return True
    def get_log(self):
        return self.recovery_log


class NullAudit:
    def __init__(self, *args, **kwargs):
        self.trace_record = {
            "task_id": kwargs.get("task_id"),
            "system": kwargs.get("system_name", "TRACE_minus_audit"),
            "steps": [],
            "failure_events": [],
            "recovery_events": [],
            "terminal_state": None,
            "goal_satisfied": False,
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
        return None


@contextmanager
def temporary_config(**updates):
    originals = {key: getattr(cfg, key) for key in updates}
    try:
        for key, value in updates.items():
            setattr(cfg, key, value)
        yield
    finally:
        for key, value in originals.items():
            setattr(cfg, key, value)


def build_agent(task, system_prompt, grounding_prompt, contradiction_prompt, variant_name):
    results_dir = os.path.join(cfg.RESULTS_DIR, "ablation", "traces", variant_name)
    agent = TRACEAgent(
        task=task,
        tool_layer=ToolLayer(task["task_id"]),
        system_prompt=system_prompt,
        grounding_prompt=grounding_prompt,
        contradiction_prompt=contradiction_prompt,
        results_dir=results_dir,
    )

    if variant_name == "TRACE_minus_monitor":
        agent.monitor = NullMonitor()
    elif variant_name == "TRACE_minus_attribution":
        agent.attributor = NullAttributor()
    elif variant_name == "TRACE_minus_recovery":
        agent.recovery = NullRecovery()
    elif variant_name == "TRACE_minus_audit":
        agent.audit = NullAudit(task_id=task["task_id"], system_name=variant_name)
    elif variant_name == "keyword_heuristic":
        import re
        def detect_F2(self, reasoning, window):
            prior = " ".join(e.get("reasoning", "") for e in window[:-1])
            tokens_now = set(re.findall(r"\w+", reasoning.lower()))
            tokens_prev = set(re.findall(r"\w+", prior.lower()))
            contradictory = (
                ("healthy" in tokens_now and "degraded" in tokens_prev) or
                ("degraded" in tokens_now and "healthy" in tokens_prev) or
                ("failed" in tokens_now and "succeeded" in tokens_prev) or
                ("succeeded" in tokens_now and "failed" in tokens_prev)
            )
            return contradictory, 1.0 if contradictory else 0.0
        agent.attributor.detect_F2 = detect_F2.__get__(agent.attributor, type(agent.attributor))
        agent.attributor.get_detector_calls = (lambda self: 0).__get__(agent.attributor, type(agent.attributor))
    return agent


def run_variant(variant_name):
    system_prompt = load_prompt("prompts/system_react.txt")
    grounding_prompt = load_prompt("prompts/grounding_check.txt")
    contradiction_prompt = load_prompt("prompts/contradiction_check.txt")

    task_files = sorted(glob.glob(os.path.join(cfg.BENCHMARK_DIR, "*.json")))
    overrides = {}
    if variant_name.startswith("k") and variant_name[1:].isdigit():
        overrides["K_WINDOW"] = int(variant_name[1:])

    output_path = os.path.join(cfg.RESULTS_DIR, "ablation", f"{variant_name}_results.jsonl")
    if os.path.exists(output_path):
        os.remove(output_path)

    with temporary_config(**overrides):
        for task_file in task_files:
            with open(task_file, encoding="utf-8") as f:
                task = json.load(f)

            sp = system_prompt.replace("{tool_list}", ", ".join(task["available_tools"]))
            agent = build_agent(task, sp, grounding_prompt, contradiction_prompt, variant_name)
            final_response, trajectory, trace = agent.run()
            record = score_task(task, final_response, trajectory, variant_name, trace)
            save_record(record, output_path)
            print(f"Done: {task['task_id']} / {variant_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TRACE ablation variants.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=[
            "TRACE_full", "TRACE_minus_monitor", "TRACE_minus_attribution",
            "TRACE_minus_recovery", "TRACE_minus_audit",
            "llm_entailment", "keyword_heuristic",
            "k3", "k5", "k7", "k10",
        ],
    )
    args = parser.parse_args()
    variant = "TRACE" if args.variant in {"TRACE_full", "llm_entailment"} else args.variant
    run_variant(variant)
