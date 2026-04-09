import json, os, glob
from config import BENCHMARK_DIR, RESULTS_DIR, RESPONSES_DIR
from tools.tool_layer import ToolLayer
from agents.vanilla_react import VanillaReActAgent
from agents.self_reflection import SelfReflectionAgent
from agents.trace_agent import TRACEAgent
from evaluation.scorer import score_task, save_record

def load_prompt(path):
    with open(path) as f: return f.read()

def run_all():
    system_prompt     = load_prompt("prompts/system_react.txt")
    reflection_prompt = load_prompt("prompts/reflection.txt")
    grounding_prompt  = load_prompt("prompts/grounding_check.txt")
    contradiction_prompt = load_prompt("prompts/contradiction_check.txt")

    task_files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "*.json")))

    for task_file in task_files:
        with open(task_file) as f:
            task = json.load(f)

        tool_layer = ToolLayer(task["task_id"])
        sp = system_prompt.replace("{tool_list}", ", ".join(task["available_tools"]))

        for system_name, AgentClass, kwargs in [
            ("vanilla_react", VanillaReActAgent, {"system_prompt": sp}),
            ("self_reflection", SelfReflectionAgent,
             {"system_prompt": sp, "reflection_prompt": reflection_prompt}),
            ("TRACE", TRACEAgent,
             {"system_prompt": sp, "grounding_prompt": grounding_prompt,
              "contradiction_prompt": contradiction_prompt,
              "results_dir": os.path.join(RESULTS_DIR, "traces")}),
        ]:
            tool_layer = ToolLayer(task["task_id"])   # fresh per system
            agent = AgentClass(task=task, tool_layer=tool_layer, **kwargs)

            if system_name == "TRACE":
                final_response, trajectory, trace = agent.run()
            else:
                final_response, trajectory = agent.run()
                trace = None

            record = score_task(task, final_response, trajectory, system_name, trace)
            out_path = os.path.join(RESULTS_DIR, "main", f"{system_name}_results.jsonl")
            save_record(record, out_path)
            print(f"Done: {task['task_id']} / {system_name}")

if __name__ == "__main__":
    run_all()
