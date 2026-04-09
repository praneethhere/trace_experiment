import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4o"
AGENT_TEMPERATURE = 0.2
DETECTOR_TEMPERATURE = 0.0
MAX_STEPS = 20          # max steps per task before forced escalation
N_MAX = 3               # max recovery attempts before s_HE
K_WINDOW = 5            # trajectory monitor window size
THETA_H = 0.25          # novelty score threshold for s_LC
THETA_GROUND = 3        # min token overlap for provisional grounding
THETA_LOOP = 0.8        # fingerprint similarity threshold for F3
N_LOOP = 3              # max identical fingerprints before F3
RHO_THRESHOLD = 2       # consecutive retries before F4
BENCHMARK_DIR = "benchmark/tasks"
RESPONSES_DIR = "tools/responses"
RESULTS_DIR = "results"
