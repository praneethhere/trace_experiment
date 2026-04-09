# TRACE Experiment Package

TRACE incident-triage benchmark
Research code for TRACE: a failure-aware correction layer for autonomous incident-triage agents.

## Included

- 50 benchmark tasks across L1-L4
- deterministic tool-response fixtures
- all prompt files
- agent implementations for Vanilla ReAct, Self-Reflection, and TRACE
- TRACE support modules
- populated results tables 6-11
- task-level JSONL outputs
- populated annotation CSVs
- TRACE audit trace files
- runnable ablation script
- utilities to recompute Table 8 agreement from annotation CSVs

## Main result files

- `results/main/results_table6.json`
- `results/main/results_table7.json`
- `results/main/results_table8.json`
- `results/ablation/results_table9.json`
- `results/ablation/results_table10.json`
- `results/ablation/results_table11.json`

## Task-level results

- `results/main/vanilla_react_results.jsonl`
- `results/main/self_reflection_results.jsonl`
- `results/main/TRACE_results.jsonl`

## Traces

TRACE audit traces are under `results/traces/`.

## Annotation utilities

To regenerate or prefill annotation templates from trace files:

```bash
python evaluation/annotator_interface.py
```

To recompute Table 8 from the annotation CSVs:

```bash
python evaluation/aggregate_results.py
```

## Running

```bash
pip install -r requirements.txt
python run_experiment.py
python run_ablation.py --variant TRACE_minus_monitor
python run_ablation.py --variant keyword_heuristic
python run_ablation.py --variant k5
```

## Note

This repository includes both runnable framework code and manuscript-ready precomputed assets.
