# Legacy KBS Result Provenance Audit

## Scope

This document records the provenance audit of the experiment artifacts
associated with the rejected Knowledge-Based Systems submission:

- Manuscript: KNOSYS-D-26-07420
- Frozen implementation tag: `kbs-v1-submission`
- TRACE v2 audit branch: `trace-v2-attribution-study`

The purpose of this audit is to determine which historical quantitative
results can be independently regenerated from the repository's preserved
raw artifacts.

The audit does not modify or reinterpret the frozen KBS submission.

## Historical artifacts

The repository contains:

- 50 TRACE result records
- 50 TRACE raw audit traces
- 50 benchmark tasks

Every TRACE result task ID has a corresponding raw trace.

## Reproducible legacy evidence

### Task success

Task success can be independently recomputed from:

1. benchmark ground-truth root cause, and
2. stored final response.

Result:

- 50/50 stored task-success labels reproduced exactly.
- TRACE successes: 38/50.
- Aggregate task success: 76%.

Therefore the historical TRACE task-success result is considered
reproducible legacy evidence.

### Raw recovery-event payloads

The recovery-event payloads in the result records match the raw traces.

Historical TRACE raw recovery events contain:

- 17 `replan` events with outcome `attempted`
- 8 `halt` events with outcome `escalated`

These event counts may be reported as descriptive historical artifacts.
They must not be interpreted as successful recoveries without an
independent recovery-success criterion.

## Non-reproducible / retired legacy metrics

The following KBS-era derived metrics do not have sufficient executable
provenance in the preserved raw artifacts and must not be reused as
scientific evidence in TRACE v2.

### Hallucination incidence

Historical result rows contain a reported aggregate-like value of 0.096.

However, raw failure events contain only:

- `state = s_HE`

The traces do not preserve F1 / `s_UR` attribution events needed to
reconstruct hallucination-positive steps.

Therefore the historical 9.6% hallucination-incidence result is retired.

### Loop frequency

Historical result rows contain a reported aggregate-like value of 0.048.

Raw traces do not preserve F3 / `s_RL` events needed to reconstruct
loop-positive steps.

Therefore the historical 4.8% loop-frequency result is retired.

### Recovery success rate

Historical result rows store 0.698 for tasks with recovery events.

The preserved events contain only intervention outcomes such as:

- `attempted`
- `escalated`

They do not record whether execution subsequently returned to a verified
healthy state.

The legacy scorer also treated `outcome == "attempted"` as a successful
recovery, which is not a valid recovery-success definition.

Therefore the historical 68% recovery-success result is retired.

### Time to recovery

The historical 2.1-step value cannot be reconstructed because the raw
artifact schema does not preserve an explicit recovery-start to
verified-recovery transition.

Therefore the historical time-to-recovery result is retired.

### Correction overhead

The historical traces contain no complete per-call LLM accounting,
token usage, provider usage metadata, latency, or cost records.

Therefore the historical 2.7-call correction-overhead result is retired.

### Tool-call efficiency

The historical scorer did not compute `informative_tool_calls`, and the
raw trace schema does not preserve an independent informative-call label.

Therefore the historical 0.71 tool-call-efficiency result is retired.

### Safe-escalation accuracy

The historical safe-escalation value depended on human annotation, but
the required annotation artifact is not present in the preserved
execution traces.

Therefore the historical 81.7% safe-escalation accuracy is retired unless
the original annotation evidence is independently recovered.

### Ablation-derived reliability metrics

Historical ablation result files repeat aggregate-like reliability values
across individual task records without sufficient raw metric provenance.

Those derived ablation metrics are retired from TRACE v2 evidence.

## TRACE v2 evidence rule

TRACE v2 must satisfy the following provenance chain:

benchmark task
→ immutable raw execution trace
→ primitive event labels
→ metered LLM/tool calls
→ deterministic scorer
→ per-run metrics
→ aggregate statistics
→ manuscript table/figure

No manuscript metric may be populated manually.

No aggregate value may be copied into individual task records.

An executed recovery intervention is not automatically a successful
recovery.

All scientific tables must be reproducible from raw artifacts by code.

## Legacy evidence classification

| Historical quantity | Status |
|---|---|
| TRACE task success (38/50, 76%) | Reproducible legacy evidence |
| Raw recovery policy/event counts | Reproducible descriptive evidence |
| Hallucination incidence (9.6%) | Retired |
| Loop frequency (4.8%) | Retired |
| Recovery success (68%) | Retired |
| Time to recovery (2.1 steps) | Retired |
| Tool-call efficiency (0.71) | Retired |
| Safe-escalation accuracy (81.7%) | Retired unless annotation evidence is recovered |
| Correction overhead (2.7 calls) | Retired |
| Derived ablation reliability metrics | Retired |

TRACE v2 experiments must be executed and scored independently of the
retired KBS-era derived metrics.
