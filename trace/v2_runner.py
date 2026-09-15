import importlib.metadata
import platform
import re
import subprocess
from datetime import date
from pathlib import Path

from agents.trace_agent import TRACEAgent
from config import (
    MODEL,
    AGENT_TEMPERATURE,
    DETECTOR_TEMPERATURE,
    MAX_STEPS,
    N_MAX,
    K_WINDOW,
    THETA_H,
    THETA_GROUND,
    THETA_LOOP,
    N_LOOP,
    RHO_THRESHOLD,
    OPENAI_MAX_RETRIES,
    OPENAI_TIMEOUT_SECONDS,
)
from tools.tool_layer import ToolLayer
from trace.llm_gateway import MeteredLLMGateway
from trace.run_artifacts import (
    build_run_artifact,
    write_run_artifact,
)


_RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._-]+$"
)

_IMPLEMENTED_TREATMENTS = {
    "predicted_attribution",
}


class V2RunExecutionError(RuntimeError):
    """
    Raised only after a started TRACE v2 execution fails and its partial
    scientific evidence has been persisted successfully.
    """

    def __init__(
        self,
        *,
        run_id,
        artifact_path,
    ):
        self.run_id = run_id
        self.artifact_path = Path(
            artifact_path
        )

        super().__init__(
            "TRACE v2 execution failed after "
            "partial evidence was persisted for "
            f"run {run_id!r}."
        )


def capture_git_source(repo_root="."):
    """
    Capture the exact source revision used for execution.

    Generated scientific evidence must never silently claim that modified
    source corresponds to the committed revision.
    """

    root = Path(repo_root)

    try:
        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise RuntimeError(
            "Unable to capture Git source provenance."
        ) from exc

    if not re.fullmatch(
        r"[0-9a-fA-F]{40}",
        commit,
    ):
        raise RuntimeError(
            "Git source probe returned an invalid commit id."
        )

    return {
        "git_commit": commit,
        "git_dirty": bool(status.strip()),
    }


def capture_config_snapshot():
    """
    Capture scientific configuration only.

    Credentials and environment secrets are intentionally excluded.
    """

    return {
        "model": MODEL,
        "agent_temperature":
            AGENT_TEMPERATURE,
        "detector_temperature":
            DETECTOR_TEMPERATURE,
        "max_steps": MAX_STEPS,
        "n_max": N_MAX,
        "k_window": K_WINDOW,
        "theta_h": THETA_H,
        "theta_ground": THETA_GROUND,
        "theta_loop": THETA_LOOP,
        "n_loop": N_LOOP,
        "rho_threshold":
            RHO_THRESHOLD,
    }


def _installed_package_version(name):
    """
    Return the installed distribution version without recording paths,
    credentials, environment variables, or other machine-local secrets.
    """
    try:
        return importlib.metadata.version(
            name
        )
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_runtime_snapshot():
    """
    Capture the execution environment needed to interpret a TRACE v2 run.

    The snapshot is deliberately narrow and secret-free: language/runtime
    identity, platform identity, exact relevant package versions, and the
    explicit provider transport policy.
    """
    package_names = (
        "openai",
        "httpx2",
        "httpcore2",
        "pydantic",
        "pydantic-core",
        "anyio",
        "jiter",
        "typing_extensions",
    )

    return {
        "python": {
            "version":
                platform.python_version(),
            "implementation":
                platform.python_implementation(),
        },

        "platform": {
            "system":
                platform.system(),
            "release":
                platform.release(),
            "machine":
                platform.machine(),
        },

        "packages": {
            name:
                _installed_package_version(
                    name
                )
            for name in package_names
        },

        "provider_transport": {
            "max_retries":
                OPENAI_MAX_RETRIES,
            "timeout_seconds":
                OPENAI_TIMEOUT_SECONDS,
        },
    }


_MODEL_SNAPSHOT_PATTERN = re.compile(
    r"^.+-(20[0-9]{2}-[0-9]{2}-[0-9]{2})$"
)


def validate_official_model_snapshot(model):
    """
    Require a structurally dated model identifier for official provider runs.

    This validates pinning structure only. It does not claim that a model
    exists or is currently available; availability is verified separately
    before real experiments.
    """
    if not isinstance(model, str):
        raise ValueError(
            "Official model must use a dated snapshot identifier."
        )

    match = _MODEL_SNAPSHOT_PATTERN.fullmatch(
        model
    )

    if match is None:
        raise ValueError(
            "Official model must use a dated snapshot identifier."
        )

    try:
        date.fromisoformat(
            match.group(1)
        )
    except ValueError as exc:
        raise ValueError(
            "Official model snapshot contains an invalid date."
        ) from exc

    return model


def _validate_run_id(run_id):
    if (
        not isinstance(run_id, str)
        or not run_id
        or not _RUN_ID_PATTERN.fullmatch(
            run_id
        )
    ):
        raise ValueError(
            "run_id must contain only letters, "
            "numbers, '.', '_' or '-'."
        )


def validate_artifact_root(
    root_dir,
    *,
    source_root=None,
):
    """
    Require TRACE v2 raw-run evidence to live outside the source checkout.

    Resolving both paths before comparison also prevents a symlink outside
    the checkout from aliasing back into the repository.
    """
    if root_dir is None:
        raise ValueError(
            "An explicit artifact root outside "
            "the source repository is required."
        )

    try:
        artifact_root = (
            Path(root_dir)
            .expanduser()
            .resolve()
        )
    except (
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        raise ValueError(
            "Artifact root must be a valid path "
            "outside the source repository."
        ) from exc

    if source_root is None:
        source_root = (
            Path(__file__)
            .resolve()
            .parents[1]
        )

    try:
        source_root = (
            Path(source_root)
            .expanduser()
            .resolve()
        )
    except (
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        raise ValueError(
            "Source root must be a valid path."
        ) from exc

    if (
        artifact_root == source_root
        or source_root
        in artifact_root.parents
    ):
        raise ValueError(
            "Artifact root must be outside "
            "the source repository."
        )

    return artifact_root


def _preflight_run_target(
    root_dir,
    run_id,
):
    """
    Reserve run identity conceptually before any inference cost.

    Existing run namespaces are immutable and cannot be reused.
    """

    _validate_run_id(run_id)

    run_dir = (
        Path(root_dir)
        / "artifacts"
        / "v2"
        / "runs"
        / run_id
    )

    if run_dir.exists():
        raise FileExistsError(
            f"Run id already exists: {run_id}"
        )


def _validate_treatment(treatment):
    if treatment not in _IMPLEMENTED_TREATMENTS:
        raise ValueError(
            "Treatment is not implemented by "
            f"this runner: {treatment!r}"
        )


def execute_trace_v2_run(
    *,
    run_id,
    task,
    treatment,
    seed,
    prompts,
    root_dir=None,
    repo_root=".",
    llm_gateway=None,
    source_probe=capture_git_source,
):
    """
    Execute one TRACE v2 predicted-attribution run and write exactly one
    immutable raw evidence bundle.

    This function deliberately does not score the run. Raw evidence is the
    execution boundary; deterministic scoring is a later stage.
    """

    # Everything below this preflight can carry inference or execution cost.
    # Reject invalid identities/treatments and existing run namespaces first.
    _validate_treatment(treatment)

    artifact_root = validate_artifact_root(
        root_dir
    )

    _preflight_run_target(
        artifact_root,
        run_id,
    )

    source = source_probe(
        repo_root
    )

    if not isinstance(source, dict):
        raise RuntimeError(
            "Source probe did not return provenance metadata."
        )

    if not source.get("git_commit"):
        raise RuntimeError(
            "Source provenance is missing git_commit."
        )

    if source.get("git_dirty"):
        raise RuntimeError(
            "Refusing TRACE v2 execution from a dirty source tree."
        )

    required_prompts = {
        "system_react",
        "grounding_check",
        "contradiction_check",
    }

    missing_prompts = (
        required_prompts
        - set(prompts)
    )

    if missing_prompts:
        raise ValueError(
            "Missing required TRACE prompts: "
            + ", ".join(
                sorted(missing_prompts)
            )
        )

    system_template = prompts[
        "system_react"
    ]

    system_rendered = (
        system_template.replace(
            "{tool_list}",
            ", ".join(
                task.get(
                    "available_tools",
                    []
                )
            ),
        )
    )

    prompt_snapshot = {
        "system_react_template":
            system_template,
        "system_react_rendered":
            system_rendered,
        "grounding_check":
            prompts["grounding_check"],
        "contradiction_check":
            prompts[
                "contradiction_check"
            ],
    }

    # Default provider construction happens only after all no-cost source,
    # identity, treatment, and prompt checks have passed.
    if llm_gateway is None:
        validate_official_model_snapshot(
            MODEL
        )
        llm_gateway = (
            MeteredLLMGateway(
                model=MODEL
            )
        )

    gateway_model = getattr(
        llm_gateway,
        "model",
        MODEL,
    )

    if gateway_model != MODEL:
        raise ValueError(
            "Injected LLM gateway model does not "
            "match the captured configuration model."
        )

    tool_layer = ToolLayer(
        task["task_id"]
    )

    agent = TRACEAgent(
        task=task,
        tool_layer=tool_layer,
        system_prompt=system_rendered,
        grounding_prompt=(
            prompts["grounding_check"]
        ),
        contradiction_prompt=(
            prompts[
                "contradiction_check"
            ]
        ),
        results_dir=None,
        persist_legacy_trace=False,
        llm_gateway=llm_gateway,
    )

    def persist_execution(
        execution,
    ):
        """
        Build and immutably persist either a completed execution or the
        partial evidence from a started execution that raised.
        """

        artifact = build_run_artifact(
            run_id=run_id,
            task=task,
            treatment=treatment,
            seed=seed,
            source=source,

            runtime_snapshot=(
                capture_runtime_snapshot()
            ),

            config_snapshot=(
                capture_config_snapshot()
            ),

            prompts=prompt_snapshot,

            llm_calls=(
                llm_gateway
                .get_call_records()
            ),

            llm_usage_totals=(
                llm_gateway
                .get_usage_totals()
            ),

            tool_calls=(
                tool_layer
                .get_call_records()
            ),

            execution=execution,
        )

        return write_run_artifact(
            artifact,
            root_dir=artifact_root,
        )


    try:
        (
            final_response,
            trajectory,
            trace_record,
        ) = agent.run()

    except Exception as exc:
        # The run has already crossed the execution boundary. Preserve all
        # evidence accumulated up to the exception instead of allowing the
        # failed attempt to disappear from the experimental record.
        trace_record = (
            agent.audit.get_trace()
        )

        execution = {
            "status": "error",

            "error": {
                "phase": "agent_run",
                "type": type(exc).__name__,
                "message": str(exc),
            },

            "trajectory":
                agent.trajectory,

            "failure_events":
                trace_record.get(
                    "failure_events",
                    [],
                ),

            "recovery_events":
                trace_record.get(
                    "recovery_events",
                    [],
                ),

            # Infrastructure/provider failure is not an agent terminal state.
            "terminal_state":
                trace_record.get(
                    "terminal_state"
                ),

            "goal_satisfied":
                trace_record.get(
                    "goal_satisfied",
                    False,
                ),

            "final_response":
                None,
        }

        artifact_path = (
            persist_execution(
                execution
            )
        )

        raise V2RunExecutionError(
            run_id=run_id,
            artifact_path=artifact_path,
        ) from exc


    execution = {
        "status": "completed",
        "error": None,

        "trajectory":
            trajectory,

        "failure_events":
            trace_record.get(
                "failure_events",
                [],
            ),

        "recovery_events":
            trace_record.get(
                "recovery_events",
                [],
            ),

        "terminal_state":
            trace_record.get(
                "terminal_state"
            ),

        "goal_satisfied":
            trace_record.get(
                "goal_satisfied",
                False,
            ),

        "final_response":
            final_response,
    }

    return persist_execution(
        execution
    )
