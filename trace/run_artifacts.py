import copy
import hashlib
import json
import re
from pathlib import Path


SCHEMA_VERSION = "trace-v2-run-artifact/1"

_RUN_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._-]+$"
)


def _canonical_bytes(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def canonical_sha256(value):
    return hashlib.sha256(
        _canonical_bytes(value)
    ).hexdigest()


def text_sha256(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def _validate_run_id(run_id):
    if (
        not isinstance(run_id, str)
        or not run_id
        or not _RUN_ID_PATTERN.fullmatch(run_id)
    ):
        raise ValueError(
            "run_id must contain only letters, "
            "numbers, '.', '_' or '-'."
        )


def _artifact_payload_for_hash(artifact):
    payload = copy.deepcopy(artifact)
    payload.pop("artifact_sha256", None)
    return payload


def compute_artifact_sha256(artifact):
    return canonical_sha256(
        _artifact_payload_for_hash(
            artifact
        )
    )


def verify_run_artifact(artifact):
    expected = artifact.get(
        "artifact_sha256"
    )

    if not isinstance(expected, str):
        return False

    return expected == compute_artifact_sha256(
        artifact
    )


def build_run_artifact(
    *,
    run_id,
    task,
    treatment,
    seed,
    source,
    config_snapshot,
    prompts,
    llm_calls,
    llm_usage_totals,
    tool_calls,
    execution,
):
    """
    Build a deterministic TRACE v2 raw-run bundle from supplied evidence.

    This function records source cleanliness exactly as supplied. The future
    official experiment runner, not this evidence container, is responsible
    for refusing dirty source trees.
    """

    _validate_run_id(run_id)

    task_snapshot = copy.deepcopy(task)
    config_copy = copy.deepcopy(
        config_snapshot
    )

    prompt_records = {}

    for name, text in copy.deepcopy(
        prompts
    ).items():
        if not isinstance(name, str):
            raise TypeError(
                "Prompt names must be strings."
            )

        if not isinstance(text, str):
            raise TypeError(
                f"Prompt {name!r} must be text."
            )

        prompt_records[name] = {
            "text": text,
            "sha256": text_sha256(text),
        }

    artifact = {
        "schema_version": SCHEMA_VERSION,

        "run_id": run_id,
        "treatment": treatment,
        "seed": seed,

        "source": copy.deepcopy(source),

        "task": {
            "snapshot": task_snapshot,
            "sha256": canonical_sha256(
                task_snapshot
            ),
        },

        "config": {
            "snapshot": config_copy,
            "sha256": canonical_sha256(
                config_copy
            ),
        },

        "prompts": prompt_records,

        "llm": {
            "calls": copy.deepcopy(
                llm_calls
            ),
            "usage_totals": copy.deepcopy(
                llm_usage_totals
            ),
        },

        "tools": {
            "calls": copy.deepcopy(
                tool_calls
            ),
        },

        "execution": copy.deepcopy(
            execution
        ),
    }

    artifact["artifact_sha256"] = (
        compute_artifact_sha256(
            artifact
        )
    )

    return artifact


def write_run_artifact(
    artifact,
    *,
    root_dir=".",
):
    """
    Persist exactly one immutable run bundle.

    Existing run.json files are never replaced.
    """

    if not verify_run_artifact(artifact):
        raise ValueError(
            "Artifact integrity verification failed "
            "before write."
        )

    run_id = artifact.get("run_id")
    _validate_run_id(run_id)

    root = Path(root_dir)

    run_dir = (
        root
        / "artifacts"
        / "v2"
        / "runs"
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = run_dir / "run.json"

    # Exclusive creation provides the immutability boundary:
    # a repeated run_id cannot silently overwrite evidence.
    with path.open(
        "x",
        encoding="utf-8",
    ) as handle:
        json.dump(
            artifact,
            handle,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    return path
