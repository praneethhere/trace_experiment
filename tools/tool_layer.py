import copy
import hashlib
import json
from pathlib import Path

from config import RESPONSES_DIR


def _canonical_sha256(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path):
    hasher = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            hasher.update(chunk)

    return hasher.hexdigest()


class ToolLayer:
    """
    Fixture-backed tool environment with append-only in-memory provenance.

    Tool arguments are preserved for scientific provenance even though the
    legacy fixture simulator does not use them to select response content.
    """

    def __init__(self, task_id):
        self.task_id = task_id

        # Retry count remains scoped per tool to preserve legacy fixture
        # selection semantics.
        self.call_counts = {}

        # Global ordered ledger for this ToolLayer/run.
        self._call_records = []
        self._next_call_id = 1

    def call(self, tool_name, args=None):
        self.call_counts[tool_name] = (
            self.call_counts.get(tool_name, 0) + 1
        )

        tool_attempt = self.call_counts[tool_name]

        call_id = self._next_call_id
        self._next_call_id += 1

        args_snapshot = copy.deepcopy(args)

        response_root = Path(RESPONSES_DIR) / self.task_id

        path_specific = (
            response_root
            / f"{tool_name}_call{tool_attempt}.json"
        )

        path_default = (
            response_root
            / f"{tool_name}.json"
        )

        selected_path = None

        if path_specific.exists():
            selected_path = path_specific
        elif path_default.exists():
            selected_path = path_default

        if selected_path is not None:
            with selected_path.open(
                encoding="utf-8"
            ) as handle:
                response = json.load(handle)["response"]

            fixture_path = (
                selected_path.resolve()
                .relative_to(
                    Path(RESPONSES_DIR).resolve()
                )
                .as_posix()
            )

            fixture_sha256 = _file_sha256(
                selected_path
            )

        else:
            response = {
                "status": "error",
                "message": (
                    f"Tool {tool_name} not available "
                    f"for this task"
                ),
            }

            fixture_path = None
            fixture_sha256 = None

        response_snapshot = copy.deepcopy(
            response
        )

        self._call_records.append({
            "call_id": call_id,

            # Explicit alias retained because future v2 artifacts may merge
            # multiple event ledgers while still needing the tool-local order.
            "tool_call_index": call_id,

            "tool_name": tool_name,
            "tool_attempt": tool_attempt,

            "args": args_snapshot,
            "args_sha256": _canonical_sha256(
                args_snapshot
            ),

            "fixture_path": fixture_path,
            "fixture_sha256": fixture_sha256,

            "response": response_snapshot,
            "response_sha256": _canonical_sha256(
                response_snapshot
            ),
        })

        return response

    def get_call_records(self):
        return copy.deepcopy(
            self._call_records
        )
