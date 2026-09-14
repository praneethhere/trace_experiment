import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.tool_layer import ToolLayer


class TestToolProvenanceContract(unittest.TestCase):

    def _make_fixture_tree(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)

        task_dir = root / "task_001"
        task_dir.mkdir(parents=True)

        default_path = task_dir / "check_status.json"
        default_path.write_text(
            json.dumps({
                "response": {
                    "status": "success",
                    "service": "api",
                }
            })
        )

        retry_path = task_dir / "check_status_call2.json"
        retry_path.write_text(
            json.dumps({
                "response": {
                    "status": "timeout",
                    "service": "api",
                }
            })
        )

        return temp, root, default_path, retry_path

    def test_tool_calls_leave_complete_ordered_provenance(self):
        """
        Every tool invocation must preserve what was called, with which
        arguments, which fixture supplied the response, and the exact output.
        """
        temp, root, default_path, retry_path = self._make_fixture_tree()

        try:
            with patch("tools.tool_layer.RESPONSES_DIR", str(root)):
                layer = ToolLayer("task_001")

                first_args = {
                    "service": "api",
                    "region": "us-east",
                }

                first = layer.call(
                    "check_status",
                    args=first_args,
                )

                second = layer.call(
                    "check_status",
                    args={"service": "api"},
                )

                records = layer.get_call_records()

            self.assertEqual(len(records), 2)

            self.assertEqual(
                [record["call_id"] for record in records],
                [1, 2],
            )

            self.assertEqual(
                [record["tool_call_index"] for record in records],
                [1, 2],
            )

            self.assertEqual(
                [record["tool_name"] for record in records],
                ["check_status", "check_status"],
            )

            self.assertEqual(
                records[0]["tool_attempt"],
                1,
            )
            self.assertEqual(
                records[1]["tool_attempt"],
                2,
            )

            self.assertEqual(
                records[0]["args"],
                {
                    "service": "api",
                    "region": "us-east",
                },
            )

            self.assertEqual(
                records[0]["response"],
                first,
            )
            self.assertEqual(
                records[1]["response"],
                second,
            )

            self.assertEqual(
                Path(records[0]["fixture_path"]).resolve(),
                default_path.resolve(),
            )
            self.assertEqual(
                Path(records[1]["fixture_path"]).resolve(),
                retry_path.resolve(),
            )

            self.assertEqual(
                len(records[0]["fixture_sha256"]),
                64,
            )
            self.assertEqual(
                len(records[1]["fixture_sha256"]),
                64,
            )

            self.assertEqual(
                len(records[0]["args_sha256"]),
                64,
            )
            self.assertEqual(
                len(records[0]["response_sha256"]),
                64,
            )

            # Mutating caller-owned data after execution must not alter
            # preserved provenance.
            first_args["service"] = "MUTATED"

            self.assertEqual(
                records[0]["args"]["service"],
                "api",
            )

        finally:
            temp.cleanup()

    def test_missing_tool_is_still_recorded(self):
        """
        Failed/missing tool invocations are scientific events too and must
        not disappear from the ledger.
        """
        with tempfile.TemporaryDirectory() as temp:
            with patch(
                "tools.tool_layer.RESPONSES_DIR",
                temp,
            ):
                layer = ToolLayer("task_404")

                response = layer.call(
                    "missing_tool",
                    args={"query": "x"},
                )

                records = layer.get_call_records()

        self.assertEqual(len(records), 1)

        record = records[0]

        self.assertEqual(
            record["tool_name"],
            "missing_tool",
        )

        self.assertIsNone(
            record["fixture_path"],
        )
        self.assertIsNone(
            record["fixture_sha256"],
        )

        self.assertEqual(
            record["response"],
            response,
        )

        self.assertEqual(
            response["status"],
            "error",
        )

        self.assertEqual(
            len(record["response_sha256"]),
            64,
        )

    def test_get_call_records_returns_defensive_copy(self):
        """
        External consumers must not be able to mutate the authoritative
        in-memory ledger accidentally.
        """
        temp, root, _, _ = self._make_fixture_tree()

        try:
            with patch("tools.tool_layer.RESPONSES_DIR", str(root)):
                layer = ToolLayer("task_001")
                layer.call("check_status")

                records = layer.get_call_records()
                records[0]["tool_name"] = "CORRUPTED"

                fresh = layer.get_call_records()

            self.assertEqual(
                fresh[0]["tool_name"],
                "check_status",
            )

        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main()
