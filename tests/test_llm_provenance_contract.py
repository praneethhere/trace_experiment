import ast
import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


class FakeCompletions:
    def __init__(self):
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        call_number = len(self.requests)

        return SimpleNamespace(
            id=f"fake-request-{call_number}",
                _request_id=f"fake-request-{call_number}",
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"response-{call_number}"
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            ),
        )


class FakeOpenAIClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(
            completions=self.completions
        )


def load_gateway_class(testcase):
    try:
        module = importlib.import_module("trace.llm_gateway")
    except ModuleNotFoundError:
        testcase.fail(
            "trace.llm_gateway does not exist; TRACE v2 has no "
            "single metered LLM provider boundary."
        )

    testcase.assertTrue(
        hasattr(module, "MeteredLLMGateway"),
        msg=(
            "trace.llm_gateway exists but does not expose "
            "MeteredLLMGateway."
        ),
    )

    return module.MeteredLLMGateway


class TestLLMProvenanceContract(unittest.TestCase):

    def test_direct_openai_access_is_isolated_to_gateway(self):
        """
        Production agent/control modules must not instantiate or import the
        provider SDK directly. All inference must cross one auditable gateway.
        """
        offenders = []

        roots = [
            ROOT / "agents",
            ROOT / "trace",
        ]

        for source_root in roots:
            for path in sorted(source_root.rglob("*.py")):
                if path.name == "llm_gateway.py":
                    continue

                tree = ast.parse(path.read_text())

                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""

                        if (
                            module == "openai"
                            or module.startswith("openai.")
                        ):
                            offenders.append(
                                str(path.relative_to(ROOT))
                            )
                            break

                    if isinstance(node, ast.Import):
                        if any(
                            alias.name == "openai"
                            or alias.name.startswith("openai.")
                            for alias in node.names
                        ):
                            offenders.append(
                                str(path.relative_to(ROOT))
                            )
                            break

        self.assertEqual(
            offenders,
            [],
            msg=(
                "Direct OpenAI access remains outside the metered "
                f"gateway: {offenders}"
            ),
        )

    def test_gateway_records_complete_call_provenance(self):
        """
        Each LLM call must leave enough evidence to reconstruct what was
        requested, why it was requested, what came back, and its token cost.
        """
        MeteredLLMGateway = load_gateway_class(self)

        client = FakeOpenAIClient()
        gateway = MeteredLLMGateway(
            client=client,
            model="gpt-4o",
        )

        messages = [
            {
                "role": "system",
                "content": "You are an engineering agent.",
            },
            {
                "role": "user",
                "content": "Diagnose the incident.",
            },
        ]

        response = gateway.complete(
            messages=messages,
            purpose="agent",
            temperature=0.2,
        )

        self.assertEqual(response, "response-1")

        records = gateway.get_call_records()

        self.assertEqual(len(records), 1)

        record = records[0]

        self.assertEqual(record["call_id"], 1)
        self.assertEqual(record["purpose"], "agent")
        self.assertEqual(record["model"], "gpt-4o")
        self.assertEqual(record["temperature"], 0.2)

        self.assertEqual(record["messages"], messages)
        self.assertEqual(
            record["response_text"],
            "response-1",
        )

        self.assertEqual(
            record["provider_request_id"],
            "fake-request-1",
        )

        self.assertEqual(
            record["usage"],
            {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
        )

        self.assertEqual(
            len(record["messages_sha256"]),
            64,
        )
        self.assertEqual(
            len(record["response_sha256"]),
            64,
        )

        # Provenance records must not alias mutable caller-owned messages.
        messages[1]["content"] = "MUTATED AFTER CALL"

        self.assertEqual(
            records[0]["messages"][1]["content"],
            "Diagnose the incident.",
        )

        # Verify the gateway actually forwarded the scientific parameters.
        request = client.completions.requests[0]

        self.assertEqual(request["model"], "gpt-4o")
        self.assertEqual(request["temperature"], 0.2)

    def test_gateway_assigns_monotonic_call_ids_and_purposes(self):
        """
        Call identity and purpose must be explicit so later experiments can
        enforce matched inference budgets by treatment and call category.
        """
        MeteredLLMGateway = load_gateway_class(self)

        client = FakeOpenAIClient()
        gateway = MeteredLLMGateway(
            client=client,
            model="gpt-4o",
        )

        gateway.complete(
            messages=[
                {
                    "role": "user",
                    "content": "normal agent reasoning",
                }
            ],
            purpose="agent",
            temperature=0.2,
        )

        gateway.complete(
            messages=[
                {
                    "role": "user",
                    "content": "grounding check",
                }
            ],
            purpose="detector_f1",
            temperature=0.0,
        )

        gateway.complete(
            messages=[
                {
                    "role": "user",
                    "content": "recovery replan",
                }
            ],
            purpose="recovery_replan",
            temperature=0.2,
        )

        records = gateway.get_call_records()

        self.assertEqual(
            [record["call_id"] for record in records],
            [1, 2, 3],
        )

        self.assertEqual(
            [record["purpose"] for record in records],
            [
                "agent",
                "detector_f1",
                "recovery_replan",
            ],
        )

        self.assertEqual(
            [
                record["usage"]["total_tokens"]
                for record in records
            ],
            [18, 18, 18],
        )


if __name__ == "__main__":
    unittest.main()
