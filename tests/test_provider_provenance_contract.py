import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config

from trace.llm_gateway import (
    MeteredLLMGateway,
)


class SuccessCompletions:
    def create(self, **kwargs):
        return SimpleNamespace(
            id="chatcmpl_completion_123",
            _request_id="req_http_456",
            model=(
                "gpt-4o-2024-11-20"
            ),
            system_fingerprint=(
                "fp_backend_789"
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="hello"
                    ),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
        )


class SuccessClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=(
                SuccessCompletions()
            )
        )


class SyntheticProviderError(
    RuntimeError
):
    def __init__(self):
        super().__init__(
            "synthetic status failure"
        )
        self.request_id = (
            "req_failed_http_999"
        )


class FailingCompletions:
    def create(self, **kwargs):
        raise SyntheticProviderError()


class FailingClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=(
                FailingCompletions()
            )
        )


class TestProviderProvenanceContract(
    unittest.TestCase
):

    def test_openai_sdk_is_exactly_pinned(self):
        requirement = (
            Path("requirements.txt")
            .read_text()
            .strip()
        )

        self.assertEqual(
            requirement,
            "openai==3.13.0",
            msg=(
                "Official experiments must not "
                "resolve an arbitrary future SDK."
            ),
        )

    def test_transport_policy_is_explicit_and_disables_sdk_retries(
        self,
    ):
        self.assertTrue(
            hasattr(
                config,
                "OPENAI_MAX_RETRIES",
            )
        )

        self.assertTrue(
            hasattr(
                config,
                "OPENAI_TIMEOUT_SECONDS",
            )
        )

        self.assertEqual(
            config.OPENAI_MAX_RETRIES,
            0,
        )

        self.assertEqual(
            config.OPENAI_TIMEOUT_SECONDS,
            120.0,
        )

        fake_client = object()

        with patch(
            "openai.OpenAI",
            return_value=fake_client,
        ) as constructor:
            gateway = MeteredLLMGateway(
                client=None,
                model="provider-test-model",
            )

        constructor.assert_called_once_with(
            max_retries=0,
            timeout=120.0,
        )

        self.assertIs(
            gateway.client,
            fake_client,
        )

    def test_success_record_separates_provider_identity_fields(
        self,
    ):
        gateway = MeteredLLMGateway(
            client=SuccessClient(),
            model="requested-model",
        )

        result = gateway.complete(
            messages=[
                {
                    "role": "user",
                    "content": "hello",
                }
            ],
            purpose="agent",
            temperature=0.2,
        )

        self.assertEqual(
            result,
            "hello",
        )

        records = (
            gateway.get_call_records()
        )

        self.assertEqual(
            len(records),
            1,
        )

        record = records[0]

        # Object identity returned in the
        # response body.
        self.assertEqual(
            record[
                "provider_completion_id"
            ],
            "chatcmpl_completion_123",
        )

        # HTTP request identity from the
        # x-request-id response header.
        self.assertEqual(
            record[
                "provider_request_id"
            ],
            "req_http_456",
        )

        self.assertNotEqual(
            record[
                "provider_completion_id"
            ],
            record[
                "provider_request_id"
            ],
        )

        self.assertEqual(
            record["provider_model"],
            "gpt-4o-2024-11-20",
        )

        self.assertEqual(
            record[
                "system_fingerprint"
            ],
            "fp_backend_789",
        )

        self.assertEqual(
            record["finish_reason"],
            "stop",
        )

    def test_failed_provider_call_preserves_request_id_when_available(
        self,
    ):
        gateway = MeteredLLMGateway(
            client=FailingClient(),
            model="requested-model",
        )

        with self.assertRaises(
            SyntheticProviderError
        ):
            gateway.complete(
                messages=[
                    {
                        "role": "user",
                        "content": "fail",
                    }
                ],
                purpose="agent",
                temperature=0.2,
            )

        records = (
            gateway.get_call_records()
        )

        self.assertEqual(
            len(records),
            1,
        )

        record = records[0]

        self.assertEqual(
            record["status"],
            "error",
        )

        self.assertEqual(
            record[
                "provider_request_id"
            ],
            "req_failed_http_999",
        )

        self.assertIsNone(
            record[
                "provider_completion_id"
            ]
        )

        self.assertIsNone(
            record["provider_model"]
        )

        self.assertIsNone(
            record[
                "system_fingerprint"
            ]
        )

        self.assertIsNone(
            record["finish_reason"]
        )


if __name__ == "__main__":
    unittest.main()
