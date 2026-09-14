import copy
import hashlib
import json
import time
from datetime import datetime, timezone

from config import MODEL


def _sha256_json(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value):
    return hashlib.sha256(
        (value or "").encode("utf-8")
    ).hexdigest()


def _usage_value(usage, name):
    if usage is None:
        return None

    if isinstance(usage, dict):
        return usage.get(name)

    return getattr(usage, name, None)


class MeteredLLMGateway:
    """
    Single provider boundary for TRACE v2 LLM inference.

    Every call receives a semantic purpose and leaves a raw provenance record
    containing the requested inputs, returned output, request identity,
    token usage, hashes, and timing metadata.
    """

    def __init__(self, client=None, model=MODEL):
        # Import lazily so offline unit tests can inject a fake client without
        # requiring the provider SDK to be installed or initialized.
        if client is None:
            from openai import OpenAI

            client = OpenAI()

        self.client = client
        self.model = model
        self._call_records = []
        self._next_call_id = 1

    def complete(
        self,
        messages,
        purpose,
        temperature,
    ):
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(
                "Every LLM call requires a non-empty provenance purpose."
            )

        call_id = self._next_call_id
        self._next_call_id += 1

        messages_snapshot = copy.deepcopy(messages)

        started_at = datetime.now(timezone.utc).isoformat()
        started_clock = time.perf_counter()

        request = {
            "model": self.model,
            "temperature": temperature,
            "messages": copy.deepcopy(messages_snapshot),
        }

        base_record = {
            "call_id": call_id,
            "purpose": purpose,
            "model": self.model,
            "temperature": temperature,
            "messages": messages_snapshot,
            "messages_sha256": _sha256_json(messages_snapshot),
            "started_at_utc": started_at,
        }

        try:
            response = self.client.chat.completions.create(**request)

            elapsed_ms = (
                time.perf_counter() - started_clock
            ) * 1000.0

            response_text = (
                response.choices[0].message.content or ""
            )

            usage = getattr(response, "usage", None)

            record = {
                **base_record,
                "status": "success",
                "provider_request_id": getattr(
                    response,
                    "id",
                    None,
                ),
                "provider_model": getattr(
                    response,
                    "model",
                    None,
                ),
                "response_text": response_text,
                "response_sha256": _sha256_text(
                    response_text
                ),
                "usage": {
                    "prompt_tokens": _usage_value(
                        usage,
                        "prompt_tokens",
                    ),
                    "completion_tokens": _usage_value(
                        usage,
                        "completion_tokens",
                    ),
                    "total_tokens": _usage_value(
                        usage,
                        "total_tokens",
                    ),
                },
                "elapsed_ms": elapsed_ms,
            }

            self._call_records.append(record)

            return response_text

        except Exception as exc:
            elapsed_ms = (
                time.perf_counter() - started_clock
            ) * 1000.0

            self._call_records.append({
                **base_record,
                "status": "error",
                "provider_request_id": None,
                "provider_model": None,
                "response_text": None,
                "response_sha256": None,
                "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
                "elapsed_ms": elapsed_ms,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })

            raise

    def get_call_records(self):
        return copy.deepcopy(self._call_records)

    def get_usage_totals(self):
        totals = {
            "calls": len(self._call_records),
            "successful_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        for record in self._call_records:
            if record["status"] == "success":
                totals["successful_calls"] += 1
            else:
                totals["failed_calls"] += 1

            usage = record.get("usage", {})

            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            ):
                value = usage.get(key)

                if isinstance(value, int):
                    totals[key] += value

        return totals
