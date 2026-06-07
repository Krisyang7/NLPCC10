"""Anthropic Messages API client exposing the same interface as the OpenAI client.

Lets the existing baseline runner talk to an Anthropic-compatible endpoint
(e.g. the aicoding relay at https://api.aicoding.sh) without changing any task,
prompt, parsing, or evaluation code. It accepts the OpenAI-style ``messages``
the runner builds (including ``image_url`` data-URLs) and converts them to the
Anthropic ``/v1/messages`` schema.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from baseline_prompting.errors import BaselineError


def _normalize_messages_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/messages"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/messages"
    return f"{base_url}/v1/messages"


class AnthropicCompatibleClient:
    """Drop-in replacement for OpenAICompatibleClient backed by /v1/messages."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        if not base_url:
            raise BaselineError(
                "base URL is required; pass --base-url or set ANTHROPIC_BASE_URL"
            )
        if not api_key:
            raise BaselineError(
                "API key is required; pass --api-key or set ANTHROPIC_AUTH_TOKEN"
            )
        self.url = _normalize_messages_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.version = anthropic_version

    @staticmethod
    def _convert_content(content: Any) -> Any:
        """Convert OpenAI-style content into Anthropic content blocks."""
        if isinstance(content, str):
            return content
        blocks: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                blocks.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                if isinstance(url, str) and url.startswith("data:"):
                    header, _, data = url.partition(",")
                    media_type = header[len("data:"):].split(";")[0] or "image/jpeg"
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        }
                    )
                # Non-data URLs are skipped: the relay may not fetch them.
        return blocks

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        system_prompt: str | None = None
        conversation: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                content = message.get("content")
                system_prompt = content if isinstance(content, str) else None
                continue
            conversation.append(
                {"role": role, "content": self._convert_content(message.get("content"))}
            )

        # NOTE: assistant prefill ("{") is intentionally not used. The aicoding
        # Bedrock relay rejects requests whose final turn is an assistant
        # prefill (HTTP 400 ValidationException). The prompt already instructs
        # JSON-only output, and extract_json_object() recovers the object.

        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "messages": conversation,
        }
        if system_prompt:
            payload["system"] = system_prompt

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.version,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise BaselineError(f"API HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise BaselineError(f"API request failed: {exc}") from exc

        try:
            obj = json.loads(body)
            parts = obj["content"]
            text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and part.get("type") == "text"
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise BaselineError(f"unexpected API response: {body[:1000]}") from exc

        return text
