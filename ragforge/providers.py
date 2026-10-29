"""Provider adapters for the optional generation step.

The retrieval, answering and evaluation paths never call a provider: they are
deterministic and work offline. This module is only for teams that want a
language model to rewrite the extractive answer in prose.

All four adapters are import-safe and key-safe. Constructing one does no I/O;
``available()`` checks configuration; ``complete()`` raises
``ProviderUnavailable`` before any socket is opened when the key or host is
missing. That is deliberate, a starter kit that crashes on import because
``OPENAI_API_KEY`` is unset is not a starter kit.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request


class ProviderUnavailable(RuntimeError):
    """The provider is not configured. No request has been attempted."""


class ProviderError(RuntimeError):
    """The provider was configured but the call failed."""


class Provider:
    name = "provider"
    env_var = ""
    default_model = ""

    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        self.model = model or self.default_model
        self.api_key = api_key

    def available(self) -> bool:  # pragma: no cover - overridden
        return False

    def status(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "available": self.available(),
            "env_var": self.env_var,
            "network": self.name != "mock",
        }

    def complete(self, prompt: str, *, system: str | None = None, timeout: float = 60.0) -> str:
        raise NotImplementedError


class MockProvider(Provider):
    """Deterministic stand-in used by tests and the offline demo.

    It does not pretend to be a model. The response is a stable digest of the
    prompt so a test can assert that the same input produces the same output,
    and the text says explicitly that no model was called.
    """

    name = "mock"
    default_model = "deterministic-mock"

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, *, system: str | None = None, timeout: float = 60.0) -> str:  # noqa: ARG002
        digest = hashlib.sha256((system or "").encode() + b"\0" + prompt.encode()).hexdigest()[:12]
        passages = prompt.count("\n[")
        return (
            f"[mock:{digest}] No model call was made. "
            f"The prompt contained {passages} numbered passage(s); the extractive answer above is the shipped path."
        )


class HttpProvider(Provider):
    """Shared request plumbing for providers that speak HTTP JSON."""

    endpoint = ""

    def _key(self) -> str:
        return self.api_key or os.environ.get(self.env_var, "")

    def available(self) -> bool:
        return bool(self._key())

    def _post(self, payload: dict, headers: dict[str, str], timeout: float) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.endpoint, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed endpoint
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise ProviderError(f"{self.name}: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{self.name}: request failed: {exc}") from exc

    def complete(self, prompt: str, *, system: str | None = None, timeout: float = 60.0) -> str:
        if not self.available():
            raise ProviderUnavailable(
                f"{self.name} is not configured. Set {self.env_var} or pass api_key=..."
            )
        payload = self._payload(prompt, system)
        data = self._post(payload, self._headers(), timeout)
        return self._extract(data)


class OpenAIProvider(HttpProvider):
    name = "openai"
    env_var = "OPENAI_API_KEY"
    default_model = "gpt-4o-mini"
    endpoint = "https://api.openai.com/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._key()}"}

    def _payload(self, prompt: str, system: str | None) -> dict:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return {"model": self.model, "messages": messages, "temperature": 0}

    def _extract(self, data: dict) -> str:
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("openai: unexpected response shape") from exc


class AnthropicProvider(HttpProvider):
    name = "anthropic"
    env_var = "ANTHROPIC_API_KEY"
    default_model = "claude-3-5-haiku-latest"
    endpoint = "https://api.anthropic.com/v1/messages"

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._key(), "anthropic-version": "2023-06-01"}

    def _payload(self, prompt: str, system: str | None) -> dict:
        body = {"model": self.model, "max_tokens": 512, "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        return body

    def _extract(self, data: dict) -> str:
        try:
            return "".join(block.get("text", "") for block in data["content"]).strip()
        except (KeyError, TypeError) as exc:
            raise ProviderError("anthropic: unexpected response shape") from exc


class OllamaProvider(HttpProvider):
    """Local models through Ollama.

    Treated as unavailable until ``OLLAMA_HOST`` is set. Probing localhost at
    import time would make a web app's startup depend on a daemon that may not
    be running, and would turn an offline kit into one that hangs.
    """

    name = "ollama"
    env_var = "OLLAMA_HOST"
    default_model = "llama3.2"

    def __init__(self, *, model: str | None = None, api_key: str | None = None, host: str | None = None) -> None:
        super().__init__(model=model, api_key=api_key)
        self.host = (host or os.environ.get("OLLAMA_HOST") or "").rstrip("/")
        self.endpoint = f"{self.host}/api/chat" if self.host else ""

    def available(self) -> bool:
        return bool(self.host)

    def _headers(self) -> dict[str, str]:
        return {}

    def _payload(self, prompt: str, system: str | None) -> dict:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return {"model": self.model, "messages": messages, "stream": False, "options": {"temperature": 0}}

    def _extract(self, data: dict) -> str:
        try:
            return data["message"]["content"].strip()
        except (KeyError, TypeError) as exc:
            raise ProviderError("ollama: unexpected response shape") from exc


PROVIDERS = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def get_provider(name: str = "mock", **options) -> Provider:
    try:
        factory = PROVIDERS[name]
    except KeyError as exc:
        raise ValueError(f"unknown provider: {name!r} (choose from {', '.join(PROVIDERS)})") from exc
    return factory(**options)


def provider_status() -> list[dict]:
    """Configuration report for ``/health``, never includes key values."""
    return [factory().status() for factory in PROVIDERS.values()]


def complete_with_fallback(prompt: str, *, provider: str = "mock", system: str | None = None) -> dict:
    """Try the requested provider, fall back to the deterministic mock.

    The fallback is returned with an explicit ``degraded`` flag rather than
    silently, an answer that quietly came from a mock must not be labelled as
    model output.
    """
    adapter = get_provider(provider)
    if not adapter.available():
        return {
            "text": MockProvider().complete(prompt, system=system),
            "provider": "mock",
            "requested": provider,
            "degraded": True,
            "reason": f"{provider} is not configured",
        }
    try:
        text = adapter.complete(prompt, system=system)
    except ProviderError as exc:
        return {
            "text": MockProvider().complete(prompt, system=system),
            "provider": "mock",
            "requested": provider,
            "degraded": True,
            "reason": str(exc),
        }
    return {"text": text, "provider": provider, "requested": provider, "degraded": False, "reason": ""}
