"""Provider adapters must be safe with no keys and no network."""

import os
import unittest
from unittest import mock

from ragforge.providers import (
    AnthropicProvider,
    MockProvider,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    ProviderUnavailable,
    complete_with_fallback,
    get_provider,
    provider_status,
)

NO_KEYS = {"OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": "", "OLLAMA_HOST": ""}


class MockProviderTests(unittest.TestCase):
    def test_is_always_available(self):
        self.assertTrue(MockProvider().available())

    def test_output_is_deterministic(self):
        prompt = "Question: what?\n\n[1] A passage."
        self.assertEqual(MockProvider().complete(prompt), MockProvider().complete(prompt))

    def test_different_prompts_differ(self):
        self.assertNotEqual(MockProvider().complete("a"), MockProvider().complete("b"))

    def test_says_it_is_not_a_model(self):
        text = MockProvider().complete("[1] passage")
        self.assertIn("No model call was made", text)


class ConfigurationTests(unittest.TestCase):
    def test_openai_is_unavailable_without_a_key(self):
        with mock.patch.dict(os.environ, NO_KEYS):
            self.assertFalse(OpenAIProvider().available())

    def test_openai_can_be_configured_explicitly(self):
        with mock.patch.dict(os.environ, NO_KEYS):
            self.assertTrue(OpenAIProvider(api_key="sk-test").available())

    def test_unconfigured_provider_raises_before_any_request(self):
        with mock.patch.dict(os.environ, NO_KEYS):
            provider = OpenAIProvider()
            with mock.patch("urllib.request.urlopen") as opener:
                with self.assertRaises(ProviderUnavailable):
                    provider.complete("hello")
                opener.assert_not_called()

    def test_anthropic_reports_its_env_var(self):
        status = AnthropicProvider().status()
        self.assertEqual(status["env_var"], "ANTHROPIC_API_KEY")
        self.assertTrue(status["network"])

    def test_ollama_needs_an_explicit_host(self):
        with mock.patch.dict(os.environ, NO_KEYS):
            self.assertFalse(OllamaProvider().available())
        provider = OllamaProvider(host="http://localhost:11434/")
        self.assertTrue(provider.available())
        self.assertEqual(provider.endpoint, "http://localhost:11434/api/chat")

    def test_unknown_provider_name(self):
        with self.assertRaises(ValueError):
            get_provider("does-not-exist")


class StatusTests(unittest.TestCase):
    def test_status_lists_every_adapter_without_secret_values(self):
        with mock.patch.dict(os.environ, {**NO_KEYS, "OPENAI_API_KEY": "sk-super-secret"}):
            statuses = provider_status()
        self.assertEqual([s["name"] for s in statuses], ["mock", "openai", "anthropic", "ollama"])
        rendered = repr(statuses)
        self.assertNotIn("sk-super-secret", rendered)
        self.assertTrue(next(s for s in statuses if s["name"] == "openai")["available"])


class FallbackTests(unittest.TestCase):
    def test_configured_mock_is_not_degraded(self):
        result = complete_with_fallback("hello", provider="mock")
        self.assertFalse(result["degraded"])
        self.assertEqual(result["provider"], "mock")

    def test_unconfigured_provider_degrades_explicitly(self):
        with mock.patch.dict(os.environ, NO_KEYS):
            result = complete_with_fallback("hello", provider="openai")
        self.assertTrue(result["degraded"])
        self.assertEqual(result["provider"], "mock")
        self.assertIn("not configured", result["reason"])

    def test_transport_failure_also_degrades(self):
        with mock.patch.dict(os.environ, {**NO_KEYS, "OPENAI_API_KEY": "sk-test"}):
            with mock.patch.object(OpenAIProvider, "complete", side_effect=ProviderError("boom")):
                result = complete_with_fallback("hello", provider="openai")
        self.assertTrue(result["degraded"])
        self.assertIn("boom", result["reason"])


if __name__ == "__main__":
    unittest.main()
