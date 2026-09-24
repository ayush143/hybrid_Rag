import os
import unittest

from hybrid_rag.main import resolve_ollama_settings


class OllamaConfigTests(unittest.TestCase):
    def test_uses_ollama_model_and_base_url_env(self):
        os.environ["OLLAMA_MODEL"] = "llama3.1"
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"

        model, base_url = resolve_ollama_settings()

        self.assertEqual(model, "llama3.1")
        self.assertEqual(base_url, "http://127.0.0.1:11434")

    def test_uses_defaults_when_env_missing(self):
        os.environ.pop("OLLAMA_MODEL", None)
        os.environ.pop("OLLAMA_BASE_URL", None)

        model, base_url = resolve_ollama_settings()

        self.assertEqual(model, "llama3.2")
        self.assertEqual(base_url, "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
