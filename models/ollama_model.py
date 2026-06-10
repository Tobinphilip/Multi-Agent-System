import json
import requests
import subprocess
import time
import sys
from .base import ModelBackend


class OllamaModel(ModelBackend):
    def __init__(self, model_name: str = "gemma4:e4b", ollama_host: str = "http://localhost:11434"):
        super().__init__()
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")

    def start(self):
        self._ensure_ollama_running()
        self._pull_model()

    def _ensure_ollama_running(self):
        try:
            requests.get(f"{self.ollama_host}/api/tags", timeout=3)
        except requests.ConnectionError:
            print("Starting Ollama...")
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(30):
                time.sleep(1)
                try:
                    requests.get(f"{self.ollama_host}/api/tags", timeout=2)
                    return
                except requests.ConnectionError:
                    continue
            print("Failed to start Ollama.")
            sys.exit(1)

    def _pull_model(self):
        result = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
        models = result.json().get("models", [])
        if not any(m["name"].startswith(self.model_name) for m in models):
            print(f"Pulling model {self.model_name}...")
            response = requests.post(
                f"{self.ollama_host}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=600,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        print(f"  {data['status']}")
            print(f"Model {self.model_name} ready.")

    @staticmethod
    def list_models(ollama_host: str = "http://localhost:11434") -> list[dict]:
        resp = requests.get(f"{ollama_host.rstrip('/')}/api/tags", timeout=10)
        return resp.json().get("models", [])

    def cleanup(self):
        try:
            requests.post(
                f"{self.ollama_host}/api/generate",
                json={"model": self.model_name, "keep_alive": "0s"},
                timeout=5,
            )
        except Exception:
            pass

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 4096, "num_ctx": 8192},
        }
        if system:
            payload["system"] = system
        response = requests.post(
            f"{self.ollama_host}/api/generate",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        self._prompt_tokens += data.get("prompt_eval_count", 0)
        self._completion_tokens += data.get("eval_count", 0)
        self._total_tokens = self._prompt_tokens + self._completion_tokens
        return data.get("response", "")

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(
            f"{self.ollama_host}/api/chat",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        self._prompt_tokens += data.get("prompt_eval_count", 0)
        self._completion_tokens += data.get("eval_count", 0)
        self._total_tokens = self._prompt_tokens + self._completion_tokens
        return data["message"]["content"]
