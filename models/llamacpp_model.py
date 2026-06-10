from pathlib import Path
from .base import ModelBackend


class LlamaCPPModel(ModelBackend):
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, verbose: bool = False):
        super().__init__()
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self._llm = None

    def start(self):
        from llama_cpp import Llama
        print(f"Loading model: {self.model_path}")
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=self.verbose,
        )
        print("Model loaded.")

    def cleanup(self):
        import gc
        if self._llm is not None:
            del self._llm
            self._llm = None
        gc.collect()

    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        if self._llm is None:
            raise RuntimeError("Model not started. Call start() first.")

        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        output = self._llm(
            full_prompt,
            max_tokens=self.n_ctx,
            temperature=temperature,
            stop=["</s>", "<|eot_id|>", "<|end|>"],
            echo=False,
        )
        usage = output.get("usage", {})
        self._prompt_tokens += usage.get("prompt_tokens", 0)
        self._completion_tokens += usage.get("completion_tokens", 0)
        self._total_tokens = self._prompt_tokens + self._completion_tokens
        return output["choices"][0]["text"].strip()

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        if self._llm is None:
            raise RuntimeError("Model not started. Call start() first.")

        output = self._llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=self.n_ctx,
        )
        usage = output.get("usage", {})
        self._prompt_tokens += usage.get("prompt_tokens", 0)
        self._completion_tokens += usage.get("completion_tokens", 0)
        self._total_tokens = self._prompt_tokens + self._completion_tokens
        return output["choices"][0]["message"]["content"].strip()
