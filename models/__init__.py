import os
from .base import ModelBackend
from .ollama_model import OllamaModel
from .llamacpp_model import LlamaCPPModel


BACKEND_OLLAMA = "ollama"
BACKEND_LLAMACPP = "llamacpp"


def create_model(backend: str | None = None, **kwargs) -> ModelBackend:
    backend = (backend or os.environ.get("AGENT_BACKEND", BACKEND_OLLAMA)).lower()

    if backend == BACKEND_LLAMACPP:
        model_path = kwargs.get("model_path") or os.environ.get(
            "AGENT_MODEL_PATH",
            input("Path to GGUF model file: "),
        )
        n_ctx = kwargs.get("n_ctx", int(os.environ.get("AGENT_N_CTX", "4096")))
        n_gpu_layers = kwargs.get("n_gpu_layers", int(os.environ.get("AGENT_N_GPU_LAYERS", "-1")))
        return LlamaCPPModel(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

    model_name = kwargs.get("model_name") or os.environ.get("AGENT_MODEL", "gemma4:e4b")
    return OllamaModel(model_name=model_name)


__all__ = [
    "ModelBackend", "OllamaModel", "LlamaCPPModel",
    "create_model",
    "BACKEND_OLLAMA", "BACKEND_LLAMACPP",
]
