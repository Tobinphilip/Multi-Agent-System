import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict


class ModelBackend(ABC):
    def __init__(self):
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._start_time = time.time()
        self._agent_calls = defaultdict(int)
        self._total_agent_calls = 0

    def track_agent(self, name: str):
        self._agent_calls[name] += 1
        self._total_agent_calls += 1

    @abstractmethod
    def start(self):
        ...

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None, temperature: float = 0.7) -> str:
        ...

    @abstractmethod
    def chat(self, messages: list, temperature: float = 0.7) -> str:
        ...

    def cleanup(self):
        ...

    def get_stats(self) -> dict:
        import psutil
        proc = psutil.Process(os.getpid())
        mem = proc.memory_info()
        return {
            "process": {
                "cpu_percent": proc.cpu_percent(interval=0),
                "memory_rss_mb": mem.rss / 1024 / 1024,
                "memory_vms_mb": mem.vms / 1024 / 1024,
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=0),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / 1024 / 1024 / 1024,
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            },
            "tokens": {
                "prompt": self._prompt_tokens,
                "completion": self._completion_tokens,
                "total": self._total_tokens,
            },
            "agents": {
                "total_calls": self._total_agent_calls,
                "by_agent": dict(self._agent_calls),
            },
            "uptime_seconds": int(time.time() - self._start_time),
        }
