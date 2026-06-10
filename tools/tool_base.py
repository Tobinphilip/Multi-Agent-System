import subprocess
import json
from abc import ABC, abstractmethod
from pathlib import Path


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def run(self, **kwargs) -> str:
        ...

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description}


class WebSearchTool(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web. Input: a search query string."

    def run(self, query: str = "") -> str:
        try:
            import urllib.parse, urllib.request
            encoded = urllib.parse.quote(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            import re
            results = re.findall(r'<a rel="nofollow" href="(.*?)".*?>(.*?)</a>', html, re.DOTALL)
            output = []
            for href, text in results[:8]:
                clean = re.sub(r"<[^>]+>", "", text).strip()
                output.append(f"{clean}\n  {href}")
            return "\n\n".join(output) if output else "No results found."
        except Exception as e:
            return f"Web search error: {e}"


class FileReadTool(Tool):
    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read a file from disk. Needs 'path' (full file path including filename)."

    def run(self, path: str = "") -> str:
        try:
            p = Path(path).expanduser().resolve()
            if not p.exists():
                return f"File not found: {path}"
            if p.is_dir():
                return f"Error: '{path}' is a directory. Provide a full file path."
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Read error: {e}"


class FileWriteTool(Tool):
    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write content to a file. Needs 'path' (full file path including filename) and 'content'."

    def run(self, path: str = "", content: str = "") -> str:
        try:
            p = Path(path).expanduser().resolve()
            if p.exists() and p.is_dir():
                return f"Error: '{path}' is a directory. Provide a full file path (e.g., '{path}/filename.txt')."
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written to {p}"
        except Exception as e:
            return f"Write error: {e}"


class FileMkdirTool(Tool):
    @property
    def name(self) -> str:
        return "file_mkdir"

    @property
    def description(self) -> str:
        return "Create a directory at the given path. Creates parent directories if needed."

    def run(self, path: str = "") -> str:
        try:
            p = Path(path).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return f"Directory created: {p}"
        except Exception as e:
            return f"Mkdir error: {e}"


class CodeExecTool(Tool):
    @property
    def name(self) -> str:
        return "code_exec"

    @property
    def description(self) -> str:
        return "Execute a shell command. Input: shell command string. Returns stdout+stderr."

    def run(self, command: str = "") -> str:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out (30s)."
        except Exception as e:
            return f"Exec error: {e}"
