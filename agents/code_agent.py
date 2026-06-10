from agents.base_agent import BaseAgent
from tools.tool_base import FileReadTool, FileWriteTool, CodeExecTool


class CodeAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "code"

    @property
    def system_prompt(self) -> str:
        return """You are a Code Agent. You write, read, and debug code.
You can read files, write files, and execute shell commands to test code.
Write clean, well-structured code."""

    @property
    def tool_instructions(self) -> str:
        return """To use a tool, output a JSON object inside a code block:
```json
{"tool": "file_write", "path": "path/to/file.py", "content": "print('hello')"}
```
IMPORTANT: 'path' must include a filename (e.g., 'script.py'), not just a directory.
Available tools: file_read, file_write, code_exec."""

    def __init__(self, model, tools=None):
        super().__init__(model, tools or [FileReadTool(), FileWriteTool(), CodeExecTool()])
