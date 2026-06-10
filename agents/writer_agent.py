from agents.base_agent import BaseAgent
from tools.tool_base import FileReadTool, FileWriteTool


class WriterAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "writer"

    @property
    def system_prompt(self) -> str:
        return """You are a Writer Agent. You create, edit, and improve written content.
You can read and write files. Produce clear, well-structured documents.
Adapt your tone and style based on the task requirements."""

    @property
    def tool_instructions(self) -> str:
        return """To use a tool, output a JSON object inside a code block:
```json
{"tool": "file_write", "path": "path/to/file.txt", "content": "file content here"}
```
IMPORTANT: 'path' must include a filename (e.g., 'output.txt'), not just a directory.
Available tools: file_read, file_write."""

    def __init__(self, model, tools=None):
        super().__init__(model, tools or [FileReadTool(), FileWriteTool()])
