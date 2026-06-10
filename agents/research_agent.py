from agents.base_agent import BaseAgent
from tools.tool_base import WebSearchTool, FileReadTool


class ResearchAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "research"

    @property
    def system_prompt(self) -> str:
        return """You are a Research Agent. Your job is to gather information, summarize findings, and answer questions thoroughly.
You can use web search to find up-to-date information.
Present your findings clearly with sources."""

    @property
    def tool_instructions(self) -> str:
        return """To use a tool, output a JSON object inside a code block:
```json
{"tool": "web_search", "query": "your search query"}
```
Available tools: web_search, file_read."""

    def __init__(self, model, tools=None):
        super().__init__(model, tools or [WebSearchTool(), FileReadTool()])
