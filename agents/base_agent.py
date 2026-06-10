import json
import re
from abc import ABC, abstractmethod
from models.base import ModelBackend
from tools.tool_base import Tool


class BaseAgent(ABC):
    def __init__(self, model: ModelBackend, tools: list[Tool] | None = None):
        self.model = model
        self.tools = tools or []

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        ...

    @property
    @abstractmethod
    def tool_instructions(self) -> str:
        ...

    def add_tool(self, tool: Tool):
        self.tools.append(tool)

    def _tool_descriptions(self) -> str:
        if not self.tools:
            return ""
        descs = []
        for t in self.tools:
            descs.append(f"- {t.name}: {t.description}")
        return "Available tools:\n" + "\n".join(descs)

    def _find_json_tool_call(self, text: str) -> dict | None:
        patterns = [
            r"<tool>(.*?)</tool>",
            r"```json\s*\n?(.*?)```",
            r"```\s*\n?(.*?)```",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.DOTALL)
            if m:
                block = m.group(1).strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        def _try_parse(s):
            for suffix in ('', '}'):
                try:
                    return json.loads(s + suffix)
                except json.JSONDecodeError:
                    continue
            return None

        for marker in ('"tool"', '"name"', '"tool_name"'):
            idx = text.find(marker)
            if idx == -1:
                continue
            start = text.rfind('{', 0, idx)
            if start == -1:
                continue
            depth, i = 0, start
            while i < len(text):
                if text[i] == '{': depth += 1
                elif text[i] == '}': depth -= 1
                if depth == 0:
                    parsed = _try_parse(text[start:i+1])
                    if parsed:
                        return parsed
                    break
                i += 1
            if depth > 0:
                parsed = _try_parse(text[start:])
                if parsed:
                    return parsed
        return None

    def _extract_tool_call(self, parsed: dict) -> tuple[str, dict] | None:
        if "tool_name" in parsed and "params" in parsed:
            return parsed["tool_name"], parsed["params"]
        if "tool" in parsed:
            return parsed["tool"], {k: v for k, v in parsed.items() if k != "tool"}
        if "name" in parsed:
            return parsed["name"], {k: v for k, v in parsed.items() if k != "name"}
        return None

    def run(self, task: str, working_dir: str = "") -> tuple[str, list]:
        steps = []
        wd_hint = f"\nIMPORTANT: Use this directory as the base for all file paths: {working_dir}\n" if working_dir else ""
        full_prompt = f"{self._tool_descriptions()}\n\n{self.tool_instructions}{wd_hint}\n\nTask: {task}"
        response = self.model.generate(prompt=full_prompt, system=self.system_prompt)

        max_iters = 12
        for _ in range(max_iters):
            parsed = self._find_json_tool_call(response)
            if not parsed:
                break

            call = self._extract_tool_call(parsed)
            if not call:
                break

            tool_name, tool_input = call
            steps.append({"type": "tool_call", "tool": tool_name, "params": tool_input})

            found = False
            for t in self.tools:
                if t.name == tool_name:
                    result = t.run(**tool_input)
                    steps.append({"type": "tool_result", "tool": tool_name, "result": result})

                    def _summarize(p):
                        d = dict(p)
                        if 'content' in d:
                            d['content'] = d['content'][:60] + '...' if len(d['content']) > 60 else d['content']
                        return json.dumps(d)
                    history = "\n".join(
                        f"  {s['type']}: {s.get('tool','')} {_summarize(s.get('params',{}))} -> {s.get('result','')[:60]}"
                        for s in steps
                    )
                    context = f"Original task: {task}"
                    if working_dir:
                        context += f"\nWorking directory: {working_dir}"
                    context += f"\n\nSteps completed so far:\n{history}\n\n"
                    context += "If the project is fully scaffolded, provide the final answer. "
                    context += "Otherwise, output your next tool call in a JSON code block:\n"
                    context += '```json\n{"tool": "...", ...}\n```'
                    response = self.model.generate(prompt=context, system=self.system_prompt)
                    found = True
                    break

            if not found:
                steps.append({"type": "tool_result", "tool": tool_name, "result": f"Unknown tool: {tool_name}"})
                break

        steps.append({"type": "final", "content": response})
        return response, steps
