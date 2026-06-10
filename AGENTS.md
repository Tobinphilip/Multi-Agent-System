# Agents — Multi-Agent LLM Framework

## Project Structure

```
Agents/
├── main.py                  # CLI entry point (--backend, --agent, --model, --interactive)
├── server.py                # FastAPI web server (SSE chat, /api/unload, /api/stats, /api/health, /api/check-model, /api/browse, /api/browse/create)
├── requirements.txt         # requests, fastapi, uvicorn, llama-cpp-python, psutil
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # BaseAgent: tool call parsing (JSON code blocks), two-step inference
│   ├── research_agent.py    # ResearchAgent: web_search + file_read
│   ├── code_agent.py        # CodeAgent: file_read/write + code_exec
│   ├── writer_agent.py      # WriterAgent: file_read/write
│   ├── project_agent.py     # ProjectAgent: file_mkdir/read/write + code_exec (scaffold projects)
│   └── orchestrator.py      # Orchestrator: auto-selects agent via LLM, returns (result, agent_name)
│
├── models/
│   ├── __init__.py          # create_model() factory, BACKEND_OLLAMA / BACKEND_LLAMACPP constants
│   ├── base.py              # ModelBackend ABC (start, generate, chat, cleanup, get_stats, track_agent)
│   ├── ollama_model.py      # Ollama HTTP backend, token tracking from API response
│   └── llamacpp_model.py    # llama-cpp-python backend with Metal, gc cleanup
│
├── tools/
│   ├── __init__.py
│   └── tool_base.py         # WebSearchTool, FileReadTool, FileWriteTool (dir detection), CodeExecTool
│
├── static/
│   └── index.html           # Web UI: chat, agent/model/backend selectors, copy btn, spinner, stats bar, conn modal, beforeunload beacon
│
├── "Start Server.command"   # double-click to start server (macOS)
├── "Stop Server.command"    # double-click to stop server (macOS)
├── xyz/                     # empty
├── xyzm/                    # empty
└── AGENTS.md                # this file
```

## Key Architecture

- **Persistent model singleton** (`_current_model`) — model stays loaded while browser tab is active
- **Four specialized agents** — Research (web), Code (read/write/exec), Writer (read/write), Project (scaffold)
- **Orchestrator** — auto-selects agent or uses forced agent; tracks which agent handled each call
- **Memory lifecycle** — model loads on first chat, unloads on `beforeunload` (tab close), SIGINT/SIGTERM, or POST `/api/unload`
- **Folder picker** — click the 📁 badge in the input area to browse, create, and select a working directory; passed to agents as context (prepended to task)
- **Stats bar** — CPU%, RAM%, process RSS, cumulative tokens, per-agent call counts, uptime, model connection status
- **Backends** — Ollama (default) or direct GGUF via llama-cpp-python with Metal GPU

## Common Commands

```bash
python3 server.py                    # Start web UI on http://127.0.0.1:8080
python3 main.py --interactive        # CLI interactive mode (Ollama default)
python3 main.py --agent code         # Force a specific agent
python3 main.py --backend llamacpp   # Use direct GGUF (requires --model with file path)
```

## Models

| Model | Size | Backend |
|---|---|---|
| gemma4:e4b (default) | ~9.6 GB | Ollama |
| gemma3:4b | ~4 GB | Ollama |
| qwen3.5:4b | ~4 GB | Ollama |
| qwen3-coder:30b | ~30 GB | Ollama |
| qwen3.5:9b | ~9 GB | Ollama |
| gemma4:26b | ~26 GB | Ollama |
