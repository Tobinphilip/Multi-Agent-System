#!/usr/bin/env python3
import json
import asyncio
import gc
import signal
import sys
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from models import create_model, BACKEND_OLLAMA, BACKEND_LLAMACPP, OllamaModel

app = FastAPI(title="Agent UI")

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)

_current_model = None
_current_backend = None
_current_model_arg = None
_agent_call_counts = defaultdict(int)
_total_agent_calls = 0


def cleanup_all():
    global _current_model
    if _current_model:
        try:
            _current_model.cleanup()
        except Exception:
            pass
        _current_model = None
    gc.collect()


def signal_handler(signum, frame):
    cleanup_all()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_or_create_model(backend: str, model_arg: str):
    global _current_model, _current_backend, _current_model_arg
    if _current_model is not None and _current_backend == backend and _current_model_arg == model_arg:
        return _current_model

    cleanup_all()
    _current_backend = backend
    _current_model_arg = model_arg

    if backend == BACKEND_LLAMACPP:
        _current_model = create_model(backend=BACKEND_LLAMACPP, model_path=model_arg)
    else:
        _current_model = create_model(backend=BACKEND_OLLAMA, model_name=model_arg)
    _current_model.start()
    return _current_model


def list_ollama_models():
    try:
        return OllamaModel.list_models()
    except Exception:
        return []


@app.get("/api/health")
async def health():
    results = {"ollama": False, "llamacpp": False}
    try:
        models = await asyncio.to_thread(list_ollama_models)
        results["ollama"] = len(models) > 0
    except Exception:
        results["ollama"] = False
    return {
        "status": "ok",
        "backends": results,
        "ollama_models": [m["name"] for m in await asyncio.to_thread(list_ollama_models)],
    }


@app.post("/api/check-model")
async def check_model(request: Request):
    body = await request.json()
    backend = body.get("backend", BACKEND_OLLAMA)
    model = body.get("model", "")

    if backend == BACKEND_LLAMACPP:
        p = Path(model).expanduser()
        if not p.exists():
            return JSONResponse({"ok": False, "error": f"File not found: {model}"})
        if not str(model).endswith(".gguf"):
            return JSONResponse({"ok": False, "error": "File must have .gguf extension"})
        return JSONResponse({"ok": True})
    try:
        models = await asyncio.to_thread(list_ollama_models)
        names = [m["name"] for m in models]
        if model and model not in names:
            return JSONResponse({"ok": False, "error": f"Model '{model}' not found. Pull it with: ollama pull {model}"})
        if not names:
            return JSONResponse({"ok": False, "error": "No models available. Pull one with: ollama pull <name>"})
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"Ollama not reachable: {e}"})


@app.get("/api/stats")
async def api_stats():
    try:
        m = _current_model if _current_model else create_model()
        stats = await asyncio.to_thread(m.get_stats)
        stats["loaded"] = _current_model is not None
        stats["agents"] = {
            "total_calls": _total_agent_calls,
            "by_agent": dict(_agent_call_counts),
        }
        return stats
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/unload")
async def api_unload():
    cleanup_all()
    return JSONResponse({"ok": True, "message": "Model unloaded and memory freed"})


@app.post("/api/shutdown")
async def api_shutdown():
    cleanup_all()
    asyncio.create_task(_shutdown())
    return JSONResponse({"ok": True, "message": "Server shutting down..."})


async def _shutdown():
    await asyncio.sleep(0.5)
    loop = asyncio.get_event_loop()
    loop.stop()


@app.get("/api/models")
async def api_list_models():
    models = await asyncio.to_thread(list_ollama_models)
    return [{"name": m["name"], "size": f"{m.get('size', 0) / 1e9:.1f} GB"} for m in models]


@app.get("/api/browse")
async def api_browse(path: str = ""):
    try:
        p = Path(path).expanduser().resolve() if path else Path.home()
        if not p.is_dir():
            return JSONResponse({"error": "Not a directory"}, status_code=400)
        entries = []
        for child in sorted(p.iterdir()):
            if child.is_dir() and not child.name.startswith("."):
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "is_dir": True,
                })
        parent = str(p.parent) if p.parent != p else None
        return {"current": str(p), "parent": parent, "entries": entries}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/browse/create")
async def api_browse_create(request: Request):
    body = await request.json()
    path = body.get("path", "").strip()
    if not path:
        return JSONResponse({"error": "Path is required"}, status_code=400)
    try:
        p = Path(path).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": str(p)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    task = body.get("task", "").strip()
    working_dir = body.get("working_dir", "").strip()
    agent_name = body.get("agent") or None
    model_arg = body.get("model", "gemma4:e4b")
    backend = body.get("backend", BACKEND_OLLAMA)

    if not task:
        return JSONResponse({"error": "Task is required"}, status_code=400)

    async def generate():
        try:
            m = await asyncio.to_thread(get_or_create_model, backend, model_arg)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Model connection failed: {e}'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status', 'content': 'Thinking...'})}\n\n"

        from agents import ResearchAgent, CodeAgent, WriterAgent, ProjectAgent, Orchestrator
        orch = Orchestrator(m)
        orch.register_agent(ResearchAgent(m))
        orch.register_agent(CodeAgent(m))
        orch.register_agent(WriterAgent(m))
        orch.register_agent(ProjectAgent(m))

        try:
            global _total_agent_calls
            _total_agent_calls += 1
            result, used_agent, steps = await asyncio.to_thread(orch.run, task, agent_name, working_dir)
            _agent_call_counts[used_agent] += 1
            for step in steps:
                yield f"data: {json.dumps({'type': 'step', 'type2': step['type'], **{k:v for k,v in step.items() if k != 'type'}})}\n\n"
            stats = await asyncio.to_thread(m.get_stats)
            stats["agents"] = {
                "total_calls": _total_agent_calls,
                "by_agent": dict(_agent_call_counts),
            }
            yield f"data: {json.dumps({'type': 'done', 'content': result, 'stats': stats})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def index():
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
