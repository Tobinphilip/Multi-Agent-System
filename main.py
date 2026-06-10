#!/usr/bin/env python3
import argparse
import os

from models import create_model, BACKEND_OLLAMA, BACKEND_LLAMACPP, OllamaModel
from agents import ResearchAgent, CodeAgent, WriterAgent, ProjectAgent, Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Local multi-agent task runner")
    parser.add_argument("task", nargs="*", help="Task description")
    parser.add_argument("--agent", "-a", choices=["research", "code", "writer", "project"],
                        help="Force a specific agent (auto-selected if omitted)")
    parser.add_argument("--model", "-m", default="gemma4:e4b",
                        help="Model name (Ollama) or path to GGUF file (llamacpp)")
    parser.add_argument("--backend", "-b", default=BACKEND_OLLAMA,
                        choices=[BACKEND_OLLAMA, BACKEND_LLAMACPP],
                        help=f"Model backend (default: {BACKEND_OLLAMA})")
    parser.add_argument("--list-models", action="store_true",
                        help="List available Ollama models and exit")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive mode")
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="Context window size for llama.cpp (default: 4096)")

    args = parser.parse_args()

    if args.list_models:
        if args.backend == BACKEND_LLAMACPP:
            print("Use --model to specify a path to a .gguf file")
            return
        ollama = OllamaModel(model_name="")
        ollama._ensure_ollama_running()
        models = OllamaModel.list_models()
        if not models:
            print("No models found. Pull one with: ollama pull <name>")
            return
        for m in models:
            name = m["name"]
            size = m.get("size", 0)
            print(f"  {name}  ({size / 1e9:.1f} GB)")
        return

    if args.backend == BACKEND_LLAMACPP:
        os.environ.setdefault("AGENT_MODEL_PATH", args.model)
        model = create_model(backend=BACKEND_LLAMACPP, model_path=args.model, n_ctx=args.n_ctx)
    else:
        model = create_model(backend=BACKEND_OLLAMA, model_name=args.model)

    model.start()

    research_agent = ResearchAgent(model)
    code_agent = CodeAgent(model)
    writer_agent = WriterAgent(model)
    project_agent = ProjectAgent(model)

    orchestrator = Orchestrator(model)
    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(code_agent)
    orchestrator.register_agent(writer_agent)
    orchestrator.register_agent(project_agent)

    if args.interactive:
        print(f"Multi-Agent System (backend: {args.backend})")
        print("Type 'quit' to exit, 'agent <name>' to force an agent.\n")
        current_agent = None
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower().startswith("agent "):
                current_agent = user_input.split(" ", 1)[1].strip()
                print(f"  Agent forced to: {current_agent}")
                continue
            result, _, _ = orchestrator.run(user_input, agent_name=current_agent)
            print(f"\n{result}\n")
        return

    if not args.task:
        parser.print_help()
        return

    task = " ".join(args.task)
    result, _, _ = orchestrator.run(task, agent_name=args.agent)
    print(result)


if __name__ == "__main__":
    main()
