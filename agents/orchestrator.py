from models.base import ModelBackend


class Orchestrator:
    def __init__(self, model: ModelBackend):
        self.model = model
        self.agents = {}

    def register_agent(self, agent):
        self.agents[agent.name] = agent

    def _choose_agent(self, task: str) -> str:
        prompt = f"""Given the following task, which agent is best suited? Choose one:
- research: for fact-finding, answering questions, web searches, summarization
- code: for writing, reading, debugging, or executing code
- writer: for creating or editing documents, emails, reports, creative writing
- project: for scaffolding projects, creating directory structures, initializing repos

Task: {task}

Respond with ONLY the agent name (research, code, writer, or project)."""
        response = self.model.generate(prompt, temperature=0.1).strip().lower()
        for name in self.agents:
            if name in response:
                return name
        return "research"

    def run(self, task: str, agent_name: str | None = None, working_dir: str = "") -> tuple[str, str, list]:
        if agent_name and agent_name in self.agents:
            agent = self.agents[agent_name]
        else:
            chosen = self._choose_agent(task)
            print(f"  -> Selected agent: {chosen}")
            agent = self.agents[chosen]
        self.model.track_agent(agent.name)
        result, steps = agent.run(task, working_dir=working_dir)
        return result, agent.name, steps
