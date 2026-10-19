from llm import LLM, parse_json
from tracer import Tracer
import time


class BaseAgent:
    name: str = "base_agent"
    role: str = "You are a helpful assistant."

    def __init__(self, llm: LLM, tracer: Tracer, memory=None):
        self.llm = llm
        self.tracer = tracer
        self.memory = memory

    def _recall_block(self, query: str, k: int = 4) -> str:
        """Retrieve relevant notes from earlier runs, formatted for the prompt."""
        if not self.memory:
            return ""
        hits = self.memory.recall(self.name, query, k)
        if not hits:
            return ""
        lines = [f"- [{h['domain']} / {h['region']}, {h['kind']}] {h['text']}" for h in hits]
        return (
            "Notes from earlier runs (unverified, possibly outdated, and possibly from a "
            "different domain or region). Use a note only if it is relevant to THIS domain "
            "AND region, and never copy it blindly:\n" + "\n".join(lines) + "\n\n"
        )

    def _remember(self, domain: str, region: str, notes: list) -> None:
        if self.memory:
            self.memory.remember(self.name, domain, region, notes)

    def _ask(self, user_prompt: str, expect_json: bool = True):
        """Call the LLM, log the exchange and its duration, optionally parse JSON."""
        self.tracer.log(self.name, "reasoning", stage="prompt", prompt=user_prompt)
        start = time.time()
        raw = self.llm.complete(self.role, user_prompt)
        elapsed = round(time.time() - start, 1)
        self.tracer.log(self.name, "reasoning", stage="response", response=raw, elapsed_s=elapsed)
        return parse_json(raw) if expect_json else raw

    def run(self, state: dict) -> dict:
        """Subclasses implement this: read from state, return a partial state update."""
        raise NotImplementedError
