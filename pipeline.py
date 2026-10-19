import json
from pathlib import Path

from llm import LLM
from tracer import Tracer
from graph import build_graph
from memory.store import MemoryStore
from report import load, build, compute_stats, render_html, decision_text

NODE_LABELS = {
    "research_node": "Research Agent",
    "funding_node": "Funding Advisor",
    "pitch_node": "Pitch Coach",
}
AGENT_LABELS = {
    "research_agent": "Research Agent",
    "funding_advisor": "Funding Advisor",
    "pitch_coach": "Pitch Coach",
}


def describe_event(r: dict):
    """Turn a trace record into a short human-readable progress line (or None to skip it)."""
    agent, event, stage = r["agent"], r["event"], r.get("stage")
    label = AGENT_LABELS.get(agent, agent)

    if agent == "orchestrator":
        if event == "handoff" and stage == "node_start":
            rev = r.get("revision", 0)
            suffix = f" (revision round {rev})" if rev else ""
            return f"▶ {NODE_LABELS.get(r.get('node'), r.get('node'))} started{suffix}"
        if event == "feedback":
            return f"↳ {decision_text(r)}"
        return None

    if event == "memory":
        if stage == "recall":
            return f"   {label}: memory recall, {r.get('used')} of {r.get('retrieved')} past notes used"
        if stage == "store":
            return f"   {label}: stored {r.get('count')} notes in memory"
        if stage and stage.endswith("_error"):
            return f"   {label}: memory problem ({stage})"
        return None

    if event == "feedback":
        if r.get("direction") == "sent":
            return f"   {label} sent feedback to {r.get('to')}"
        return f"   {label} received feedback and is revising"

    if event == "reasoning" and stage == "response":
        secs = r.get("elapsed_s")
        return f"   {label}: model call finished" + (f" in {secs}s" if secs else "")

    return None


def run_pipeline(domain, region, review=False, use_memory=True, on_event=None):
    """Run the full multi-agent workflow. Returns the final state and run details."""
    tracer = Tracer(on_event=on_event)
    memory = MemoryStore(tracer) if use_memory else None
    app = build_graph(LLM(), tracer, memory)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/workflow.mmd").write_text(app.get_graph().draw_mermaid())

    state = app.invoke(
        {
            "domain": domain,
            "region": region,
            "feedback": [],
            "revision_count": 0,
            "force_review": review,
        },
        config={"recursion_limit": 40},
    )

    name = f"{domain}_{region}_outline.json".replace(" ", "_")
    outline_path = Path("outputs") / name
    outline_path.write_text(json.dumps(state["pitch_outline"], indent=2))
    return {"state": state, "tracer": tracer, "outline_path": outline_path}


def make_report(tracer) -> Path:
    """Build the HTML trace report for a finished run."""
    records = load(tracer.path)
    items = build(records)
    per, total = compute_stats(records, items)
    out = Path("outputs") / f"report_{tracer.run_id}.html"
    out.write_text(render_html(str(tracer.path), records, items, per, total), encoding="utf-8")
    return out
