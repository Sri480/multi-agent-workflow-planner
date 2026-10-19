from langgraph.graph import StateGraph, START, END

from state import WorkflowState
from agents.research import ResearchAgent
from agents.funding import FundingAdvisor
from agents.pitch_coach import PitchCoach

MAX_REVISIONS = 2   # loop guard: maximum feedback rounds


def traced(node_name, agent, tracer):
    """Wrap an agent so the orchestrator logs the context going in and out."""

    def node(state: WorkflowState) -> dict:
        tracer.log(
            "orchestrator", "handoff", stage="node_start", node=node_name,
            revision=state.get("revision_count", 0),
            state_keys_available=sorted(state.keys()),
        )
        update = agent.run(state)
        tracer.log(
            "orchestrator", "handoff", stage="node_end", node=node_name,
            revision=state.get("revision_count", 0),
            state_keys_updated=sorted(update.keys()),
        )
        return update

    return node


def build_graph(llm, tracer, memory=None):
    research = ResearchAgent(llm, tracer, memory=memory)
    funding = FundingAdvisor(llm, tracer, memory=memory)
    coach = PitchCoach(llm, tracer)

    # ---------- orchestration nodes (route feedback between agents) ----------

    def request_research_node(state: WorkflowState) -> dict:
        """Funding Advisor asked Research for more information."""
        requests = [str(r) for r in state["funding"].get("requests_to_research", [])]
        feedback = [{
            "from": "funding_advisor",
            "to": "research_agent",
            "issue": "The Funding Advisor needs more information to recommend well.",
            "request": " | ".join(requests),
        }]
        tracer.log("orchestrator", "feedback", stage="routed",
                   source="funding_advisor", to="research_agent", requests=requests)
        return {"feedback": feedback,
                "revision_count": state.get("revision_count", 0) + 1}

    def dispatch_feedback_node(state: WorkflowState) -> dict:
        """Pitch Coach sent feedback: deliver it to the peers, and tell the coach to revise after."""
        items = list(state["pitch_outline"].get("feedback_for_peers", []))
        items.append({
            "from": "orchestrator",
            "to": "pitch_coach",
            "issue": "Peers are revising their work based on your feedback.",
            "request": "Revise the outline using the updated research and funding.",
        })
        tracer.log("orchestrator", "feedback", stage="routed",
                   source="pitch_coach", targets=[i["to"] for i in items])
        return {"feedback": items,
                "revision_count": state.get("revision_count", 0) + 1}

    # ---------- routers (decide the next step) ----------

    def route_after_funding(state: WorkflowState) -> str:
        requests = state["funding"].get("requests_to_research") or []
        if requests and state.get("revision_count", 0) == 0:
            return "request_research_node"
        if requests:
            tracer.log("orchestrator", "feedback", stage="funding_requests_skipped",
                       reason="already revised once", requests=len(requests))
        return "pitch_node"

    def route_after_pitch(state: WorkflowState) -> str:
        outline = state["pitch_outline"]
        revisions = state.get("revision_count", 0)
        pending = outline.get("feedback_for_peers", [])

        if pending and revisions < MAX_REVISIONS:
            tracer.log("orchestrator", "feedback", stage="loop_continue",
                       revision=revisions, items=len(pending))
            return "dispatch_feedback_node"
        if pending:
            tracer.log("orchestrator", "feedback", stage="loop_guard_stop",
                       revision=revisions, note="max revisions reached; finalising")
        else:
            tracer.log("orchestrator", "feedback", stage="loop_end",
                       revision=revisions, ready=outline.get("ready"),
                       note="no outstanding feedback")
        return END

    def route_feedback_target(state: WorkflowState) -> str:
        # Research feeds Funding via a fixed edge, so start at research if it was addressed.
        targets = {f.get("to") for f in state.get("feedback", [])}
        return "research_node" if "research_agent" in targets else "funding_node"

    # ---------- graph ----------

    graph = StateGraph(WorkflowState)

    graph.add_node("research_node", traced("research_node", research, tracer))
    graph.add_node("funding_node", traced("funding_node", funding, tracer))
    graph.add_node("pitch_node", traced("pitch_node", coach, tracer))
    graph.add_node("request_research_node", request_research_node)
    graph.add_node("dispatch_feedback_node", dispatch_feedback_node)

    graph.add_edge(START, "research_node")
    graph.add_edge("research_node", "funding_node")

    graph.add_conditional_edges(
        "funding_node", route_after_funding,
        {"request_research_node": "request_research_node", "pitch_node": "pitch_node"},
    )
    graph.add_edge("request_research_node", "research_node")

    graph.add_conditional_edges(
        "pitch_node", route_after_pitch,
        {"dispatch_feedback_node": "dispatch_feedback_node", END: END},
    )
    graph.add_conditional_edges(
        "dispatch_feedback_node", route_feedback_target,
        {"research_node": "research_node", "funding_node": "funding_node"},
    )

    return graph.compile()
