import json
from agents.base import BaseAgent

SCHEMA = """
Return ONLY valid JSON with exactly this structure:
{
  "domain": "string",
  "summary": "2-3 sentence overview of the domain today",
  "market_overview": {
    "size_estimate": "approximate figure for the stated region, labelled as an estimate, with its geographic scope stated",
    "growth_outlook": "string"
  },
  "subsegments": [
      {
        "name": "e.g. payments, digital lending, wealthtech",
        "size_estimate": "approximate, labelled as an estimate",
        "growth_outlook": "string",
        "regulatory_notes": "key licences or rules in the stated region"
      }
  ],
  "trends": [
    {"name": "string", "description": "string", "why_it_matters": "string"}
  ],
  "target_customers": ["string"],
  "competitors": [
    {"name": "string", "positioning": "string"}
  ],
  "opportunities": ["string"],
  "risks": ["string"],
  "regulatory_notes": ["string"]
}
Provide 4-5 trends, 3-4 subsegments, 3-4 competitors (prioritise players active in the stated region), and 3 items for each remaining list.
"""

def research_notes(r: dict) -> list:
    """Turn a research result into small, retrievable notes."""
    notes = []
    for t in r.get("trends", []):
        notes.append({"kind": "trend", "text":
            f"Trend: {t.get('name')}. {t.get('description')} Why it matters: {t.get('why_it_matters')}"})
    for s in r.get("subsegments", []):
        notes.append({"kind": "subsegment", "text":
            f"Subsegment: {s.get('name')}. Size: {s.get('size_estimate')}; "
            f"growth: {s.get('growth_outlook')}; regulation: {s.get('regulatory_notes')}"})
    for c in r.get("competitors", []):
        notes.append({"kind": "competitor", "text": f"Competitor: {c.get('name')}: {c.get('positioning')}"})
    for risk in r.get("risks", []):
        notes.append({"kind": "risk", "text": f"Risk: {risk}"})
    return notes

class ResearchAgent(BaseAgent):
    name = "research_agent"
    role = (
        "You are a market research analyst embedded in a startup accelerator. "
        "You analyse a startup domain and report market trends, target customers, "
        "competitors, opportunities, risks, and regulatory context. "
        "You do NOT recommend funding sources and you do NOT structure pitch decks; "
        "other specialists handle those. "
        "You have no live web access, so label all figures as estimates and never "
        "invent precise statistics. You reply ONLY with valid JSON."
    )

    def run(self, state: dict) -> dict:
        domain = state["domain"]
        region = state.get("region", "Global (not specified)")
        self.tracer.log(self.name, "input", domain=domain, region=region)

        # Feedback from other agents, if any (used by the Layer 7 loop)
        feedback = [f for f in state.get("feedback", []) if f.get("to") == self.name]
        previous = state.get("research")

        if feedback and previous:
            self.tracer.log(self.name, "feedback", received=feedback)
            prompt = (
                f"Domain: {domain}\nRegion: {region}\n\n"
                f"Your previous research:\n{json.dumps(previous, indent=2)}\n\n"
                f"Feedback from teammates:\n{json.dumps(feedback, indent=2)}\n\n"
                "Revise your research to address the feedback. Keep what was good, "
                "fix the gaps.\n" + SCHEMA
            )
        else:
            prompt = f"Research the startup domain: {domain}\nRegion: {region}\n" + SCHEMA

        memory_block = self._recall_block(f"{domain} {region} market trends, competitors, risks")
        result = self._ask(memory_block + prompt)
        self._remember(domain, region, research_notes(result))

        self.tracer.log(self.name, "output", trends=len(result.get("trends", [])))
        self.tracer.log(
            self.name,
            "handoff",
            to="funding_advisor",
            summary=f"{len(result.get('trends', []))} trends, "
                    f"{len(result.get('opportunities', []))} opportunities",
        )
        return {"research": result}

