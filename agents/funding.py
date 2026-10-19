import json
from agents.base import BaseAgent

SCHEMA = """
Return ONLY valid JSON with exactly this structure:
{
  "domain": "string",
  "region": "string",
  "stage_assumption": "the startup stage you assumed (e.g. pre-seed)",
  "strategy_summary": "2-3 sentences on the recommended funding path",
  "recommendations": [
    {
      "name": "string",
      "type": "grant | accelerator | competition | government_program | venture_capital | angel | debt",
      "provider": "string",
      "fit_reason": "why it fits, referencing a specific trend or opportunity from the research",
      "typical_range": "approximate range, labelled as an estimate",
      "eligibility_notes": "string",
      "verify": "what the founder must confirm (deadlines, current eligibility)"
    }
  ],
  "risks_and_gaps": ["string"],
  "requests_to_research": ["questions you need the Research Agent to answer, or empty list"]
}
Provide 5-6 recommendations covering at least three different types.
"""

def funding_notes(f: dict) -> list:
    """Turn funding recommendations into small, retrievable notes."""
    return [
        {"kind": "funding_option", "text":
            f"{r.get('name')} ({r.get('type')}), provider {r.get('provider')}. "
            f"Fit: {r.get('fit_reason')} Range: {r.get('typical_range')}. Verify: {r.get('verify')}"}
        for r in f.get("recommendations", [])
    ]

class FundingAdvisor(BaseAgent):
    name = "funding_advisor"
    role = (
        "You are a funding advisor embedded in a startup accelerator. "
        "Given market research about a domain, you recommend grants, accelerators, "
        "competitions, and investor types that suit an early-stage founder. "
        "You do NOT perform market research and you do NOT structure pitch decks; "
        "other specialists handle those. "
        "You have no live web access: programs, amounts, and deadlines may have "
        "changed, so give approximate ranges, never invent exact deadlines or "
        "figures, and tell the founder what to verify. "
        "Tie every recommendation to the research you were given. "
        "You reply ONLY with valid JSON."
    )

    def run(self, state: dict) -> dict:
        domain = state["domain"]
        region = state.get("region", "Global (not specified)")
        research = state.get("research")
        if not research:
            raise ValueError("FundingAdvisor needs state['research']. Run ResearchAgent first.")

        # Log the context handoff that this agent received
        self.tracer.log(
            self.name,
            "input",
            domain=domain,
            region=region,
            received_from="research_agent",
            research_keys=list(research.keys()),
        )

        feedback = [f for f in state.get("feedback", []) if f.get("to") == self.name]
        previous = state.get("funding")

        context = (
            f"Domain: {domain}\n"
            f"Region: {region}\n\n"
            f"Research from the Research Agent:\n{json.dumps(research, indent=2)}\n\n"
        )

        if feedback and previous:
            self.tracer.log(self.name, "feedback", received=feedback)
            prompt = (
                context
                + f"Your previous recommendations:\n{json.dumps(previous, indent=2)}\n\n"
                + f"Feedback from teammates:\n{json.dumps(feedback, indent=2)}\n\n"
                + "Revise your recommendations to address the feedback. "
                + "Keep what was good, fix the gaps.\n"
                + SCHEMA
            )
        else:
            prompt = context + "Recommend funding options for a founder in this domain.\n" + SCHEMA

        memory_block = self._recall_block(f"{domain} {region} funding grants accelerators investors")
        result = self._ask(memory_block + prompt)

        self._remember(domain, region, funding_notes(result))

        recs = result.get("recommendations", [])
        self.tracer.log(
            self.name,
            "output",
            recommendations=len(recs),
            types=sorted({r.get("type", "?") for r in recs}),
            requests_to_research=len(result.get("requests_to_research", [])),
        )
        self.tracer.log(
            self.name,
            "handoff",
            to="pitch_coach",
            summary=f"{len(recs)} funding options",
        )
        return {"funding": result}
