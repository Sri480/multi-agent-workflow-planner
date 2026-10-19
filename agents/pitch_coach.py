import json
from agents.base import BaseAgent

SCHEMA = """
Return ONLY valid JSON with exactly this structure:
{
  "domain": "string",
  "deck_title": "string",
  "narrative_arc": "2-3 sentences: the story the deck tells",
  "slides": [
    {
      "number": 1,
      "title": "string",
      "purpose": "what this slide must achieve with investors",
      "key_points": ["string"],
      "data_source": "research | funding | founder | mixed",
      "evidence_used": "which specific trend, competitor, program, etc. this slide draws on, or 'none'",
      "visual_suggestion": "string",
      "speaker_note": "one or two sentences"
    }
  ],
  "founder_todo": ["facts only the founder can supply"],
  "feedback_for_peers": [
    {
      "to": "research_agent | funding_advisor",
      "issue": "what is missing or weak",
      "request": "the specific thing you need them to provide or fix"
    }
  ],
  "ready": true
}
Rules:
- Produce 10-12 slides covering: title, problem, solution, market opportunity, product,
  business model, go-to-market, competition, team, financial ask and use of funds, roadmap, closing.
- Slides on market, competition, go-to-market, and funding ask MUST name the specific
  trends, subsegments, competitors, and funding programs from the material (by name).
  The problem slide must draw on a research trend or risk.
- Set "data_source" to the dominant origin of the slide's content.
- Feedback to peers must be something the teammate can answer from general knowledge,
  WITHOUT live web access and WITHOUT knowing the founder's private choices.
  Never request live verification, contact details, or named individuals.
  Anything only the founder can supply belongs in founder_todo, not in feedback.
- Set "ready" to false ONLY if a gap that a peer can actually fix would materially hurt
  the deck. If the only gaps are founder-specific, set it to true and leave
  feedback_for_peers as an empty list.
- Use at most 2 feedback items in total.
"""


class PitchCoach(BaseAgent):
    name = "pitch_coach"
    role = (
        "You are a pitch coach embedded in a startup accelerator. "
        "You turn market research and funding recommendations into a compelling, "
        "investor-ready pitch deck outline. "
        "You do NOT perform new market research and you do NOT search for funding programs; "
        "you only use the material you are given. "
        "You do not know the founder's team, product details, or traction, so write "
        "'[Founder to fill: ...]' instead of inventing them. "
        "Never state a number that is not in the material you were given. "
        "You also review your inputs and tell teammates precisely what is missing. "
        "You reply ONLY with valid JSON."
    )

    def run(self, state: dict) -> dict:
        domain = state["domain"]
        research = state.get("research")
        funding = state.get("funding")
        if not research or not funding:
            raise ValueError("PitchCoach needs state['research'] and state['funding'].")

        # Log the context handoffs this agent received
        self.tracer.log(
            self.name,
            "input",
            domain=domain,
            received_from=["research_agent", "funding_advisor"],
            research_keys=list(research.keys()),
            funding_options=len(funding.get("recommendations", [])),
        )

        feedback = [f for f in state.get("feedback", []) if f.get("to") == self.name]
        previous = state.get("pitch_outline")

        context = (
            f"Domain: {domain}\n\n"
            f"Research from the Research Agent:\n{json.dumps(research, indent=2)}\n\n"
            f"Funding recommendations from the Funding Advisor:\n{json.dumps(funding, indent=2)}\n\n"
        )

        if feedback and previous:
            self.tracer.log(self.name, "feedback", direction="received", received=feedback)
            prompt = (
                context
                + f"Your previous outline:\n{json.dumps(previous, indent=2)}\n\n"
                + f"Feedback:\n{json.dumps(feedback, indent=2)}\n\n"
                + "Revise the outline using the updated research and funding above. "
                + "Keep what was good, fix the gaps.\n"
                + SCHEMA
            )
        else:
            prompt = context + "Create the pitch deck outline and review the inputs.\n" + SCHEMA

        # Demo mode: guarantee one visible review round on the first pass
        if state.get("force_review") and not previous:
            prompt += (
                "\nMANDATORY REVIEW ROUND (overrides the rule about 'ready'): "
                "set \"ready\" to false and include exactly ONE feedback item, addressed "
                "to the peer whose output could most improve the deck. It must still obey "
                "the feedback rules: answerable from general knowledge, no live verification, "
                "no founder-only information.\n"
            )
        result = self._ask(prompt)

        # Normalise feedback so every item is addressed to a valid peer
        valid_peers = {"research_agent", "funding_advisor"}
        result["feedback_for_peers"] = [
            {**f, "from": self.name}
            for f in result.get("feedback_for_peers", [])
            if f.get("to") in valid_peers
        ]

        # Keep 'ready' consistent with the feedback: outstanding feedback means not ready
        if result["feedback_for_peers"]:
            result["ready"] = False

        slides = result.get("slides", [])
        self.tracer.log(
            self.name,
            "output",
            slides=len(slides),
            sources=sorted({s.get("data_source", "?") for s in slides}),
            ready=result.get("ready"),
        )

        for fb in result["feedback_for_peers"]:
            self.tracer.log(
                self.name, "feedback", direction="sent",
                to=fb["to"], issue=fb.get("issue"), request=fb.get("request"),
            )

        self.tracer.log(
            self.name,
            "handoff",
            to="founder",
            summary=f"{len(slides)}-slide outline, ready={result.get('ready')}",
        )
        return {"pitch_outline": result}
