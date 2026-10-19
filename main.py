import json
import argparse
from pathlib import Path
from memory.store import MemoryStore

from llm import LLM
from tracer import Tracer
from graph import build_graph


def print_outline(outline: dict):
    print(f"\n{outline['deck_title']}")
    print(outline["narrative_arc"], "\n")
    for s in outline["slides"]:
        print(f"{s['number']:>2}. {s['title']}  [{s['data_source']}]")
        for p in s["key_points"]:
            print(f"      - {p}")
    print("\nFounder to-do:")
    for t in outline.get("founder_todo", []):
        print(f"  * {t}")
    print("\nReady:", outline["ready"])
    print("Feedback for peers:", outline["feedback_for_peers"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent pitch deck planner")
    parser.add_argument("domain", nargs="?", default="fintech")
    parser.add_argument("region", nargs="?", default="US")
    parser.add_argument("--review", action="store_true",
                        help="demo mode: guarantee one feedback round between agents")
    parser.add_argument("--no-memory", action="store_true",
                        help="disable vector memory (for before/after comparison)")
    args = parser.parse_args()

    tracer = Tracer()
    memory = None if args.no_memory else MemoryStore(tracer)
    app = build_graph(LLM(), tracer, memory)

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/workflow.mmd").write_text(app.get_graph().draw_mermaid())

    final_state = app.invoke(
        {
            "domain": args.domain,
            "region": args.region,
            "feedback": [],
            "revision_count": 0,
            "force_review": args.review,
        },
        config={"recursion_limit": 40},
    )

    print_outline(final_state["pitch_outline"])

    print("\nVerify before applying (from the Funding Advisor):")
    for r in final_state["funding"].get("recommendations", []):
        print(f"  * {r.get('name')}: {r.get('verify', '')}")

    print(f"\nFeedback rounds used: {final_state.get('revision_count', 0)}")

    name = f"{args.domain}_{args.region}_outline.json".replace(" ", "_")
    out_path = Path("outputs") / name
    out_path.write_text(json.dumps(final_state["pitch_outline"], indent=2))
    print("Saved:", out_path)
    print("Mermaid diagram:", "outputs/workflow.mmd")
    print("Log:", tracer.path)
