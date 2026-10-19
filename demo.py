# ..............................................................
# Model-written text can contain square brackets (like [Founder to fill: ...]), which rich would read as formatting. Everything from the model goes through escape() so it prints literally.
# ..............................................................

import argparse
import time

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from pipeline import run_pipeline, make_report, describe_event

console = Console()


def on_event(record):
    line = describe_event(record)
    if line:
        console.print(escape(line))


def show_result(state):
    outline = state["pitch_outline"]
    console.print()
    console.print(Panel(escape(outline["narrative_arc"]),
                        title=escape(outline["deck_title"]), border_style="cyan"))

    table = Table(title="Pitch deck outline", show_lines=True)
    table.add_column("#", justify="right", width=3)
    table.add_column("Slide", style="bold")
    table.add_column("Source", width=9)
    table.add_column("Key points")
    for s in outline["slides"]:
        points = "\n".join(f"• {escape(str(p))}" for p in s["key_points"])
        table.add_row(str(s["number"]), escape(s["title"]), s["data_source"], points)
    console.print(table)

    console.print("\n[bold]Founder to-do[/bold] (only you can supply these)")
    for t in outline.get("founder_todo", []):
        console.print(f"  • {escape(str(t))}")

    console.print("\n[bold]Verify before applying[/bold] (from the Funding Advisor; programs change)")
    for r in state["funding"].get("recommendations", []):
        console.print(f"  • [cyan]{escape(str(r.get('name')))}[/cyan]: {escape(str(r.get('verify', '')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-agent pitch deck planner (terminal demo)")
    parser.add_argument("domain", nargs="?")
    parser.add_argument("region", nargs="?")
    parser.add_argument("--review", action="store_true",
                        help="demo mode: guarantee one feedback round between agents")
    parser.add_argument("--no-memory", action="store_true", help="disable vector memory")
    args = parser.parse_args()

    if args.domain:
        domain, region = args.domain, args.region or "US"
    else:
        console.rule("[bold]Startup Accelerator: Multi-Agent Planner")
        domain = Prompt.ask("Startup domain", default="fintech")
        region = Prompt.ask("Region", default="US")

    console.rule(f"[bold]{escape(domain)} · {escape(region)}")
    start = time.time()
    with console.status("[bold cyan]Agents are working (a few minutes)..."):
        result = run_pipeline(domain, region, args.review, not args.no_memory, on_event=on_event)

    show_result(result["state"])
    report_path = make_report(result["tracer"])

    console.print()
    console.rule("[bold]Done")
    console.print(f"Feedback rounds used : {result['state'].get('revision_count', 0)}")
    console.print(f"Total time           : {time.time() - start:.0f}s")
    console.print(f"Outline (JSON)       : {result['outline_path']}")
    console.print(f"Trace report (HTML)  : {report_path}")
    console.print(f"Raw log (JSONL)      : {result['tracer'].path}")

