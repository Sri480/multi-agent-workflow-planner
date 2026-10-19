"""report.py: turn a JSONL trace into a terminal narrative and a self-contained HTML report."""
import sys
import glob
import os
import json
import html
from pathlib import Path
from collections import defaultdict

AGENTS = {
    "research_agent": ("Research Agent", "#2563eb", "R"),
    "funding_advisor": ("Funding Advisor", "#059669", "F"),
    "pitch_coach": ("Pitch Coach", "#d97706", "C"),
}
NODE_AGENT = {
    "research_node": "research_agent",
    "funding_node": "funding_advisor",
    "pitch_node": "pitch_coach",
}


def esc(x):
    return html.escape(str(x))


def cut(x, n=170):
    x = str(x)
    return x if len(x) <= n else x[: n - 1] + "…"


# ---------------------------------------------------------------- parsing

def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def build(records):
    """Split the log into ordered items: agent steps and orchestrator decisions."""
    items, current = [], None
    for r in records:
        a, e, st = r["agent"], r["event"], r.get("stage")
        if a == "orchestrator" and e == "handoff" and st == "node_start":
            current = {
                "type": "step", "node": r["node"],
                "agent": NODE_AGENT.get(r["node"], r["node"]),
                "revision": r.get("revision", 0),
                "start": r["ts"], "end": r["ts"], "events": [],
            }
            items.append(current)
        elif a == "orchestrator" and e == "handoff" and st == "node_end":
            if current:
                current["end"] = r["ts"]
            current = None
        elif a == "orchestrator":
            items.append({"type": "decision", "rec": r})
        elif current is not None:
            current["events"].append(r)
    for it in items:
        if it["type"] == "step":
            it["s"] = summarise(it)
    return items


def summarise(step):
    s = {"calls": [], "recall": None, "store": None, "feedback_in": [],
         "feedback_out": [], "output": {}, "handoff": None, "input": {},
         "prompt": "", "response": "", "errors": []}
    skip = ("ts", "run_id", "agent", "event")
    for r in step["events"]:
        e, st = r["event"], r.get("stage")
        if e == "input":
            s["input"] = r
        elif e == "output":
            s["output"] = {k: v for k, v in r.items() if k not in skip}
        elif e == "handoff":
            s["handoff"] = r
        elif e == "reasoning" and st == "prompt":
            s["prompt"] = r.get("prompt", "")
        elif e == "reasoning" and st == "response":
            s["response"] = r.get("response", "")
            s["calls"].append(r.get("elapsed_s"))
        elif e == "memory":
            if st == "recall":
                s["recall"] = r
            elif st == "store":
                s["store"] = r
            elif st and st.endswith("_error"):
                s["errors"].append(r)
        elif e == "feedback":
            if r.get("direction") == "sent":
                s["feedback_out"].append(r)
            else:
                s["feedback_in"].extend(r.get("received", []))
    return s


# ---------------------------------------------------------------- narrative

def decision_text(rec):
    st = rec.get("stage")
    if st == "routed":
        if rec.get("targets"):
            return "Coach feedback delivered to: " + ", ".join(sorted(set(rec["targets"])))
        return f"{rec.get('source')} requests routed to {rec.get('to')}"
    if st == "funding_requests_skipped":
        return "Loop guard: a second funding → research round was refused"
    if st == "loop_continue":
        return (f"Coach has {rec.get('items')} outstanding feedback item(s): "
                f"starting revision round {rec.get('revision', 0) + 1}")
    if st == "loop_guard_stop":
        return "Loop guard: maximum revisions reached, finalising"
    if st == "loop_end":
        return f"No outstanding feedback: finishing (ready={rec.get('ready')})"
    return st or "orchestrator decision"


def step_lines(it):
    s, a, inp = it["s"], it["agent"], it["s"]["input"]
    lines = []
    if a == "research_agent":
        lines.append(f"Researched '{inp.get('domain')}' for region '{inp.get('region', 'n/a')}'.")
    elif a == "funding_advisor":
        lines.append(f"Read the research ({len(inp.get('research_keys', []))} sections) "
                     f"for region '{inp.get('region', 'n/a')}'.")
    else:
        lines.append(f"Read the research ({len(inp.get('research_keys', []))} sections) "
                     f"and {inp.get('funding_options', '?')} funding options.")
    for f in s["feedback_in"]:
        lines.append(f"Received feedback from {f.get('from', '?')}: "
                     f"{cut(f.get('issue', ''))} → asked: {cut(f.get('request', ''))}")
    rec = s["recall"]
    if rec:
        lines.append(f"Memory recall: {rec.get('used')} of {rec.get('retrieved')} past notes were used.")
    if s["output"]:
        lines.append("Produced: " + ", ".join(f"{k}={v}" for k, v in s["output"].items()))
    for f in s["feedback_out"]:
        lines.append(f"Sent feedback to {f.get('to')}: {cut(f.get('issue', ''))}")
    if s["store"]:
        lines.append(f"Stored {s['store'].get('count')} notes in memory.")
    for err in s["errors"]:
        lines.append(f"Memory problem ({err.get('stage')}): {cut(err.get('error', ''))}")
    h = s["handoff"]
    if h:
        lines.append(f"Handed off to {h.get('to')}: {h.get('summary', '')}")
    return lines


def narrate(items):
    n = 0
    for it in items:
        if it["type"] == "decision":
            print(f"      ↳ {decision_text(it['rec'])}")
            continue
        n += 1
        label = AGENTS.get(it["agent"], (it["agent"],))[0]
        rev = f", revision round {it['revision']}" if it["revision"] else ""
        print(f"\n{n}. {label}{rev}  ({it['end'] - it['start']:.0f}s)")
        for line in step_lines(it):
            print("   - " + line)


# ---------------------------------------------------------------- stats

def compute_stats(records, items):
    steps = [i for i in items if i["type"] == "step"]
    per = defaultdict(lambda: defaultdict(float))
    for it in steps:
        p, s = per[it["agent"]], it["s"]
        p["runs"] += 1
        p["calls"] += len(s["calls"])
        p["model_s"] += sum(c for c in s["calls"] if c is not None)
        p["fb_recv"] += len([f for f in s["feedback_in"] if f.get("from") != "orchestrator"])
        p["fb_sent"] += len(s["feedback_out"])
        if s["recall"]:
            p["retrieved"] += s["recall"].get("retrieved", 0)
            p["used"] += s["recall"].get("used", 0)
    total = {
        "wall": records[-1]["ts"] - records[0]["ts"],
        "steps": len(steps),
        "calls": sum(len(i["s"]["calls"]) for i in steps),
        "model_s": sum(c for i in steps for c in i["s"]["calls"] if c is not None),
        "rounds": max([i["revision"] for i in steps] or [0]),
    }
    return per, total


# ---------------------------------------------------------------- HTML

CSS = """
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--card:#f6f7f9;--line:#dcdfe4}
@media (prefers-color-scheme: dark){:root{--bg:#16181c;--fg:#e8e8e8;--muted:#9aa0a6;--card:#1f2228;--line:#34383f}}
body{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.5rem;margin:0 0 .2rem}
h2{font-size:1.15rem;margin:2rem 0 .6rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}
.muted{color:var(--muted)}
.chips{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center;margin:.6rem 0}
.chip{padding:.15rem .65rem;border-radius:999px;color:#fff;font-weight:600;font-size:.85rem}
.arrow{color:var(--muted)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.6rem}
.stat{background:var(--card);border-radius:8px;padding:.6rem .8rem}
.stat b{display:block;font-size:1.3rem}
.card{background:var(--card);border-left:5px solid var(--c);border-radius:6px;padding:.7rem 1rem;margin:.6rem 0}
.card h3{margin:0 0 .3rem;font-size:1rem}
.card ul{margin:.3rem 0;padding-left:1.2rem}
.decision{margin:.4rem 0 .4rem 1rem;padding:.25rem .7rem;border:1px dashed var(--line);border-radius:6px;color:var(--muted);font-size:.9rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.4rem 0}
th,td{border-bottom:1px solid var(--line);padding:.3rem .5rem;text-align:left;vertical-align:top}
.used{color:#059669;font-weight:600}
.skip{color:var(--muted)}
details{margin:.3rem 0}
summary{cursor:pointer;color:var(--muted)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:.5rem;font-size:.8rem;max-height:320px;overflow:auto}
"""


def hits_table(rec):
    hits, used = rec.get("hits", []), rec.get("used", 0)
    if not hits:
        return '<p class="muted">No past notes were available (cold start).</p>'
    rows = ""
    for j, h in enumerate(hits):
        flag = '<span class="used">used</span>' if j < used else '<span class="skip">too far</span>'
        rows += (f"<tr><td>{esc(h.get('distance'))}</td><td>{flag}</td><td>{esc(h.get('kind'))}</td>"
                 f"<td>{esc(h.get('domain'))} / {esc(h.get('region'))}</td>"
                 f"<td>{esc(h.get('preview'))}</td></tr>")
    return ("<table><tr><th>Distance</th><th></th><th>Kind</th><th>From</th><th>Note</th></tr>"
            + rows + "</table>")


def step_html(n, it):
    s = it["s"]
    label, color, _ = AGENTS.get(it["agent"], (it["agent"], "#888", "?"))
    rev = f" · revision round {it['revision']}" if it["revision"] else ""
    calls = [c for c in s["calls"] if c is not None]
    call_txt = f" · model call {', '.join(f'{c:.0f}s' for c in calls)}" if calls else ""
    lis = "".join(f"<li>{esc(l)}</li>" for l in step_lines(it))
    prompt = (f"<details><summary>Prompt sent to the model</summary><pre>{esc(cut(s['prompt'], 4000))}</pre></details>"
              f"<details><summary>Model response</summary><pre>{esc(cut(s['response'], 4000))}</pre></details>")
    return (f'<div class="card" style="--c:{color}"><h3>{n}. {esc(label)}{esc(rev)} '
            f'<span class="muted">({it["end"] - it["start"]:.0f}s{esc(call_txt)})</span></h3>'
            f"<ul>{lis}</ul>{prompt}</div>")


def render_html(path, records, items, per, total):
    steps = [i for i in items if i["type"] == "step"]
    first = next((r for r in records if r["agent"] == "research_agent" and r["event"] == "input"), {})
    domain = first.get("domain", "?")
    region = first.get("region", "n/a")

    chips = ""
    for k, it in enumerate(steps):
        _, color, short = AGENTS.get(it["agent"], ("?", "#888", "?"))
        sub = f"<sub>{it['revision']}</sub>" if it["revision"] else ""
        chips += f'<span class="chip" style="background:{color}">{short}{sub}</span>'
        if k < len(steps) - 1:
            chips += '<span class="arrow">→</span>'

    stats = "".join(
        f'<div class="stat"><b>{v}</b>{l}</div>' for l, v in [
            ("wall time", f"{total['wall']:.0f}s"),
            ("agent runs", total["steps"]),
            ("model calls", total["calls"]),
            ("model time", f"{total['model_s']:.0f}s"),
            ("feedback rounds", total["rounds"]),
        ])

    body, n = "", 0
    for it in items:
        if it["type"] == "decision":
            body += f'<div class="decision">↳ {esc(decision_text(it["rec"]))}</div>'
        else:
            n += 1
            body += step_html(n, it)

    ledger = ""
    for k, it in enumerate([i for i in items if i["type"] == "step"], 1):
        for f in it["s"]["feedback_in"]:
            if f.get("from") == "orchestrator":
                continue
            ledger += (f"<tr><td>{esc(f.get('from'))}</td><td>{esc(f.get('to'))}</td>"
                       f"<td>{esc(f.get('issue'))}</td><td>{esc(f.get('request'))}</td></tr>")
    ledger = ("<table><tr><th>From</th><th>To</th><th>Issue</th><th>Request</th></tr>" + ledger + "</table>"
              if ledger else '<p class="muted">No peer feedback was exchanged in this run.</p>')

    mem = ""
    for it in steps:
        rec = it["s"]["recall"]
        if rec:
            mem += (f"<p><b>{esc(AGENTS.get(it['agent'], (it['agent'],))[0])}</b> "
                    f"(run {it['revision']}) searched memory for "
                    f"<i>{esc(rec.get('query'))}</i>: {rec.get('used')} of {rec.get('retrieved')} notes used.</p>"
                    + hits_table(rec))
    mem = mem or '<p class="muted">Memory was not used in this run.</p>'

    contrib = ""
    for agent, p in per.items():
        contrib += (f"<tr><td>{esc(AGENTS.get(agent, (agent,))[0])}</td><td>{int(p['runs'])}</td>"
                    f"<td>{int(p['calls'])}</td><td>{p['model_s']:.0f}s</td><td>{int(p['fb_sent'])}</td>"
                    f"<td>{int(p['fb_recv'])}</td><td>{int(p['used'])} / {int(p['retrieved'])}</td></tr>")

    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Run {esc(records[0]['run_id'])}</title>
<style>{CSS}</style></head><body>
<h1>Multi-agent run report</h1>
<div class="muted">Run {esc(records[0]['run_id'])} · domain: {esc(domain)} · region: {esc(region)} · log: {esc(path)}</div>
<div class="stats" style="margin-top:1rem">{stats}</div>
<h2>Flow</h2>
<div class="chips">{chips}</div>
<div class="muted">R = Research Agent, F = Funding Advisor, C = Pitch Coach. A subscript is the revision round.</div>
<h2>Step by step</h2>{body}
<h2>Feedback exchanged between agents</h2>{ledger}
<h2>Memory use</h2>{mem}
<h2>Contribution per agent</h2>
<table><tr><th>Agent</th><th>Runs</th><th>Model calls</th><th>Model time</th><th>Feedback sent</th>
<th>Feedback received</th><th>Memory used / retrieved</th></tr>{contrib}</table>
</body></html>"""


# ---------------------------------------------------------------- main

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("logs/run_*.jsonl"), key=os.path.getmtime)
    records = load(path)
    items = build(records)
    if not any(i["type"] == "step" for i in items):
        sys.exit("No orchestrator events found. Use a log created by the Layer 6 or later graph.")

    per, total = compute_stats(records, items)
    print(f"Trace: {path}")
    narrate(items)

    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs") / f"report_{records[0]['run_id']}.html"
    out.write_text(render_html(path, records, items, per, total), encoding="utf-8")
    print(f"\nHTML report: {out}  (open it in a browser)")
