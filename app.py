# .........................................................................
# Streamlit reruns the whole script on every click, including the download buttons. So results are kept in st.session_state, or they'd vanish when someone clicks Download.
# .........................................................................
import json
import time

import streamlit as st
import streamlit.components.v1 as components

from pipeline import run_pipeline, make_report, describe_event

st.set_page_config(page_title="Startup Accelerator Planner", layout="wide")
st.title("Multi-Agent Pitch Deck Planner")
st.caption("Research Agent → Funding Advisor → Pitch Coach, with feedback loops and vector memory")

with st.sidebar:
    st.header("Founder input")
    preset = st.selectbox("Startup domain",
                          ["fintech", "healthtech", "agritech", "edtech", "climate tech", "Other..."])
    domain = st.text_input("Your domain") if preset == "Other..." else preset
    region = st.selectbox("Region", ["US", "India", "EU", "UK", "Global"])
    review = st.checkbox("Guarantee one feedback round (demo mode)", value=False)
    use_memory = st.checkbox("Use vector memory", value=True)
    run = st.button("Generate pitch deck", type="primary")
    st.caption("A run takes roughly 3 to 5 minutes (up to 9 model calls).")

if run:
    if not domain or not domain.strip():
        st.warning("Please enter a startup domain.")
        st.stop()

    st.session_state.pop("result", None)
    start = time.time()
    with st.status("Agents are working...", expanded=True) as status:

        def on_event(record):
            line = describe_event(record)
            if line:
                status.write(line.strip())

        try:
            res = run_pipeline(domain.strip(), region, review, use_memory, on_event=on_event)
            report_path = make_report(res["tracer"])
        except Exception as e:
            status.update(label="Run failed", state="error")
            st.exception(e)
            st.stop()
        status.update(label=f"Done in {time.time() - start:.0f}s", state="complete", expanded=False)

    st.session_state["result"] = {
        "state": res["state"],
        "domain": domain.strip(),
        "region": region,
        "elapsed": time.time() - start,
        "report_html": report_path.read_text(encoding="utf-8"),
        "report_name": report_path.name,
    }

result = st.session_state.get("result")

if not result:
    st.info("Choose a domain and region in the sidebar, then press **Generate pitch deck**.")
else:
    state = result["state"]
    outline = state["pitch_outline"]
    funding = state["funding"]
    research = state["research"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Slides", len(outline["slides"]))
    c2.metric("Feedback rounds", state.get("revision_count", 0))
    c3.metric("Run time", f"{result['elapsed']:.0f}s")

    tab_deck, tab_fund, tab_research, tab_trace = st.tabs(
        ["Pitch deck", "Funding options", "Research", "Agent trace"]
    )

    with tab_deck:
        st.subheader(outline["deck_title"])
        st.write(outline["narrative_arc"])
        for s in outline["slides"]:
            with st.expander(f"{s['number']}. {s['title']}   ·   source: {s['data_source']}"):
                st.markdown(f"**Purpose:** {s.get('purpose', '')}")
                for p in s["key_points"]:
                    st.markdown(f"- {p}")
                st.markdown(f"**Evidence used:** {s.get('evidence_used', '')}")
                st.markdown(f"**Visual:** {s.get('visual_suggestion', '')}")
                st.markdown(f"**Speaker note:** {s.get('speaker_note', '')}")
        st.subheader("Founder to-do")
        for t in outline.get("founder_todo", []):
            st.markdown(f"- {t}")
        st.download_button("Download outline (JSON)", json.dumps(outline, indent=2),
                           file_name=f"{result['domain']}_{result['region']}_outline.json".replace(" ", "_"),
                           mime="application/json")

    with tab_fund:
        st.write(funding.get("strategy_summary", ""))
        for r in funding.get("recommendations", []):
            with st.expander(f"{r.get('name')}  ·  {r.get('type')}"):
                st.markdown(f"**Provider:** {r.get('provider')}")
                st.markdown(f"**Why it fits:** {r.get('fit_reason')}")
                st.markdown(f"**Typical range (estimate):** {r.get('typical_range')}")
                st.markdown(f"**Eligibility:** {r.get('eligibility_notes')}")
                st.warning(f"Verify before applying: {r.get('verify')}")

    with tab_research:
        st.write(research.get("summary", ""))
        st.markdown("**Trends**")
        for t in research.get("trends", []):
            st.markdown(f"- **{t.get('name')}**: {t.get('description')}")
        st.markdown("**Competitors**")
        for c in research.get("competitors", []):
            st.markdown(f"- **{c.get('name')}**: {c.get('positioning')}")
        st.caption("Research comes from the model's training knowledge (no live web access): "
                   "treat figures as estimates.")

    with tab_trace:
        components.html(result["report_html"], height=900, scrolling=True)
        st.download_button("Download trace report (HTML)", result["report_html"],
                           file_name=result["report_name"], mime="text/html")

