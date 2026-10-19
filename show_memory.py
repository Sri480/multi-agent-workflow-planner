import sys
from collections import Counter
import chromadb

col = chromadb.PersistentClient(path="memory_store").get_or_create_collection(
    "agent_insights", metadata={"hnsw:space": "cosine"}
)
print("Total notes:", col.count())
if col.count() == 0:
    sys.exit()

metas = col.get(include=["metadatas"])["metadatas"]
counts = Counter((m["agent"], m["domain"], m["region"], m["kind"]) for m in metas)
for (agent, domain, region, kind), n in sorted(counts.items()):
    print(f"{agent:<16} {domain:<12} {region:<8} {kind:<14} {n}")

if len(sys.argv) > 1:
    q = " ".join(sys.argv[1:])
    res = col.query(query_texts=[q], n_results=min(5, col.count()))
    print(f"\nNearest to: {q!r}")
    for d, dist, m in zip(res["documents"][0], res["distances"][0], res["metadatas"][0]):
        print(f"  {dist:.3f} [{m['agent']} | {m['domain']}/{m['region']}] {d[:90]}")
