import time
import chromadb

MAX_DISTANCE = 0.75   # cosine distance: lower = more similar. Tune using the logged distances.


class MemoryStore:
    """Vector memory shared across runs. Each agent recalls its own past notes."""

    def __init__(self, tracer, path: str = "memory_store"):
        self.tracer = tracer
        self.run_id = tracer.run_id
        client = chromadb.PersistentClient(path=path)
        self._col = client.get_or_create_collection(
            "agent_insights", metadata={"hnsw:space": "cosine"}
        )

    def remember(self, agent: str, domain: str, region: str, notes: list) -> None:
        """Store notes for this agent, replacing this run's earlier version (revisions)."""
        notes = [n for n in notes if n.get("text")]
        if not notes:
            return
        try:
            self._col.delete(where={"$and": [{"run_id": self.run_id}, {"agent": agent}]})
            self._col.add(
                ids=[f"{self.run_id}:{agent}:{i}" for i in range(len(notes))],
                documents=[n["text"] for n in notes],
                metadatas=[
                    {
                        "agent": agent,
                        "kind": n.get("kind", "note"),
                        "domain": domain,
                        "region": region,
                        "run_id": self.run_id,
                        "ts": int(time.time()),
                    }
                    for n in notes
                ],
            )
            self.tracer.log(
                agent, "memory", stage="store", count=len(notes),
                kinds=sorted({n.get("kind", "note") for n in notes}),
                total_in_store=self._col.count(),
            )
        except Exception as e:
            self.tracer.log(agent, "memory", stage="store_error", error=str(e)[:200])

    def recall(self, agent: str, query: str, k: int = 4) -> list:
        """Return relevant notes from earlier runs (never the current run)."""
        try:
            total = self._col.count()
            hits = []
            if total:
                res = self._col.query(
                    query_texts=[query],
                    n_results=min(k, total),
                    where={"$and": [{"agent": agent}, {"run_id": {"$ne": self.run_id}}]},
                )
                hits = [
                    {"text": d, "distance": dist, **m}
                    for d, dist, m in zip(
                        res["documents"][0], res["distances"][0], res["metadatas"][0]
                    )
                ]
        except Exception as e:
            self.tracer.log(agent, "memory", stage="recall_error", error=str(e)[:200])
            return []

        kept = [h for h in hits if h["distance"] <= MAX_DISTANCE]
        self.tracer.log(
            agent, "memory", stage="recall", query=query,
            retrieved=len(hits), used=len(kept),
            hits=[
                {
                    "distance": round(h["distance"], 3),
                    "kind": h["kind"],
                    "domain": h["domain"],
                    "region": h["region"],
                    "from_run": h["run_id"],
                    "preview": h["text"][:80],
                }
                for h in hits
            ],
        )
        return kept
