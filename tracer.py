import json
import time
import uuid
from pathlib import Path


class Tracer:
    """Writes one JSON object per line. Optionally forwards each event to a callback (live UI)."""

    def __init__(self, log_dir: str = "logs", on_event=None):
        self.run_id = uuid.uuid4().hex[:8]
        Path(log_dir).mkdir(exist_ok=True)
        self.path = Path(log_dir) / f"run_{self.run_id}.jsonl"
        self.on_event = on_event

    def log(self, agent: str, event: str, **data):
        record = {
            "ts": round(time.time(), 3),
            "run_id": self.run_id,
            "agent": agent,
            "event": event,   # input | reasoning | handoff | feedback | output | memory
            **data,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.on_event:
            try:
                self.on_event(record)
            except Exception:
                pass   # a UI problem must never crash the workflow
        return record

