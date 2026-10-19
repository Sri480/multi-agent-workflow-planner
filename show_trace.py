import sys
import glob
import os
import json

# Use the log file given on the command line, or default to the newest one
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    path = max(glob.glob("logs/run_*.jsonl"), key=os.path.getmtime)

print("Trace:", path, "\n")

for r in map(json.loads, open(path)):
    detail = " ".join(
        str(x)
        for x in (r.get("stage") or r.get("node") or r.get("direction"), r.get("to"))
        if x
    )
    print(f"{r['agent']:<16} {r['event']:<10} {detail}")
