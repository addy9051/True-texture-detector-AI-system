"""List models available to this Groq account (ids only, never the key)."""

import os
import sys

import requests

key = os.environ.get("GROQ_API_KEY")
if not key:
    sys.exit("GROQ_API_KEY not set in this shell.")
r = requests.get("https://api.groq.com/openai/v1/models",
                 headers={"Authorization": f"Bearer {key}"}, timeout=30)
r.raise_for_status()
ids = sorted(m["id"] for m in r.json()["data"])
print(f"{len(ids)} models available:")
for i in ids:
    print(" ", i)
