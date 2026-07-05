"""Versioned prompt + config — what the Release step ships back into the agent.

The concierge's "system prompt" is skill.md (procedural memory) plus the model
config. A release records a *blessed* snapshot: the skill.md hash (prompt
version), provider, model, and the tunable knobs. This is the "Improved System
Prompt + Config" arrow feeding back into the agent run.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.concierge.concierge import MAX_QUESTIONS, _SKILL_PATH

RELEASES_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "releases.json"


def prompt_version() -> str:
    """Short hash of skill.md — changes whenever the interview policy changes."""
    text = _SKILL_PATH.read_text(encoding="utf-8") if _SKILL_PATH.exists() else ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def current_config(provider: str = "", model_id: str = "") -> dict:
    return {
        "prompt_version": prompt_version(),   # skill.md hash
        "provider": provider,
        "model_id": model_id,
        "max_questions": MAX_QUESTIONS,
    }


def load_releases(path: Path = RELEASES_PATH) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def record_release(config: dict, pass_rate: float, note: str = "",
                   path: Path = RELEASES_PATH) -> dict:
    releases = load_releases(path)
    entry = {
        "version": len(releases) + 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pass_rate": round(pass_rate, 3),
        "note": note,
        "config": config,
    }
    releases.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(releases, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def blessed_config(path: Path = RELEASES_PATH) -> dict | None:
    """The latest released config — what a production agent should run."""
    releases = load_releases(path)
    return releases[-1]["config"] if releases else None
