"""Episodic memory: concierge sessions in SQLite.

Replaces the append-only seller_insights.jsonl with a queryable store. Commonly
filtered diagnosis fields (case_class, seller_action, substitution, the two
mismatch flags) are promoted to columns so the dashboard and seller logic get
real SQL recency/filtering; the full diagnosis + transcript are kept as JSON.

load_sessions() returns rows in the same dict shape the JSONL used, so readers
(app.py, seller_escalation) work unchanged. This is Stage 1 of the episodic
memory plan — SQL recency now; a vector store for relevance is a later stage.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "insights.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_asin TEXT,
    title TEXT,
    timestamp TEXT,
    model_id TEXT,
    cost_usd REAL,
    case_class TEXT,
    root_cause_category TEXT,
    seller_action TEXT,
    suspected_substitution TEXT,
    material_issue_suspected INTEGER,
    weather_suitability_mismatch INTEGER,
    diagnosis_json TEXT,
    transcript_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_asin ON sessions(parent_asin);
CREATE INDEX IF NOT EXISTS idx_sessions_ts ON sessions(timestamp);
CREATE INDEX IF NOT EXISTS idx_sessions_case ON sessions(case_class);
"""


def _to_int(v):
    return None if v is None else int(bool(v))


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_session(row: dict, db_path: Path = DB_PATH) -> None:
    """Persist one concierge session. `row` is the dict the scripts already
    build: parent_asin, title, timestamp, diagnosis, transcript, cost_usd, model_id."""
    dx = row.get("diagnosis") or {}
    with _connect(db_path) as conn:
        conn.execute(
            """INSERT INTO sessions
               (parent_asin, title, timestamp, model_id, cost_usd, case_class,
                root_cause_category, seller_action, suspected_substitution,
                material_issue_suspected, weather_suitability_mismatch,
                diagnosis_json, transcript_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (row.get("parent_asin"), row.get("title"),
             row.get("timestamp") or datetime.now(timezone.utc).isoformat(),
             row.get("model_id"), row.get("cost_usd"),
             dx.get("case_class"), dx.get("root_cause_category"),
             dx.get("seller_action"), dx.get("suspected_substitution"),
             _to_int(dx.get("material_issue_suspected")),
             _to_int(dx.get("weather_suitability_mismatch")),
             json.dumps(dx, ensure_ascii=False),
             json.dumps(row.get("transcript") or [], ensure_ascii=False)))


def _row_to_dict(r: sqlite3.Row) -> dict:
    return {
        "id": r["id"],
        "parent_asin": r["parent_asin"],
        "title": r["title"],
        "timestamp": r["timestamp"],
        "model_id": r["model_id"],
        "cost_usd": r["cost_usd"],
        "diagnosis": json.loads(r["diagnosis_json"]) if r["diagnosis_json"] else {},
        "transcript": json.loads(r["transcript_json"]) if r["transcript_json"] else [],
    }


def load_sessions(db_path: Path = DB_PATH) -> list[dict]:
    """All sessions oldest-first (same shape the JSONL used)."""
    if not db_path.exists():
        return []
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY timestamp").fetchall()
    return [_row_to_dict(r) for r in rows]


def recent_sessions(limit: int = 20, parent_asin: str | None = None,
                    case_class: str | None = None, db_path: Path = DB_PATH) -> list[dict]:
    """SQL recency query — newest first, optionally filtered by ASIN / case class."""
    if not db_path.exists():
        return []
    clauses, params = [], []
    if parent_asin:
        clauses.append("parent_asin = ?")
        params.append(parent_asin)
    if case_class:
        clauses.append("case_class = ?")
        params.append(case_class)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    with _connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM sessions {where} ORDER BY timestamp DESC LIMIT ?",
            params).fetchall()
    return [_row_to_dict(r) for r in rows]


def count(db_path: Path = DB_PATH) -> int:
    if not db_path.exists():
        return 0
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]


def migrate_jsonl(jsonl_path: Path, db_path: Path = DB_PATH) -> int:
    """One-time import of legacy seller_insights.jsonl rows. Skips if the DB
    already has rows. Returns the number imported."""
    if count(db_path) > 0 or not Path(jsonl_path).exists():
        return 0
    imported = 0
    with Path(jsonl_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                save_session(json.loads(line), db_path)
                imported += 1
    return imported
