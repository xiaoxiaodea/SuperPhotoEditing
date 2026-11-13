from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            step TEXT NOT NULL,
            error TEXT,
            original_name TEXT,
            original_file TEXT,
            technical_json TEXT,
            critique_json TEXT,
            plan_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS variants (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            slot TEXT NOT NULL,
            look_id TEXT NOT NULL,
            title TEXT NOT NULL,
            why TEXT,
            recipe_json TEXT NOT NULL,
            filename TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            chosen_slot TEXT NOT NULL,
            look_id TEXT NOT NULL,
            technical_score INTEGER NOT NULL,
            style_score INTEGER NOT NULL,
            restraint_score INTEGER NOT NULL,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        );
        CREATE TABLE IF NOT EXISTS ratings (
            job_id TEXT NOT NULL REFERENCES jobs(id),
            slot TEXT NOT NULL CHECK(slot IN ('A','B','C')),
            verdict INTEGER NOT NULL CHECK(verdict IN (-1,0,1)),
            method_id TEXT NOT NULL,
            scene_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(job_id, slot)
        );
        CREATE TABLE IF NOT EXISTS refinements (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id),
            slot TEXT NOT NULL CHECK(slot IN ('A','B','C')),
            parent_id TEXT REFERENCES refinements(id),
            prompt TEXT NOT NULL,
            status TEXT NOT NULL,
            step TEXT NOT NULL,
            summary TEXT,
            error TEXT,
            recipe_json TEXT,
            filename TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS refinements_job ON refinements(job_id, created_at);
        CREATE UNIQUE INDEX IF NOT EXISTS refinements_one_active
            ON refinements(job_id, slot) WHERE status IN ('queued', 'running');
        """
    )
    conn.commit()


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def insert_job(conn: sqlite3.Connection, job_id: str, original_name: str) -> None:
    now = utcnow()
    conn.execute(
        """
        INSERT INTO jobs (id, status, step, original_name, created_at, updated_at)
        VALUES (?, 'queued', 'queued', ?, ?, ?)
        """,
        (job_id, original_name, now, now),
    )
    conn.commit()


def update_job(conn: sqlite3.Connection, job_id: str, **fields: Any) -> None:
    fields["updated_at"] = utcnow()
    keys = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE jobs SET {keys} WHERE id = ?",
        [*fields.values(), job_id],
    )
    conn.commit()


def get_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def replace_variants(conn: sqlite3.Connection, job_id: str, variants: list[dict[str, Any]]) -> None:
    conn.execute("DELETE FROM variants WHERE job_id = ?", (job_id,))
    for v in variants:
        conn.execute(
            """
            INSERT INTO variants (id, job_id, slot, look_id, title, why, recipe_json, filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                v["id"],
                job_id,
                v["slot"],
                v["look_id"],
                v["title"],
                v.get("why", ""),
                v["recipe_json"] if isinstance(v["recipe_json"], str) else json.dumps(v["recipe_json"]),
                v["filename"],
            ),
        )
    conn.commit()


def get_variants(conn: sqlite3.Connection, job_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM variants WHERE job_id = ? ORDER BY slot", (job_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def insert_feedback(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO feedback (
            id, job_id, chosen_slot, look_id, technical_score, style_score,
            restraint_score, notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["id"],
            row["job_id"],
            row["chosen_slot"],
            row["look_id"],
            row["technical_score"],
            row["style_score"],
            row["restraint_score"],
            row.get("notes", ""),
            utcnow(),
        ),
    )
    conn.commit()


def look_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    wins = conn.execute(
        """
        SELECT look_id, SUM(CASE WHEN style_score >= 4 AND technical_score >= 3
                                THEN 1 ELSE 0 END) AS n,
               AVG(technical_score) AS technical,
               AVG(style_score) AS style,
               AVG(restraint_score) AS restraint
        FROM feedback GROUP BY look_id
        """
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) AS n FROM feedback").fetchone()["n"]
    return {
        "total_picks": total,
        "looks": [
            {
                "look_id": r["look_id"],
                "wins": r["n"],
                "technical": r["technical"],
                "style": r["style"],
                "restraint": r["restraint"],
            }
            for r in wins
        ],
    }


def get_ratings(conn: sqlite3.Connection, job_id: str) -> dict[str, str | None]:
    names = {1: "like", -1: "dislike", 0: None}
    return {r["slot"]: names[r["verdict"]] for r in conn.execute(
        "SELECT slot, verdict FROM ratings WHERE job_id = ?", (job_id,))}


def set_rating(conn: sqlite3.Connection, job_id: str, slot: str, verdict: int,
               method_id: str, scene: dict) -> None:
    # Keep neutral rows as explicit undo markers, including for legacy feedback.
    conn.execute("""INSERT INTO ratings VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id, slot) DO UPDATE SET verdict=excluded.verdict,
        method_id=excluded.method_id, scene_json=excluded.scene_json, updated_at=excluded.updated_at""",
        (job_id, slot, verdict, method_id, json.dumps(scene), utcnow()))
    conn.commit()


def get_refinement(conn: sqlite3.Connection, refinement_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM refinements WHERE id = ?", (refinement_id,)).fetchone()
    return dict(row) if row else None


def get_refinements(conn: sqlite3.Connection, job_id: str) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM refinements WHERE job_id = ? ORDER BY created_at, rowid", (job_id,))]


def insert_refinement(conn: sqlite3.Connection, refinement_id: str, job_id: str,
                      slot: str, parent_id: str | None, prompt: str) -> None:
    conn.execute("""INSERT INTO refinements
        (id, job_id, slot, parent_id, prompt, status, step, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'queued', 'queued', ?, ?)""",
        (refinement_id, job_id, slot, parent_id, prompt, utcnow(), utcnow()))
    conn.commit()


def update_refinement(conn: sqlite3.Connection, refinement_id: str, **fields: Any) -> None:
    allowed = {"status", "step", "summary", "error", "recipe_json", "filename"}
    if not fields.keys() <= allowed:
        raise ValueError("Unknown refinement field")
    fields["updated_at"] = utcnow()
    conn.execute(f"UPDATE refinements SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                 [*fields.values(), refinement_id])
    conn.commit()
