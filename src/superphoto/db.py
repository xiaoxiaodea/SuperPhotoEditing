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
        """ + _RATINGS.format(name="ratings") + _REFINEMENTS.format(name="refinements") + """
        CREATE TABLE IF NOT EXISTS directions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            role TEXT NOT NULL,
            look_id TEXT,
            origin_job TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS style_batches (
            id TEXT PRIMARY KEY,
            source_job_id TEXT NOT NULL REFERENCES jobs(id),
            source_slot TEXT NOT NULL,
            source_version TEXT,
            title TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            status TEXT NOT NULL,
            total_count INTEGER NOT NULL,
            done_count INTEGER NOT NULL DEFAULT 0,
            error_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS style_batch_items (
            id TEXT PRIMARY KEY,
            batch_id TEXT NOT NULL REFERENCES style_batches(id),
            original_name TEXT NOT NULL,
            original_file TEXT NOT NULL,
            status TEXT NOT NULL,
            step TEXT NOT NULL,
            error TEXT,
            output_filename TEXT,
            develop_json TEXT,
            metrics_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    _widen_slots(conn)
    conn.executescript(_INDEXES)
    conn.commit()


# A photo gets up to len(SLOT_LETTERS) versions (A is the natural finish). Ratings: -1 pass (this treatment is not
# for this photo; a weak, scene-local signal), 1 keep, 2 favourite (the version the user would deliver), 0 undone.
SLOT_LETTERS = "ABCDEFGHI"
VERDICTS = {"pass": -1, "keep": 1, "favorite": 2}
VERDICT_NAMES = {-1: "pass", 0: None, 1: "keep", 2: "favorite"}
_SLOT_CHECK = f"CHECK(length(slot) = 1 AND slot BETWEEN 'A' AND '{SLOT_LETTERS[-1]}')"
# {name} lets the migration build the new table beside the old one; references stay on the final names.
_RATINGS = """
        CREATE TABLE IF NOT EXISTS {name} (
            job_id TEXT NOT NULL REFERENCES jobs(id),
            slot TEXT NOT NULL {slot_check},
            verdict INTEGER NOT NULL CHECK(verdict IN (-1,0,1,2)),
            method_id TEXT NOT NULL,
            scene_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(job_id, slot)
        );""".replace("{slot_check}", _SLOT_CHECK)
_REFINEMENTS = """
        CREATE TABLE IF NOT EXISTS {name} (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id),
            slot TEXT NOT NULL {slot_check},
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
        );""".replace("{slot_check}", _SLOT_CHECK)
_INDEXES = """
        CREATE INDEX IF NOT EXISTS refinements_job ON refinements(job_id, created_at);
        CREATE UNIQUE INDEX IF NOT EXISTS refinements_one_active
            ON refinements(job_id, slot) WHERE status IN ('queued', 'running');
        CREATE INDEX IF NOT EXISTS style_batch_items_batch ON style_batch_items(batch_id, created_at);"""


def _widen_slots(conn: sqlite3.Connection) -> None:
    """Rebuild ratings / refinements made when a photo had only A/B/C and a rating was like or dislike."""
    old = [name for name, sql in conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' AND name IN ('ratings', 'refinements')")
        if "IN ('A','B','C')" in (sql or "")]
    if not old:
        return
    # SQLite's table-rebuild order: build the new table beside the old one, copy, drop the old, rename the new.
    # (Renaming the old table first would rewrite refinements' self-reference to the temporary name.)
    steps = ["BEGIN;"]
    for name in old:
        steps += [(_RATINGS if name == "ratings" else _REFINEMENTS).format(name=f"{name}_wide"),
                  f"INSERT INTO {name}_wide SELECT * FROM {name};",
                  f"DROP TABLE {name};",
                  f"ALTER TABLE {name}_wide RENAME TO {name};"]
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")      # foreign_keys cannot change inside a transaction.
    try:
        conn.executescript("\n".join([*steps, "COMMIT;"]))
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def insert_job(conn: sqlite3.Connection, job_id: str, original_name: str,
               original_file: str | None = None) -> None:
    now = utcnow()
    conn.execute(
        """
        INSERT INTO jobs (id, status, step, original_name, original_file, created_at, updated_at)
        VALUES (?, 'queued', 'queued', ?, ?, ?, ?)
        """,
        (job_id, original_name, original_file, now, now),
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
    return {r["slot"]: VERDICT_NAMES[r["verdict"]] for r in conn.execute(
        "SELECT slot, verdict FROM ratings WHERE job_id = ?", (job_id,))}


def set_rating(conn: sqlite3.Connection, job_id: str, slot: str, verdict: int,
               method_id: str, scene: dict) -> None:
    with tx(conn):
        if verdict == VERDICTS["favorite"]:
            # One favourite per photo: an earlier favourite stays a kept version.
            conn.execute("UPDATE ratings SET verdict = ?, updated_at = ? WHERE job_id = ? AND slot != ? AND verdict = ?",
                         (VERDICTS["keep"], utcnow(), job_id, slot, VERDICTS["favorite"]))
        # Keep neutral rows as explicit undo markers, including for legacy feedback.
        conn.execute("""INSERT INTO ratings VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, slot) DO UPDATE SET verdict=excluded.verdict,
            method_id=excluded.method_id, scene_json=excluded.scene_json, updated_at=excluded.updated_at""",
            (job_id, slot, verdict, method_id, json.dumps(scene), utcnow()))


def insert_direction(conn: sqlite3.Connection, direction_id: str, title: str, role: str,
                     look_id: str | None = None, origin_job: str | None = None) -> None:
    """Remember a direction the planner proposed, so later photos can be offered it and its ratings add up."""
    conn.execute("INSERT OR IGNORE INTO directions VALUES (?, ?, ?, ?, ?, ?)",
                 (direction_id, title, role, look_id, origin_job, utcnow()))
    conn.commit()


def get_directions(conn: sqlite3.Connection) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM directions ORDER BY created_at, id")]


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


def insert_style_batch(conn: sqlite3.Connection, batch_id: str, source_job_id: str, source_slot: str,
                       source_version: str | None, title: str, profile: dict[str, Any], total_count: int) -> None:
    now = utcnow()
    conn.execute(
        """INSERT INTO style_batches
           (id, source_job_id, source_slot, source_version, title, profile_json, status,
            total_count, done_count, error_count, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, 0, 0, ?, ?)""",
        (batch_id, source_job_id, source_slot, source_version, title,
         json.dumps(profile, ensure_ascii=False), int(total_count), now, now),
    )
    conn.commit()


def insert_style_batch_item(conn: sqlite3.Connection, item_id: str, batch_id: str,
                            original_name: str, original_file: str) -> None:
    now = utcnow()
    conn.execute(
        """INSERT INTO style_batch_items
           (id, batch_id, original_name, original_file, status, step, created_at, updated_at)
           VALUES (?, ?, ?, ?, 'queued', 'queued', ?, ?)""",
        (item_id, batch_id, original_name, original_file, now, now),
    )
    conn.commit()


def get_style_batch(conn: sqlite3.Connection, batch_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM style_batches WHERE id = ?", (batch_id,)).fetchone()
    return dict(row) if row else None


def list_style_batches(conn: sqlite3.Connection, *, source_job_id: str | None = None,
                       limit: int = 20) -> list[dict[str, Any]]:
    if source_job_id:
        rows = conn.execute(
            "SELECT * FROM style_batches WHERE source_job_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (source_job_id, int(limit)),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM style_batches ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(r) for r in rows]


def get_style_batch_item(conn: sqlite3.Connection, item_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM style_batch_items WHERE id = ?", (item_id,)).fetchone()
    return dict(row) if row else None


def get_style_batch_items(conn: sqlite3.Connection, batch_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM style_batch_items WHERE batch_id = ? ORDER BY created_at, rowid",
        (batch_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_style_batch_item(conn: sqlite3.Connection, item_id: str, **fields: Any) -> None:
    allowed = {"status", "step", "error", "output_filename", "develop_json", "metrics_json"}
    if not fields.keys() <= allowed:
        raise ValueError("Unknown style_batch_items field")
    fields["updated_at"] = utcnow()
    conn.execute(f"UPDATE style_batch_items SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                 [*fields.values(), item_id])
    conn.commit()


def refresh_style_batch_counts(conn: sqlite3.Connection, batch_id: str) -> None:
    rows = conn.execute("SELECT status FROM style_batch_items WHERE batch_id = ?", (batch_id,)).fetchall()
    statuses = [r["status"] for r in rows]
    total = len(statuses)
    done = sum(1 for s in statuses if s == "ready")
    err = sum(1 for s in statuses if s == "error")
    running = any(s == "running" for s in statuses)
    queued = any(s == "queued" for s in statuses)
    if running:
        status = "running"
    elif queued:
        status = "running" if (done or err) else "queued"
    elif total > 0 and done == 0 and err == total:
        status = "error"
    else:
        status = "ready"
    conn.execute(
        """UPDATE style_batches SET status = ?, total_count = ?, done_count = ?, error_count = ?, updated_at = ?
           WHERE id = ?""",
        (status, total, done, err, utcnow(), batch_id),
    )
    conn.commit()
