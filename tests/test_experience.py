import json
import sqlite3

import pytest
from PIL import Image

from superphoto import experience as ex
from superphoto.db import connect, get_ratings, init_db, insert_job, replace_variants, set_rating, update_job

OLD_SCHEMA = """
CREATE TABLE jobs (id TEXT PRIMARY KEY, status TEXT NOT NULL, step TEXT NOT NULL, error TEXT, original_name TEXT,
    original_file TEXT, technical_json TEXT, critique_json TEXT, plan_json TEXT, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL);
CREATE TABLE ratings (
    job_id TEXT NOT NULL REFERENCES jobs(id),
    slot TEXT NOT NULL CHECK(slot IN ('A','B','C')),
    verdict INTEGER NOT NULL CHECK(verdict IN (-1,0,1)),
    method_id TEXT NOT NULL, scene_json TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY(job_id, slot));
CREATE TABLE refinements (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id),
    slot TEXT NOT NULL CHECK(slot IN ('A','B','C')),
    parent_id TEXT REFERENCES refinements(id),
    prompt TEXT NOT NULL, status TEXT NOT NULL, step TEXT NOT NULL, summary TEXT, error TEXT, recipe_json TEXT,
    filename TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX refinements_job ON refinements(job_id, created_at);
CREATE UNIQUE INDEX refinements_one_active ON refinements(job_id, slot) WHERE status IN ('queued', 'running');
INSERT INTO jobs VALUES ('photo', 'ready', 'done', NULL, 'p.jpg', NULL, NULL, NULL, NULL, 't', 't');
INSERT INTO ratings VALUES ('photo', 'B', -1, 'sculpted', '{}', 't');
INSERT INTO refinements VALUES ('r1', 'photo', 'B', NULL, 'brighter', 'ready', 'done', NULL, NULL, NULL, NULL, 't', 't');
INSERT INTO refinements VALUES ('r2', 'photo', 'B', 'r1', 'warmer', 'running', 'x', NULL, NULL, NULL, NULL, 't', 't');
"""


def test_an_abc_database_is_widened_to_nine_versions_and_keeps_its_records(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as old:
        old.executescript(OLD_SCHEMA)
    conn = connect(path)
    init_db(conn)
    init_db(conn)   # idempotent
    assert get_ratings(conn, "photo") == {"B": "pass"}
    assert [r["id"] for r in conn.execute("SELECT id FROM refinements ORDER BY id")] == ["r1", "r2"]
    schema = {r["name"]: r["sql"] for r in conn.execute("SELECT name, sql FROM sqlite_master WHERE sql IS NOT NULL")}
    assert "REFERENCES refinements(id)" in schema["refinements"] and "_wide" not in "".join(schema.values())
    assert {"refinements_job", "refinements_one_active", "directions"} <= schema.keys()
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    set_rating(conn, "photo", "F", 2, "direction:airy", {})
    assert get_ratings(conn, "photo") == {"B": "pass", "F": "favorite"}
    with pytest.raises(sqlite3.IntegrityError):
        set_rating(conn, "photo", "J", 1, "direction:airy", {})
    with pytest.raises(sqlite3.IntegrityError):    # the one-active-refinement rule survived the rebuild
        conn.execute("INSERT INTO refinements VALUES ('r3', 'photo', 'B', 'r2', 'x', 'queued', 'q', NULL, NULL,"
                     " NULL, NULL, 't', 't')")


def test_elements_and_light_are_kept_to_the_vocabulary():
    assert ex.clean_elements(["Snow", "dunes", "camels", "unicorn", "snow", "Sky"]) == ["snow", "sand", "animal", "sky"]
    assert ex.clean_light("Golden") == "golden" and ex.clean_light("moonlight") == ""


def _scene(elements, light="golden", genre="landscape", luma=.4):
    class Tech:
        mean_luma, saturation_mean, temperature_bias, source_kind = luma, .3, .05, "raw"
    return ex.profile({"elements": elements, "light_kind": light, "genre": genre}, Tech())


def test_similarity_counts_shared_elements_then_light():
    snow = _scene(["snow", "sky", "mountain"])
    assert ex.similarity(snow, snow) == pytest.approx(1.0)
    assert ex.similarity(snow, _scene(["snow", "sky", "mountain"], light="overcast")) < .5
    assert ex.similarity(snow, _scene(["snow", "sky"])) > ex.similarity(snow, _scene(["snow", "sand"]))
    assert ex.similarity(snow, _scene(["face", "interior"], genre="portrait")) == 0
    assert ex.similarity(snow, {"version": 1, "family": "landscape"}) == 0


def _render(tmp_path, name, value, colour=(0, 0, 0)):
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", (64, 48), tuple(min(255, value + c) for c in colour)).save(path)
    return path


def _photo(conn, tmp_path, job_id, scene, versions):
    """versions: slot -> (direction id, brightness, colour push, plan intent)."""
    insert_job(conn, job_id, f"{job_id}.jpg")
    update_job(conn, job_id, status="ready")
    rows = []
    library = ex.library(conn)
    for slot, (direction, value, colour, intent) in versions.items():
        render = _render(tmp_path, f"{job_id}-{slot}", value, colour)
        sidecar = {"plan": {"title": library[direction].title, "intent": intent,
                            "tone": {"black": .14, "mid": .70, "white": .93}},
                   "style_workflow": {"method_id": ex.METHOD_PREFIX + direction},
                   "experience": ex.sidecar_block(scene, library[direction], intent, render)}
        rows.append({"id": f"{job_id}-{slot}", "slot": slot, "look_id": "directed", "title": library[direction].title,
                     "recipe_json": json.dumps(sidecar), "filename": render.name})
    replace_variants(conn, job_id, rows)


def test_picks_on_similar_pictures_become_weights_taste_and_exemplars(tmp_path):
    conn = connect(tmp_path / "x.db")
    init_db(conn)
    snow = _scene(["snow", "sky", "mountain"])
    for i in range(2):
        _photo(conn, tmp_path, f"snow{i}", snow, {"A": ("natural", 110, (0, 0, 0), "自然"),
                                                   "B": ("dramatic", 70, (0, 0, 0), "压暗"),
                                                   "C": ("airy", 150, (12, 6, 0), "通透明亮的雪")})
        set_rating(conn, f"snow{i}", "C", 2 if i else 1, "direction:airy", snow)
        set_rating(conn, f"snow{i}", "B", -1, "direction:dramatic", snow)
    desert = _scene(["sand", "animal"], light="hard_sun")
    _photo(conn, tmp_path, "desert", desert, {"A": ("natural", 120, (0, 0, 0), "自然"),
                                              "B": ("dramatic", 80, (0, 0, 0), "沙丘光影")})
    set_rating(conn, "desert", "B", 2, "direction:dramatic", desert)

    now = _scene(["snow", "sky"])
    evidence = ex.direction_evidence(conn, now)
    assert evidence["airy"]["favorite"] == 1 and evidence["airy"]["keep"] == 1 and evidence["airy"]["weight"] > 2
    assert evidence["dramatic"]["pass"] == 2 and evidence["dramatic"]["favorite"] == 0   # the desert pick is elsewhere
    assert evidence["dramatic"]["weight"] < 1 and evidence["dramatic"]["shared"] == ["sky", "snow"]
    weights = ex.weights(conn, now, ["natural", "dramatic", "airy"])
    assert weights == {"natural": 1.0, "dramatic": evidence["dramatic"]["weight"], "airy": evidence["airy"]["weight"]}
    assert ex.display_order({"A": weights["natural"], "B": weights["dramatic"], "C": weights["airy"]}) == ["C", "A", "B"]
    assert ex.display_order({"A": 1.0, "B": 1.0}) == ["A", "B"]

    leaning = ex.taste(conn, now)
    assert leaning["kept"]["mid"] > .1 and leaning["kept"]["warmth"] > .02 and leaning["passed"]["mid"] < -.1
    block = ex.experience_block(conn, now)
    assert "通透明亮 (airy): favorite 1, keep 1" in block and "dramatic): pass 2" in block
    assert "brighter mids" in block and "warmer" in block and "通透明亮的雪" in block
    assert "not a ban" in block
    assert ex.experience_block(conn, _scene(["face"], light="indoor", genre="portrait")) == ""


def test_a_direction_the_planner_proposes_is_remembered_under_a_stable_id(tmp_path):
    conn = connect(tmp_path / "x.db")
    init_db(conn)
    first = ex.remember_direction(conn, "冷调极简", "Minimal cool palette ...", look="cine_teal", job_id="j1")
    again = ex.remember_direction(conn, "冷调极简", "another wording", job_id="j2")
    assert first.id == again.id and first.id.startswith("own:") and again.role == "Minimal cool palette ..."
    library = ex.library(conn)
    assert library[first.id].look == "cine_teal" and library[first.id].origin == "planner"
    assert library["natural"].natural and library["natural"].vignette_floor == -10
    assert all(f"look:{lid}" in library for lid in ex.LOOKS)
    assert ex.variant_profile({"recipe_json": json.dumps({"experience": {"profile": _scene(["sky"])}})})["elements"] == ["sky"]
    assert ex.variant_profile({"recipe_json": "{}"}) == {}
