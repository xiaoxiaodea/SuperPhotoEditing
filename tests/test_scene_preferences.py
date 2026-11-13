import json

from superphoto.db import connect, get_ratings, init_db, insert_feedback, insert_job, look_stats, replace_variants, set_rating, update_job
from superphoto.directions import design_directions
from superphoto.finishing_knowledge import METHODS
from superphoto.models import Critique, Recipe
from superphoto.scene_preferences import job_profile, preference_samples, scene_profile, select_methods, variant_method
from test_celebrity_styles import _tech


def saved_job(conn, job_id="photo", *, genre="landscape", subject="河上的船与对岸建筑"):
    critique, technical = Critique(genre=genre, subject_note=subject), _tech()
    insert_job(conn, job_id, "photo.jpg")
    update_job(conn, job_id, status="ready", critique_json=critique.model_dump_json(), technical_json=technical.model_dump_json())
    replace_variants(conn, job_id, [{"id": f"{job_id}-{slot}", "slot": slot, "look_id": "obsolete-lut",
        "title": METHODS[method]["title"], "recipe_json": json.dumps({"style_workflow": {"method_id": method}}),
        "filename": f"variant_{slot}.jpg"} for slot, method in zip("ABC", ("natural", "sculpted", "soft_warm"))])
    return critique, technical


def test_dislike_reduces_actual_method_frequency_only_for_similar_scenes(tmp_path):
    conn = connect(tmp_path / "prefs.db")
    init_db(conn)
    critique, technical = saved_job(conn)
    before = select_methods(conn, critique, technical, seed="first")
    profile = scene_profile(critique, technical)
    set_rating(conn, "photo", "B", -1, "sculpted", profile)
    after = select_methods(conn, critique, technical, seed="first")
    assert after["weights"]["sculpted"] < before["weights"]["sculpted"] * .25
    results = [select_methods(conn, critique, technical, seed=str(i))["selected"] for i in range(300)]
    frequency = sum("sculpted" in r for r in results) / len(results)
    assert .05 < frequency < .60, frequency
    assert all(len(set(r)) == 3 for r in results)
    # A different subject family and substantially different light/color are
    # independent, even when they offer the same finishing method.
    portrait = critique.model_copy(update={"genre": "portrait", "subject_note": "室内人像"})
    unrelated = select_methods(conn, portrait, technical, seed="first")
    assert unrelated["matched_feedback"] == 0
    assert unrelated["weights"]["sculpted"] == before["weights"]["sculpted"]
    different_light = select_methods(conn, critique, _tech(mean_luma=.80, aqua_fraction=.70), seed="first")
    assert different_light["matched_feedback"] == 0
    similar = select_methods(conn, critique, _tech(mean_luma=.32), seed="first")
    assert similar["matched_feedback"] == 1
    # Natural isn't a mandatory slot immune to negative feedback.
    set_rating(conn, "photo", "A", -1, "natural", profile)
    assert any("natural" not in select_methods(conn, critique, technical, seed=str(i))["selected"] for i in range(20))


def test_independent_ratings_are_idempotent_reversible_and_positive_boosts(tmp_path):
    conn = connect(tmp_path / "prefs.db")
    init_db(conn)
    critique, technical = saved_job(conn)
    profile = scene_profile(critique, technical)
    for _ in range(5):
        set_rating(conn, "photo", "A", 1, "natural", profile)
        set_rating(conn, "photo", "B", -1, "sculpted", profile)
        set_rating(conn, "photo", "C", 1, "soft_warm", profile)
    assert len(preference_samples(conn)) == 3
    assert get_ratings(conn, "photo") == {"A": "like", "B": "dislike", "C": "like"}
    assert select_methods(conn, critique, technical, seed="x")["weights"]["soft_warm"] > 3
    set_rating(conn, "photo", "B", 1, "sculpted", profile)
    assert select_methods(conn, critique, technical, seed="x")["weights"]["sculpted"] > 3
    set_rating(conn, "photo", "B", 0, "sculpted", profile)
    assert select_methods(conn, critique, technical, seed="x")["weights"]["sculpted"] == 3
    init_db(conn)
    assert get_ratings(conn, "photo")["B"] is None


def test_legacy_negative_is_not_a_win_and_undo_does_not_resurrect_it(tmp_path):
    conn = connect(tmp_path / "prefs.db")
    init_db(conn)
    critique, technical = saved_job(conn)
    for i in range(2):
        insert_feedback(conn, {"id": str(i), "job_id": "photo", "chosen_slot": "B", "look_id": "obsolete-lut",
                               "technical_score": 3, "style_score": 2, "restraint_score": 1})
    assert look_stats(conn)["looks"][0]["wins"] == 0
    assert len(preference_samples(conn)) == 1
    assert preference_samples(conn)[0]["verdict"] == -1
    set_rating(conn, "photo", "B", 0, "sculpted", scene_profile(critique, technical))
    assert preference_samples(conn) == []
    assert variant_method({"title": "未知旧风格", "recipe_json": "{}", "slot": "B"}) == ""
    assert job_profile({"critique_json": None, "technical_json": None}) == {}


def test_personalized_selection_reaches_recipe_design_and_persisted_identity(tmp_path, monkeypatch):
    from superphoto.config import get_settings
    from superphoto.preference import build_plans
    from superphoto.celebrity_styles import build_style_decision
    critique, technical = Critique(genre="landscape"), _tech()
    methods = ["sculpted", "soft_neutral", "soft_warm"]
    directions = design_directions(get_settings(), tmp_path / "unused.jpg", critique, technical, methods=methods)
    assert [d.method_id for d in directions] == methods
    assert [d.title for d in directions] == [METHODS[m]["title"] for m in methods]
    assert all(d.recipe != Recipe() for d in directions)
    critique.directions = directions
    plans = build_plans(build_style_decision(technical, critique).look_ids, critique, technical)
    assert [p.method_id for p in plans] == methods
