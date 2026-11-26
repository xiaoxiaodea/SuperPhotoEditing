"""Unit tests for the full-corpus 35-look master colour library, RGB split curves, look WB merging,
grading.global support, scene-aware shortlisting, /api/directed-looks, night-scene highlight guard,
and Style Sync curve/WB fidelity.
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from superphoto import app as app_mod, directed, looks, raster_negative, style_sync


def test_all_35_looks_have_valid_structure_signed_mixer_hues_and_monotonic_rgb_curves():
    assert len(looks.LOOKS) == 35
    expected_new = {
        # 6 remaining LeoCarbin personal families (100% LeoCarbin coverage)
        "mist_violet",
        "sunset_crimson",
        "wildlife_rich",
        "urban_street",
        "botanical_jade",
        "still_life_editorial",
        # 15 cinema/film/drama/anime archetypes (100% 素材a coverage)
        "japanese_airy_film",
        "wong_kar_wai",
        "k_drama_warm",
        "anime_sky_blue",
        "wes_pastel",
        "twilight_blue_drama",
        "dune_amber",
        "european_classic",
        "cyber_neon",
        "period_ink",
        "j_drama_cool",
        "youth_film_mint",
        "summer_nostalgia",
        "crime_amber_grit",
        "retro_americana",
        "noir_steel",
    }
    assert expected_new.issubset(looks.LOOKS.keys())

    for lid, spec in looks.LOOKS.items():
        assert spec.get("title") and spec.get("use"), f"{lid} missing title/use"
        assert spec.get("category"), f"{lid} missing category"
        assert spec.get("affinity"), f"{lid} missing scene affinity"
        dev = spec.get("develop") or {}
        # All mixer hues must stay on signed -100..100 scale (never wrapped 0..360!)
        for band, hsl in (dev.get("mixer") or {}).items():
            assert band in directed.MIXER_BANDS, f"{lid} invalid mixer band {band}"
            for key, val in hsl.items():
                assert -100.0 <= float(val) <= 100.0, f"{lid}.mixer.{band}.{key}={val} out of range"
        # RGB split curves must be strictly monotonic in x, within [0, 1], and never touch curve.master
        curves = dev.get("curve") or {}
        assert "master" not in curves, f"{lid} must not override tone solver's curve.master"
        for ch, pts in curves.items():
            assert ch in ("red", "green", "blue"), f"{lid} unexpected curve channel {ch}"
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            assert xs[0] == 0.0 and xs[-1] == 1.0, f"{lid}.curve.{ch} must span 0..1"
            assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1)), f"{lid}.curve.{ch} non-monotonic x"
            assert all(0.0 <= y <= 1.0 for y in ys), f"{lid}.curve.{ch} y out of [0,1]"
        # Grading wheels must be supported keys with valid hue/sat
        grading = dev.get("grading") or {}
        for k, v in grading.items():
            if k in directed.GRADING_WHEELS:
                assert 0.0 <= float(v.get("hue", 0.0)) <= 360.0
                assert 0.0 <= float(v.get("sat", 0.0)) <= 45.0


def test_shortlist_for_scene_and_catalog_api():
    night_list = looks.shortlist_for_scene({"mid": 0.14, "warmth": -0.05, "chroma": 0.10}, limit=20)
    snow_list = looks.shortlist_for_scene({"mid": 0.72, "warmth": -0.06, "chroma": 0.05}, limit=20)
    assert len(night_list) == 20 and len(snow_list) == 20
    # Night scenes must surface night cinema looks; high-key scenes must surface snow/mist/pastel looks
    assert {"wong_kar_wai", "night_blue", "twilight_blue_drama", "cyber_neon"}.issubset(set(night_list))
    assert {"snow_luminous", "mist_violet", "wes_pastel"}.issubset(set(snow_list))

    items = looks.catalog()
    assert len(items) == 35
    assert all({"id", "title", "category", "use", "has_rgb_curves", "has_wb"} <= set(it) for it in items)

    client = TestClient(app_mod.app)
    resp = client.get("/api/directed-looks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 35 and len(body["looks"]) == 35


def test_merged_develop_interpolates_rgb_curves_wb_and_global_grading_by_strength():
    plan_half = {
        "develop": {"wb": {"temp": 200.0, "tint": 2.0}},
        "look": {"id": "wong_kar_wai", "strength": 0.5},
    }
    merged = directed._merged_develop(plan_half)

    # WB: look WB * 0.5 + model (+200, +2)
    raw_wb = looks.LOOKS["wong_kar_wai"]["develop"]["wb"]
    assert merged["wb"]["temp"] == pytest.approx(200.0 + raw_wb["temp"] * 0.5, abs=0.1)
    assert merged["wb"]["tint"] == pytest.approx(2.0 + raw_wb["tint"] * 0.5, abs=0.1)

    # RGB curve midpoint interpolation toward identity y=x
    raw_blue = looks.LOOKS["wong_kar_wai"]["develop"]["curve"]["blue"]
    merged_blue = merged["curve"]["blue"]
    assert len(merged_blue) == len(raw_blue)
    for (rx, ry), (mx, my) in zip(raw_blue, merged_blue):
        assert mx == pytest.approx(rx, abs=1e-4)
        assert my == pytest.approx(rx + (ry - rx) * 0.5, abs=1e-3)

    # grading.global scaled by strength
    raw_glob = looks.LOOKS["wong_kar_wai"]["develop"]["grading"]["global"]
    assert merged["grading"]["global"]["hue"] == pytest.approx(raw_glob["hue"], abs=1e-3)
    assert merged["grading"]["global"]["sat"] == pytest.approx(raw_glob["sat"] * 0.5, abs=1e-2)

    # Compiling with a solved master curve keeps both master and RGB split curves cleanly
    doc, _notes = directed.compile_document(plan_half, "jpeg", curve=[[0.0, 0.05], [0.5, 0.52], [1.0, 0.94]])
    assert doc["curve"]["master"][0] == [0.0, 0.05]
    assert "blue" in doc["curve"] and "green" in doc["curve"] and "red" in doc["curve"]
    assert doc["wb"] == {
        "mode": "custom",
        "temp": round(6500.0 + 200.0 + raw_wb["temp"] * 0.5, 1),
        "tint": round(2.0 + raw_wb["tint"] * 0.5, 2),
    }
    assert "global" in doc["grading"]


def test_compile_document_clamps_extreme_cool_look_wb_safely():
    plan = {
        "develop": {"wb": {"temp": -2200.0, "tint": -38.0}},
        "look": {"id": "twilight_blue_drama", "strength": 1.0},
    }
    doc, _ = directed.compile_document(plan, "jpeg")
    # MERGED_WB_RANGES clamps temp at -3200 -> 6500 - 3200 = 3300K (>= 2000K safety floor)
    assert doc["wb"]["temp"] >= 2000.0
    assert doc["wb"]["temp"] == pytest.approx(3300.0, abs=0.1)
    assert doc["wb"]["tint"] == pytest.approx(-45.0, abs=0.1)


def test_expanded_effects_and_vibrance_ranges_pass_plan_sanitization():
    raw = {
        "title": "柔光胶片",
        "intent": "测试扩展滑块区间",
        "develop": {
            "color": {"vibrance": 52, "saturation": -22},
            "effects": {"dehaze": -38, "texture": -38, "clarity": 8},
            "grading": {"global": {"hue": 200, "sat": 20, "lum": 0}},
        },
        "look": {"id": "japanese_airy_film", "strength": 0.85},
    }
    clean, _notes = directed.sanitize_plan(raw, slot="C")
    assert clean["develop"]["color"]["vibrance"] == 52.0
    assert clean["develop"]["effects"]["dehaze"] == -38.0
    assert clean["develop"]["effects"]["texture"] == -38.0
    assert clean["develop"]["grading"]["global"]["sat"] == 20.0
    assert clean["look"] == {"id": "japanese_airy_film", "strength": 0.85}


def test_style_sync_preserves_and_scales_rgb_curves_and_wide_wb():
    profile = {
        "style_develop": {
            "rgb_curves": {
                "red": [[0.0, 0.0], [0.5, 0.40], [1.0, 1.0]],
                "blue": [[0.0, 0.0], [0.5, 0.60], [1.0, 1.0]],
            },
            "grading": {"global": {"hue": 200.0, "sat": 30.0, "lum": 0.0}},
        },
        "wb_offset": {"temp": -2300.0, "tint": -24.0},
        "anchor_baseline": {"warmth": 0.0, "tint_bias": 0.0},
    }
    doc = style_sync._compose_base_document(
        profile, "jpeg", {"warmth": 0.0, "tint_bias": 0.0}, match_wb=False, style_strength=0.5,
    )
    assert doc["curve"]["red"][1] == [0.5, 0.45]
    assert doc["curve"]["blue"][1] == [0.5, 0.55]
    assert doc["grading"]["global"]["sat"] == pytest.approx(15.0, abs=0.05)
    assert doc["wb"] == {"mode": "custom", "temp": 5350.0, "tint": -12.0}


def test_night_scene_emitters_skip_synthetic_highlight_headroom():
    # Dark night sky (0.04) with bright orange fireworks/lamps (1.0, 0.70, 0.25) -> mean luma < 0.22
    night = np.full((64, 64, 3), 0.04, dtype=np.float32)
    night[20:40, 20:40] = (1.0, 0.70, 0.25)
    out, pct = raster_negative.reconstruct_partial_highlights(night)
    assert pct == 0.0
    assert np.allclose(out, night, atol=1e-6)


def test_merged_develop_preserves_unscaled_shape_params_and_local_light_and_masking():
    plan = {"develop": {}, "look": {"id": "wes_pastel", "strength": 0.5}}
    merged = directed._merged_develop(plan)
    # Local edge-aware shadows/highlights scale with strength
    assert merged["light"]["shadows"] == pytest.approx(18.0, abs=0.1)
    assert merged["light"]["highlights"] == pytest.approx(-12.0, abs=0.1)
    # Sharpen amount scales with strength, while sharpen_masking & grain size/roughness stay unscaled
    assert merged["detail"]["sharpen_amount"] == pytest.approx(19.0, abs=0.1)
    assert merged["detail"]["sharpen_masking"] == pytest.approx(35.0, abs=0.1)
    assert merged["grain"]["size"] == pytest.approx(24.0, abs=0.1)
    assert merged["grain"]["roughness"] == pytest.approx(75.0, abs=0.1)

    # Plan sharpening + look sharpening must take max rather than double-stacking to 80+
    stacked = directed._merged_develop({
        "develop": {"detail": {"sharpen_amount": 40.0, "nr_luminance": 18.0}},
        "look": {"id": "cine_teal", "strength": 0.7},
    })
    assert stacked["detail"]["sharpen_amount"] <= 46.0
    assert stacked["detail"]["sharpen_amount"] == pytest.approx(max(40.0, 64.0 * 0.7), abs=0.2)
    assert stacked["detail"]["nr_luminance"] == pytest.approx(18.0, abs=0.1)


def test_strict_8_dimension_quality_audit_catches_blur_banding_and_missing_grain(tmp_path):
    from PIL import Image, ImageFilter
    from superphoto.quality_audit import audit_against_reference

    rng = np.random.default_rng(42)
    h, w = 256, 256
    # Construct crisp synthetic photographer reference with sharp branches + fine organic film grain
    y = np.broadcast_to(np.linspace(0.25, 0.85, h, dtype=np.float32)[:, None], (h, w))
    ref = np.stack([y * 0.9, y * 0.95, np.clip(y * 1.05, 0, 1)], axis=-1).copy()
    ref[90:170, 110:150] = (0.12, 0.14, 0.12)
    # Crisp bright snow flecks inside the dark tree
    ref[100:160:6, 115:145:5] = (0.92, 0.94, 0.96)
    # Organic fine 35mm film grain (~6 LSB)
    ref = np.clip(ref + rng.normal(0.0, 6.0 / 255.0, size=ref.shape).astype(np.float32), 0.0, 1.0)

    ref_path = tmp_path / "ref.jpg"
    Image.fromarray((ref * 255).astype(np.uint8)).save(ref_path, quality=95)

    # 1. Identical image -> EXCELLENT with zero defects
    self_res = audit_against_reference(ref_path, ref_path)
    assert self_res["verdict"] == "EXCELLENT"
    assert self_res["strict_pct"] >= 99.0
    assert self_res["defects"] == []

    # 2. Blurred plastic candidate (loses snow detail AND grain!) -> catches detail_loss & grain_mismatch
    blur_path = tmp_path / "blur.jpg"
    Image.open(ref_path).filter(ImageFilter.GaussianBlur(radius=2.2)).save(blur_path, quality=95)
    blur_res = audit_against_reference(blur_path, ref_path)
    codes = {d["code"] for d in blur_res["defects"]}
    assert "detail_loss" in codes
    assert "grain_mismatch" in codes
    assert blur_res["verdict"] == "NEEDS_WORK"

    # 3. Medium-scale colour-banded candidate (harsh quantized colour steps in smooth sky) -> catches color_banding
    banded = ref.copy()
    banded[:80, :, 0] = np.round(banded[:80, :, 0] * 4.0) / 4.0
    banded[:80, :, 2] = np.clip(1.0 - banded[:80, :, 0], 0.0, 1.0)
    band_path = tmp_path / "band.jpg"
    Image.fromarray((banded * 255).astype(np.uint8)).save(band_path, quality=95)
    band_res = audit_against_reference(band_path, ref_path)
    assert band_res["antibanding_pct"] < 84.0
    assert any(d["code"] == "color_banding" for d in band_res["defects"])


def test_light_stylization_first_portfolio_shortlist_quota_and_display_order():
    from superphoto import experience

    # 1. Every look is classified cleanly; shortlist caps strong looks (<= 5) so light options dominate
    strong_cnt = sum(1 for lid in looks.LOOKS if looks.is_strong_look(lid))
    light_cnt = len(looks.LOOKS) - strong_cnt
    assert light_cnt > strong_cnt and strong_cnt == len(looks.STRONG_LOOK_IDS)

    night_short = looks.shortlist_for_scene({"mid": 0.14, "warmth": -0.05, "chroma": 0.10}, limit=20, max_strong=5)
    assert len(night_short) == 20
    assert sum(1 for lid in night_short if looks.is_strong_look(lid)) <= 5

    # Catalog & API expose intensity tier
    cat = looks.catalog()
    assert all(item["intensity"] in ("light", "strong") for item in cat)
    assert any("[轻量风格" in line for line in looks.look_lines())
    assert any("[强风格" in line for line in looks.look_lines())

    # 2. _rebalance_stylization enforces at most MAX_STRONG_STYLIZED=1 strong slot and moderates excess slots
    plans = {
        "A": {"title": "自然完成", "develop": {}, "look": None},
        "B": {"title": "清透电影", "develop": {}, "look": {"id": "clean_cinema", "strength": 0.95}},
        "C": {"title": "港风胶片", "develop": {}, "look": {"id": "wong_kar_wai", "strength": 0.85}},
        "D": {"title": "暮色蓝调", "develop": {"wb": {"temp": -1200.0, "tint": -18.0}},
              "look": {"id": "twilight_blue_drama", "strength": 0.90}},
        "E": {"title": "赛博霓虹", "develop": {"grading": {"global": {"hue": 200, "sat": 28}}},
              "look": {"id": "cyber_neon", "strength": 0.80}},
    }
    directions = {
        "A": {"id": "natural", "title": "自然完成"},
        "B": {"id": "look:clean_cinema", "title": "通透电影感"},
        "C": {"id": "look:wong_kar_wai", "title": "王家卫"},
        "D": {"id": "look:twilight_blue_drama", "title": "蓝调暮色"},
        "E": {"id": "look:cyber_neon", "title": "赛博霓虹"},
    }
    notes: dict[str, list[str]] = {}
    # User previously liked D slightly more -> D should be the ONE strong slot kept!
    directed._rebalance_stylization(plans, directions, notes, weights={"look:twilight_blue_drama": 1.4})

    # B (light look) capped to <= LIGHT_LOOK_STRENGTH_CAP
    assert plans["B"]["look"]["strength"] == pytest.approx(directed.LIGHT_LOOK_STRENGTH_CAP)
    # D (highest-weighted strong slot) kept intact
    assert plans["D"]["look"]["strength"] == pytest.approx(0.90)
    assert directed.is_strong_plan(plans["D"], directions["D"]) is True
    # C and E (excess strong slots) demoted to lightweight stylization (strength <= 0.45, WB/global moderated)
    assert plans["C"]["look"]["strength"] == pytest.approx(directed.DEMOTED_STRONG_STRENGTH_CAP)
    assert plans["E"]["look"]["strength"] == pytest.approx(directed.DEMOTED_STRONG_STRENGTH_CAP)
    assert plans["E"]["develop"]["grading"]["global"]["sat"] <= 12.0
    assert directed.is_strong_plan(plans["C"], directions["C"]) is False
    assert directed.is_strong_plan(plans["E"], directions["E"]) is False
    assert sum(1 for s in "BCDE" if directed.is_strong_plan(plans[s], directions[s])) == 1

    # 3. Display ordering places Natural + Light-Stylized versions ahead of Strong on equal weights,
    #    while still letting a strongly user-favored strong direction rise to the top.
    assert experience.display_order({"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0}, strong_slots={"B"}) == ["A", "C", "D", "B"]
    assert experience.display_order({"A": 1.0, "B": 1.8, "C": 1.0}, strong_slots={"B"}) == ["B", "A", "C"]


def test_speed_optimizations_mask_inheritance_shared_reconcile_and_shortlist():
    from types import SimpleNamespace
    from PIL import Image
    from superphoto import experience, tone

    # 1. Compact mask inheritance expands cleanly into independent full masks before sanitize_plan
    plans_raw = {
        "A": {
            "title": "自然",
            "masks": [{
                "name": "sky", "purpose": "压天空",
                "shape": {"kind": "linear", "start": {"x": 0.5, "y": 0.0}, "end": {"x": 0.5, "y": 0.4}},
                "limit": {"kind": "luminance", "lo": 0.7, "hi": 1.0, "feather": 0.05},
                "adjust": {"dehaze": 10},
                "target": {"box": [0.2, 0.05, 0.8, 0.3], "median": 0.65},
            }],
        },
        "B": {
            "title": "通透",
            "masks": [
                {"use": "sky", "adjust": {"dehaze": 18}, "target": {"median": 0.60}},
            ],
        },
    }
    expanded = directed._expand_inherited_masks(plans_raw)
    mask_b = expanded["B"]["masks"][0]
    assert mask_b["name"] == "sky"
    assert mask_b["shape"] == plans_raw["A"]["masks"][0]["shape"]
    assert mask_b["limit"] == plans_raw["A"]["masks"][0]["limit"]
    assert mask_b["adjust"] == {"dehaze": 18}
    assert mask_b["target"] == {"box": [0.2, 0.05, 0.8, 0.3], "median": 0.60}

    # 2. Pure mathematical `shared` contradiction is reconciled deterministically without a VLM roundtrip
    arr = np.full((128, 128, 3), 235, dtype=np.uint8)
    arr[70:, :] = 50
    img = Image.fromarray(arr)
    anchors = tone.measure(tone.load_small(img))
    plan_shared = {
        "title": "A",
        "tone": {k: round(float(v), 3) for k, v in anchors.items()},
        "masks": [{
            "name": "sky", "purpose": "压天空",
            "shape": {"kind": "linear", "start": {"x": 0.5, "y": 0.0}, "end": {"x": 0.5, "y": 0.55}},
            "adjust": {"dehaze": 10},
            "target": {"box": [0.2, 0.05, 0.8, 0.35], "median": 0.72},
        }],
    }
    assert any(item["kind"] == "shared" for item in directed.plan_reach(plan_shared, img))
    plans = {"A": plan_shared}
    notes: dict[str, list[str]] = {}
    reconciled = directed._reconcile_shared_reach(plans, img, notes)
    assert reconciled == ["A"]
    assert not any(item["kind"] == "shared" for item in directed.plan_reach(plans["A"], img))

    # 3. 6-version prompt uses scene-shortlisted looks (10) & profiles (5) instead of dumping all 35 + 15
    tech = SimpleNamespace(model_dump=lambda: {"source_kind": "jpeg", "width": 1800, "height": 3200, "mean_luma": 0.45})
    full_prompt = directed.plan_prompt(tech, (1800, 3200))
    multi_prompt = directed.plan_prompt(tech, (1800, 3200), library=experience.library())
    # Even with the ~6k DIRECTION LIBRARY included, shortlisted multi_prompt is much shorter than unshortlisted full_prompt
    assert len(multi_prompt) < len(full_prompt) - 5000
    assert "OUTPUT COMPACTNESS" in multi_prompt

    # 4. Labeled contact sheet packs 7 vertical frames into 1 compact image for local Ollama damage_check
    panels = [("ORIG", img)] + [(f"VERSION {s}", img) for s in "ABCDEF"]
    sheet = directed._labeled_contact_sheet(panels, tile_edge=320)
    assert sheet.mode == "RGB" and sheet.width == 4 * 320 and sheet.height == 2 * (320 + 28)

    # 5. subtle_film with missing look automatically receives an unused light-tier film look,
    #    and duplicate look IDs across sibling slots are deduplicated deterministically.
    sf_plans = {
        "A": {"title": "自然"},
        "B": {"title": "青橙", "look": {"id": "cine_teal", "strength": 0.65}},
        "C": {"title": "轻量胶片", "develop": {"detail": {"sharpen_amount": 38}}},
        "D": {"title": "青橙2", "look": {"id": "cine_teal", "strength": 0.70}},
    }
    sf_dirs = {
        "A": {"id": "natural", "title": "自然"},
        "B": {"id": "look:cine_teal", "title": "青橙"},
        "C": {"id": "subtle_film", "title": "轻量胶片"},
        "D": {"id": "look:cine_teal", "title": "青橙"},
    }
    sf_notes: dict[str, list[str]] = {}
    directed._rebalance_stylization(sf_plans, sf_dirs, sf_notes, measured_looks={"vintage_olive": {"distance": 5.8}})
    assert sf_plans["C"].get("look") is not None and not looks.is_strong_look(sf_plans["C"]["look"]["id"])
    assert sf_plans["D"]["look"]["id"] != sf_plans["B"]["look"]["id"]

    # 6. Ultra-compact integer delta schema expands losslessly into standard reading, directions, plans & elements
    compact_raw = {
        "r": {"sc": "雾天城市", "g": "landscape", "sb": "高楼与道路", "box": [10, 20, 90, 80],
              "lt": "阴天散射光", "el": ["sky", "mist", "building"], "lk": "fog",
              "pr": ["画面偏灰"], "ps": ["远处高楼"]},
        "A": {"t": "自然通透", "w": "清晰还原", "tn": [3, 14, 48, 82, 96, 58],
              "dev": {"color": {"vibrance": 6}, "detail": {"sharpen_amount": 38}},
              "ms": [{"n": "sky", "p": "压雾", "s": ["lin", 50, 0, 50, 40],
                      "l": ["lum", 70, 100, 5], "a": {"dehaze": 12}, "tg": [20, 5, 80, 30, 74]}]},
        "v": {
            "B": {"d": "clean_tone", "t": "清透光影", "w": "通透立体", "tn": [1, 10, 52, 86, 98, 56],
                  "mm": {"sky": {"a": {"dehaze": 18}, "m": 70}}},
            "C": {"d": "subtle_film", "t": "轻量胶片", "w": "暖调胶片", "tn": [4, 16, 48, 82, 95, 55],
                  "lk": ["vintage_olive", 0.62]},
        },
    }
    std, ext_el = directed._expand_ultra_compact(compact_raw, ["A", "B", "C"])
    assert ext_el is not None and ext_el["light_kind"] == "fog" and "sky" in ext_el["elements"]
    assert std["plans"]["A"]["tone"]["mid"] == pytest.approx(0.48) and std["plans"]["A"]["tone"]["chroma"] == pytest.approx(5.8)
    # Regression guard: integer `1` in a 0..100 tone array must expand to 0.01 (NOT 1.0 pure white!)
    assert std["plans"]["B"]["tone"]["black"] == pytest.approx(0.01) and std["plans"]["B"]["tone"]["white"] == pytest.approx(0.98)
    assert std["plans"]["A"]["masks"][0]["shape"] == {"kind": "linear", "start": {"x": 0.5, "y": 0.0}, "end": {"x": 0.5, "y": 0.4}}
    assert std["plans"]["B"]["masks"][0]["adjust"]["dehaze"] == 18 and std["plans"]["B"]["masks"][0]["target"]["median"] == pytest.approx(0.70)
    assert std["plans"]["C"]["look"] == {"id": "vintage_olive", "strength": 0.62}
    assert std["plans"]["C"]["develop"]["detail"]["sharpen_amount"] == 38



