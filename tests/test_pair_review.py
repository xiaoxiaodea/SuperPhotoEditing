import base64
import hashlib
from io import BytesIO
import json

from PIL import Image
import pytest

from superphoto import pair_review as pair
from superphoto.config import Settings
from superphoto.models import EditIntent


@pytest.fixture
def photos(tmp_path):
    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    Image.new("RGB", (120, 80), (200, 100, 70)).save(first)
    Image.new("RGB", (120, 80), (60, 120, 200)).save(second)
    return first, second


def _answer(choice, confidence=.9, *, label="forward"):
    return {"choice": choice, "confidence": confidence, "evidence": [{
        "region": "中央受光区域", "image_1": f"{label}图一的可见受光表现",
        "image_2": f"{label}图二的可见受光表现", "reason": "比较局部光线与整体关系，强度本身不代表优劣"}]}


def _reference_answer(choice, confidence=.9):
    response = _answer(choice, confidence)
    response["evidence"][0].update(
        reference_observation="R中央明亮光带与两侧暗部明确分开，但亮部仍有纹理",
        reason="根据两版与R同一区域光带和暗部组织的接近程度判断，不以更浓更暗作为标准")
    return response


@pytest.fixture
def vision(monkeypatch):
    calls = []
    monkeypatch.setattr(pair, "probe", lambda _: {"ready": True, "resolved_model": "test-vision"})

    def install(answers):
        replies = iter(answers)

        def chat(settings, model, prompt, images, *, response_schema, audit):
            assert len(images) == 1
            binary = base64.b64decode(images[0])
            with Image.open(BytesIO(binary)) as board:
                image_info = {"sha256": hashlib.sha256(binary).hexdigest(),
                              "width": board.width, "height": board.height}
            audit["request"] = {"prompt": prompt, "images": [image_info], "format": response_schema}
            calls.append({"prompt": prompt, "images": images, "audit": audit})
            response = next(replies)
            if isinstance(response, Exception):
                audit["error"] = {"type": type(response).__name__}
                raise response
            raw = response if isinstance(response, str) else json.dumps(response)
            audit["response"] = {"done": True, "message": {"content": raw}}
            return raw

        monkeypatch.setattr(pair, "chat_vision", chat)
        return calls
    return install


def test_board_numbers_equal_cells_preserve_ratio_and_source_files(photos):
    first, second = photos
    originals = [path.read_bytes() for path in photos]
    first_preview, second_preview = pair._preview(first), pair._preview(second)
    board_b64 = pair.build_pair_board(first_preview, second_preview)
    with Image.open(BytesIO(base64.b64decode(board_b64))) as board:
        assert board.size == (1920, 693)  # 950 × 633 image cells plus the 60px label/margins.
        assert board.info.get("icc_profile")
        assert board.getpixel((480, 340))[0] > board.getpixel((480, 340))[2]
        assert board.getpixel((1440, 340))[2] > board.getpixel((1440, 340))[0]
        assert all(abs(channel - 128) <= 2 for channel in board.getpixel((960, 340)))
        # Both label strips contain visible white numbering on the neutral board.
        assert board.crop((455, 5, 505, 50)).convert("L").getextrema()[1] > 230
        assert board.crop((1415, 5, 1465, 50)).convert("L").getextrema()[1] > 230
    assert [path.read_bytes() for path in photos] == originals


def test_board_does_not_stretch_portrait_and_keeps_reference_in_fixed_lower_row(photos, tmp_path):
    portrait = tmp_path / "portrait.jpg"
    Image.new("RGB", (60, 120), (30, 210, 90)).save(portrait)
    previews = [pair._preview(portrait), pair._preview(photos[0])]
    reference = pair._preview(photos[1])
    boards = [Image.open(BytesIO(base64.b64decode(pair.build_pair_board(*items, reference))))
              for items in (previews, previews[::-1])]
    try:
        assert boards[0].size == boards[1].size
        # A 1:2 portrait fits to 325×650, centered within its 950px-wide cell.
        assert all(abs(channel - 128) <= 3 for channel in boards[0].getpixel((100, 340)))
        assert boards[0].getpixel((480, 340))[1] > 190
        assert boards[1].getpixel((1440, 340))[1] > 190
        row_start = 710 + 10
        # JPEG MCU boundaries can couple a few edge pixels; the reference's body
        # must have the same appearance and position after reversing the top row.
        assert boards[0].getpixel((960, row_start + 300)) == boards[1].getpixel((960, row_start + 300))
    finally:
        for board in boards:
            board.close()


@pytest.mark.parametrize("choices,expected_index", [([2, 1], 1), ([1, 2], 0)])
def test_both_orders_must_map_to_the_same_physical_winner(photos, vision, choices, expected_index):
    calls = vision([_answer(choices[0], .95), _answer(choices[1], .8, label="reverse")])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in photos]
    assert result.outcome == "winner" and result.winner_id == hashes[expected_index]
    assert result.first_id == hashes[0] and result.second_id == hashes[1]
    assert result.confidence == .8 and result.source == "ollama-pair"
    assert [order.image_ids for order in result.orders] == [hashes, hashes[::-1]]
    assert result.orders[0].preview_hashes == result.orders[1].preview_hashes[::-1]
    assert calls[0]["prompt"] == calls[1]["prompt"]
    assert result.orders[0].board_sha256 != result.orders[1].board_sha256
    for order, call in zip(result.orders, calls):
        assert order.transport_audit == call["audit"]
        assert order.board_sha256 == call["audit"]["request"]["images"][0]["sha256"]
        assert json.loads(order.raw_response)["choice"] == order.choice


@pytest.mark.parametrize("choice", [1, 2])
def test_always_preferring_a_display_position_is_uncertain_not_a_physical_winner(photos, vision, choice):
    vision([_answer(choice), _answer(choice, label="reverse")])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    assert result.outcome == "uncertain" and result.winner_id is None
    assert result.confidence == 0
    assert "胜者改变" in result.notes[0]


def test_reversing_api_arguments_does_not_change_the_physical_winner(photos, vision):
    vision([_answer(2), _answer(1), _answer(1), _answer(2)])
    forward = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    reverse = pair.compare_finished_pair(Settings(), *photos[::-1], EditIntent())
    assert forward.outcome == reverse.outcome == "winner"
    assert forward.winner_id == reverse.winner_id == hashlib.sha256(photos[1].read_bytes()).hexdigest()
    assert forward.first_id == reverse.second_id


def test_evidence_stays_attached_to_its_physical_image_in_each_order(photos, vision):
    vision([_answer(2, label="ab"), _answer(1, label="ba")])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    observations = []
    for order in result.orders:
        evidence = order.evidence[0]
        observations.append({order.image_ids[0]: evidence.image_1, order.image_ids[1]: evidence.image_2})
    assert observations[0][result.first_id] == "ab图一的可见受光表现"
    assert observations[1][result.first_id] == "ba图二的可见受光表现"
    assert observations[0][result.second_id] == "ab图二的可见受光表现"
    assert observations[1][result.second_id] == "ba图一的可见受光表现"


def test_two_explicit_ties_are_distinct_from_inability_to_judge(photos, vision):
    vision([_answer("tie", .95), _answer("tie", .85)])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    assert result.outcome == "tie" and result.winner_id is None and result.confidence == .85


@pytest.mark.parametrize("answers", [
    [_answer(2), _answer("tie")],
    [_answer("tie"), _answer(1)],
    [_answer("uncertain"), _answer("uncertain")],
    [_answer(2, .95), _answer(1, .74)],
    [_answer("tie", .7), _answer("tie", .9)],
    [_answer(2, 0), _answer(1, .99)],
])
def test_partial_agreement_or_low_confidence_remains_uncertain(photos, vision, answers):
    vision(answers)
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    assert result.outcome == "uncertain" and result.winner_id is None


@pytest.mark.parametrize("failed_response", [RuntimeError("vision unavailable"), "not JSON", "{}",
                                           _answer(1, float("nan")), _answer(1, float("inf")),
                                           {**_answer(1), "choice": True},
                                           {**_answer(1), "evidence": []}])
def test_second_order_failure_cannot_promote_first_order_and_preserves_audit(photos, vision, failed_response):
    calls = vision([_answer(2), failed_response])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    assert result.outcome == "unavailable" and result.winner_id is None
    assert result.source == "unavailable" and len(result.orders) == 2
    assert result.orders[0].choice == 2 and result.orders[1].error
    assert result.orders[1].transport_audit == calls[1]["audit"]
    if not isinstance(failed_response, Exception):
        assert result.orders[1].raw_response


def test_identical_previews_skip_probe_and_cannot_have_a_winner(photos, monkeypatch):
    preview = pair._preview(photos[0])
    monkeypatch.setattr(pair, "_preview", lambda _: preview)
    monkeypatch.setattr(pair, "probe", lambda _: pytest.fail("same preview requires no model"))
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    assert result.outcome == "tie" and result.source == "identical-render"
    assert result.winner_id is None and result.orders == []
    assert result.first_id != result.second_id  # Equality is explicitly at preview scale.


def test_prompt_is_role_neutral_and_omits_prior_reference_or_recipe_claims(photos, vision):
    calls = vision([_answer(2), _answer(1)])
    intent = EditIntent(reference_ids=["famous-author"], reference_checks=["previous-praise"])
    pair.compare_finished_pair(Settings(), *photos, intent)
    prompt = calls[0]["prompt"]
    assert not any(word in prompt for word in ("famous-author", "previous-praise", "候选", "当前版本", "精修前", "原片", "QA"))
    assert "更暗、更亮、更浓、更锐或变化更大都不自动代表更好" in prompt
    assert len(calls[0]["images"]) == len(calls[1]["images"]) == 1


def test_optional_reference_uses_same_board_location_and_same_prompt(photos, vision):
    calls = vision([_reference_answer(2), _reference_answer(1)])
    reference = pair._preview(photos[0])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent(), reference_preview=reference)
    assert result.outcome == "winner"
    assert calls[0]["prompt"] == calls[1]["prompt"] and "下排标记R" in calls[0]["prompt"]
    assert calls[0]["audit"]["request"]["images"][0]["height"] > 1000


@pytest.mark.parametrize("missing_reference", ["missing", "", "   ", None])
def test_reference_comparison_without_actual_r_observation_is_unavailable_not_tie(photos, vision, missing_reference):
    invalid = _reference_answer("tie")
    if missing_reference == "missing":
        del invalid["evidence"][0]["reference_observation"]
    else:
        invalid["evidence"][0]["reference_observation"] = missing_reference
    vision([_reference_answer("tie"), invalid])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent(),
                                         reference_preview=pair._preview(photos[0]))
    assert result.outcome == "unavailable" and result.winner_id is None
    assert result.orders[0].choice == "tie" and result.orders[1].error
    assert result.orders[1].raw_response == json.dumps(invalid)


def test_reference_schema_requires_r_evidence_and_prompt_uses_intent_only_as_constraints(photos, vision):
    calls = vision([_reference_answer(2), _reference_answer(1)])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent(natural_color=True),
                                         reference_preview=pair._preview(photos[0]))
    schema = calls[0]["audit"]["request"]["format"]
    evidence_type = schema["properties"]["evidence"]["items"]["$ref"].rsplit("/", 1)[1]
    assert "reference_observation" in schema["$defs"][evidence_type]["required"]
    assert schema["$defs"][evidence_type]["properties"]["reference_observation"]["minLength"] == 1
    assert "任务是判断哪版更接近R可见的明暗关系" in calls[0]["prompt"]
    assert "共同意图只作保护约束，不能替代R观察" in calls[0]["prompt"]
    assert "reason必须解释两版与R的关系" in calls[0]["prompt"]
    assert all(order.evidence[0].reference_observation.startswith("R中央") for order in result.orders)
    assert all(order["evidence"][0]["reference_observation"] for order in result.model_dump()["orders"])


def test_reference_evidence_can_support_a_real_tie_without_forcing_a_winner(photos, vision):
    vision([_reference_answer("tie"), _reference_answer("tie")])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent(),
                                         reference_preview=pair._preview(photos[0]))
    assert result.outcome == "tie" and result.winner_id is None


def test_ordinary_comparison_does_not_require_or_invent_a_reference(photos, vision):
    calls = vision([_answer(2), _answer(1)])
    result = pair.compare_finished_pair(Settings(), *photos, EditIntent())
    assert result.outcome == "winner"
    assert all(order.evidence[0].reference_observation is None for order in result.orders)
    schema = calls[0]["audit"]["request"]["format"]
    evidence_type = schema["properties"]["evidence"]["items"]["$ref"].rsplit("/", 1)[1]
    assert "reference_observation" not in schema["$defs"][evidence_type]["required"]
    assert "参考接近程度比较" not in calls[0]["prompt"]
    assert "共同意图：" in calls[0]["prompt"]


@pytest.mark.parametrize("enabled,ready", [(False, True), (True, False)])
def test_disabled_or_missing_model_does_not_assert_a_quality_order(photos, monkeypatch, enabled, ready):
    monkeypatch.setattr(pair, "probe", lambda _: {"ready": ready})
    monkeypatch.setattr(pair, "chat_vision", lambda *a, **kw: pytest.fail("model should not run"))
    result = pair.compare_finished_pair(Settings(visual_review_enabled=enabled), *photos, EditIntent())
    assert result.outcome == "unavailable" and result.winner_id is None
