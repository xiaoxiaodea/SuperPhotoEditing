from __future__ import annotations

from superphoto.analysis import analyze, correction_from_technical
from superphoto.color import srgb_to_linear
from superphoto.ingest import load_linear


def test_underexposed_scene_wants_positive_ev(scene) -> None:
    linear, _ = load_linear(scene)
    tech = analyze(linear, scene)
    corr = correction_from_technical(tech)
    assert tech.mean_luma < 0.25
    assert corr.exposure_ev > 0.15


def test_blue_green_scene_is_landscape(bright_scene) -> None:
    linear, _ = load_linear(bright_scene)
    tech = analyze(linear, bright_scene)
    from superphoto.analysis import guess_genre

    assert guess_genre(tech) == "landscape"


def test_load_jpeg_is_linearish(scene) -> None:
    linear, meta = load_linear(scene)
    assert meta["source_kind"] == "jpeg"
    assert linear.dtype.kind == "f"
    # A mid sRGB value should not stay numerically equal after linearization.
    from PIL import Image
    import numpy as np

    srgb = np.asarray(Image.open(scene).convert("RGB"), dtype=np.float32) / 255.0
    assert not np.allclose(linear, srgb)
    assert np.allclose(linear, srgb_to_linear(srgb), atol=1e-5)
