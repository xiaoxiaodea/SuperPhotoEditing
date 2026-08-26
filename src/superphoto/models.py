from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

HueName = Literal["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
SourceKind = Literal["raw", "jpeg", "png", "tiff", "unknown"]
JobStatus = Literal["queued", "running", "ready", "error"]
JobStep = Literal[
    "queued",
    "ingest",
    "analyze",
    "critique",
    "render",
    "qa",
    "ready",
    "error",
]


class HueShift(BaseModel):
    hue: float = 0.0  # degrees, -30..30
    sat: float = 0.0  # -1..1
    lum: float = 0.0  # -1..1


class Grade(BaseModel):
    shadows: tuple[float, float, float] = (0.0, 0.0, 0.0)
    midtones: tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlights: tuple[float, float, float] = (0.0, 0.0, 0.0)


class LocalEdit(BaseModel):
    """Layered, mask-based finish. 0 is a no-op."""

    subject_clarity: float = 0.0
    subject_sharpen: float = 0.0
    subject_lift: float = 0.0
    subject_vibrance: float = 0.0
    background_darken: float = 0.0
    background_desat: float = 0.0
    sky_dehaze: float = 0.0
    color_clean: float = 0.0
    clutter_suppress: float = 0.0
    defect_heal: float = 0.0

    def is_noop(self) -> bool:
        return max(abs(v) for v in self.model_dump().values()) < 1e-4

    def merged(self, other: "LocalEdit") -> "LocalEdit":
        data = self.model_dump()
        for key, value in other.model_dump().items():
            data[key] = data[key] + value
        return LocalEdit.model_validate(data)


class RegionBox(BaseModel):
    left: float = 0.0
    top: float = 0.0
    right: float = 1.0
    bottom: float = 1.0
    note: str = ""
    kind: str = "clutter"


class ToneCurve(BaseModel):
    """Display-referred 1D curves. Points are (x, y) in 0..1."""

    rgb: list[tuple[float, float]] = Field(default_factory=list)
    r: list[tuple[float, float]] = Field(default_factory=list)
    g: list[tuple[float, float]] = Field(default_factory=list)
    b: list[tuple[float, float]] = Field(default_factory=list)

    def is_identity(self) -> bool:
        return not (self.rgb or self.r or self.g or self.b)


class Recipe(BaseModel):
    """Parametric edit. Units are Lightroom-like: 0 is no-op."""

    exposure_ev: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    vignette: float = 0.0
    grain: float = 0.0
    hsl: dict[str, HueShift] = Field(default_factory=dict)
    grade: Grade = Field(default_factory=Grade)
    curve: ToneCurve = Field(default_factory=ToneCurve)
    local: LocalEdit = Field(default_factory=LocalEdit)

    def merged(self, other: Recipe) -> Recipe:
        hsl = {k: v.model_copy() for k, v in self.hsl.items()}
        for name, shift in other.hsl.items():
            base = hsl.get(name, HueShift())
            hsl[name] = HueShift(
                hue=base.hue + shift.hue,
                sat=base.sat + shift.sat,
                lum=base.lum + shift.lum,
            )
        g1, g2 = self.grade, other.grade
        return Recipe(
            exposure_ev=self.exposure_ev + other.exposure_ev,
            temperature=self.temperature + other.temperature,
            tint=self.tint + other.tint,
            highlights=self.highlights + other.highlights,
            shadows=self.shadows + other.shadows,
            whites=self.whites + other.whites,
            blacks=self.blacks + other.blacks,
            contrast=self.contrast + other.contrast,
            saturation=self.saturation + other.saturation,
            vibrance=self.vibrance + other.vibrance,
            clarity=self.clarity + other.clarity,
            dehaze=self.dehaze + other.dehaze,
            vignette=self.vignette + other.vignette,
            grain=self.grain + other.grain,
            hsl=hsl,
            grade=Grade(
                shadows=_add3(g1.shadows, g2.shadows),
                midtones=_add3(g1.midtones, g2.midtones),
                highlights=_add3(g1.highlights, g2.highlights),
            ),
            curve=other.curve if not other.curve.is_identity() else self.curve,
            local=self.local.merged(other.local),
        )

    def clamped(self) -> Recipe:
        def clip(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        hsl = {
            k: HueShift(
                hue=clip(s.hue, -40, 40),
                sat=clip(s.sat, -1, 1),
                lum=clip(s.lum, -1, 1),
            )
            for k, s in self.hsl.items()
        }
        return Recipe(
            exposure_ev=clip(self.exposure_ev, -2.5, 2.5),
            temperature=clip(self.temperature, -80, 80),
            tint=clip(self.tint, -50, 50),
            highlights=clip(self.highlights, -1, 1),
            shadows=clip(self.shadows, -1, 1),
            whites=clip(self.whites, -1, 1),
            blacks=clip(self.blacks, -1, 1),
            contrast=clip(self.contrast, -0.8, 0.8),
            saturation=clip(self.saturation, -0.8, 0.8),
            vibrance=clip(self.vibrance, -0.8, 0.8),
            clarity=clip(self.clarity, -0.6, 0.8),
            dehaze=clip(self.dehaze, -0.4, 0.6),
            vignette=clip(self.vignette, -0.6, 0.4),
            grain=clip(self.grain, 0.0, 0.4),
            hsl=hsl,
            grade=Grade(
                shadows=_clip3(self.grade.shadows, -0.25),
                midtones=_clip3(self.grade.midtones, -0.2),
                highlights=_clip3(self.grade.highlights, -0.25),
            ),
            curve=self.curve,
            local=LocalEdit(
                subject_clarity=clip(self.local.subject_clarity, 0.0, 0.8),
                subject_sharpen=clip(self.local.subject_sharpen, 0.0, 0.8),
                subject_lift=clip(self.local.subject_lift, -0.08, 0.16),
                subject_vibrance=clip(self.local.subject_vibrance, -0.2, 0.35),
                background_darken=clip(self.local.background_darken, 0.0, 0.28),
                background_desat=clip(self.local.background_desat, 0.0, 0.45),
                sky_dehaze=clip(self.local.sky_dehaze, 0.0, 0.4),
                color_clean=clip(self.local.color_clean, 0.0, 0.7),
                clutter_suppress=clip(self.local.clutter_suppress, 0.0, 0.8),
                defect_heal=clip(self.local.defect_heal, 0.0, 1.0),
            ),
        )


class Look(BaseModel):
    id: str
    title: str
    title_en: str
    description: str
    genre_tags: list[str] = Field(default_factory=list)
    moves: list[str] = Field(default_factory=list)
    recipe: Recipe = Field(default_factory=Recipe)
    source: Literal["builtin", "xmp"] = "builtin"


class Issue(BaseModel):
    dim: str
    severity: float = Field(ge=0, le=1)
    note: str


class CorrectionHint(BaseModel):
    exposure_ev: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0

    def as_recipe(self) -> Recipe:
        return Recipe(**self.model_dump())


class GeometryHint(BaseModel):
    """Normalized crop box and a small straighten. apply=False means leave the frame."""

    rotate_deg: float = 0.0
    crop_left: float = 0.0
    crop_top: float = 0.0
    crop_right: float = 1.0
    crop_bottom: float = 1.0
    apply: bool = False
    why: str = ""

    def is_noop(self) -> bool:
        if not self.apply:
            return True
        return (
            abs(self.rotate_deg) < 0.08
            and self.crop_left <= 1e-3
            and self.crop_top <= 1e-3
            and self.crop_right >= 0.997
            and self.crop_bottom >= 0.997
        )


class Critique(BaseModel):
    genre: str = "general"
    summary: str = ""
    issues: list[Issue] = Field(default_factory=list)
    do_not: list[str] = Field(default_factory=list)
    correction: CorrectionHint = Field(default_factory=CorrectionHint)
    looks: list[str] = Field(default_factory=list)
    source: Literal["ollama", "grok", "local"] = "local"
    lut_mix: float = 0.55
    geometry: GeometryHint = Field(default_factory=GeometryHint)
    clutter: list[RegionBox] = Field(default_factory=list)
    subject_note: str = ""


class TechnicalReport(BaseModel):
    width: int
    height: int
    source_kind: SourceKind
    mean_luma: float
    median_luma: float
    std_luma: float
    highlight_clip: float
    shadow_clip: float
    dynamic_spread: float
    sharpness: float
    noise: float
    saturation_mean: float
    temperature_bias: float
    top_third_blue: float
    skin_fraction: float
    green_fraction: float
    histogram: list[float]
    shadow_p05: float = 0.0
    highlight_p95: float = 1.0
    exif: dict[str, str] = Field(default_factory=dict)
    suggested_exposure_ev: float = 0.0
    aqua_fraction: float = 0.0
    magenta_fraction: float = 0.0
    orange_fraction: float = 0.0
    horizon_tilt_deg: float = 0.0
    horizon_confidence: float = 0.0
    subject_left: float = 0.0
    subject_top: float = 0.0
    subject_right: float = 1.0
    subject_bottom: float = 1.0
    margin_waste: float = 0.0
    thirds_error: float = 0.0


class VariantPlan(BaseModel):
    slot: Literal["A", "B", "C"]
    look_id: str
    title: str
    why: str
    recipe: Recipe
    moves: list[str] = Field(default_factory=list)
    lut_mix: float = 1.0
    geometry: GeometryHint = Field(default_factory=GeometryHint)


class FeedbackIn(BaseModel):
    chosen_slot: Literal["A", "B", "C"]
    technical_score: int = Field(ge=1, le=5, default=3)
    style_score: int = Field(ge=1, le=5, default=3)
    restraint_score: int = Field(ge=1, le=5, default=3)
    notes: str = ""


def _add3(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _clip3(a: tuple[float, float, float], bound: float) -> tuple[float, float, float]:
    hi = abs(bound)
    return (
        max(-hi, min(hi, a[0])),
        max(-hi, min(hi, a[1])),
        max(-hi, min(hi, a[2])),
    )
