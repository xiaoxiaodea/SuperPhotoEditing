from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

HueName = Literal["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
SourceKind = Literal["raw", "jpeg", "png", "tiff", "unknown"]
JobStatus = Literal["queued", "running", "ready", "error"]
JobStep = Literal[
    "queued",
    "ingest",
    "analyze",
    "restore",
    "critique",
    "render",
    "qa",
    "lightroom",
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


class MaskAdjustment(BaseModel):
    """Absolute, bounded local settings on a native semantic Lightroom mask."""

    target: Literal["subject", "sky", "background"]
    exposure_ev: float = Field(default=0.0, ge=-2.0, le=2.0)
    contrast: float = Field(default=0.0, ge=-0.50, le=0.50)
    whites: float = Field(default=0.0, ge=-0.75, le=0.75)
    blacks: float = Field(default=0.0, ge=-0.50, le=0.50)
    highlights: float = Field(default=0.0, ge=-0.40, le=0.20)
    shadows: float = Field(default=0.0, ge=-0.25, le=0.35)
    temperature: float = Field(default=0.0, ge=-12.0, le=12.0)
    tint: float = Field(default=0.0, ge=-10.0, le=10.0)
    saturation: float = Field(default=0.0, ge=-0.25, le=0.15)
    clarity: float = Field(default=0.0, ge=-0.15, le=0.15)
    texture: float = Field(default=0.0, ge=-0.20, le=0.20)
    dehaze: float = Field(default=0.0, ge=-0.10, le=0.15)
    reason: str = ""

    def is_noop(self) -> bool:
        return all(abs(v) < 1e-4 for v in self.model_dump(exclude={"target", "reason"}).values())

    def for_source(self, source: SourceKind) -> "MaskAdjustment":
        """Keep RAW sky recovery out of developed files and other targets."""
        raw_sky = source == "raw" and self.target == "sky"
        limits = {"exposure_ev": (-2 if raw_sky else -.75, .75),
                  "contrast": (-.50, .50) if raw_sky else (-.30, .30),
                  "whites": (-.75, .75) if raw_sky else (-.30, .30),
                  "blacks": (-.50, .50) if raw_sky else (-.25, .25)}
        return self.model_copy(update={key: max(lo, min(hi, getattr(self, key)))
                                       for key, (lo, hi) in limits.items()})


class EditIntent(BaseModel):
    goal: str = "保留现场光色，让主体与环境的层次更清楚"
    preserve: list[str] = Field(default_factory=list, max_length=6)
    natural_color: bool = True
    preserve_atmosphere: bool = True
    preserve_black_depth: bool = True
    locked_controls: list[Literal["exposure_ev", "temperature", "tint", "hue", "grain", "clarity", "dehaze"]] = Field(default_factory=list, max_length=7)
    subject_box: RegionBox | None = None
    subject_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    masks: list[MaskAdjustment] = Field(default_factory=list, max_length=3)
    reference_ids: list[str] = Field(default_factory=list, max_length=3)
    reference_checks: list[str] = Field(default_factory=list, max_length=4)
    resource_lesson_ids: list[str] = Field(default_factory=list, max_length=12)
    resource_checks: list[str] = Field(default_factory=list, max_length=4)


class ParametricTone(BaseModel):
    """Additive Lightroom parametric-curve moves; split points are absolute."""

    shadows: float = 0.0
    darks: float = 0.0
    lights: float = 0.0
    highlights: float = 0.0
    shadow_split: float | None = None
    midtone_split: float | None = None
    highlight_split: float | None = None

    def merged(self, other: "ParametricTone") -> "ParametricTone":
        return ParametricTone(
            shadows=self.shadows + other.shadows,
            darks=self.darks + other.darks,
            lights=self.lights + other.lights,
            highlights=self.highlights + other.highlights,
            shadow_split=other.shadow_split if other.shadow_split is not None else self.shadow_split,
            midtone_split=other.midtone_split if other.midtone_split is not None else self.midtone_split,
            highlight_split=other.highlight_split if other.highlight_split is not None else self.highlight_split,
        )


class DetailControls(BaseModel):
    """Absolute Lightroom detail-panel values; None preserves the current value."""

    sharpness: float | None = None
    sharpen_radius: float | None = None
    sharpen_detail: float | None = None
    sharpen_masking: float | None = None
    luminance_nr: float | None = None
    luminance_nr_detail: float | None = None
    luminance_nr_contrast: float | None = None
    color_nr: float | None = None
    color_nr_detail: float | None = None
    color_nr_smoothness: float | None = None


class OpticsControls(BaseModel):
    """Absolute lens and perspective settings; None leaves Lightroom untouched."""

    lens_profile_enable: bool | None = None
    manual_distortion: float | None = None
    perspective_vertical: float | None = None
    perspective_horizontal: float | None = None
    perspective_rotate: float | None = None
    perspective_scale: float | None = None
    perspective_aspect: float | None = None
    perspective_upright: int | None = None


class FinishControls(BaseModel):
    """Absolute shape controls for vignette and grain."""

    vignette_midpoint: float | None = None
    vignette_roundness: float | None = None
    vignette_feather: float | None = None
    vignette_style: int | None = None
    grain_size: float | None = None
    grain_roughness: float | None = None


class CropControls(BaseModel):
    """Normalized Lightroom crop and straighten values."""

    top: float | None = None
    left: float | None = None
    bottom: float | None = None
    right: float | None = None
    angle: float | None = None


class ColorGradingControls(BaseModel):
    """Native Lightroom color wheels; None preserves the existing value."""
    shadow_hue: float | None = Field(default=None, ge=0, le=360)
    shadow_sat: float | None = Field(default=None, ge=0, le=100)
    shadow_lum: float | None = Field(default=None, ge=-100, le=100)
    midtone_hue: float | None = Field(default=None, ge=0, le=360)
    midtone_sat: float | None = Field(default=None, ge=0, le=100)
    midtone_lum: float | None = Field(default=None, ge=-100, le=100)
    highlight_hue: float | None = Field(default=None, ge=0, le=360)
    highlight_sat: float | None = Field(default=None, ge=0, le=100)
    highlight_lum: float | None = Field(default=None, ge=-100, le=100)
    blending: float | None = Field(default=None, ge=0, le=100)
    balance: float | None = Field(default=None, ge=-100, le=100)


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
    texture: float = 0.0
    clarity: float = 0.0
    dehaze: float = 0.0
    vignette: float = 0.0
    grain: float = 0.0
    hsl: dict[str, HueShift] = Field(default_factory=dict)
    grade: Grade = Field(default_factory=Grade)
    color_grading: ColorGradingControls = Field(default_factory=ColorGradingControls)
    curve: ToneCurve = Field(default_factory=ToneCurve)
    parametric_tone: ParametricTone = Field(default_factory=ParametricTone)
    detail: DetailControls = Field(default_factory=DetailControls)
    optics: OpticsControls = Field(default_factory=OpticsControls)
    finish: FinishControls = Field(default_factory=FinishControls)
    crop: CropControls = Field(default_factory=CropControls)
    local: LocalEdit = Field(default_factory=LocalEdit)
    masks: list[MaskAdjustment] = Field(default_factory=list, max_length=3)

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
            texture=self.texture + other.texture,
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
            color_grading=_overlay_optional(self.color_grading, other.color_grading),
            parametric_tone=self.parametric_tone.merged(other.parametric_tone),
            detail=_overlay_optional(self.detail, other.detail),
            optics=_overlay_optional(self.optics, other.optics),
            finish=_overlay_optional(self.finish, other.finish),
            crop=_overlay_optional(self.crop, other.crop),
            local=self.local.merged(other.local),
            masks=list({m.target: m.model_copy(deep=True) for m in [*self.masks, *other.masks]}.values()),
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
            texture=clip(self.texture, -0.8, 0.8),
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
            color_grading=self.color_grading.model_copy(deep=True),
            parametric_tone=ParametricTone(
                shadows=clip(self.parametric_tone.shadows, -1.0, 1.0),
                darks=clip(self.parametric_tone.darks, -1.0, 1.0),
                lights=clip(self.parametric_tone.lights, -1.0, 1.0),
                highlights=clip(self.parametric_tone.highlights, -1.0, 1.0),
                shadow_split=_clip_optional(self.parametric_tone.shadow_split, 0.0, 100.0),
                midtone_split=_clip_optional(self.parametric_tone.midtone_split, 0.0, 100.0),
                highlight_split=_clip_optional(self.parametric_tone.highlight_split, 0.0, 100.0),
            ),
            detail=DetailControls(
                sharpness=_clip_optional(self.detail.sharpness, 0.0, 150.0),
                sharpen_radius=_clip_optional(self.detail.sharpen_radius, 0.5, 3.0),
                sharpen_detail=_clip_optional(self.detail.sharpen_detail, 0.0, 100.0),
                sharpen_masking=_clip_optional(self.detail.sharpen_masking, 0.0, 100.0),
                luminance_nr=_clip_optional(self.detail.luminance_nr, 0.0, 100.0),
                luminance_nr_detail=_clip_optional(self.detail.luminance_nr_detail, 0.0, 100.0),
                luminance_nr_contrast=_clip_optional(self.detail.luminance_nr_contrast, 0.0, 100.0),
                color_nr=_clip_optional(self.detail.color_nr, 0.0, 100.0),
                color_nr_detail=_clip_optional(self.detail.color_nr_detail, 0.0, 100.0),
                color_nr_smoothness=_clip_optional(self.detail.color_nr_smoothness, 0.0, 100.0),
            ),
            optics=OpticsControls(
                lens_profile_enable=self.optics.lens_profile_enable,
                manual_distortion=_clip_optional(self.optics.manual_distortion, -100.0, 100.0),
                perspective_vertical=_clip_optional(self.optics.perspective_vertical, -100.0, 100.0),
                perspective_horizontal=_clip_optional(self.optics.perspective_horizontal, -100.0, 100.0),
                perspective_rotate=_clip_optional(self.optics.perspective_rotate, -10.0, 10.0),
                perspective_scale=_clip_optional(self.optics.perspective_scale, 50.0, 150.0),
                perspective_aspect=_clip_optional(self.optics.perspective_aspect, -100.0, 100.0),
                perspective_upright=(
                    int(clip(float(self.optics.perspective_upright), 0.0, 5.0))
                    if self.optics.perspective_upright is not None
                    else None
                ),
            ),
            finish=FinishControls(
                vignette_midpoint=_clip_optional(self.finish.vignette_midpoint, 0.0, 100.0),
                vignette_roundness=_clip_optional(self.finish.vignette_roundness, -100.0, 100.0),
                vignette_feather=_clip_optional(self.finish.vignette_feather, 0.0, 100.0),
                vignette_style=(
                    int(clip(float(self.finish.vignette_style), 1.0, 3.0))
                    if self.finish.vignette_style is not None
                    else None
                ),
                grain_size=_clip_optional(self.finish.grain_size, 0.0, 100.0),
                grain_roughness=_clip_optional(self.finish.grain_roughness, 0.0, 100.0),
            ),
            crop=CropControls(
                top=_clip_optional(self.crop.top, 0.0, 1.0),
                left=_clip_optional(self.crop.left, 0.0, 1.0),
                bottom=_clip_optional(self.crop.bottom, 0.0, 1.0),
                right=_clip_optional(self.crop.right, 0.0, 1.0),
                angle=_clip_optional(self.crop.angle, -45.0, 45.0),
            ),
            masks=[m.model_copy(deep=True) for m in self.masks],
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
    source: Literal["builtin", "xmp", "master"] = "builtin"


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


class ArtDirection(BaseModel):
    slot: Literal["A", "B", "C"]
    method_id: str = ""
    look_id: str = ""
    preset_amount: float = Field(default=0.0, ge=0, le=1)
    title: str
    goal: str
    moves: list[str] = Field(default_factory=list, max_length=5)
    recipe: Recipe = Field(default_factory=Recipe)
    resource_adjustments: list[dict] = Field(default_factory=list)
    source: Literal["ollama", "openai", "xai", "vault", "scene"] = "scene"
    vision_provider: str = ""
    transport_audit: dict = Field(default_factory=dict)


class Critique(BaseModel):
    genre: str = "general"
    summary: str = ""
    issues: list[Issue] = Field(default_factory=list)
    do_not: list[str] = Field(default_factory=list)
    correction: CorrectionHint = Field(default_factory=CorrectionHint)
    looks: list[str] = Field(default_factory=list)
    # grok remains valid for previously saved jobs; new calls use actual providers.
    # vault is a model configured in the client; its identity stays in the vault.
    source: Literal["ollama", "grok", "openai", "xai", "vault", "local"] = "local"
    vision_provider: str = ""
    transport_audit: dict = Field(default_factory=dict)
    lut_mix: float = 0.55
    geometry: GeometryHint = Field(default_factory=GeometryHint)
    clutter: list[RegionBox] = Field(default_factory=list)
    subject_note: str = ""
    intent: EditIntent = Field(default_factory=EditIntent)
    directions: list[ArtDirection] = Field(default_factory=list, max_length=3)


class TechnicalReport(BaseModel):
    width: int
    height: int
    source_kind: SourceKind
    mean_luma: float
    median_luma: float
    std_luma: float
    highlight_clip: float
    shadow_clip: float
    black_clip: float = 0.0
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
    # Linear-light measurements excluding clipped channels and deep shadows.
    # None distinguishes old reports from measured zero; white areas alone do
    # not establish that the remaining subject is overexposed.
    unclipped_fraction: float = 0.0
    unclipped_median_luma: float | None = None
    white_clip_fraction: float = 0.0
    chromatic_fraction: float = 0.0
    chromatic_median_luma: float | None = None
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


EthicsRegime = Literal["documentary", "portrait", "landscape", "product", "commercial"]


class StyleMatch(BaseModel):
    look_id: str
    title: str
    match_score: float = Field(ge=0, le=1)
    reason: str
    family: str = ""
    source: str = ""
    evidence: dict[str, float] = Field(default_factory=dict)
    resource_lesson_ids: list[str] = Field(default_factory=list, max_length=12)


class StyleDecision(BaseModel):
    tone: str
    color: str
    purpose: str
    route_id: str
    route_title: str
    route_reason: str
    look_ids: list[str] = Field(default_factory=list)
    candidates: list[StyleMatch] = Field(default_factory=list)
    ranked_candidates: list[StyleMatch] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    selection_version: int = 2
    workflow_steps: list[str] = Field(default_factory=list)
    manual_gaps: list[str] = Field(default_factory=list)


class VariantPlan(BaseModel):
    slot: Literal["A", "B", "C"]
    method_id: str = ""
    look_id: str
    title: str
    why: str
    recipe: Recipe
    moves: list[str] = Field(default_factory=list)
    lut_mix: float = 1.0
    geometry: GeometryHint = Field(default_factory=GeometryHint)
    primary: Recipe = Field(default_factory=Recipe)
    ethics: EthicsRegime = "commercial"
    genre: str = "general"
    master: dict | None = None
    route_label: str = ""
    match_score: float = 0.0
    adaptation_notes: list[str] = Field(default_factory=list)
    manual_gaps: list[str] = Field(default_factory=list)
    intent: EditIntent = Field(default_factory=EditIntent)


# A directed photo carries up to nine versions (db.SLOT_LETTERS); the legacy pipeline still makes A/B/C.
Slot = Literal["A", "B", "C", "D", "E", "F", "G", "H", "I"]


class RatingIn(BaseModel):
    # pass / keep / favorite; like and dislike are kept for older clients and mean keep and pass.
    verdict: Literal["pass", "keep", "favorite", "like", "dislike"] | None


class RefinementIn(BaseModel):
    slot: Slot
    prompt: str = Field(min_length=2, max_length=1000)
    request_id: UUID
    parent_id: UUID | None = None

    @field_validator("prompt", mode="before")
    @classmethod
    def trim_prompt(cls, value):
        return value.strip() if isinstance(value, str) else value


class FullExportIn(BaseModel):
    # Full-size delivery of one finish: the version shown (a refinement id, or none for the first finish).
    slot: Slot
    version: UUID | None = None
    format: Literal["jpeg", "tiff"] = "jpeg"


class FeedbackIn(BaseModel):
    chosen_slot: Slot
    reaction: Literal[
        "keep",
        "too_dark",
        "too_bright",
        "too_strong",
        "too_flat",
        "color_wrong",
        "subject_weak",
    ] = "keep"
    # Kept for API compatibility. The simple reaction is converted into these
    # learning signals when the caller does not provide expert ratings.
    technical_score: int | None = Field(ge=1, le=5, default=None)
    style_score: int | None = Field(ge=1, le=5, default=None)
    restraint_score: int | None = Field(ge=1, le=5, default=None)
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


def _clip_optional(value: float | None, lo: float, hi: float) -> float | None:
    if value is None:
        return None
    return max(lo, min(hi, float(value)))


def _overlay_optional(base: BaseModel, other: BaseModel):
    """Preserve absolute controls unless the later recipe explicitly supplies one."""
    data = base.model_dump()
    for key, value in other.model_dump().items():
        if value is not None:
            data[key] = value
    return type(base).model_validate(data)
