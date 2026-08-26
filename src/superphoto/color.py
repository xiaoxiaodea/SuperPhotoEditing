from __future__ import annotations

import numpy as np

EPS = 1e-6

# Rec.709 luma
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

HUE_CENTERS = {
    "red": 0.0,
    "orange": 30.0,
    "yellow": 60.0,
    "green": 120.0,
    "aqua": 180.0,
    "blue": 240.0,
    "purple": 270.0,
    "magenta": 300.0,
}


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.where(
        x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.minimum(x, 20.0), 1 / 2.4) - 0.055
    ).astype(np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ _LUMA


def revalue(rgb: np.ndarray, old: np.ndarray, new: np.ndarray) -> np.ndarray:
    scale = new / np.maximum(old, EPS)
    return rgb * scale[..., None]


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rgb_to_hsl(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    lum = (maxc + minc) / 2.0
    diff = maxc - minc
    sat = np.zeros_like(lum)
    chroma = diff > EPS
    sat[chroma] = diff[chroma] / np.maximum(1.0 - np.abs(2.0 * lum[chroma] - 1.0), EPS)
    hue = np.zeros_like(lum)
    mask_r = chroma & (maxc == r)
    mask_g = chroma & (maxc == g)
    mask_b = chroma & (maxc == b)
    hue[mask_r] = (g[mask_r] - b[mask_r]) / np.maximum(diff[mask_r], EPS)
    hue[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / np.maximum(diff[mask_g], EPS)
    hue[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / np.maximum(diff[mask_b], EPS)
    hue = (hue / 6.0) % 1.0
    return hue, sat, lum


def hsl_to_rgb(h: np.ndarray, s: np.ndarray, l: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    l = np.clip(l, 0.0, 1.0)
    h = np.mod(h, 1.0)
    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    def hue2rgb(p_, q_, t):
        t = np.mod(t, 1.0)
        out = np.copy(p_)
        out = np.where(t < 1 / 6, p_ + (q_ - p_) * 6 * t, out)
        out = np.where((t >= 1 / 6) & (t < 1 / 2), q_, out)
        out = np.where((t >= 1 / 2) & (t < 2 / 3), p_ + (q_ - p_) * (2 / 3 - t) * 6, out)
        return out

    r = hue2rgb(p, q, h + 1 / 3)
    g = hue2rgb(p, q, h)
    b = hue2rgb(p, q, h - 1 / 3)
    gray = s < 1e-5
    rgb = np.stack([r, g, b], axis=-1)
    rgb[gray] = l[gray][..., None]
    return rgb.astype(np.float32)


def hue_weights(hue01: np.ndarray, center_deg: float, width_deg: float = 50.0) -> np.ndarray:
    h = hue01 * 360.0
    d = np.abs((h - center_deg + 180.0) % 360.0 - 180.0)
    w = np.clip(1.0 - d / width_deg, 0.0, 1.0)
    return w * w


def circular_mean_hue(hue01: np.ndarray, weights: np.ndarray | None = None) -> float:
    ang = hue01 * 2 * np.pi
    if weights is None:
        x = np.mean(np.cos(ang))
        y = np.mean(np.sin(ang))
    else:
        w = weights.astype(np.float64)
        s = np.sum(w) + EPS
        x = np.sum(np.cos(ang) * w) / s
        y = np.sum(np.sin(ang) * w) / s
    return float((np.arctan2(y, x) / (2 * np.pi)) % 1.0)
