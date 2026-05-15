"""
Chroma Clean node for ComfyUI (Badman).

Refines a primary matte (from BiRefNet / RMBG / similar segmentation nets)
for a known-color backdrop. Two operations in one pass:

  1. Chroma punch-out — pixels inside the matte that are actually the
     backdrop color (LAB distance below threshold) are removed. Fixes the
     classic "segmentation net filled in a cavity" failure, e.g. green
     leaking through a hole between hair strands.

  2. Despill — limits the dominant channel of the backdrop color so it
     cannot exceed the other two. Kills the color tint on edge halos
     where the backdrop bled into the subject.

Input mask convention: 1=foreground, 0=background (BiRefNet / RMBG default).
Flip invert_input_mask if wiring from LoadImage's alpha instead.
"""

from __future__ import annotations

import numpy as np
import torch
import cv2


def _hex_to_rgb(hex_str: str, default=(0, 254, 0)) -> tuple[int, int, int]:
    s = (hex_str or "").strip().lstrip("#")
    try:
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) != 6:
            return default
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return default


def _bg_similarity_lab(rgb_u8: np.ndarray, bg_rgb_u8: tuple[int, int, int],
                       threshold: float, feather: float) -> np.ndarray:
    """
    Per-pixel 0..1 where 1 = pixel is essentially bg_color, 0 = far from it.
    Distance measured in LAB so "how close to the backdrop" is perceptual
    rather than raw-RGB (which would over-count hue and under-count luminance).

        dist <= threshold                      -> 1
        threshold < dist < threshold + feather -> linear ramp
        dist >= threshold + feather            -> 0
    """
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_patch = np.array([[bg_rgb_u8]], dtype=np.uint8)
    bg_lab = cv2.cvtColor(bg_patch, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    diff = lab - bg_lab
    dist = np.sqrt((diff * diff).sum(axis=-1))
    ramp = max(float(feather), 1e-3)
    return np.clip((threshold + ramp - dist) / ramp, 0.0, 1.0)


def _despill_channel_limit(rgb_f: np.ndarray, bg_rgb_u8: tuple[int, int, int],
                           strength: float) -> np.ndarray:
    """
    Standard channel-limit despill: where the dominant backdrop channel
    exceeds the other two, pull it down toward that ceiling.

    Applied unconditionally — a character pixel whose dominant channel
    matches the backdrop's will also be dampened. Drop strength toward 0
    on characters whose palette overlaps the backdrop hue.
    """
    if strength <= 0:
        return rgb_f
    dom = int(np.argmax(bg_rgb_u8))  # 0=R, 1=G, 2=B
    others = [i for i in range(3) if i != dom]
    other_max = np.maximum(rgb_f[..., others[0]], rgb_f[..., others[1]])
    excess = np.maximum(rgb_f[..., dom] - other_max, 0.0)
    out = rgb_f.copy()
    out[..., dom] = rgb_f[..., dom] - float(strength) * excess
    return out


class ChromaCleanNode:
    """Punch out chroma leaks inside a matte and despill color contamination."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha_mask": ("MASK",),
                "bg_color": ("STRING", {
                    "default": "#00FE00",
                    "tooltip": "Hex of the backdrop color to remove (e.g. #00FE00 greenscreen, #0047AB bluescreen).",
                }),
                "chroma_threshold": ("FLOAT", {
                    "default": 28.0, "min": 0.0, "max": 120.0, "step": 0.5,
                    "tooltip": "LAB distance: pixels closer than this to bg_color are treated as backdrop.",
                }),
                "chroma_feather": ("FLOAT", {
                    "default": 12.0, "min": 0.1, "max": 60.0, "step": 0.5,
                    "tooltip": "Width of the soft ramp above threshold. Higher = softer edge.",
                }),
                "despill_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0 = off. 1 = clamp dominant bg channel to max of the other two.",
                }),
                "invert_input_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "BiRefNet/RMBG use 1=foreground (leave OFF). LoadImage alpha uses 1=background (turn ON).",
                }),
                "invert_output_mask": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Output as 1=foreground (OFF) or 1=background (ON, for chaining into Color Match).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "chroma_mask")
    FUNCTION = "run"
    CATEGORY = "Badman/matte"

    def run(self, image, alpha_mask, bg_color, chroma_threshold, chroma_feather,
            despill_strength, invert_input_mask, invert_output_mask):

        bg_rgb = _hex_to_rgb(bg_color)

        imgs = image.detach().cpu().numpy()  # [B, H, W, 3] float 0..1, RGB
        b, h, w, _ = imgs.shape

        m = alpha_mask.detach().cpu().numpy()
        if m.ndim == 2:
            m = m[None]

        out_img = np.empty_like(imgs)
        out_mask = np.empty((b, h, w), dtype=np.float32)
        out_chroma = np.empty((b, h, w), dtype=np.float32)

        for i in range(b):
            rgb_f = imgs[i]
            rgb_u8 = np.clip(rgb_f * 255.0 + 0.5, 0, 255).astype(np.uint8)

            mi = m[i] if i < m.shape[0] else m[-1]
            mi = mi.astype(np.float32)
            if mi.shape != (h, w):
                mi = cv2.resize(mi, (w, h), interpolation=cv2.INTER_LINEAR)

            fg = (1.0 - mi) if invert_input_mask else mi
            fg = np.clip(fg, 0.0, 1.0)

            sim = _bg_similarity_lab(rgb_u8, bg_rgb, chroma_threshold, chroma_feather)

            refined = fg * (1.0 - sim)
            despilled = _despill_channel_limit(rgb_f, bg_rgb, despill_strength)

            out_img[i] = despilled
            out_mask[i] = refined
            out_chroma[i] = sim

        if invert_output_mask:
            out_mask = 1.0 - out_mask

        return (
            torch.from_numpy(np.ascontiguousarray(out_img.astype(np.float32))),
            torch.from_numpy(out_mask),
            torch.from_numpy(out_chroma),
        )
