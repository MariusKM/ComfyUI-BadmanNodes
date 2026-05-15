"""
Tiled Background Removal node for ComfyUI (Badman).

Wraps RMBG-2.0 / BiRefNet (from the comfyui-rmbg pack) with a tile-and-
reassemble strategy so that large source images keep per-tile detail that a
single fixed-resolution model pass would lose to its internal downsample.

Per-image flow:
  1. (optional) Global-context pass — one inference on the full image at the
     model's native resolution to get a coarse semantic mask.
  2. Tile grid — step = tile_size - tile_overlap, last row/col clamped so the
     edge is covered. One model inference per tile at its native resolution.
  3. Feather-weighted reassembly — each tile contributes via a cosine 2D
     weight that fades to near-zero at the tile edges. Accumulator =
     sum(mask * weight) / sum(weight).
  4. (optional) Blend the global coarse mask back in by `global_weight`.
  5. Post-process once at full res: sensitivity (RMBG only), blur, offset,
     invert, refine-foreground, alpha/color composite. Mirrors what the
     reference nodes do on their single-pass masks.

Depends on the already-installed `comfyui-rmbg` pack for model loaders and
the shared post-processing helpers — we never copy model code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter


# ---------- runtime dependency on comfyui-rmbg ----------

_RMBG_PACK = Path(__file__).parent.parent / "comfyui-rmbg" / "py"
if not _RMBG_PACK.exists():
    raise ImportError(
        f"Tiled Background Removal needs the comfyui-rmbg pack installed at "
        f"{_RMBG_PACK.parent}. Install it via ComfyUI Manager and restart."
    )
if str(_RMBG_PACK) not in sys.path:
    sys.path.insert(0, str(_RMBG_PACK))

from AILab_RMBG import RMBGModel, AVAILABLE_MODELS, refine_foreground  # noqa: E402
from AILab_BiRefNet import BiRefNetModel, MODEL_CONFIG  # noqa: E402
from AILab_utils import tensor2pil, pil2tensor  # noqa: E402


# ---------- tiling helpers ----------

def _tile_starts(dim: int, tile_size: int, overlap: int) -> list[int]:
    """Start coordinates covering [0, dim) with tiles of `tile_size`.

    If `dim <= tile_size`, returns a single start at 0 (the caller will crop
    a smaller-than-tile_size tile at the image edge). Otherwise steps by
    `tile_size - overlap` and clamps the final start so the last tile's right
    edge sits exactly at `dim`.
    """
    if dim <= tile_size:
        return [0]
    step = max(tile_size - overlap, 1)
    starts = list(range(0, dim - tile_size + 1, step))
    last = dim - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _feather_weight(h: int, w: int, overlap: int) -> np.ndarray:
    """2D cosine ramp: ~1 inside, fades to small values over `overlap` px
    at each edge. Dimensions can be smaller than overlap*2 (edge-of-image
    tiles) — the ramps get clamped to half the tile dimension."""
    def _profile(n: int, edge: int) -> np.ndarray:
        edge = max(0, min(edge, n // 2))
        out = np.ones(n, dtype=np.float32)
        if edge <= 0:
            return out
        t = (np.arange(edge) + 1) / (edge + 1)
        ramp = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
        out[:edge] = ramp
        out[-edge:] = ramp[::-1]
        return out

    py = _profile(h, overlap)
    px = _profile(w, overlap)
    return np.outer(py, px)


# ---------- model routing ----------

_RMBG_FAMILY = "rmbg"
_BIREF_FAMILY = "biref"


def _resolve_process_res(model_name: str, family: str, user_res: int, tile_size: int) -> int:
    """Decide the internal resize resolution for one inference call.

    `user_res == 0` means auto: for BiRefNet use the per-model `default_res`
    (with the same rounding rules the reference node uses); for RMBG use the
    tile size the caller handed us.
    """
    if family == _BIREF_FAMILY:
        cfg = MODEL_CONFIG[model_name]
        res = user_res if user_res > 0 else cfg.get("default_res", 1024)
        if cfg.get("force_res", False):
            base = 512
            res = ((res + base - 1) // base) * base
        else:
            res = (res // 32) * 32
        return max(res, 32)
    # RMBG
    res = user_res if user_res > 0 else tile_size
    return max((res // 8) * 8, 256)


def _run_model(model_instance, family: str, model_name: str,
               tile_tensor_hw3: torch.Tensor, params: dict) -> Image.Image:
    """Call the right model API and return a single PIL L-mask sized to the tile.

    RMBG's `process_image` accepts a list (or single tensor) and returns
    `list[PIL]`; BiRefNet's takes a single 3D tensor and returns one PIL.
    """
    if family == _RMBG_FAMILY:
        masks = model_instance.process_image([tile_tensor_hw3], model_name, params)
        if isinstance(masks, list):
            mask = masks[0]
        else:
            mask = masks
    else:
        mask = model_instance.process_image(tile_tensor_hw3, params)
    return mask.convert("L")


# ---------- small local helpers ----------

def _hex_to_rgba(hex_color: str) -> tuple[int, int, int, int]:
    s = (hex_color or "").strip().lstrip("#")
    if len(s) == 6:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255
    if len(s) == 8:
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), int(s[6:8], 16)
    raise ValueError(f"Invalid color format: {hex_color!r}")


def _apply_mask_post(mask_np_01: np.ndarray, params: dict, family: str,
                     safezone_np_01: np.ndarray | None = None) -> Image.Image:
    """Replicate the reference nodes' post-processing on a full-size mask.

    Input: mask as float32 in [0, 1]. Output: PIL L-mode mask.
    If `safezone_np_01` is given, its values are unioned in (max) after
    blur/offset but before invert — so guaranteed-foreground regions survive
    erosion, don't get softened by blur, and still follow the invert toggle.
    """
    m = mask_np_01
    if family == _RMBG_FAMILY:
        # Reference RMBG node re-applies sensitivity here too; keep parity.
        m = m * (1.0 + (1.0 - float(params.get("sensitivity", 1.0))))
        m = np.clip(m, 0.0, 1.0)
    mask_pil = Image.fromarray(np.clip(m * 255.0, 0, 255).astype(np.uint8), mode="L")

    mask_blur = int(params.get("mask_blur", 0))
    if mask_blur > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_blur))

    mask_offset = int(params.get("mask_offset", 0))
    if mask_offset > 0:
        for _ in range(mask_offset):
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
    elif mask_offset < 0:
        for _ in range(-mask_offset):
            mask_pil = mask_pil.filter(ImageFilter.MinFilter(3))

    if safezone_np_01 is not None:
        arr = np.array(mask_pil).astype(np.float32) / 255.0
        arr = np.maximum(arr, np.clip(safezone_np_01, 0.0, 1.0))
        mask_pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")

    if params.get("invert_output", False):
        mask_pil = Image.fromarray(255 - np.array(mask_pil))

    return mask_pil


def _resize_mask_to(mask_np_01: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize a 2-D [0,1] float mask to (w, h) using PIL BILINEAR."""
    if mask_np_01.shape == (h, w):
        return mask_np_01.astype(np.float32)
    pil = Image.fromarray(
        np.clip(mask_np_01 * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).resize((w, h), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


# ---------- the node ----------

class TiledRMBGNode:
    """Tiled BiRefNet / RMBG-2.0 inference with feathered reassembly."""

    def __init__(self):
        # Keep both loaders alive across invocations so their internal
        # model-version tracking avoids redundant reloads when the user
        # switches between models in the same family.
        self._rmbg = RMBGModel()
        self._biref = BiRefNetModel()

    @classmethod
    def INPUT_TYPES(cls):
        model_choices = ["RMBG-2.0"] + list(MODEL_CONFIG.keys())
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image batch."}),
                "model": (model_choices, {
                    "default": "BiRefNet-general",
                    "tooltip": "Background-removal model. RMBG-2.0 or any BiRefNet variant.",
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Pixel size of each inference tile. 1024 matches most models' native res.",
                }),
                "tile_overlap": ("INT", {
                    "default": 128, "min": 0, "max": 512, "step": 16,
                    "tooltip": "Pixels shared between neighbouring tiles. Higher = smoother seams, more compute.",
                }),
                "use_global_context": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also run the model once on the full image to anchor semantics. Blended with tiles via global_weight.",
                }),
                "global_weight": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "When global context is on: weight of the coarse full-image mask in the final blend.",
                }),
                "sensitivity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "RMBG-2.0 only. Lower = more aggressive mask. Ignored by BiRefNet.",
                }),
                "process_res": ("INT", {
                    "default": 0, "min": 0, "max": 2560, "step": 8,
                    "tooltip": "Internal inference res per tile. 0 = auto (tile_size for RMBG, per-model default for BiRefNet).",
                }),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "mask_offset": ("INT", {"default": 0, "min": -64, "max": 64, "step": 1}),
                "invert_output": ("BOOLEAN", {"default": False}),
                "refine_foreground": ("BOOLEAN", {"default": False}),
                "background": (["Alpha", "Color"], {"default": "Alpha"}),
                "background_color": ("COLORCODE", {"default": "#222222"}),
                "safezone_mask": ("MASK", {
                    "tooltip": "Optional. Regions painted 1 in this mask are force-kept as foreground (the model can't carve them out). Use for eyes, small features that the matte net drops. Same Comfy 1=foreground convention as the main output.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "MASK_IMAGE")
    FUNCTION = "process_image"
    CATEGORY = "Badman/matte"

    # ----- main entrypoint -----

    def process_image(self, image, model, **params):
        family = _RMBG_FAMILY if model == "RMBG-2.0" else _BIREF_FAMILY
        model_instance = self._rmbg if family == _RMBG_FAMILY else self._biref

        cache_ok, message = model_instance.check_model_cache(model)
        if not cache_ok:
            print(f"[Badman TiledRMBG] cache miss: {message} — downloading")
            dl_ok, dl_msg = model_instance.download_model(model)
            if not dl_ok:
                raise RuntimeError(f"Model download failed: {dl_msg}")
        # BiRefNetModel is lazy: needs explicit load_model to move weights onto device.
        if family == _BIREF_FAMILY:
            model_instance.load_model(model)

        tile_size = int(params.get("tile_size", 1024))
        tile_overlap = int(params.get("tile_overlap", 128))
        tile_overlap = min(tile_overlap, max(tile_size // 2 - 1, 0))

        use_global = bool(params.get("use_global_context", True))
        global_weight = float(params.get("global_weight", 0.35))

        # Per-tile and per-global param dicts differ only in process_res.
        # Tile: run at tile's native size (auto) or user override.
        # Global: run at the reference node's default — 1024 for RMBG, per-model for BiRefNet.
        tile_res = _resolve_process_res(model, family, int(params.get("process_res", 0)), tile_size)
        global_res = _resolve_process_res(model, family, int(params.get("process_res", 0)), 1024)
        base_params = {
            "sensitivity": float(params.get("sensitivity", 1.0)),
        }
        tile_params = dict(base_params, process_res=tile_res)
        global_params = dict(base_params, process_res=global_res)

        # Safezone mask: optional per-image MASK that force-protects pixels as
        # foreground after post-processing. Normalised to [B, H, W] up front;
        # per-image resize to frame dims happens inside the loop.
        safezone = params.get("safezone_mask", None)
        if safezone is not None:
            sz_all = safezone.detach().cpu().numpy().astype(np.float32)
            if sz_all.ndim == 2:
                sz_all = sz_all[None]
        else:
            sz_all = None

        out_images: list[torch.Tensor] = []
        out_masks: list[torch.Tensor] = []

        for i, img_hw3 in enumerate(image):  # shape (H, W, 3), float 0..1
            h, w = int(img_hw3.shape[0]), int(img_hw3.shape[1])

            if sz_all is not None:
                si = sz_all[i] if i < sz_all.shape[0] else sz_all[-1]
                safezone_for_frame = _resize_mask_to(si, w, h)
            else:
                safezone_for_frame = None

            # --- 1. (optional) global context pass ---
            if use_global:
                coarse_pil = _run_model(model_instance, family, model, img_hw3, global_params)
                if coarse_pil.size != (w, h):
                    coarse_pil = coarse_pil.resize((w, h), Image.BILINEAR)
                coarse = np.asarray(coarse_pil, dtype=np.float32) / 255.0
            else:
                coarse = None

            # --- 2. tile grid inference ---
            mask_sum = np.zeros((h, w), dtype=np.float32)
            weight_sum = np.zeros((h, w), dtype=np.float32)
            y_starts = _tile_starts(h, tile_size, tile_overlap)
            x_starts = _tile_starts(w, tile_size, tile_overlap)
            for ty in y_starts:
                ty2 = min(ty + tile_size, h)
                for tx in x_starts:
                    tx2 = min(tx + tile_size, w)
                    tile = img_hw3[ty:ty2, tx:tx2]
                    m_pil = _run_model(model_instance, family, model, tile, tile_params)
                    # model's process_image already interpolates back to tile dims
                    m_np = np.asarray(m_pil, dtype=np.float32) / 255.0
                    th, tw = m_np.shape[:2]
                    w_tile = _feather_weight(th, tw, tile_overlap)
                    mask_sum[ty:ty + th, tx:tx + tw] += m_np * w_tile
                    weight_sum[ty:ty + th, tx:tx + tw] += w_tile

            detail = mask_sum / np.maximum(weight_sum, 1e-6)

            # --- 3. blend with global context ---
            if coarse is not None:
                gw = float(np.clip(global_weight, 0.0, 1.0))
                combined = gw * coarse + (1.0 - gw) * detail
            else:
                combined = detail
            combined = np.clip(combined, 0.0, 1.0)

            # --- 4. post-processing on full-res mask ---
            mask_pil = _apply_mask_post(combined, params, family, safezone_for_frame)

            # --- 5. compose output image (match reference node shape) ---
            orig_pil = tensor2pil(img_hw3)

            if bool(params.get("refine_foreground", False)):
                img_bchw = torch.from_numpy(
                    np.array(orig_pil)
                ).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                mask_b1hw = torch.from_numpy(
                    np.array(mask_pil)
                ).unsqueeze(0).unsqueeze(0).float() / 255.0
                refined = refine_foreground(img_bchw, mask_b1hw)
                refined_pil = tensor2pil(refined[0].permute(1, 2, 0))
                r, g, b = refined_pil.split()
                foreground = Image.merge("RGBA", (r, g, b, mask_pil))
            else:
                r, g, b, _a = orig_pil.convert("RGBA").split()
                foreground = Image.merge("RGBA", (r, g, b, mask_pil))

            if params.get("background", "Alpha") == "Color":
                rgba = _hex_to_rgba(params.get("background_color", "#222222"))
                bg = Image.new("RGBA", orig_pil.size, rgba)
                composite = Image.alpha_composite(bg, foreground)
                out_images.append(pil2tensor(composite.convert("RGB")))
            else:
                out_images.append(pil2tensor(foreground))

            out_masks.append(pil2tensor(mask_pil))

        # --- 6. stack and return ---
        mask_image_chans = []
        for mt in out_masks:
            mi = mt.reshape((-1, 1, mt.shape[-2], mt.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
            mask_image_chans.append(mi)

        return (
            torch.cat(out_images, dim=0),
            torch.cat(out_masks, dim=0),
            torch.cat(mask_image_chans, dim=0),
        )
