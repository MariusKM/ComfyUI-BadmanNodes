"""
Color Match node for ComfyUI (Badman).

One node, two modes controlled by `use_preset`:

  build:   learn a color correction from a reference pair (source_ref -> target_ref)
           and apply it to `image`. Optionally save the learned model to a preset.
  preset:  load a previously saved preset and apply it to `image`.

Methods (build mode):
  reinhard: LAB mean+std transfer. Low-parameter, safe across palettes.
  hist:     LAB per-channel histogram matching. Stronger but can clip when applied
            to a palette different from the reference pair.

Presets are stored as JSON in ComfyUI-BadmanNodes/presets/. A companion JS extension
in ./web handles widget show/hide based on `use_preset` and refreshes the preset
dropdown after save. A small HTTP route (/badman/color_match/presets) serves the
current preset list to the frontend.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import cv2


PRESETS_DIR = Path(__file__).parent / "presets"
PRESETS_DIR.mkdir(exist_ok=True)


# ---------- tensor <-> numpy helpers ----------

def _image_to_np(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0 + 0.5).astype(np.uint8)
    return arr[..., ::-1]  # RGB -> BGR


def _np_to_image(arr_bgr: np.ndarray) -> torch.Tensor:
    rgb = arr_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb))


def _mask_to_np(mask: torch.Tensor | None, shape_hw: tuple[int, int]) -> np.ndarray:
    """Comfy masks come in inverted (1=background, 0=foreground), so flip here."""
    if mask is None:
        return np.ones(shape_hw, dtype=bool)
    m = mask.detach().cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m < 0.5


# ---------- color math ----------

def _build_channel_lut(src_vals: np.ndarray, tgt_vals: np.ndarray) -> np.ndarray:
    src_hist, _ = np.histogram(src_vals, bins=256, range=(0, 256))
    tgt_hist, _ = np.histogram(tgt_vals, bins=256, range=(0, 256))
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
    src_cdf /= max(src_cdf[-1], 1)
    tgt_cdf /= max(tgt_cdf[-1], 1)
    lut = np.interp(src_cdf, tgt_cdf, np.arange(256))
    return np.clip(lut, 0, 255).astype(np.uint8)


def _build_hist_model(src_bgr, tgt_bgr, src_mask, tgt_mask):
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB)
    luts = [
        _build_channel_lut(src_lab[:, :, c][src_mask], tgt_lab[:, :, c][tgt_mask])
        for c in range(3)
    ]
    return ("hist", {"luts": luts})


def _build_reinhard_model(src_bgr, tgt_bgr, src_mask, tgt_mask):
    src_lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    stats = {}
    for name, lab, mask in (("src", src_lab, src_mask), ("tgt", tgt_lab, tgt_mask)):
        means, stds = [], []
        for c in range(3):
            vals = lab[:, :, c][mask]
            means.append(float(vals.mean()))
            stds.append(float(vals.std() + 1e-6))
        stats[name + "_mean"] = np.array(means, dtype=np.float32)
        stats[name + "_std"] = np.array(stds, dtype=np.float32)
    return ("reinhard", stats)


def _apply_model(bgr: np.ndarray, model, mask: np.ndarray, strength: float) -> np.ndarray:
    method, data = model
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_out = lab.copy()

    if method == "hist":
        luts = data["luts"]
        for c in range(3):
            channel = lab[:, :, c]
            mapped = luts[c][channel.astype(np.uint8)].astype(np.float32)
            blended = (1.0 - strength) * channel + strength * mapped
            lab_out[:, :, c] = np.where(mask, blended, channel)
    else:
        src_mean, src_std = data["src_mean"], data["src_std"]
        tgt_mean, tgt_std = data["tgt_mean"], data["tgt_std"]
        for c in range(3):
            channel = lab[:, :, c]
            shifted = (channel - src_mean[c]) * (tgt_std[c] / src_std[c]) + tgt_mean[c]
            blended = (1.0 - strength) * channel + strength * shifted
            lab_out[:, :, c] = np.where(mask, blended, channel)

    lab_out = np.clip(lab_out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)


def _apply_to_batch(imgs_bgr, model, image_mask, strength):
    out = np.empty_like(imgs_bgr)
    if image_mask is not None:
        m = image_mask.detach().cpu().numpy()
        if m.ndim == 2:
            m = m[None]
        full_mask = None
    else:
        m = None
        # Frames in a batch share shape — build the "everything is foreground"
        # mask once and reuse it per frame.
        full_mask = np.ones(imgs_bgr.shape[1:3], dtype=bool)

    for i in range(imgs_bgr.shape[0]):
        frame = imgs_bgr[i]
        if m is not None:
            mi = m[i] if i < m.shape[0] else m[-1]
            if mi.shape != frame.shape[:2]:
                mi = cv2.resize(mi, (frame.shape[1], frame.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            # Comfy masks are inverted (1=background), so foreground is < 0.5.
            frame_mask = mi < 0.5
        else:
            frame_mask = full_mask
        out[i] = _apply_model(frame, model, frame_mask, float(strength))
    return out


# ---------- preset serialization ----------

def _serialize_model(model) -> dict:
    method, data = model
    if method == "hist":
        return {
            "method": "hist",
            "luts": [lut.tolist() for lut in data["luts"]],
        }
    return {
        "method": "reinhard",
        "src_mean": data["src_mean"].tolist(),
        "src_std": data["src_std"].tolist(),
        "tgt_mean": data["tgt_mean"].tolist(),
        "tgt_std": data["tgt_std"].tolist(),
    }


def _deserialize_model(d: dict):
    method = d["method"]
    if method == "hist":
        luts = [np.array(lut, dtype=np.uint8) for lut in d["luts"]]
        return ("hist", {"luts": luts})
    return ("reinhard", {
        "src_mean": np.array(d["src_mean"], dtype=np.float32),
        "src_std": np.array(d["src_std"], dtype=np.float32),
        "tgt_mean": np.array(d["tgt_mean"], dtype=np.float32),
        "tgt_std": np.array(d["tgt_std"], dtype=np.float32),
    })


def _list_presets() -> list[str]:
    names = sorted(p.stem for p in PRESETS_DIR.glob("*.json"))
    return names or ["<none>"]


def _sanitize_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip()


def _presets_signature() -> str:
    """Cheap fingerprint of the presets dir so IS_CHANGED can invalidate
    cached node outputs whenever a preset is added, removed, or overwritten."""
    try:
        parts = [
            f"{p.name}:{p.stat().st_mtime_ns}"
            for p in sorted(PRESETS_DIR.glob("*.json"))
        ]
    except OSError:
        return "unreadable"
    return "|".join(parts) or "empty"


def _combine_reinhard(models: list[tuple]):
    """Model-average across reinhard presets - equal weight per preset."""
    keys = ("src_mean", "src_std", "tgt_mean", "tgt_std")
    acc = {k: np.zeros(3, dtype=np.float64) for k in keys}
    for _, data in models:
        for k in keys:
            acc[k] += data[k].astype(np.float64)
    n = len(models)
    return ("reinhard", {k: (acc[k] / n).astype(np.float32) for k in keys})


def _combine_hist(models: list[tuple]):
    """Pointwise mean of per-channel LUTs across hist presets."""
    acc = [np.zeros(256, dtype=np.float64) for _ in range(3)]
    for _, data in models:
        for c in range(3):
            acc[c] += data["luts"][c].astype(np.float64)
    n = len(models)
    luts = [np.clip(a / n, 0, 255).astype(np.uint8) for a in acc]
    return ("hist", {"luts": luts})


# ---------- Preview cache (per-node, seeded on execute) ----------

# LRU-capped so deleted nodes don't pin their cached image tensors forever.
_PREVIEW_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_PREVIEW_CACHE_MAX = 20
_PREVIEW_MAX_SIDE = 384  # downscale for fast live preview


def _preview_cache_put(node_id: str, entry: dict) -> None:
    _PREVIEW_CACHE[node_id] = entry
    _PREVIEW_CACHE.move_to_end(node_id)
    while len(_PREVIEW_CACHE) > _PREVIEW_CACHE_MAX:
        _PREVIEW_CACHE.popitem(last=False)


def _preview_cache_get(node_id: str) -> dict | None:
    entry = _PREVIEW_CACHE.get(node_id)
    if entry is not None:
        _PREVIEW_CACHE.move_to_end(node_id)
    return entry


def _downscale_bgr(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return bgr, 1.0
    new_w, new_h = int(round(w * s)), int(round(h * s))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA), s


def _mask_tensor_to_bool(mask_tensor, target_hw):
    if mask_tensor is None:
        return np.ones(target_hw, dtype=bool)
    m = mask_tensor.detach().cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    if m.shape != target_hw:
        m = cv2.resize(m, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m < 0.5


# ---------- HTTP routes ----------

try:
    from server import PromptServer
    from aiohttp import web
    import base64

    @PromptServer.instance.routes.get("/badman/color_match/presets")
    async def _badman_list_presets(request):
        return web.json_response({"presets": _list_presets()})

    @PromptServer.instance.routes.post("/badman/color_match/preview")
    async def _badman_preview(request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid json"}, status=400)

        node_id = str(data.get("node_id", ""))
        cache = _preview_cache_get(node_id)
        if not cache or cache.get("image") is None:
            return web.json_response(
                {"error": "no cached image yet - run the workflow once"},
                status=400,
            )

        use_preset = bool(data.get("use_preset"))
        method = data.get("method", "reinhard")
        try:
            strength = float(data.get("strength", 1.0))
        except (TypeError, ValueError):
            strength = 1.0

        if use_preset:
            name = data.get("preset_name") or ""
            if name in ("", "<none>"):
                return web.json_response({"error": "no preset selected"}, status=400)
            path = PRESETS_DIR / f"{name}.json"
            if not path.exists():
                return web.json_response({"error": f"preset not found: {name}"}, status=400)
            model = _deserialize_model(json.loads(path.read_text()))
        else:
            src_ref = cache.get("source_ref")
            tgt_ref = cache.get("target_ref")
            if src_ref is None or tgt_ref is None:
                return web.json_response(
                    {"error": "no refs cached - run build mode once"},
                    status=400,
                )
            src_bgr = _image_to_np(src_ref)[0]
            tgt_bgr = _image_to_np(tgt_ref)[0]
            src_mask = _mask_to_np(cache.get("source_mask"), src_bgr.shape[:2])
            tgt_mask = _mask_to_np(cache.get("target_mask"), tgt_bgr.shape[:2])
            if method == "hist":
                model = _build_hist_model(src_bgr, tgt_bgr, src_mask, tgt_mask)
            else:
                model = _build_reinhard_model(src_bgr, tgt_bgr, src_mask, tgt_mask)

        imgs_bgr = _image_to_np(cache["image"])  # [B,H,W,3]
        frame = np.ascontiguousarray(imgs_bgr[0])
        frame, _ = _downscale_bgr(frame, _PREVIEW_MAX_SIDE)
        frame_mask = _mask_tensor_to_bool(cache.get("image_mask"), frame.shape[:2])

        out_bgr = _apply_model(frame, model, frame_mask, strength)
        ok, buf = cv2.imencode(".png", out_bgr)
        if not ok:
            return web.json_response({"error": "encode failed"}, status=500)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return web.json_response({"image": f"data:image/png;base64,{b64}"})

except Exception:
    pass


# ---------- Comfy node ----------

class ColorMatchNode:
    """All-in-one color match: build + apply, or load preset + apply."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "use_preset": ("BOOLEAN", {
                    "default": False,
                    "label_on": "preset",
                    "label_off": "build",
                }),
                "preset_name": (_list_presets(),),
                "method": (["reinhard", "hist"], {"default": "reinhard"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "save_as": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "source_ref": ("IMAGE",),
                "target_ref": ("IMAGE",),
                "source_mask": ("MASK",),
                "target_mask": ("MASK",),
                "image_mask": ("MASK",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "preset_path")
    FUNCTION = "run"
    CATEGORY = "Badman/color"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Re-run when any preset file on disk changes: keeps preset-mode output
        # fresh after a preset is overwritten, and helps surface validation
        # errors early if a referenced preset was deleted.
        return _presets_signature()

    def run(self, image, use_preset, preset_name, method, strength, save_as, overwrite,
            source_ref=None, target_ref=None,
            source_mask=None, target_mask=None, image_mask=None,
            unique_id=None):

        if unique_id is not None:
            _preview_cache_put(str(unique_id), {
                "image": image,
                "image_mask": image_mask,
                "source_ref": source_ref,
                "target_ref": target_ref,
                "source_mask": source_mask,
                "target_mask": target_mask,
            })

        saved_path = ""

        if use_preset:
            if preset_name in ("", "<none>"):
                raise ValueError("use_preset=True but no preset is selected.")
            path = PRESETS_DIR / f"{preset_name}.json"
            if not path.exists():
                raise FileNotFoundError(f"Preset not found: {path}")
            model = _deserialize_model(json.loads(path.read_text()))
        else:
            if source_ref is None or target_ref is None:
                raise ValueError("Build mode requires source_ref and target_ref.")
            src_bgr = _image_to_np(source_ref)[0]
            tgt_bgr = _image_to_np(target_ref)[0]
            src_mask = _mask_to_np(source_mask, src_bgr.shape[:2])
            tgt_mask = _mask_to_np(target_mask, tgt_bgr.shape[:2])
            if method == "hist":
                model = _build_hist_model(src_bgr, tgt_bgr, src_mask, tgt_mask)
            else:
                model = _build_reinhard_model(src_bgr, tgt_bgr, src_mask, tgt_mask)

            if save_as.strip():
                safe = _sanitize_name(save_as)
                if not safe:
                    raise ValueError("save_as must contain at least one alphanumeric char.")
                out_path = PRESETS_DIR / f"{safe}.json"
                if out_path.exists() and not overwrite:
                    raise FileExistsError(f"Preset '{safe}' exists and overwrite=False")
                out_path.write_text(json.dumps(_serialize_model(model), indent=2))
                saved_path = str(out_path)
                print(f"[Badman ColorMatch] saved preset: {out_path}")

        imgs_bgr = _image_to_np(image)
        out = _apply_to_batch(imgs_bgr, model, image_mask, strength)

        # Pass through whatever was wired in; None when nothing was.
        # We don't compute a mask here, so returning a synthetic one would
        # mislead downstream nodes about which pixels were corrected.
        if image_mask is not None:
            out_mask = image_mask.detach().clone().float()
            if out_mask.ndim == 2:
                out_mask = out_mask[None]
        else:
            out_mask = None

        return (_np_to_image(out), out_mask, saved_path)


class ColorMatchCombinePresets:
    """Combine any number (>=2) of saved presets into a new preset.

    Reinhard: averages the 4 stat vectors across selected presets (equal weight).
    Hist:     averages the 3 per-channel LUTs (pointwise mean, uint8).
    All selected presets must share the same method.

    The node declares MAX_SLOTS preset inputs; the JS extension shows/hides them
    via + / - buttons so the UI starts compact (2 slots) and grows on demand.
    Unused slots stay at "<none>" and are skipped.
    """

    MAX_SLOTS = 10

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            f"preset_{i+1}": (_list_presets(),)
            for i in range(cls.MAX_SLOTS)
        }
        required["save_as"] = ("STRING", {"default": ""})
        required["overwrite"] = ("BOOLEAN", {"default": False})
        return {"required": required}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("name", "path")
    FUNCTION = "run"
    CATEGORY = "Badman/color"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return _presets_signature()

    def run(self, save_as, overwrite, **presets):
        selected_names = [
            presets[f"preset_{i+1}"] for i in range(self.MAX_SLOTS)
            if presets.get(f"preset_{i+1}") not in ("", "<none>", None)
        ]
        # Deduplicate while preserving order
        seen = set()
        selected_names = [n for n in selected_names if not (n in seen or seen.add(n))]
        if len(selected_names) < 2:
            raise ValueError("Select at least two distinct presets to combine.")

        safe = _sanitize_name(save_as)
        if not safe:
            raise ValueError("save_as must contain at least one alphanumeric char.")
        out_path = PRESETS_DIR / f"{safe}.json"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Preset '{safe}' exists and overwrite=False")

        models = []
        methods = set()
        for n in selected_names:
            p = PRESETS_DIR / f"{n}.json"
            if not p.exists():
                raise FileNotFoundError(f"Preset not found: {p}")
            m = _deserialize_model(json.loads(p.read_text()))
            methods.add(m[0])
            models.append(m)

        if len(methods) > 1:
            raise ValueError(f"All presets must share one method, got: {methods}")
        method = methods.pop()

        if method == "reinhard":
            combined = _combine_reinhard(models)
        else:
            combined = _combine_hist(models)

        out_path.write_text(json.dumps(_serialize_model(combined), indent=2))
        print(f"[Badman ColorMatch] combined {len(selected_names)} preset(s) "
              f"({method}) -> {out_path}")
        return (safe, str(out_path))
