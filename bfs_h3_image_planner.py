"""BFS Image Planner: the Shot Planner for one still image and MiniMax H3 Image.

The same panel and the same per-shot tools as BFS Shot Planner, without anything that only exists for a video
(timeline, cuts, shots, continuity, audio, queue loop, stored shots, mask videos, tracking):

- the image is <Picture 1>, the picture MiniMax H3 Image edits; the reference pictures follow as <Picture 2>, ...
- prompt with placeholders ({target}, {details}, {details@Picture n}, {image}), 🤖 Auto prompt, picture descriptions
- what changes: a SAM 3 selection (text or points) with Full frame / Frame + paste / Mask only / Crop / Crop + mask,
  🎨 Colored Mark (overlay / hatching / normal map), {target} described by the VLM, mask settings
- generation size and the second pass size

    BFS Image Planner ─plan─► BFS Image Planner H3 Conditioning ─positive/latent─► sampler ─► VAE Decode
                      └──────────────────────────────────────────► BFS Image Planner Paste Back ◄─┘

All state lives in the hidden `plan` widget (JSON), written by the panel (web/js/bfs_image_planner.js).
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import sys

import numpy as np
import torch

PLAN_TYPE = "BFS_IMAGE_PLAN"
TAG = "[BFS Image Planner]"
CANVAS_MULTIPLE = 32            # nodes_minimax_h3.CANVAS_MULTIPLE
H3_SHORT_EDGE = (1024, 2048)    # H3 Image: image_1's short edge is kept inside this range
PH3_SIZE = 2912                 # p-h3-image: image_1 fits 2912^2 of area (down only)
MAX_PICTURES = 100
WORK_SIDE = 1024                # SAM 3 works on the image at this longest side
VARIANTS = ("MiniMax H3 Image", "p-h3-image")
SIZE_MODES = ("native", "megapixels", "original", "custom")
MODES = ("frame", "paste", "mask", "crop", "cropmask")
INPAINT_MODES = ("per plan (panel)", "off", "only the mask")
AUTO_REVISION = "image-v1"

DEFAULT_PLAN = {
    "image": "", "variant": VARIANTS[0], "global_refs": [], "global_prompt": "", "picture_colors": [],
    "generation_size_mode": "native", "megapixels": 4.0, "multiple": 32, "custom_width": 2048, "custom_height": 2048,
    "mask": {}, "mask_cfg": {}, "crop": False, "inpaint": False, "paste": False, "paste_text": "", "strength": 1.0,
    "color_guide": False, "color_guide_picture": 2, "color_guide_style": "overlay", "color_guide_opacity": 0.35,
    "color_guide_regions": [], "target": "", "vlm_cfg": {}, "picture_details": {}, "picture_cfg": {},
    "auto_prompt": {}, "auto_written": {}, "image_note": "", "second_pass": {},
}
AUTO_DEFAULTS = {"enabled": False, "style": "structured", "prefix": ""}


def _sl():
    """bfs_shot_loop (the video Shot Planner): SAM 3, Colored Mark, paste back and VLM helpers are shared."""
    try:
        from . import bfs_shot_loop as SL   # type: ignore
        return SL
    except ImportError:
        pass
    if "bfs_shot_loop" in sys.modules:
        return sys.modules["bfs_shot_loop"]
    spec = importlib.util.spec_from_file_location("bfs_shot_loop", os.path.join(os.path.dirname(__file__), "bfs_shot_loop.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bfs_shot_loop"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- plan

def load_plan(text) -> dict:
    if isinstance(text, dict):
        p = text
    else:
        try:
            p = json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{TAG} the plan is not valid JSON ({exc})") from exc
    out = json.loads(json.dumps(DEFAULT_PLAN))
    out.update({k: v for k, v in p.items() if v is not None})
    out["global_refs"] = [str(x or "") for x in (out.get("global_refs") or [])][:MAX_PICTURES - 1]
    while out["global_refs"] and not out["global_refs"][-1]:
        out["global_refs"].pop()
    return out


def ref_names(p: dict) -> list[str]:
    return [x for x in p.get("global_refs") or [] if x]


def picture_names(p: dict) -> list[str]:
    """<Picture 1> = the image, then the references."""
    return [p.get("image") or "<image>"] + ref_names(p)


def mode_of(p: dict) -> str:
    crop, inpaint, paste = bool(p.get("crop")), bool(p.get("inpaint")), bool(p.get("paste"))
    if crop:
        return "cropmask" if inpaint else "crop"
    if inpaint:
        return "mask"
    return "paste" if paste else "frame"


def mask_spec(p: dict, override: dict | None = None) -> dict:
    SL = _sl()
    spec = SL.mask_spec(p.get("mask"), p.get("mask_cfg"))
    spec["temporal_expand"] = 0       # one image: nothing to expand over time
    if isinstance(override, dict):
        for k, v in override.items():
            if k in ("text", "points", "threshold", "max_objects", "invert", "fill_holes", "blockify", "padding",
                     "expand", "feather", "paste"):
                spec[k] = v
    spec["key"] = 0
    return spec


def has_selection(spec: dict) -> bool:
    return bool(spec.get("points") or str(spec.get("text") or "").strip())


# ---------------------------------------------------------------------------- size

def _grid(x: float, multiple: int) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def generation_size(w: int, h: int, p: dict) -> tuple[int, int]:
    """The canvas: native (what the model's own node picks), megapixels, the original size or custom."""
    mode = p.get("generation_size_mode") or "native"
    mult = max(CANVAS_MULTIPLE, int(p.get("multiple") or 32) // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
    if mode == "custom":
        return _grid(float(p.get("custom_width") or 2048), mult), _grid(float(p.get("custom_height") or 2048), mult)
    if mode == "megapixels":
        s = math.sqrt(float(p.get("megapixels") or 4.0) * 1024 * 1024 / (w * h))
        return _grid(w * s, mult), _grid(h * s, mult)
    if mode == "original":
        return _grid(w, mult), _grid(h, mult)
    if p.get("variant") == VARIANTS[1]:
        s = min(1.0, math.sqrt(PH3_SIZE ** 2 / (w * h)))
    else:
        lo, hi = H3_SHORT_EDGE
        s = min(max(min(w, h), lo), hi) / min(w, h)
    return _grid(w * s, CANVAS_MULTIPLE), _grid(h * s, CANVAS_MULTIPLE)


def resize(img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    if img.shape[2] == w and img.shape[1] == h:
        return img[..., :3]
    x = torch.nn.functional.interpolate(img[..., :3].movedim(-1, 1).float(), size=(h, w), mode="bicubic",
                                        align_corners=False, antialias=True)
    return x.clamp(0, 1).movedim(1, -1)


def resize_mask(m: torch.Tensor, w: int, h: int) -> torch.Tensor:
    if m.ndim == 2:
        m = m[None]
    return torch.nn.functional.interpolate(m[:, None].float(), size=(h, w), mode="nearest")[:, 0]


# ---------------------------------------------------------------------------- masks (SAM 3 on one image)

_MASKS: dict[tuple, torch.Tensor] = {}
_EXT: dict[str, torch.Tensor] = {}    # the planner's `mask` input from its last run (image key -> mask)


def image_key(img: torch.Tensor) -> str:
    return _sl()._image_key(img)


def _working(img: torch.Tensor) -> torch.Tensor:
    H, W = img.shape[1:3]
    s = min(1.0, WORK_SIDE / max(H, W))
    return resize(img, max(32, int(W * s) // 2 * 2), max(32, int(H * s) // 2 * 2))


def segment(img: torch.Tensor, spec: dict) -> torch.Tensor:
    """[H,W] 0/1 mask of the selection on the image (source size): the planner's mask input, else SAM 3 (points win
    over the text). Cached; shaped by invert / fill holes / blockify."""
    SL = _sl()
    H, W = img.shape[1:3]
    key = image_key(img)
    if key in _EXT:
        m = resize_mask(_EXT[key], W, H)[0]
    else:
        if not has_selection(spec):
            raise ValueError("the selection needs a text or points (Edit tab)")
        ck = (key, json.dumps({k: spec.get(k) for k in ("text", "points", "threshold", "max_objects")}, sort_keys=True))
        if ck not in _MASKS:
            small = _working(img)
            SL._status("Segmenting the image with SAM 3", force=True)
            _MASKS[ck] = (SL.segment_frames(small, dict(spec, key=0))[:1] > 0.5).to(torch.uint8).cpu()
            SL._node_boundary()
        m = resize_mask(_MASKS[ck], W, H)[0]
    return SL.shape_mask(m[None], spec)[0]


def mark_masks(img: torch.Tensor, p: dict, first: torch.Tensor) -> list[tuple[torch.Tensor, dict]]:
    """Every Colored Mark region: (mask at source size, mark) - the selection first, then the extra subjects."""
    SL = _sl()
    marks = SL.color_marks(p, p)
    out = [(first, marks[0])]
    for x in marks[1:]:
        spec = dict(SL.DEFAULT_MASK, text=x.get("text") or "", points=x.get("points") or [], key=0, max_objects=1,
                    temporal_expand=0)
        out.append((segment(img, spec), x))
    return out


# ---------------------------------------------------------------------------- prompt

def picture_markers(text: str, n_refs: int) -> str:
    """{details} / {details2} = the first / second reference (<Picture 2> / <Picture 3>), {image} = <Picture 1>."""
    return (str(text or "").replace("{details2}", "{details@Picture 3}").replace("{details}", "{details@Picture 2}")
            .replace("{image}", "{details@Picture 1}"))


def auto_for(p: dict) -> str:
    """What a written prompt was made from (the panel computes the same string): request, pictures, marks, mode."""
    SL = _sl()
    marks = []
    if p.get("color_guide"):
        marks = [[x["picture"], x["color"]] for x in SL.color_marks(p, p)]
        if (p.get("color_guide_style") or "overlay") != "overlay":
            marks.append(p.get("color_guide_style"))
    a = {**AUTO_DEFAULTS, **(p.get("auto_prompt") or {})}
    return json.dumps([p.get("global_prompt") or "", picture_names(p), marks, mode_of(p),
                       str((p.get("mask") or {}).get("text") or ""), p.get("target") or "", a["style"], a["prefix"],
                       AUTO_REVISION], separators=(",", ":"), ensure_ascii=False)


AUTO_RULES = {
    "structured": (
        "Write the prompt in this structure, in English, with nothing before or after it:\n"
        "subject_definitions: one line per picture used, '<Picture n> is ...' (what it shows and its role: <Picture 1> "
        "is the image to edit, the others give appearances, objects, styles or settings).\n"
        "summary: [image editing + reference generation] one or two sentences saying exactly what the result shows.\n"
        "retention_analysis: what keeps exactly as in <Picture 1> (layout, background, lighting, camera, framing, "
        "everyone and everything not changed) and what is taken from each reference picture."),
    "short": ("Write the prompt as a short image-edit request: two to four sentences in English, one paragraph, no "
              "section names, no lists, no markdown. Name pictures only as <Picture n>."),
}
AUTO_QUESTION = """You write the prompt for MiniMax H3 Image, an image editing model. You see {n} picture(s) in order: <Picture 1> is the image to edit{refs_txt}.
{marks_txt}{mode_txt}{target_txt}The user's request: {request}
{details_txt}{rules}
Keep everything the request does not change exactly as in <Picture 1>. Never invent pictures that are not given."""


def write_prompt(clip, p: dict, img: torch.Tensor, refs: list, details: dict) -> str:
    """🤖 Auto prompt: the VLM writes the H3 Image prompt from the image, the references and the request."""
    SL = _sl()
    a = {**AUTO_DEFAULTS, **(p.get("auto_prompt") or {})}
    vc = SL.vlm_cfg(p.get("vlm_cfg"))
    names = picture_names(p)
    refs_txt = "".join(f", <Picture {k + 2}> is a reference" for k in range(len(refs)))
    marks_txt = ""
    if p.get("color_guide"):
        marks = SL.color_marks(p, p)
        style = p.get("color_guide_style") or "overlay"
        marks_txt = "In <Picture 1> the subjects to change are colour-marked: " + "; ".join(
            f"the subject {SL.mark_phrase(SL._COLOR_GUIDE_NAMES.get(x['color'], x['color']), style)} becomes the "
            f"character of <Picture {x['picture']}>" for x in marks) + ". The result must not show the marks.\n"
    mode = mode_of(p)
    mode_txt = {"mask": "Only the selected region is regenerated.\n", "cropmask": "Only the selected region is regenerated.\n",
                "crop": "<Picture 1> is a close crop around the subject.\n"}.get(mode, "")
    target_txt = f"What changes: {p['target']}.\n" if p.get("target") else ""
    dl = [f"<Picture {k + 1}>: {details[n]}" for k, n in enumerate(names) if details.get(n)]
    details_txt = ("Descriptions: " + " ".join(dl) + "\n") if dl else ""
    q = AUTO_QUESTION.format(n=1 + len(refs), refs_txt=refs_txt, marks_txt=marks_txt, mode_txt=mode_txt,
                             target_txt=target_txt, request=p.get("global_prompt") or "(none: describe a faithful edit)",
                             details_txt=details_txt, rules=AUTO_RULES.get(a["style"], AUTO_RULES["structured"]))
    side = max(224, min(2048, int(vc.get("image_max_side") or 1024)))
    batch = SL._letterbox_batch([img] + list(refs), side)
    text = SL.vlm_generate(clip, q, batch, max(256, int(vc.get("max_tokens") or 1024)),
                           float(vc.get("temperature") or 0), int(vc.get("seed") or 0))
    text = SL._clean_written(text) if hasattr(SL, "_clean_written") else str(text).strip()
    prefix = str(a.get("prefix") or "").strip()
    if prefix and not text.startswith(prefix):
        text = f"{prefix} {text}"
    return text.strip()


IMAGE_Q = """Look at this picture, which will be edited with an AI image model. Answer with one JSON object only, no code fence:
{{"segment": "the main person or object a user would edit, as a short English noun phrase a segmentation model understands (e.g. 'woman in a black top', 'red car'); never on-screen text",
"image": "one or two sentences for an image prompt: framing, camera angle, where the subject is and what they do, the setting and the light; not the person's face or clothes"}}{extra}"""


def analyse_image(clip, img: torch.Tensor, cfg: dict) -> dict:
    SL = _sl()
    extra = ("\n" + cfg["instruction"].strip()) if str(cfg.get("instruction") or "").strip() else ""
    side = max(224, min(4096, int(cfg.get("image_max_side") or 1024)))
    text = SL.vlm_generate(clip, IMAGE_Q.format(extra=extra), SL._letterbox_batch([img], side), int(cfg.get("max_tokens") or 512))
    out = SL._parse_json(text)
    return {"segment": str(out.get("segment") or "").strip(), "image": str(out.get("image") or "").strip(),
            "raw": "" if out else text}


def target_crop(img: torch.Tensor, mask: torch.Tensor, pad: float = 0.12) -> np.ndarray | None:
    """The selected subject cut from the image (RGB uint8), for 🧑 Describe."""
    if not bool(mask.any()):
        return None
    fr = (img[0, ..., :3].clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    ys, xs = np.where(mask.numpy() > 0.5)
    H, W = fr.shape[:2]
    py, px = int((ys.max() - ys.min()) * pad) + 4, int((xs.max() - xs.min()) * pad) + 4
    return fr[max(0, ys.min() - py):min(H, ys.max() + py), max(0, xs.min() - px):min(W, xs.max() + px)]


def build_prompt(p: dict, prompt_in: str | None, img: torch.Tensor, refs: list, vlm=None, mask=None,
                 log: str = TAG) -> tuple[str, dict]:
    """The final prompt and what was filled (auto prompt, target, image description, picture details)."""
    SL = _sl()
    vc = SL.vlm_cfg(p.get("vlm_cfg"))
    names = picture_names(p)
    pics = [img] + list(refs)
    details = dict(p.get("picture_details") or {})
    filled = {}
    a = {**AUTO_DEFAULTS, **(p.get("auto_prompt") or {})}
    request = prompt_in if prompt_in is not None and str(prompt_in).strip() else (p.get("global_prompt") or "")
    text, written = request, False
    if a["enabled"]:
        got = p.get("auto_written") or {}
        if got.get("text") and got.get("for") == auto_for(p):
            text, written = got["text"], True
        elif vlm is not None:
            text, written = write_prompt(vlm, dict(p, global_prompt=request), img, refs, details), True
            filled["auto_written"] = {"text": text, "for": auto_for(p)}
            print(f"{log}: prompt written by the VLM ({len(text.split())} words)", flush=True)
        elif got.get("text"):
            raise ValueError("The written prompt is older than the request / pictures / marks: use 🤖 Write again in "
                             "the panel, or connect the VLM to the planner.")
        else:
            print(f"{log}: Auto prompt is on but there is no written prompt and no VLM: the request is the prompt", flush=True)
    text = picture_markers(text, len(refs))
    if "{shot}" in text:
        note = p.get("image_note") or ""
        if not note and vlm is not None and vc.get("auto_shot", True):
            note = analyse_image(vlm, img, vc)["image"]
            filled["image_note"] = note
        text = text.replace("{shot}", note)
    # descriptions nobody wrote: described now, one picture at a time
    for m in list(SL.PICTURE_DETAILS.finditer(text)):
        k = int(m.group(1)) - 1
        if 0 <= k < len(names) and not details.get(names[k]) and vlm is not None:
            details[names[k]] = SL.describe_picture(vlm, pics[k], text, m.group(0), SL.picture_cfg(vc, p, names[k]))
            filled.setdefault("picture_details", {})[names[k]] = details[names[k]]
            print(f"{log}: {m.group(0)} = {details[names[k]]!r} (described automatically)", flush=True)
    text = SL.fill_picture_details(text, names, pics, details, tag=log)
    target = p.get("target") or ""
    if "{target}" in text and not target and vlm is not None and mask is not None:
        crop = target_crop(img, mask)
        if crop is not None:
            slot = SL.prompt_slot(text, "{target}") if vc.get("fit_prompt", True) else ""
            target = SL.describe_target(vlm, crop, slot=slot, temperature=float(vc.get("temperature") or 0),
                                        seed=int(vc.get("seed") or 0), image_max_side=int(vc.get("image_max_side") or 1024))
            filled["target"] = target
            print(f"{log}: {{target}} = {target!r} (described automatically)", flush=True)
    text = SL.fill_target(text, target)
    if p.get("color_guide") and not written:
        text = SL.color_guide_prompt(text, SL.color_marks(p, p), p.get("color_guide_style"))
    return text, filled


# ---------------------------------------------------------------------------- planning

def make_plan(p: dict, image: torch.Tensor | None = None, prompt: str | None = None, vlm=None,
              ext_mask: torch.Tensor | None = None) -> dict:
    SL = _sl()
    src = image[:1, ..., :3].float().cpu() if image is not None else SL._load_image(p.get("image"))
    if src is None:
        raise ValueError(f"{TAG} no image: pick one in the panel (Image tab) or connect the image input")
    key = image_key(src)
    if ext_mask is not None:
        m = ext_mask.float().cpu()
        if m.ndim == 4:                        # an IMAGE used as a mask
            m = m[..., 0]
        _EXT[key] = m[:1]
    else:
        _EXT.pop(key, None)
    refs = [r for r in (SL._load_image(n) for n in ref_names(p)) if r is not None]
    SH, SW = src.shape[1:3]
    mode = mode_of(p)
    spec = mask_spec(p)
    needs_mask = mode != "frame" or p.get("color_guide") or ("{target}" in (p.get("global_prompt") or "") and not p.get("target"))
    mask = segment(src, spec) if needs_mask and (has_selection(spec) or key in _EXT) else None
    if (mode != "frame" or p.get("color_guide")) and mask is None:
        raise ValueError(f"{TAG} {mode if mode != 'frame' else 'Colored Mark'} needs a selection: a text or points in "
                         "the Edit tab (or the planner's mask input)")
    crop, image_gen, full = None, None, None
    if mask is not None and p.get("crop"):
        box = SL.crop_box(mask[None], float(spec["padding"]))
        if box is None:
            raise ValueError(f"{TAG} Crop: the selection found nothing on the image")
        x0, y0, x1, y1 = SL.box_px(box, SW, SH)
        W, H = generation_size(x1 - x0, y1 - y0, p)
        image_gen = resize(src[:, y0:y1, x0:x1], W, H)
        crop = {"box": box, "mask": mask[None].float(), "paste": spec["paste"], "expand": int(spec["expand"]),
                "feather": int(spec["feather"])}
        gen_mask = resize_mask(mask[y0:y1, x0:x1], W, H)[0]
    else:
        W, H = generation_size(SW, SH, p)
        image_gen = resize(src, W, H)
        gen_mask = resize_mask(mask, W, H)[0] if mask is not None else None
    marks = []
    if p.get("color_guide") and mask is not None:
        for mk, x in mark_masks(src, p, mask):
            if crop is not None:
                if len(marks):
                    raise ValueError(f"{TAG} Colored Mark: more than one marked subject needs Crop off")
                mk = mk[y0:y1, x0:x1]
            marks.append((resize_mask(mk, W, H)[0], x))
    text, filled = build_prompt(p, prompt, image_gen, refs, vlm, mask, TAG)
    sp = {**SL.SECOND_PASS_DEFAULTS, **(p.get("second_pass") or {})}
    UW, UH = SL.second_pass_size(W, H, sp)
    return {
        "image": image_gen, "source": src, "refs": refs, "ref_names": ref_names(p), "prompt": text,
        "width": W, "height": H, "mode": mode, "mask": gen_mask, "crop": crop, "full_frames": src if crop else None,
        "inpaint": bool(p.get("inpaint")), "strength": float(p.get("strength") or 1.0), "paste": mode == "paste",
        "paste_text": p.get("paste_text") or "", "spec": spec, "source_mask": mask,
        "color_guide": bool(p.get("color_guide")), "marks": marks,
        "color_guide_style": p.get("color_guide_style") or "overlay",
        "color_guide_opacity": float(p.get("color_guide_opacity") or 0.35),
        "variant": p.get("variant") or VARIANTS[0], "second_pass": bool(sp["enabled"]), "upscale": (UW, UH),
        "filled": filled,
    }


def summary(r: dict) -> str:
    lines = [f"{r['width']}x{r['height']} ({r['variant']}), mode {r['mode']}"
             + (", Colored Mark" if r["color_guide"] else "") + (f", second pass {r['upscale'][0]}x{r['upscale'][1]}" if r["second_pass"] else ""),
             "<Picture 1> = the image" + "".join(f", <Picture {k + 2}> = {n}" for k, n in enumerate(r["ref_names"])),
             f"prompt: {r['prompt']}"]
    return "\n".join(lines)


def _notify(event: str, data: dict) -> None:
    try:
        from server import PromptServer
        PromptServer.instance.send_sync(event, data)
    except Exception:  # noqa: BLE001 - no server in tests
        pass


class BFSImagePlanner:
    """The Shot Planner for one image (MiniMax H3 Image): pictures, prompt and what changes, in a panel."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("STRING", {"default": json.dumps(DEFAULT_PLAN), "multiline": True,
                                    "tooltip": "The panel writes this: the image, pictures, prompt and edit settings."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional: the image to edit from the graph (wins over the panel's image)."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Overrides the panel's prompt (with 🤖 Auto prompt: the request)."}),
                "mask": ("MASK,IMAGE", {"tooltip": "Optional mask of the image (white = the subject): replaces SAM 3."}),
                "vlm": ("CLIP", {"tooltip": "Optional Qwen3-VL (CLIPLoader): picture descriptions, {target}, {shot}, "
                                            "🤖 Auto prompt and the panel's VLM buttons."}),
            },
        }

    RETURN_TYPES = (PLAN_TYPE, "IMAGE", "IMAGE", "INT", "STRING", "INT", "INT", "MASK", "STRING", "BOOLEAN", "INT", "INT")
    RETURN_NAMES = ("plan", "image", "refs", "ref_count", "prompt", "width", "height", "mask", "summary",
                    "second_pass", "upscale_width", "upscale_height")
    OUTPUT_IS_LIST = (False, False, True, False, False, False, False, False, False, False, False, False)
    OUTPUT_TOOLTIPS = (
        "Everything for BFS Image Planner H3 Conditioning and Paste Back.",
        "<Picture 1> at the generation size (the crop, in the crop modes), without marks.",
        "The reference pictures (<Picture 2>...), one per item (a grey image when there is none).",
        "How many reference pictures.", "The final prompt.", "Generation width.", "Generation height.",
        "The selection at the generation size (empty when there is none).", "Human-readable plan.",
        "Run tab > Second pass: on or off.", "Second pass width.", "Second pass height.")
    FUNCTION = "run"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = ("Shot Planner for one still image and MiniMax H3 Image: the image is <Picture 1>, the references "
                   "follow; prompt helpers, SAM 3 selection, crop / mask-only / paste modes and Colored Mark in a panel.")

    @classmethod
    def IS_CHANGED(cls, plan, **kwargs):
        return plan

    def run(self, plan, image=None, prompt=None, mask=None, vlm=None):
        SL = _sl()
        if vlm is not None:
            _VLM["clip"] = vlm
        p = load_plan(plan)
        r = make_plan(p, image, prompt, vlm, mask)
        if r["filled"]:
            _notify("bfs-imageplanner-filled", {"image": p.get("image") or "", **r["filled"]})
        text = summary(r)
        print(f"{TAG} {text}", flush=True)
        empty = torch.zeros(r["height"], r["width"])
        return (r, r["image"], r["refs"] or [SL._grey()], len(r["refs"]), r["prompt"], r["width"], r["height"],
                (r["mask"] if r["mask"] is not None else empty)[None], text, r["second_pass"], *r["upscale"])


_VLM: dict = {}


# ---------------------------------------------------------------------------- conditioning

def guide_image(r: dict, normals=None, source_blur: str = "off", blur_strength: float = 1.0,
                blur_segmenter: str = "SAM 3.1", blur_effect: str = "blur", log: str = "") -> torch.Tensor:
    """<Picture 1> as the model sees it: source blur, then the Colored Marks."""
    SL = _sl()
    g = r["image"]
    if source_blur and source_blur != "off":
        try:
            from .bfs_source_blur import blur_source
        except ImportError:
            from bfs_source_blur import blur_source
        area = r["mask"][None] if r.get("mask") is not None else None
        g = blur_source(g, source_blur, float(blur_strength), log, blur_segmenter, blur_effect, area)
    for mk, x in r.get("marks") or []:
        g = SL.paint_mark(g, mk, x["color"], r["color_guide_opacity"], r["color_guide_style"], normals)
    return g


def image_inpaint_latent(latent: dict, vae, image: torch.Tensor, mask: torch.Tensor, strength: float = 1.0) -> dict:
    """The H3 Image latent [1, C, 1, h, w] starting from the image, with a noise mask (1 = regenerate, grown by one
    latent cell; strength < 1 keeps some of the original)."""
    z = latent["samples"]
    h, w = z.shape[-2], z.shape[-1]
    enc = vae.encode(resize(image, w * 16, h * 16)[..., :3])
    while enc.ndim < z.ndim:
        enc = enc.unsqueeze(2)
    m = torch.nn.functional.interpolate(mask[None, None].float(), size=(h, w), mode="area")[0, 0]
    m = torch.nn.functional.max_pool2d((m > 0.01).float()[None, None], 3, 1, 1)[0, 0]
    out = dict(latent)
    out["samples"] = enc.to(z)
    out["noise_mask"] = (m * float(min(1.0, max(0.0, strength))))[None, None, None].to(z)
    return out


def condition(clip, vae, r: dict, inpaint: str = INPAINT_MODES[0], prompt: str | None = None, normals=None,
              source_blur: str = "off", blur_strength: float = 1.0, blur_segmenter: str = "SAM 3.1",
              blur_effect: str = "blur", node=None, references: list | None = None):
    log = f"{TAG} conditioning"
    g = guide_image(r, normals, source_blur, blur_strength, blur_segmenter, blur_effect, log)
    refs = list(r.get("refs") or []) if references is None else list(references)
    if references is not None and len(refs) != len(r.get("refs") or []):
        print(f"{log}: {len(refs)} connected reference(s) instead of the planner's {len(r.get('refs') or [])}: "
              "the prompt's <Picture n> numbers follow the connected ones", flush=True)
    pictures = [g] + refs
    if len(pictures) > MAX_PICTURES:
        raise ValueError(f"{log}: {len(pictures)} pictures; the model takes up to {MAX_PICTURES}")
    text = prompt if prompt is not None and str(prompt).strip() else r["prompt"]
    if node is None:
        from comfy_extras import nodes_minimax_h3 as H3   # type: ignore
        node = H3.PH3ImageConditioning if r.get("variant") == VARIANTS[1] else H3.MiniMaxH3ImageConditioning
    out = node.execute(clip=clip, vae=vae, prompt=text, width=r["width"], height=r["height"],
                       images={f"image_{k}": x for k, x in enumerate(pictures, 1)})
    positive, latent = (out.args if hasattr(out, "args") else out)[:2]
    use = r.get("inpaint") if inpaint == INPAINT_MODES[0] else inpaint == INPAINT_MODES[2]
    if use:
        if r.get("mask") is None:
            raise ValueError(f"{log}: inpaint 'only the mask' needs a selection on the image")
        if isinstance(latent.get("samples"), torch.Tensor):
            latent = image_inpaint_latent(latent, vae, r["image"], r["mask"], r.get("strength", 1.0))
            print(f"{log}: only the selection is regenerated (strength {r.get('strength', 1.0):.2f})", flush=True)
        else:
            print(f"{log}: this latent has no plain image tensor: inpaint skipped", flush=True)
    return positive, latent, text, g


class BFSImagePlannerH3Conditioning:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from .bfs_source_blur import EFFECTS, SEGMENTERS, SOURCE_BLUR
        except ImportError:
            EFFECTS, SEGMENTERS, SOURCE_BLUR = ["blur"], ["SAM 3.1"], ["off"]
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The H3 Image VAE (also needed for 'only the mask')."}),
                "plan": (PLAN_TYPE, {"tooltip": "From BFS Image Planner."}),
                "inpaint": (list(INPAINT_MODES), {"default": INPAINT_MODES[0],
                            "tooltip": "per plan: the panel's Generate mode decides. only the mask: regenerate only "
                                       "the selection. off: the whole canvas."}),
                "source_blur": (SOURCE_BLUR, {"default": "off", "tooltip": "Blur (or invert) the source subject in "
                                "<Picture 1> so the model does not copy it."}),
                "source_blur_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.05}),
                "source_blur_segmenter": (SEGMENTERS, {"default": SEGMENTERS[0]}),
                "source_blur_effect": (EFFECTS, {"default": EFFECTS[0]}),
            },
            "optional": {
                "references": ("IMAGE", {"tooltip": "Optional: replaces the planner's references, in the same order "
                                                    "(e.g. its refs output after background removal). A list or a "
                                                    "batch; every picture is one <Picture n> from 2 on."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Overrides the planner's final prompt."}),
                "normals": ("IMAGE", {"tooltip": "MoGe normals of the planner's image output (Render MoGe Geometry, "
                                                 "normal_directx), for the Colored Mark 'normal map' style."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "IMAGE")
    RETURN_NAMES = ("positive", "latent", "prompt", "guide")
    OUTPUT_TOOLTIPS = ("H3 Image conditioning with every picture.", "The image latent (with a noise mask in mask-only).",
                       "The prompt that was encoded.", "<Picture 1> as the model sees it (marks, blur).")
    FUNCTION = "run"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = ("Native MiniMax H3 Image (or p-h3-image) conditioning from BFS Image Planner: <Picture 1> with its "
                   "Colored Marks and source blur, the references, the prompt and the generation size; 'only the mask' "
                   "regenerates the selection only.")

    # the references come as a list (one item per picture, e.g. after a per-picture background removal)
    INPUT_IS_LIST = True

    def run(self, clip, vae, plan, inpaint, source_blur, source_blur_strength, source_blur_segmenter, source_blur_effect,
            references=None, prompt=None, normals=None):
        one = lambda v, d=None: (v[0] if v else d) if isinstance(v, list) else (d if v is None else v)
        refs = None
        if references is not None:
            refs = [x[i:i + 1] for x in (references if isinstance(references, list) else [references])
                    if x is not None for i in range(x.shape[0])]
        positive, latent, text, g = condition(one(clip), one(vae), one(plan), one(inpaint, INPAINT_MODES[0]), one(prompt),
                                              one(normals), one(source_blur, "off"), float(one(source_blur_strength, 1.0)),
                                              one(source_blur_segmenter, "SAM 3.1"), one(source_blur_effect, "blur"),
                                              references=refs)
        return (positive, latent, text, g)


# ---------------------------------------------------------------------------- paste back

def paste_result(result: torch.Tensor, r: dict, new_outline: bool = True, **finish) -> torch.Tensor:
    """Crop modes: the result goes back into the full image. Frame + paste: only the subject is pasted onto the
    original (plus the new subject's outline found by SAM 3 near it). Other modes: the result as it is."""
    SL = _sl()
    res = result[:1, ..., :3].float()
    if r.get("crop"):
        return SL.uncrop(res, {"crop": r["crop"], "full_frames": r["full_frames"]}, **finish)
    if not r.get("paste"):
        return res
    base = r["image"]
    H, W = base.shape[1:3]
    res = resize(res, W, H)
    m = r["mask"].float()[None]
    for mk, _ in (r.get("marks") or [])[1:]:
        m = torch.maximum(m, mk.float()[None])
    if new_outline and bool(m.any()):
        x0, y0, x1, y1 = SL.box_px(SL.crop_box(m, 0.5), W, H)
        what = str(r.get("paste_text") or "").strip() or "person"
        new = SL.segment_frames(_working(res), dict(SL.DEFAULT_MASK, text=what, max_objects=8))
        SL._node_boundary()
        new = resize_mask(new[:1], W, H)
        near = torch.zeros(1, H, W)
        near[:, y0:y1, x0:x1] = 1
        m = torch.maximum(m, new * near)
    sp = r.get("spec") or {}
    crop = {"box": [0.0, 0.0, 1.0, 1.0], "mask": m, "paste": "mask",
            "expand": int(sp.get("expand") if sp.get("expand") is not None else 16),
            "feather": int(sp.get("feather") if sp.get("feather") is not None else 12)}
    return SL.uncrop(res, {"crop": crop, "full_frames": base}, **finish)


class BFSImagePlannerPasteBack:
    @classmethod
    def INPUT_TYPES(cls):
        SL = _sl()
        return {
            "required": {
                "plan": (PLAN_TYPE,),
                "image": ("IMAGE", {"tooltip": "The decoded result."}),
                "new_outline": ("BOOLEAN", {"default": True, "tooltip": "Frame + paste: also paste the new subject's "
                                            "outline (SAM 3 on the result), so a bigger silhouette is not cut."}),
                "edge_hardness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "match_colors": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                           "tooltip": "Colour-match the pasted result to the original (LAB)."}),
                "match_region": (SL.MATCH_REGIONS, {"default": SL.MATCH_REGIONS[0]}),
                "seamless": ("BOOLEAN", {"default": False, "tooltip": "Poisson seamless clone at the paste."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "comparison", "original")
    OUTPUT_TOOLTIPS = ("The final image: pasted back in the crop and Frame + paste modes, else the result.",
                       "Original | result, side by side.", "The original at the result's size (e.g. for BFS Video Compare).")
    FUNCTION = "run"
    CATEGORY = "BFS/MiniMax H3"

    def run(self, plan, image, new_outline, edge_hardness, match_colors, match_region, seamless):
        out = paste_result(image, plan, new_outline, edge_hardness=edge_hardness, match_colors=match_colors,
                           match_region=match_region, seamless=seamless)
        # a crop is pasted into the source image; every other mode stays at the generation size
        orig = resize(plan["source"] if plan.get("crop") else plan["image"], out.shape[2], out.shape[1])
        return (out, torch.cat([orig, out], dim=2), orig)


# ---------------------------------------------------------------------------- panel routes

def _jpeg(img: np.ndarray, quality: int = 88) -> str:
    import cv2
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def mask_preview(p: dict, override: dict | None = None) -> dict:
    """The image with the selection tinted (and the crop box), as a data URL, plus its coverage."""
    SL = _sl()
    src = SL._load_image(p.get("image"))
    if src is None:
        raise ValueError("pick an image first")
    spec = mask_spec(p, override)
    m = segment(src, spec)
    small = _working(src)
    H, W = small.shape[1:3]
    ms = resize_mask(m, W, H)
    color = (override or {}).get("preview_color") or "#ff283c"
    style = p.get("color_guide_style") or "overlay" if p.get("color_guide") else "overlay"
    opacity = float(p.get("color_guide_opacity") or 0.35) if p.get("color_guide") else 0.55
    out = SL.paint_mark(small, ms, color, opacity, style if style != "normal map" else "overlay")[0]
    fr = (out.clamp(0, 1) * 255).round().to(torch.uint8).numpy().copy()
    box = SL.crop_box(m[None], float(spec["padding"])) if bool(m.any()) else None
    if box is not None and p.get("crop"):
        import cv2
        x0, y0, x1, y1 = SL.box_px(box, W, H)
        cv2.rectangle(fr, (x0, y0), (x1 - 1, y1 - 1), (255, 220, 0), 2)
    return {"src": _jpeg(fr), "coverage": float(m.float().mean()), "empty": not bool(m.any()), "box": box}


def _setup_routes():
    try:
        from aiohttp import web
        from server import PromptServer
        import folder_paths
    except Exception:  # noqa: BLE001 - outside ComfyUI
        return
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def clip_or_fail():
        clip = _VLM.get("clip")
        if clip is None:
            raise ValueError("The panel has no VLM yet. Connect a Qwen3-VL (CLIPLoader) to the planner's vlm input and "
                             "queue the workflow once.")
        return clip

    async def run(fn):
        return await _sl()._off_loop(fn)

    def handler(fn):
        async def h(request):
            try:
                body = await request.json() if request.method == "POST" else {}
                return web.json_response(await fn(body))
            except Exception as exc:  # noqa: BLE001
                return web.json_response({"error": f"{type(exc).__name__}: {exc}"}, status=500)
        return h

    routes = PromptServer.instance.routes

    @routes.get("/bfs/imageplanner/files")
    async def _files(request):
        root = folder_paths.get_input_directory()
        out = []
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.lower().endswith(exts):
                    out.append(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/"))
        return web.json_response({"images": sorted(out, key=str.lower), "vlm": _VLM.get("clip") is not None})

    async def _mask(body):
        p = load_plan(body.get("plan") or {})
        return await run(lambda: mask_preview(p, body.get("mask")))

    async def _vlm(body):
        p, clip = load_plan(body.get("plan") or {}), clip_or_fail()
        SL = _sl()
        return await run(lambda: analyse_image(clip, SL._load_image(p["image"]), SL.vlm_cfg(p.get("vlm_cfg"))))

    async def _target(body):
        p, SL = load_plan(body.get("plan") or {}), _sl()

        def go():
            src = SL._load_image(p["image"])
            crop = target_crop(src, segment(src, mask_spec(p)))
            if crop is None:
                raise ValueError("the selection found nothing")
            clip = _VLM.get("clip")
            desc = ""
            if clip is not None:
                vc = SL.vlm_cfg(p.get("vlm_cfg"))
                text = picture_markers(p.get("global_prompt") or "", 0)
                slot = SL.prompt_slot(text, "{target}") if vc.get("fit_prompt", True) and "{target}" in text else ""
                desc = SL.describe_target(clip, crop, slot=slot, temperature=float(vc.get("temperature") or 0),
                                          seed=int(vc.get("seed") or 0), image_max_side=int(vc.get("image_max_side") or 1024))
            return {"desc": desc, "crop": _jpeg(crop)}
        return await run(go)

    async def _describe(body):
        p, clip, SL = load_plan(body.get("plan") or {}), clip_or_fail(), _sl()
        want = [str(x) for x in body.get("pictures") or picture_names(p)]

        def go():
            vc = SL.vlm_cfg(p.get("vlm_cfg"))
            names = picture_names(p)
            text = picture_markers(p.get("global_prompt") or "", len(names) - 1)
            out = {}
            for k, name in enumerate(names):
                if name not in want:
                    continue
                img = SL._load_image(name)
                marker = "{details@Picture %d}" % (k + 1)
                out[name] = SL.describe_picture(clip, img, text, marker if marker in text else "", SL.picture_cfg(vc, p, name))
            return {"pictures": out}
        return await run(go)

    async def _autoprompt(body):
        p, clip, SL = load_plan(body.get("plan") or {}), clip_or_fail(), _sl()

        def go():
            src = SL._load_image(p["image"])
            refs = [SL._load_image(n) for n in ref_names(p)]
            return {"text": write_prompt(clip, p, src, refs, dict(p.get("picture_details") or {})), "for": auto_for(p)}
        return await run(go)

    async def _size(body):
        p = load_plan(body.get("plan") or {})
        w, h = int(body.get("w") or 0), int(body.get("h") or 0)
        if not (w and h):
            return {"width": 0, "height": 0}
        W, H = generation_size(w, h, p)
        sp = {**_sl().SECOND_PASS_DEFAULTS, **(p.get("second_pass") or {})}
        UW, UH = _sl().second_pass_size(W, H, sp)
        return {"width": W, "height": H, "upscale_width": UW, "upscale_height": UH}

    for path, fn in (("mask", _mask), ("vlm", _vlm), ("target", _target), ("describe", _describe),
                     ("autoprompt", _autoprompt), ("size", _size)):
        routes.post(f"/bfs/imageplanner/{path}")(handler(fn))


_setup_routes()

NODE_CLASS_MAPPINGS = {
    "BFSImagePlanner": BFSImagePlanner,
    "BFSImagePlannerH3Conditioning": BFSImagePlannerH3Conditioning,
    "BFSImagePlannerPasteBack": BFSImagePlannerPasteBack,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSImagePlanner": "BFS Image Planner (H3 Image)",
    "BFSImagePlannerH3Conditioning": "BFS Image Planner H3 Conditioning",
    "BFSImagePlannerPasteBack": "BFS Image Planner Paste Back",
}
