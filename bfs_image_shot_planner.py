"""BFS Image Shot Planner: the Shot Planner for still images and MiniMax H3 Image.

Every input image is one "shot": it gets its own prompt and its own reference pictures, the planner outputs the
shots as a ComfyUI list, so every node downstream (conditioning, sampler, decode) runs once per image, and
BFS Image Shot Join collects the results again.

    images ─► BFS Image Shot Planner ─shots─► BFS Image Shot H3 Conditioning ─positive/latent─► sampler ─► VAE Decode
                                                                                                        │
              BFS Image Shot Join ◄──────────────────────────── images ───────────────────────────────────┘

In the H3 Image prompt the shot's image is <Picture 1> (the picture to edit) and its references are <Picture 2>,
<Picture 3>, ... in the order given (with `source_as_picture_1` off the references start at <Picture 1> and the
image only sets the size: text-to-image with references).
"""
from __future__ import annotations

import importlib.util
import math
import os
import re
import sys

import torch

CANVAS_MULTIPLE = 32            # the H3 Image canvas grid (nodes_minimax_h3.CANVAS_MULTIPLE)
H3_SHORT_EDGE = (1024, 2048)    # H3 Image: the edit canvas keeps image_1's short edge inside this range
PH3_AREA = 2912 * 2912          # p-h3-image: the canvas fits image_1 inside this area (down only)
MAX_PICTURES = 100              # the H3 Image conditioning autogrow capacity

VARIANTS = ("MiniMax H3 Image", "p-h3-image")
SIZE_MODES = ("native", "megapixels", "source size", "custom")
SEED_MODES = ("increment", "fixed")
FITS = ("pad", "crop", "stretch")
DESCRIBE_PRESETS = ("short", "object", "full body", "head / face", "face attributes", "outfit", "scene / setting")
SHOT_TYPE = "BFS_IMAGE_SHOT"
TAG = "[BFS Image Shot Planner]"

_BLOCK_SPLIT = re.compile(r"^[ \t]*---+[ \t]*$", re.M)
_DETAILS_AT = re.compile(r"\{details@Picture\s*(\d+)\}")


# ---------------------------------------------------------------------------- parsing

def split_images(items) -> list[torch.Tensor]:
    """A list of IMAGE batches (or one batch) -> single images [1, H, W, C]."""
    if items is None:
        return []
    if isinstance(items, torch.Tensor):
        items = [items]
    out = []
    for batch in items:
        if batch is None:
            continue
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        out.extend(batch[i:i + 1] for i in range(batch.shape[0]))
    return out


def parse_selection(text: str, n: int) -> list[int]:
    """'1-3, 5' -> [0, 1, 2, 4] (1-based, inclusive ranges; '' or 'all' = every shot). Out-of-range numbers are an
    error, so a typo does not silently skip shots."""
    t = str(text or "").strip().lower()
    if t in ("", "all", "*"):
        return list(range(n))
    picked = set()
    for part in re.split(r"[,\s;]+", t):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"{TAG} run: cannot read {part!r} (write e.g. '1-3, 5', or leave it empty for all)")
        a = int(m.group(1))
        b = int(m.group(2) or a)
        if a > b:
            a, b = b, a
        if a < 1 or b > n:
            raise ValueError(f"{TAG} run: {part!r} is outside 1-{n} (there are {n} images)")
        picked.update(range(a - 1, b))
    return sorted(picked)


def split_blocks(text: str) -> list[str]:
    """Per-shot texts: blocks separated by a line with only '---'. Block k belongs to shot k; an empty block (or a
    missing one) means 'use the global prompt'."""
    t = str(text or "")
    if not t.strip():
        return []
    return [b.strip() for b in _BLOCK_SPLIT.split(t)]


def parse_refs(spec: str, pool: int, default: str = "all", where: str = "") -> list[int]:
    """Which pool references a shot uses, in <Picture> order: 'all', 'none', or '2, 1, 3' / '1-3' (1-based; order and
    repeats are kept as written). '' falls back to `default`."""
    t = str(spec or "").strip().lower()
    if not t:
        t = str(default or "all").strip().lower() or "all"
    if t in ("all", "*"):
        return list(range(pool))
    if t in ("none", "-", "0"):
        return []
    out = []
    for part in re.split(r"[,\s;]+", t):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"{TAG} {where}references: cannot read {part!r} (write 'all', 'none' or e.g. '1, 3')")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        step = 1 if b >= a else -1
        for k in range(a, b + step, step):
            if not 1 <= k <= pool:
                raise ValueError(f"{TAG} {where}references: reference {k} does not exist "
                                 f"({pool} reference picture{'s' if pool != 1 else ''} connected)")
            out.append(k - 1)
    return out


# ---------------------------------------------------------------------------- sizes

def _grid(x: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, int(round(x / multiple)) * multiple)


def native_size(w: int, h: int, variant: str = VARIANTS[0]) -> tuple[int, int]:
    """The canvas the native conditioning node picks for an image_1 of w x h (width = height = 0)."""
    if variant == VARIANTS[1]:
        scale = min(1.0, math.sqrt(PH3_AREA / (w * h)))
    else:
        lo, hi = H3_SHORT_EDGE
        scale = min(max(min(w, h), lo), hi) / min(w, h)
    return _grid(w * scale), _grid(h * scale)


def shot_size(w: int, h: int, mode: str, megapixels: float, width: int, height: int, multiple: int,
              variant: str = VARIANTS[0]) -> tuple[int, int]:
    multiple = max(CANVAS_MULTIPLE, int(multiple) // CANVAS_MULTIPLE * CANVAS_MULTIPLE)
    if mode == "megapixels":
        scale = math.sqrt(float(megapixels) * 1024 * 1024 / (w * h))
        return _grid(w * scale, multiple), _grid(h * scale, multiple)
    if mode == "source size":
        return _grid(w, multiple), _grid(h, multiple)
    if mode == "custom":
        return _grid(int(width), multiple), _grid(int(height), multiple)
    return native_size(w, h, variant)


# ---------------------------------------------------------------------------- prompts

def picture_numbers(source_is_picture_1: bool, n_refs: int) -> dict:
    """Where things sit in the <Picture n> numbering of one shot."""
    first = 2 if source_is_picture_1 else 1
    return {"source": 1 if source_is_picture_1 else 0, "refs": list(range(first, first + n_refs))}


def expand_prompt(text: str, index: int, count: int, name: str, source_is_picture_1: bool, n_refs: int) -> str:
    """Plain placeholders: {index} / {count} / {name}; {details} / {details2} = the first / second reference,
    {image} = the shot's own image (resolved to {details@Picture n}, which the VLM fills)."""
    nums = picture_numbers(source_is_picture_1, n_refs)
    t = str(text or "")
    t = t.replace("{index}", str(index + 1)).replace("{count}", str(count)).replace("{name}", name)
    first = 2 if source_is_picture_1 else 1
    for key, k in (("{details}", 0), ("{details2}", 1)):
        # a missing reference keeps its would-be number, so the fill step reports it instead of describing another
        t = t.replace(key, "{details@Picture %d}" % (nums["refs"][k] if k < n_refs else first + k))
    # Picture 0 = the shot's image when it is not one of the pictures (text-to-image)
    return t.replace("{image}", "{details@Picture %d}" % nums["source"])


def _shot_loop():
    """bfs_shot_loop (the video planner) for its VLM description helpers, loaded only when a VLM is connected."""
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


def fill_details(text: str, pictures: list, source: torch.Tensor, source_is_picture_1: bool, vlm=None,
                 cfg: dict | None = None, tag: str = TAG, describe=None) -> str:
    """{details@Picture k} = a short VLM description of picture k (k = 0 means the shot's image when it is not one of
    the pictures). Without a VLM the marker becomes 'character' and the log says so."""
    if "{details@" not in (text or ""):
        return text
    describer = describe
    if describer is None and vlm is not None:
        SL = _shot_loop()
        base = SL.vlm_cfg(cfg or {})
        describer = lambda image, prompt, marker: SL.describe_picture(vlm, image, prompt, marker, base)

    def sub(m):
        k = int(m.group(1))
        image = source if k == 0 else (pictures[k - 1] if 1 <= k <= len(pictures) else None)
        if image is None:
            print(f"{tag}: {m.group(0)}: the shot has no picture {k}", flush=True)
            return "character"
        if describer is None:
            print(f"{tag}: {m.group(0)}: connect a VLM (vlm input) to describe it; using 'character'", flush=True)
            return "character"
        d = str(describer(image, text, m.group(0)) or "").strip()
        if d:
            print(f"{tag}: {m.group(0)} = {d!r} (described automatically)", flush=True)
        return d or "character"
    return _DETAILS_AT.sub(sub, text)


# ---------------------------------------------------------------------------- planning

def _first(v, default=None):
    """INPUT_IS_LIST hands every input as a list: the single value of a widget input."""
    if isinstance(v, list):
        return v[0] if v else default
    return default if v is None else v


def build_shots(images: list, refs: list, *, global_prompt: str = "", shot_prompts: str = "",
                prompt_list: list | None = None, ref_assign: str = "", default_refs: str = "all",
                source_as_picture_1: bool = True, run: str = "", variant: str = VARIANTS[0],
                size_mode: str = "native", megapixels: float = 4.0, width: int = 2048, height: int = 2048,
                multiple: int = CANVAS_MULTIPLE, seed: int = 0, seed_mode: str = "increment",
                names: list | None = None) -> list[dict]:
    """The shots of a plan (only the ones that run), before the VLM fills any {details}."""
    n = len(images)
    if n == 0:
        raise ValueError(f"{TAG}: connect at least one image")
    blocks = split_blocks(shot_prompts)
    if len(blocks) > n:
        print(f"{TAG}: shot_prompts has {len(blocks)} blocks for {n} images; the extra ones are ignored", flush=True)
    ref_lines = str(ref_assign or "").splitlines()
    prompt_list = [p for p in (prompt_list or []) if p is not None]
    shots = []
    for i in parse_selection(run, n):
        img = images[i]
        h, w = int(img.shape[1]), int(img.shape[2])
        own = blocks[i] if i < len(blocks) else ""
        if i < len(prompt_list) and str(prompt_list[i]).strip():
            own = str(prompt_list[i]).strip()          # a connected prompt list beats the typed blocks
        prompt = own or str(global_prompt or "").strip()
        ids = parse_refs(ref_lines[i] if i < len(ref_lines) else "", len(refs), default_refs, f"shot {i + 1}: ")
        pictures = [refs[k] for k in ids]
        npics = len(pictures) + (1 if source_as_picture_1 else 0)
        if npics > MAX_PICTURES:
            raise ValueError(f"{TAG} shot {i + 1}: {npics} pictures; MiniMax H3 Image takes up to {MAX_PICTURES}")
        name = (names[i] if names and i < len(names) and names[i] else f"image_{i + 1:03d}")
        gw, gh = shot_size(w, h, size_mode, megapixels, width, height, multiple, variant)
        if not prompt:
            print(f"{TAG} shot {i + 1}: no prompt (neither its own nor the global one)", flush=True)
        shots.append({
            "index": i, "count": n, "name": name, "image": img, "refs": pictures, "ref_ids": [k + 1 for k in ids],
            "prompt": expand_prompt(prompt, i, n, name, source_as_picture_1, len(pictures)),
            "width": gw, "height": gh, "source_width": w, "source_height": h,
            "seed": int(seed) + (i if seed_mode == "increment" else 0),
            "source_is_picture_1": bool(source_as_picture_1), "variant": variant,
        })
    if not shots:
        raise ValueError(f"{TAG}: no image is selected to run")
    return shots


def shot_pictures(shot: dict) -> list[torch.Tensor]:
    """The pictures of a shot in <Picture n> order (the image to edit first, when it is one)."""
    pics = [shot["image"]] if shot.get("source_is_picture_1", True) else []
    return pics + [r for r in shot.get("refs") or [] if r is not None]


def summary(shots: list[dict], n_images: int, n_refs: int) -> str:
    lines = [f"{len(shots)} of {n_images} image(s) run, {n_refs} reference picture(s) in the pool"]
    for s in shots:
        pics = ["<Picture 1> = the image"] if s["source_is_picture_1"] else []
        first = 2 if s["source_is_picture_1"] else 1
        pics += [f"<Picture {first + j}> = ref {k}" for j, k in enumerate(s["ref_ids"])]
        p = " ".join(s["prompt"].split())
        lines.append(f"#{s['index'] + 1} {s['name']}: {s['source_width']}x{s['source_height']} -> "
                     f"{s['width']}x{s['height']}, seed {s['seed']}; {', '.join(pics) or 'no pictures'}; "
                     f"prompt: {p[:160] + ('...' if len(p) > 160 else '') or '(empty)'}")
    return "\n".join(lines)


class BFSImageShotPlanner:
    """The Shot Planner for still images: one shot per image, with its own prompt and references."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to edit, one shot each: a batch, or a list (e.g. a "
                                                "folder loader) when they have different sizes."}),
                "global_prompt": ("STRING", {"default": "", "multiline": True,
                                             "tooltip": "Prompt for every shot without its own. <Picture 1> is the "
                                                        "shot's image, <Picture 2>, <Picture 3>... its references. "
                                                        "Placeholders: {index} {count} {name}; {details} / {details2} "
                                                        "/ {image} / {details@Picture n} are described by the VLM."}),
                "shot_prompts": ("STRING", {"default": "", "multiline": True,
                                            "tooltip": "Optional own prompt per image: blocks separated by a line "
                                                       "with only ---, in image order. An empty block uses the "
                                                       "global prompt."}),
                "ref_assign": ("STRING", {"default": "", "multiline": True,
                                          "tooltip": "Optional references per image, one line per image: 'all', "
                                                     "'none' or numbers of the `references` pictures (e.g. '2, 1' = "
                                                     "<Picture 2> is reference 2, <Picture 3> reference 1). An empty "
                                                     "or missing line uses default_refs."}),
                "default_refs": ("STRING", {"default": "all",
                                            "tooltip": "References of images without a ref_assign line: 'all', "
                                                       "'none' or e.g. '1, 2'."}),
                "source_as_picture_1": ("BOOLEAN", {"default": True,
                                                    "tooltip": "On: the image is <Picture 1>, the picture H3 Image "
                                                               "edits. Off: text-to-image with the references only; "
                                                               "the image just sets the size."}),
                "run": ("STRING", {"default": "", "tooltip": "Which images run, e.g. '1-3, 5'. Empty = all."}),
                "variant": (list(VARIANTS), {"default": VARIANTS[0],
                                             "tooltip": "The image model: sets the native canvas size and which "
                                                        "native conditioning node is used."}),
                "size_mode": (list(SIZE_MODES), {"default": "native",
                                                 "tooltip": "native: what the model's own node picks (H3 Image: "
                                                            "short edge 1024-2048). megapixels: that area, keeping "
                                                            "the aspect. source size: the image's own size. custom: "
                                                            "width x height. Always on the 32 px grid."}),
                "megapixels": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "width": ("INT", {"default": 2048, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 2048, "min": 32, "max": 16384, "step": 32}),
                "multiple": ("INT", {"default": 32, "min": 32, "max": 256, "step": 32,
                                     "tooltip": "Size grid for megapixels / source size / custom."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "seed_mode": (list(SEED_MODES), {"default": "increment",
                                                 "tooltip": "increment: image k gets seed + k - 1. fixed: every "
                                                            "image gets the seed."}),
                "describe_preset": (list(DESCRIBE_PRESETS), {"default": "short",
                                                             "tooltip": "How the VLM writes {details} descriptions."}),
            },
            "optional": {
                "references": ("IMAGE", {"tooltip": "The reference pool: a batch or a list; numbered 1, 2, 3... in "
                                                    "order (ref_assign / default_refs pick from it)."}),
                "prompts": ("STRING", {"forceInput": True,
                                       "tooltip": "Optional list of prompts (one per image, image order); a "
                                                  "non-empty one beats the shot_prompts block."}),
                "names": ("STRING", {"forceInput": True,
                                     "tooltip": "Optional list of names (e.g. the file names), for {name} and the "
                                                "summary."}),
                "vlm": ("CLIP", {"tooltip": "Optional Qwen3-VL text encoder (CLIPLoader) to fill {details}, "
                                            "{details2}, {image} and {details@Picture n}."}),
            },
        }

    RETURN_TYPES = (SHOT_TYPE, "IMAGE", "STRING", "INT", "INT", "STRING", "IMAGE")
    RETURN_NAMES = ("shots", "images", "prompts", "seeds", "count", "summary", "refs")
    OUTPUT_IS_LIST = (True, True, True, True, False, False, True)
    OUTPUT_TOOLTIPS = (
        "One item per image that runs. Every node that receives this list runs once per image: connect it to BFS "
        "Image Shot H3 Conditioning (or BFS Image Shot Unpack), sample, decode, then BFS Image Shot Join.",
        "The image of every shot (list).", "The final prompt of every shot (list).", "The seed of every shot (list).",
        "How many images run.", "Human-readable plan.", "The reference pool, one picture per item (list).")
    FUNCTION = "plan"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = ("Shot Planner for still images (MiniMax H3 Image): every image is one shot with its own prompt and "
                   "reference pictures; the rest of the graph runs once per image. Collect with BFS Image Shot Join.")

    def plan(self, images, global_prompt, shot_prompts, ref_assign, default_refs, source_as_picture_1, run, variant,
             size_mode, megapixels, width, height, multiple, seed, seed_mode, describe_preset, references=None,
             prompts=None, names=None, vlm=None):
        imgs = split_images(images)
        refs = split_images(references)
        shots = build_shots(
            imgs, refs, global_prompt=_first(global_prompt, ""), shot_prompts=_first(shot_prompts, ""),
            prompt_list=prompts, ref_assign=_first(ref_assign, ""), default_refs=_first(default_refs, "all"),
            source_as_picture_1=bool(_first(source_as_picture_1, True)), run=_first(run, ""),
            variant=_first(variant, VARIANTS[0]), size_mode=_first(size_mode, "native"),
            megapixels=float(_first(megapixels, 4.0)), width=int(_first(width, 2048)), height=int(_first(height, 2048)),
            multiple=int(_first(multiple, 32)), seed=int(_first(seed, 0)), seed_mode=_first(seed_mode, "increment"),
            names=[str(x) for x in names] if names else None)
        clip = _first(vlm)
        cfg = {"describe_preset": _first(describe_preset, "short")}
        for s in shots:
            s["prompt"] = fill_details(s["prompt"], shot_pictures(s), s["image"], s["source_is_picture_1"], clip, cfg,
                                       tag=f"{TAG} shot {s['index'] + 1}")
        text = summary(shots, len(imgs), len(refs))
        print(text, flush=True)
        pool = refs or [torch.full((1, 64, 64, 3), 0.5)]
        return (shots, [s["image"] for s in shots], [s["prompt"] for s in shots], [s["seed"] for s in shots],
                len(shots), text, pool)


# ---------------------------------------------------------------------------- per shot

def _conditioning_node(variant: str):
    from comfy_extras import nodes_minimax_h3 as H3   # type: ignore
    return H3.PH3ImageConditioning if variant == VARIANTS[1] else H3.MiniMaxH3ImageConditioning


def condition_shot(clip, vae, shot: dict, prompt: str | None = None, width: int = 0, height: int = 0,
                   extra: list | None = None, node=None):
    pictures = shot_pictures(shot) + [x for x in (extra or []) if x is not None]
    if len(pictures) > MAX_PICTURES:
        raise ValueError(f"BFS Image Shot H3 Conditioning: {len(pictures)} pictures; the model takes up to {MAX_PICTURES}")
    text = prompt if prompt is not None and str(prompt).strip() else shot.get("prompt", "")
    w = int(width) or int(shot.get("width") or 0)
    h = int(height) or int(shot.get("height") or 0)
    node = node or _conditioning_node(shot.get("variant", VARIANTS[0]))
    images = {f"image_{k}": pic for k, pic in enumerate(pictures, 1)}
    out = node.execute(clip=clip, vae=vae, prompt=text, width=w, height=h, images=images or None)
    positive, latent = (out.args if hasattr(out, "args") else out)[:2]
    return positive, latent, text


class BFSImageShotH3Conditioning:
    """Native MiniMax H3 Image conditioning for one shot of BFS Image Shot Planner."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "The H3 Image VAE: the pictures are also anchored as reference latents."}),
                "shot": (SHOT_TYPE, {"tooltip": "From BFS Image Shot Planner (or Repack)."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32,
                                  "tooltip": "0 = the planner's size for this shot."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 32}),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Overrides the shot's prompt."}),
                "extra_picture": ("IMAGE", {"tooltip": "Optional picture appended after the shot's references "
                                                       "(the next <Picture n>), e.g. a marked guide; a batch adds "
                                                       "every frame as its own picture."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "INT")
    RETURN_NAMES = ("positive", "latent", "prompt", "seed")
    FUNCTION = "condition"
    CATEGORY = "BFS/MiniMax H3"
    DESCRIPTION = ("MiniMax H3 Image conditioning for one image shot: the shot's image is <Picture 1>, its references "
                   "follow. Sample the latent (e.g. BasicGuider + SamplerCustomAdvanced) and decode with the H3 Image VAE.")

    def condition(self, clip, vae, shot, width=0, height=0, prompt=None, extra_picture=None):
        positive, latent, text = condition_shot(clip, vae, shot, prompt, width, height, split_images(extra_picture))
        return (positive, latent, text, int(shot.get("seed", 0)))


class BFSImageShotUnpack:
    """One image shot as plain values, for any model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shot": (SHOT_TYPE,)}}

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "STRING", "INT", "INT", "INT", "INT", "STRING", SHOT_TYPE)
    RETURN_NAMES = ("image", "refs", "ref_count", "prompt", "width", "height", "seed", "index", "name", "shot")
    OUTPUT_IS_LIST = (False, True, False, False, False, False, False, False, False, False)
    FUNCTION = "unpack"
    CATEGORY = "BFS/shot loop"
    DESCRIPTION = "The image, references (a list), prompt, size, seed and index of one image shot."

    def unpack(self, shot):
        refs = list(shot.get("refs") or [])
        return (shot["image"], refs or [torch.full((1, 64, 64, 3), 0.5)], len(refs), shot.get("prompt", ""),
                int(shot["width"]), int(shot["height"]), int(shot.get("seed", 0)), int(shot["index"]) + 1,
                shot.get("name", ""), shot)


class BFSImageShotRepack:
    """Edited pieces back into an image shot (unconnected inputs keep the shot's values)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shot": (SHOT_TYPE,)},
                "optional": {"image": ("IMAGE",), "refs": ("IMAGE", {"tooltip": "Replaces every reference: each "
                                                                               "frame of the batch is one picture."}),
                             "prompt": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = (SHOT_TYPE,)
    RETURN_NAMES = ("shot",)
    FUNCTION = "repack"
    CATEGORY = "BFS/shot loop"

    def repack(self, shot, image=None, refs=None, prompt=None):
        s = dict(shot)
        if image is not None:
            s["image"] = image[:1]
        if refs is not None:
            s["refs"] = split_images(refs)
            s["ref_ids"] = list(range(1, len(s["refs"]) + 1))
        if prompt is not None and str(prompt).strip():
            s["prompt"] = str(prompt)
        return (s,)


# ---------------------------------------------------------------------------- join

def fit_image(img: torch.Tensor, w: int, h: int, fit: str = "pad", fill: float = 0.0) -> torch.Tensor:
    """[B, H, W, C] -> [B, h, w, C]: pad (letterbox), crop (fill and centre-crop) or stretch."""
    ih, iw = int(img.shape[1]), int(img.shape[2])
    if (iw, ih) == (w, h):
        return img
    x = img.movedim(-1, 1).float()
    if fit == "stretch":
        return torch.nn.functional.interpolate(x, size=(h, w), mode="bicubic", align_corners=False).clamp(0, 1).movedim(1, -1)
    scale = (min if fit == "pad" else max)(w / iw, h / ih)
    nw, nh = max(1, round(iw * scale)), max(1, round(ih * scale))
    x = torch.nn.functional.interpolate(x, size=(nh, nw), mode="bicubic", align_corners=False).clamp(0, 1)
    if fit == "pad":
        out = torch.full((x.shape[0], x.shape[1], h, w), float(fill), dtype=x.dtype)
        top, left = (h - nh) // 2, (w - nw) // 2
        out[:, :, top:top + nh, left:left + nw] = x
    else:
        top, left = (nh - h) // 2, (nw - w) // 2
        out = x[:, :, top:top + h, left:left + w]
    return out.movedim(1, -1)


def join_images(results: list, shots: list | None = None, fit: str = "pad", size: str = "first") -> tuple:
    """-> (batch of results, batch of originals, batch of side-by-side comparisons), all at one size."""
    frames = [im[..., :3] for im in split_images(results)]
    if not frames:
        raise ValueError("BFS Image Shot Join: no images")
    if size == "largest":
        tw = max(int(f.shape[2]) for f in frames)
        th = max(int(f.shape[1]) for f in frames)
    else:
        th, tw = int(frames[0].shape[1]), int(frames[0].shape[2])
    batch = torch.cat([fit_image(f, tw, th, fit) for f in frames], 0)
    shots = list(shots or [])
    if shots and len(shots) != len(frames):
        print(f"BFS Image Shot Join: {len(frames)} results for {len(shots)} shots; originals follow the shot order",
              flush=True)
    origs = []
    for k, f in enumerate(frames):
        src = shots[k]["image"][..., :3] if k < len(shots) else f
        origs.append(fit_image(src.to(batch.dtype), tw, th, fit))
    originals = torch.cat(origs, 0)
    comparison = torch.cat([originals, batch], dim=2)
    return batch, originals, comparison


class BFSImageShotJoin:
    """Collects the decoded image shots into one batch (plus the originals and a before | after comparison)."""

    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The decoded result of every shot (the list from VAE Decode)."}),
                "fit": (list(FITS), {"default": "pad", "tooltip": "How results of other sizes are brought to the "
                                                                   "batch size (the list output keeps them as they are)."}),
                "size": (["first", "largest"], {"default": "first"}),
            },
            "optional": {"shots": (SHOT_TYPE, {"tooltip": "The planner's shots list: gives the originals and the "
                                                         "comparison."})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "originals", "comparison", "image_list")
    OUTPUT_IS_LIST = (False, False, False, True)
    OUTPUT_TOOLTIPS = ("Every result in one batch, at one size.", "The shots' source images, at the same size.",
                       "Original | result, side by side.", "Every result at its own size (a list, for Save Image).")
    FUNCTION = "join"
    CATEGORY = "BFS/shot loop"

    def join(self, images, fit, size, shots=None):
        batch, originals, comparison = join_images(images, shots, _first(fit, "pad"), _first(size, "first"))
        return (batch, originals, comparison, [im[..., :3] for im in split_images(images)])


NODE_CLASS_MAPPINGS = {
    "BFSImageShotPlanner": BFSImageShotPlanner,
    "BFSImageShotH3Conditioning": BFSImageShotH3Conditioning,
    "BFSImageShotUnpack": BFSImageShotUnpack,
    "BFSImageShotRepack": BFSImageShotRepack,
    "BFSImageShotJoin": BFSImageShotJoin,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BFSImageShotPlanner": "BFS Image Shot Planner (H3 Image)",
    "BFSImageShotH3Conditioning": "BFS Image Shot H3 Image Conditioning",
    "BFSImageShotUnpack": "BFS Image Shot Unpack",
    "BFSImageShotRepack": "BFS Image Shot Repack",
    "BFSImageShotJoin": "BFS Image Shot Join",
}
