from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
_TMP = tempfile.mkdtemp(prefix="bfs_imageplanner_test_")

# folder_paths / comfy_execution stubs, as in test_bfs_shot_loop (the modules only need a few calls)
_fp = types.ModuleType("folder_paths")
_fp.get_temp_directory = lambda: _TMP
_fp.get_input_directory = lambda: _TMP
_fp.get_annotated_filepath = lambda name: str(Path(_TMP) / name)
sys.modules.setdefault("folder_paths", _fp)
_ce = types.ModuleType("comfy_execution")
_gu = types.ModuleType("comfy_execution.graph_utils")
_gu.ExecutionBlocker = type("ExecutionBlocker", (), {"__init__": lambda self, m: None})
sys.modules.setdefault("comfy_execution", _ce)
sys.modules.setdefault("comfy_execution.graph_utils", _gu)

if "bfs_shot_loop" not in sys.modules:
    spec = importlib.util.spec_from_file_location("bfs_shot_loop", ROOT / "bfs_shot_loop.py")
    SL = importlib.util.module_from_spec(spec)
    sys.modules["bfs_shot_loop"] = SL
    spec.loader.exec_module(SL)
spec = importlib.util.spec_from_file_location("bfs_h3_image_planner", ROOT / "bfs_h3_image_planner.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def save_png(name, value, h=60, w=80):
    from PIL import Image
    arr = np.full((h, w, 3), int(value * 255), np.uint8)
    Image.fromarray(arr).save(Path(_TMP) / name)
    return name


def src_image(h=600, w=800):
    img = torch.full((1, h, w, 3), 0.2)
    img[:, 200:400, 300:500] = 0.8
    return img


def box_mask(h=600, w=800):
    m = torch.zeros(1, h, w)
    m[:, 200:400, 300:500] = 1
    return m


class FakeNode:
    @classmethod
    def execute(cls, clip, vae, prompt, width, height, images=None):
        cls.got = dict(prompt=prompt, width=width, height=height, images=images)
        return ([[torch.zeros(1), {}]], {"samples": torch.zeros(1, 24, 1, height // 16, width // 16)})


class FakeVae:
    def encode(self, px):
        return torch.ones(1, 24, 1, px.shape[1] // 16, px.shape[2] // 16)


def plan(**kw):
    p = {"image": "", "global_refs": [save_png("ref_a.png", .6), save_png("ref_b.png", .7)],
         "global_prompt": "Replace the person in <Picture 1> with the {details}.",
         "picture_details": {"ref_a.png": "the knight in silver armour"}, "mask_cfg": {"expand": 0, "fill_holes": False}}
    p.update(kw)
    return M.load_plan(p)


class SizeTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(M.generation_size(512, 768, M.load_plan({})), (1024, 1536))
        self.assertEqual(M.generation_size(4000, 3000, M.load_plan({})), (2720, 2048))
        self.assertEqual(M.generation_size(4000, 3000, M.load_plan({"variant": "p-h3-image"}))[1], 2528)
        self.assertEqual(M.generation_size(1000, 500, M.load_plan({"generation_size_mode": "original"})), (992, 512))
        self.assertEqual(M.generation_size(1000, 500, M.load_plan({"generation_size_mode": "custom", "custom_width": 1024,
                                                                   "custom_height": 768})), (1024, 768))
        w, h = M.generation_size(1000, 500, M.load_plan({"generation_size_mode": "megapixels", "megapixels": 1}))
        self.assertAlmostEqual(w * h / 2 ** 20, 1, delta=0.06)


class PromptTest(unittest.TestCase):
    def test_markers_and_modes(self):
        self.assertEqual(M.picture_markers("{details} {details2} {image}", 2),
                         "{details@Picture 2} {details@Picture 3} {details@Picture 1}")
        self.assertEqual(M.mode_of({"crop": True, "inpaint": True}), "cropmask")
        self.assertEqual(M.mode_of({"paste": True}), "paste")

    def test_auto_for_is_the_panels_string(self):
        p = plan(image="img.png", color_guide=True, color_guide_style="hatching",
                 color_guide_regions=[{"picture": 3, "text": "the man"}], mask={"text": "woman"})
        got = json.loads(M.auto_for(p))
        self.assertEqual(got[1], ["img.png", "ref_a.png", "ref_b.png"])
        self.assertEqual(got[2], [[2, "#9b30ff"], [3, "#ff2020"], "hatching"])
        self.assertEqual(got[3:], ["frame", "woman", "", "structured", "", M.AUTO_REVISION])
        self.assertNotIn(", ", M.auto_for(p).replace("<Picture 1> with", ""))   # JSON.stringify separators


class PlanTest(unittest.TestCase):
    def test_frame_mode(self):
        r = M.make_plan(plan(), src_image())
        self.assertEqual((r["width"], r["height"]), (1376, 1024))
        self.assertEqual(r["prompt"], "Replace the person in <Picture 1> with the knight in silver armour.")
        self.assertEqual([round(float(x.mean()), 1) for x in r["refs"]], [.6, .7])
        self.assertIsNone(r["mask"])

    def test_selection_is_required(self):
        with self.assertRaises(ValueError):
            M.make_plan(plan(inpaint=True), src_image())

    def test_crop_mode_and_paste_back(self):
        r = M.make_plan(plan(crop=True, mask_cfg={"padding": 0.0, "expand": 0, "fill_holes": False}), src_image(),
                        ext_mask=box_mask())
        self.assertEqual(r["mode"], "crop")
        self.assertEqual((r["width"], r["height"]), (1024, 1024))      # the 200x200 crop, native size
        self.assertGreater(float(r["image"].mean()), 0.75)            # the crop shows the bright subject
        out = M.paste_result(torch.zeros(1, r["height"], r["width"], 3), r)
        self.assertEqual(tuple(out.shape), (1, 600, 800, 3))
        self.assertLess(float(out[0, 300, 400].mean()), 0.1)          # the result is inside the box
        self.assertAlmostEqual(float(out[0, 50, 50].mean()), 0.2, places=2)   # the rest is the original

    def test_colored_mark(self):
        r = M.make_plan(plan(color_guide=True, color_guide_picture=2), src_image(), ext_mask=box_mask())
        self.assertEqual(len(r["marks"]), 1)
        self.assertIn("<Picture 2>", r["prompt"])
        self.assertIn("Color bindings", r["prompt"])
        g = M.guide_image(r)
        self.assertGreater(float((g - r["image"]).abs().max()), 0.1)   # the mark is painted on <Picture 1>

    def test_conditioning_and_inpaint(self):
        r = M.make_plan(plan(inpaint=True, strength=0.8), src_image(), ext_mask=box_mask())
        pos, lat, text, g = M.condition(None, FakeVae(), r, node=FakeNode)
        imgs = FakeNode.got["images"]
        self.assertEqual(list(imgs), ["image_1", "image_2", "image_3"])
        self.assertEqual((FakeNode.got["width"], FakeNode.got["height"]), (1376, 1024))
        self.assertEqual(tuple(lat["noise_mask"].shape), (1, 1, 1, 64, 86))
        self.assertAlmostEqual(float(lat["noise_mask"].max()), 0.8, places=5)
        self.assertEqual(float(lat["noise_mask"][0, 0, 0, 2, 2]), 0.0)
        self.assertEqual(float(lat["samples"].mean()), 1.0)            # starts from the encoded image
        _, lat2, _, _ = M.condition(None, FakeVae(), r, inpaint="off", node=FakeNode)
        self.assertNotIn("noise_mask", lat2)

    def test_paste_mode(self):
        r = M.make_plan(plan(paste=True), src_image(), ext_mask=box_mask())
        out = M.paste_result(torch.zeros(1, r["height"], r["width"], 3), r, new_outline=False)
        self.assertEqual(tuple(out.shape[1:3]), (r["height"], r["width"]))
        self.assertAlmostEqual(float(out[0, 10, 10].mean()), 0.2, places=2)
        self.assertLess(float(out[0, r["height"] // 2, r["width"] // 2].mean()), 0.05)

    def test_node_outputs(self):
        node = M.BFSImagePlanner()
        out = node.run(json.dumps(plan()), image=src_image())
        self.assertEqual(len(out), len(M.BFSImagePlanner.RETURN_TYPES))
        self.assertEqual(out[3], 2)
        self.assertEqual(tuple(out[7].shape), (1, 1024, 1376))
        pb = M.BFSImagePlannerPasteBack().run(out[0], torch.zeros(1, 1024, 1376, 3), False, 0.0, 0.0,
                                               SL.MATCH_REGIONS[0], False)
        self.assertEqual(tuple(pb[1].shape), (1, 1024, 2752, 3))


class ReferencesInputTest(unittest.TestCase):
    def test_connected_references_replace_the_plans(self):
        r = M.make_plan(plan(), src_image())
        old = M.condition
        seen = {}
        M.condition = lambda *a, **k: seen.update(k) or (None, None, "", None)
        try:
            M.BFSImagePlannerH3Conditioning().run([None], [None], [r], ["off"], ["off"], [1.0], ["SAM 3.1"], ["blur"],
                                                 references=[torch.full((1, 8, 8, 3), .3), torch.full((2, 8, 8, 3), .4)])
        finally:
            M.condition = old
        self.assertEqual([round(float(x.mean()), 1) for x in seen["references"]], [.3, .4, .4])
        FakeNode.got = None
        M.condition(None, FakeVae(), r, inpaint="off", node=FakeNode, references=[torch.full((1, 8, 8, 3), .9)])
        self.assertEqual(round(float(FakeNode.got["images"]["image_2"].mean()), 1), .9)
        self.assertEqual(len(FakeNode.got["images"]), 2)


SL = sys.modules["bfs_shot_loop"]

if __name__ == "__main__":
    unittest.main()
