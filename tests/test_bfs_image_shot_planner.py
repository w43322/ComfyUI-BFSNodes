import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch

if os.environ.get("COMFYUI_PATH"):
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
spec = importlib.util.spec_from_file_location(
    "bfs_image_shot_planner", Path(__file__).resolve().parents[1] / "bfs_image_shot_planner.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def img(v, h=64, w=96):
    return torch.full((1, h, w, 3), float(v))


class Clip:
    def tokenize(self, prompt, **kwargs):
        self.prompt, self.images = prompt, kwargs["images"]
        return {"text": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {}]]


class Vae:
    def encode(self, pixels):
        return torch.zeros(1, 24, 1, pixels.shape[1] // 16, pixels.shape[2] // 16)


class FakeNode:
    """Stands in for the native conditioning node: records what it receives."""

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, images=None):
        cls.got = dict(prompt=prompt, width=width, height=height, images=images)
        return ([[torch.zeros(1), {}]], {"samples": torch.zeros(1, 24, 1, height // 16, width // 16)})


class ParsingTest(unittest.TestCase):
    def test_selection(self):
        self.assertEqual(M.parse_selection("", 4), [0, 1, 2, 3])
        self.assertEqual(M.parse_selection("1-2, 4", 4), [0, 1, 3])
        self.assertEqual(M.parse_selection("3-1", 4), [0, 1, 2])
        with self.assertRaises(ValueError):
            M.parse_selection("5", 4)
        with self.assertRaises(ValueError):
            M.parse_selection("a", 4)

    def test_blocks(self):
        self.assertEqual(M.split_blocks(""), [])
        self.assertEqual(M.split_blocks("one\n---\n\n---\nthree\nlines"), ["one", "", "three\nlines"])

    def test_refs(self):
        self.assertEqual(M.parse_refs("", 3), [0, 1, 2])
        self.assertEqual(M.parse_refs("none", 3), [])
        self.assertEqual(M.parse_refs("3, 1", 3), [2, 0])
        self.assertEqual(M.parse_refs("3-2", 3), [2, 1])
        self.assertEqual(M.parse_refs("", 3, default="2"), [1])
        with self.assertRaises(ValueError):
            M.parse_refs("4", 3)

    def test_split_images(self):
        batch = torch.cat([img(0), img(1)], 0)
        self.assertEqual(len(M.split_images([batch, img(2, 32, 32)])), 3)
        self.assertEqual(len(M.split_images(batch)), 2)
        self.assertEqual(M.split_images(None), [])


class SizeTest(unittest.TestCase):
    def test_native_h3_short_edge_range(self):
        self.assertEqual(M.native_size(512, 768), (1024, 1536))
        self.assertEqual(M.native_size(4000, 3000), (2720, 2048))
        self.assertEqual(M.native_size(1500, 1200), (1504, 1216))

    def test_native_ph3_area(self):
        w, h = M.native_size(8000, 6000, "p-h3-image")
        self.assertLessEqual(w * h, 2912 * 2912 * 1.01)
        self.assertEqual(M.native_size(512, 512, "p-h3-image"), (512, 512))

    def test_modes(self):
        w, h = M.shot_size(1000, 500, "megapixels", 1.0, 0, 0, 32)
        self.assertEqual((w % 32, h % 32), (0, 0))
        self.assertAlmostEqual(w * h / 1024 ** 2, 1.0, delta=0.06)
        self.assertEqual(M.shot_size(1000, 500, "source size", 1, 0, 0, 32), (992, 512))
        self.assertEqual(M.shot_size(1000, 500, "custom", 1, 1024, 768, 64), (1024, 768))


class PlannerTest(unittest.TestCase):
    def run_planner(self, **kw):
        args = dict(images=[torch.cat([img(.1), img(.2)], 0), img(.3, 90, 90)], global_prompt=["global {index}/{count}"],
                    shot_prompts=["first <Picture 2>\n---\n\n---\nthird"], ref_assign=["2, 1\nnone"],
                    default_refs=["all"], source_as_picture_1=[True], run=[""], variant=["MiniMax H3 Image"],
                    size_mode=["source size"], megapixels=[4.0], width=[2048], height=[2048], multiple=[32], seed=[10],
                    seed_mode=["increment"], describe_preset=["short"], references=[img(.7), img(.8)])
        args.update(kw)
        return M.BFSImageShotPlanner().plan(**args)

    def test_outputs_are_lists_per_shot(self):
        shots, images, prompts, seeds, count, text, pool = self.run_planner()
        self.assertEqual(count, 3)
        self.assertEqual(len(M.BFSImageShotPlanner.OUTPUT_IS_LIST), len(M.BFSImageShotPlanner.RETURN_TYPES))
        self.assertEqual(prompts, ["first <Picture 2>", "global 2/3", "third"])
        self.assertEqual(seeds, [10, 11, 12])
        self.assertEqual([s["ref_ids"] for s in shots], [[2, 1], [], [1, 2]])
        self.assertEqual(round(float(shots[0]["refs"][0].mean()), 1), .8)
        self.assertEqual((shots[2]["width"], shots[2]["height"]), (96, 96))
        self.assertEqual(len(pool), 2)
        self.assertIn("<Picture 2> = ref 2", text)

    def test_run_selection_and_prompt_list(self):
        shots, _, prompts, seeds, count, _, _ = self.run_planner(run=["3"], prompts=["", "", "from list"],
                                                                  seed_mode=["fixed"])
        self.assertEqual(count, 1)
        self.assertEqual(shots[0]["index"], 2)
        self.assertEqual(prompts, ["from list"])
        self.assertEqual(seeds, [10])

    def test_no_references_gives_grey_pool(self):
        shots, *_, pool = self.run_planner(references=None, ref_assign=[""])
        self.assertEqual(shots[0]["refs"], [])
        self.assertEqual(len(pool), 1)

    def test_details_without_vlm(self):
        shots, _, prompts, *_ = self.run_planner(global_prompt=["swap to {details}"], shot_prompts=[""],
                                                 ref_assign=[""])
        self.assertEqual(prompts[0], "swap to character")


class DetailsTest(unittest.TestCase):
    def test_numbering_follows_source_picture(self):
        t = M.expand_prompt("{details} {details2} {image}", 0, 1, "a", True, 2)
        self.assertEqual(t, "{details@Picture 2} {details@Picture 3} {details@Picture 1}")
        t = M.expand_prompt("{details} {image}", 0, 1, "a", False, 1)
        self.assertEqual(t, "{details@Picture 1} {details@Picture 0}")

    def test_fill_uses_the_right_picture(self):
        src, r1 = img(.1), img(.9)
        seen = []
        describe = lambda image, prompt, marker: (seen.append(round(float(image.mean()), 1)) or f"d{marker[-2]}")
        out = M.fill_details("a {details@Picture 2}, b {details@Picture 1}, c {details@Picture 5}", [src, r1], src,
                             True, describe=describe)
        self.assertEqual(out, "a d2, b d1, c character")
        self.assertEqual(seen, [.9, .1])


class ConditioningTest(unittest.TestCase):
    def shot(self, **kw):
        s = {"index": 0, "image": img(.1), "refs": [img(.5), img(.6)], "prompt": "p", "width": 1024, "height": 768,
             "seed": 3, "source_is_picture_1": True, "variant": "MiniMax H3 Image"}
        s.update(kw)
        return s

    def test_picture_order_and_size(self):
        pos, lat, text = M.condition_shot(None, None, self.shot(), extra=[img(.7)], node=FakeNode)
        got = FakeNode.got
        self.assertEqual(list(got["images"]), ["image_1", "image_2", "image_3", "image_4"])
        self.assertEqual([round(float(v.mean()), 1) for v in got["images"].values()], [.1, .5, .6, .7])
        self.assertEqual((got["width"], got["height"]), (1024, 768))
        self.assertEqual(text, "p")

    def test_text_to_image_and_overrides(self):
        M.condition_shot(None, None, self.shot(source_is_picture_1=False, refs=[]), prompt="new", width=512,
                         height=512, node=FakeNode)
        self.assertIsNone(FakeNode.got["images"])
        self.assertEqual((FakeNode.got["prompt"], FakeNode.got["width"]), ("new", 512))

    @unittest.skipUnless(os.environ.get("COMFYUI_PATH"), "needs COMFYUI_PATH for the native H3 Image node")
    def test_native_h3_image_node(self):
        clip = Clip()
        positive, latent, _ = M.BFSImageShotH3Conditioning().condition(clip, Vae(), self.shot(width=0, height=0))[:3]
        self.assertEqual(len(positive[0][1]["minimax_refs"]), 3)
        self.assertEqual([round(float(i.mean()), 1) for i in clip.images], [.1, .5, .6])
        self.assertEqual(tuple(latent["samples"].shape[:3]), (1, 24, 1))
        clip = Clip()
        _, latent, _ = M.BFSImageShotH3Conditioning().condition(clip, Vae(), self.shot())[:3]
        self.assertEqual(tuple(latent["samples"].shape), (1, 24, 1, 48, 64))
        _, latent, _ = M.BFSImageShotH3Conditioning().condition(clip, Vae(), self.shot(variant="p-h3-image"))[:3]
        self.assertEqual(latent["samples"].shape[-2:], (48, 64))


class UnpackJoinTest(unittest.TestCase):
    def test_unpack_repack(self):
        s = {"index": 1, "image": img(.1), "refs": [], "prompt": "p", "width": 64, "height": 32, "seed": 4, "name": "n"}
        out = M.BFSImageShotUnpack().unpack(s)
        self.assertEqual(out[2:9], (0, "p", 64, 32, 4, 2, "n"))
        s2 = M.BFSImageShotRepack().repack(s, refs=torch.cat([img(.3), img(.4)], 0), prompt="q")[0]
        self.assertEqual((len(s2["refs"]), s2["prompt"], s["prompt"]), (2, "q", "p"))

    def test_join_sizes(self):
        shots = [{"image": img(.1, 64, 96)}, {"image": img(.2, 50, 50)}]
        res = [img(.5, 64, 96), img(.6, 32, 32)]
        for fit in M.FITS:
            batch, orig, comp, lst = M.BFSImageShotJoin().join(res, [fit], ["first"], shots)
            self.assertEqual(tuple(batch.shape), (2, 64, 96, 3))
            self.assertEqual(tuple(orig.shape), (2, 64, 96, 3))
            self.assertEqual(tuple(comp.shape), (2, 64, 192, 3))
            self.assertEqual([tuple(x.shape) for x in lst], [(1, 64, 96, 3), (1, 32, 32, 3)])
        batch, *_ = M.BFSImageShotJoin().join(res, ["pad"], ["largest"])
        self.assertEqual(tuple(batch.shape), (2, 64, 96, 3))
        # pad letterboxes with black, crop fills the frame
        pad = M.fit_image(img(1, 32, 32), 64, 32, "pad")
        self.assertEqual(float(pad[0, 0, 0, 0]), 0.0)
        self.assertEqual(float(M.fit_image(img(1, 32, 32), 64, 32, "crop").min()), 1.0)


if __name__ == "__main__":
    unittest.main()
