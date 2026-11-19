#!/usr/bin/env python3
"""Tests for train.py, the training runner of the pod. Standard library only, Python 3.11 or newer (train.py reads
TOML with tomllib, as on the pod); skipped on older Pythons.

    python -m unittest discover tests      # from the repository root

The pod's /workspace is a temporary folder (AINVFX_ROOT): fake trainer environments whose `python` is this one and
whose `accelerate` runs the script it is given, fake musubi-tuner and sd-scripts scripts that write what the real ones
write (checkpoints, ComfyUI copies, their arguments), the course's models.json with dummy base files, and datasets.
`Units` tests the pure functions; `Runs` runs the real commands through subprocess, as on the pod.
"""
import contextlib, importlib.util, io, json, os, shutil, subprocess, sys, tempfile, time, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TMP = Path(tempfile.mkdtemp(prefix="ainvfx-train-"))
os.environ["AINVFX_ROOT"] = str(TMP)                  # before train.py is imported: its folders follow
os.environ["AINVFX_COMFY_URL"] = "http://127.0.0.1:9"  # no ComfyUI here: the request to free its memory fails at once
sys.path.insert(0, str(REPO))
FAKE_AV = TMP / "fake_av"                             # a stand-in for PyAV, which train.py reads the clips with
FAKE_AV.mkdir()
(FAKE_AV / "av.py").write_text('''"""A stand-in for PyAV in the tests. A clip file holds JSON: {"frames": 97, "fps": 25.0, "sound": true}.
A file that holds anything else cannot be read, as PyAV fails on a broken file."""
import io
import json


class Stream:
    def __init__(self, kind, frames=0, fps=0.0):
        self.type, self.frames, self.average_rate, self.base_rate = kind, frames, fps, fps


class Streams(list):
    @property
    def video(self):
        return [s for s in self if s.type == "video"]


class Packet:
    size = 1


class Container:
    def __init__(self, path):
        with io.open(path, encoding="utf-8") as f:
            d = json.load(f)
        self.streams = Streams([Stream("video", d.get("frames", 0), d.get("fps", 25.0))]
                               + ([Stream("audio")] if d.get("sound") else []))
        self.packets = d.get("packets", 0)

    def demux(self, stream):
        return [Packet() for _ in range(self.packets)]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def open(path):
    return Container(path)
''', encoding="utf-8")
(FAKE_AV / "torch.py").write_text('''"""A stand-in for PyTorch in the tests: what the caption code touches."""
bfloat16 = "bfloat16"


class no_grad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
''', encoding="utf-8")
(FAKE_AV / "qwen_vl_utils.py").write_text("def process_vision_info(messages):\n    return None, None\n", encoding="utf-8")
(FAKE_AV / "transformers.py").write_text('''"""A stand-in for Qwen3-VL in the tests. Its first answer uses the word "scene"; with FAKE_CAPTION=obey it follows the
instruction at once; asked to rewrite, it gives the caption with another word, or, with FAKE_CAPTION=stubborn, the same.
FAKE_CAPTION_LOG names a file that receives every chat it is given."""
import json
import os

FIRST = "A rocket stands in the scene, framed from above."


class _Ids:
    shape = (1, 0)


class _Inputs(dict):
    input_ids = _Ids()

    def to(self, device):
        return self


class _Out:
    def __init__(self, text):
        self.text = text

    def __getitem__(self, key):
        return self


class AutoProcessor:
    @classmethod
    def from_pretrained(cls, name):
        return cls()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages

    def __call__(self, text, images=None, videos=None, padding=True, return_tensors="pt"):
        return _Inputs(messages=text[0])

    def batch_decode(self, out, skip_special_tokens=True):
        return [out.text]


class Qwen3VLForConditionalGeneration:
    @classmethod
    def from_pretrained(cls, name, dtype=None, device_map=None):
        return cls()

    def generate(self, messages, max_new_tokens=0, do_sample=False):
        if os.environ.get("FAKE_CAPTION_LOG"):
            with open(os.environ["FAKE_CAPTION_LOG"], "a", encoding="utf-8") as f:
                f.write(json.dumps(messages) + "\\n")
        mode, asked = os.environ.get("FAKE_CAPTION", ""), messages[-1]["content"][-1]["text"]
        if asked.startswith("Rewrite"):
            caption = asked.rsplit("\\n\\n", 1)[1]
            return _Out(caption if mode == "stubborn" else caption.replace("scene", "crater"))
        ruled = any(m["role"] == "system" for m in messages)
        return _Out(FIRST.replace("in the scene", "on the pad") if ruled and mode == "obey" else FIRST)
''', encoding="utf-8")
sys.path.insert(0, str(FAKE_AV))
os.environ["PYTHONPATH"] = os.pathsep.join([str(FAKE_AV)] + [p for p in [os.environ.get("PYTHONPATH")] if p])
try:
    import tomllib
    import train
except ImportError:                                    # Python older than 3.11
    tomllib = train = None

FAKE_CACHE = "import sys\nprint('cached', sys.argv[0])\n"
FAKE_CONVERT = """import shutil, sys
a = sys.argv[1:]
shutil.copy(a[a.index("--input") + 1], a[a.index("--output") + 1])
"""
FAKE_TRAIN = """import json, os, sys, time
a = sys.argv[1:]
def val(f):
    return a[a.index(f) + 1] if f in a else None
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
if os.environ.get("FAKE_FAIL"):
    print("the fake trainer fails on purpose")
    sys.exit(3)
out, name = val("--output_dir"), val("--output_name")
with open(os.path.join(out, "args.json"), "w") as f:
    json.dump(a, f)
for w in (name + "-step00000250.safetensors", name + ".safetensors"):
    open(os.path.join(out, w), "wb").write(b"weights")
    if "ltx2" in sys.argv[0]:
        open(os.path.join(out, w.replace(".safetensors", ".comfy.safetensors")), "wb").write(b"comfy")
print("steps: 100%| 250/250")
"""


def clip(path, frames, fps=25.0, sound=False, caption=None):
    """A clip for the stand-in PyAV, with its caption when one is given."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(frames=frames, fps=fps, sound=sound)), encoding="utf-8")
    if caption:
        path.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8")

def setUpModule():
    if not train:
        return
    for home, scripts in train.TRAINERS.values():
        bin_ = home / ".venv" / "bin"
        bin_.mkdir(parents=True, exist_ok=True)
        (bin_ / "python").symlink_to(sys.executable)
        acc = bin_ / "accelerate"          # drops `launch` and its options, runs the script with this Python
        acc.write_text('#!/bin/sh\nshift\nwhile [ "${1#--}" != "$1" ]; do shift 2; done\nexec "%s" "$@"\n' % sys.executable)
        acc.chmod(0o755)
        (home / scripts).mkdir(parents=True, exist_ok=True)
    m = train.MUSUBI / "src" / "musubi_tuner"
    for name in ("zimage_cache_latents.py", "zimage_cache_text_encoder_outputs.py", "ltx2_cache_latents.py",
                 "ltx2_cache_text_encoder_outputs.py"):
        (m / name).write_text(FAKE_CACHE)
    for name in ("zimage_train_network.py", "ltx2_train_network.py"):
        (m / name).write_text(FAKE_TRAIN)
    (m / "convert_lora.py").write_text(FAKE_CONVERT)
    (train.SDSCRIPTS / "moss_image_train_network.py").write_text(FAKE_TRAIN)
    spec = json.loads((REPO / "models.json").read_text(encoding="utf-8"))
    train.MODELS_JSON.write_text(json.dumps(spec), encoding="utf-8")
    for family in ("zimage", "ltx", "moss"):            # base files already on the pod: no download in the tests
        for f in train.family_files(family).values():
            f["dest"].parent.mkdir(parents=True, exist_ok=True)
            f["dest"].write_bytes(b"x")
    images = train.DATASETS / "chars"
    images.mkdir(parents=True)
    for stem in ("a", "b", "c"):
        (images / (stem + ".png")).write_bytes(b"png")
        (images / (stem + ".txt")).write_text("mychr, a person number {} -- in a street\n".format(stem))
    for sub in ("targets", "control"):
        (train.DATASETS / "fx" / sub).mkdir(parents=True)
    for stem in ("s1", "s2"):
        (train.DATASETS / "fx" / "targets" / (stem + ".mp4")).write_text(json.dumps(dict(frames=130, fps=25.0)))
        (train.DATASETS / "fx" / "targets" / (stem + ".txt")).write_text("fxw, smoke rises from a chimney\n")
        (train.DATASETS / "fx" / "control" / (stem + ".mp4")).write_text(json.dumps(dict(frames=130, fps=25.0)))


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


@unittest.skipIf(train is None, "train.py runs on the pod, with Python 3.11 or newer")
class Units(unittest.TestCase):
    def test_every_recipe_has_its_base_files_and_its_placeholders(self):
        """Each recipe names a family of the course's models.json, and its commands fill without a missing value."""
        spec = json.loads((REPO / "models.json").read_text(encoding="utf-8"))
        for name, r in train.RECIPES.items():
            roles = spec["training"][r["family"]]
            self.assertTrue(all(k in spec["files"] for k in roles.values()), name)
            values = dict({role: "/m/" + role for role in roles}, config="c.toml", prompts="p.txt", run="/r", name="n")
            stage = dict(steps=None, lr=None, rank=None, start=None)
            cmds = train.stage_commands(name, stage, values)
            self.assertEqual(cmds[-1][0], "training")
            self.assertFalse(any("{" in a and "}" in a for _, argv in cmds for a in argv if "betas" not in a), name)
            self.assertIn(r.get("trainer", "musubi-tuner"), train.TRAINERS)
        self.assertEqual(train.RECIPES["moss"]["trainer"], "sd-scripts")

    def test_rank_and_alpha_change_together(self):
        args = train.RECIPES["qwen_edit"]["args"]
        out = train.with_rank(args, 32)
        self.assertEqual((train.arg_value(out, "--network_dim"), train.arg_value(out, "--network_alpha")), ("32", "2"))
        out = train.with_rank(train.RECIPES["zimage"]["args"], 64)
        self.assertEqual((train.arg_value(out, "--network_dim"), train.arg_value(out, "--network_alpha")), ("64", "32"))
        self.assertEqual(train.set_flag(["--a", "1"], "--a", "2"), ["--a", "2"])
        self.assertEqual(train.set_flag(["--a", "1"], "--b", "3"), ["--a", "1", "--b", "3"])

    def test_configs_written_for_a_folder(self):
        """musubi-tuner's format for its recipes, with the fork's traps avoided (fps a decimal, one cache per
        block); sd-scripts' format for moss; both read back as blocks of the right kind."""
        iclora = tomllib.loads(train.make_config("ltx_iclora", train.DATASETS / "fx"))
        block = iclora["datasets"][0]
        self.assertTrue(block["video_directory"].endswith("fx/targets"))
        self.assertTrue(block["reference_directory"].endswith("fx/control"))
        self.assertIsInstance(block["target_fps"], float)
        self.assertNotEqual(block["cache_directory"], block["reference_cache_directory"])
        moss = train.make_config("moss", train.DATASETS / "chars")
        self.assertEqual(tomllib.loads(moss)["datasets"][0]["subsets"][0]["image_dir"], str(train.DATASETS / "chars"))
        blocks = train.dataset_blocks(moss, "moss.toml")
        self.assertEqual((blocks[0]["_format"], blocks[0]["image_directory"]), ("sd-scripts", str(train.DATASETS / "chars")))
        infos = train.check_blocks("moss", blocks, "moss.toml")
        self.assertEqual(len(infos[0]["files"]), 3)

    def test_a_dataset_that_is_not_ready_stops_before_the_gpu(self):
        text = train.make_config("zimage", train.DATASETS / "chars")
        with self.assertRaises(SystemExit):           # a musubi-tuner config given to the sd-scripts recipe
            train.check_blocks("moss", train.dataset_blocks(text, "z.toml"), "z.toml")
        with self.assertRaises(SystemExit):           # a whole number of frames per second
            train.check_blocks("ltx", train.dataset_blocks(
                '[general]\nresolution = [960, 544]\n[[datasets]]\nvideo_directory = "{}"\ncache_directory = "/c"\n'
                'target_fps = 25\n'.format(train.DATASETS / "fx" / "targets"), "t.toml"), "t.toml")
        lonely = train.DATASETS / "lonely"
        (lonely / "targets").mkdir(parents=True)
        (lonely / "control").mkdir()
        (lonely / "targets" / "x.png").write_bytes(b"png")
        (lonely / "targets" / "x.txt").write_text("w, x")
        with self.assertRaises(SystemExit):           # a result without its image to edit
            train.check_blocks("qwen_edit", train.dataset_blocks(train.make_config("qwen_edit", lonely), "l"), "l")
        bare = train.DATASETS / "bare"
        bare.mkdir()
        (bare / "y.png").write_bytes(b"png")
        with self.assertRaises(SystemExit):           # no caption at all
            train.check_blocks("zimage", train.dataset_blocks(train.make_config("zimage", bare), "b"), "b")

    def test_sample_prompts(self):
        """Two captions of the dataset and the recipe's subject with the trigger; for pairs, each with its control
        file; a caption cannot open an option of the line."""
        info = train.check_blocks("ltx_iclora", train.dataset_blocks(train.make_config("ltx_iclora", train.DATASETS / "fx"),
                                                                     "fx"), "fx")[0]
        text, trigger = train.make_prompts("ltx_iclora", info, None)
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        self.assertEqual((trigger, len(lines)), ("fxw", 3))
        self.assertIn("--v {} ".format(train.DATASETS / "fx" / "control" / "s1.mp4"), lines[0])
        self.assertTrue(lines[2].startswith("fxw, a woman walks"))
        self.assertNotIn(" -- in", train.sample_line(train.RECIPES["zimage"], "a -- in b", None))

    def test_stages(self):
        stages, trigger = train.parse_stages("ltx_iclora", [
            "s1", "--dataset", "fx", "--trigger", "fxw", "--then", "s2", "--from", "s1", "--steps", "100", "--lr", "5e-5"])
        self.assertEqual([s["name"] for s in stages], ["s1", "s2"])
        self.assertEqual(stages[1]["dataset"], "fx", "a stage without --dataset takes the one before")
        self.assertEqual(stages[1]["start"], str(train.RUNS / "s1" / "s1.safetensors"))
        self.assertEqual(trigger, "fxw")
        for bad in (["x", "--dataset", "fx", "--from", "y"], ["x y", "--dataset", "fx"], ["x", "--dataset", "fx", "--then"],
                    ["x", "--dataset", "fx", "--then", "z", "--trigger", "t"], ["x", "--dataset", "fx", "--then", "x"],
                    ["x", "--dataset", "fx", "--from", "y_comfy.safetensors", "--steps", "5"]):
            with self.assertRaises(SystemExit, msg=bad):
                train.parse_stages("ltx_iclora", bad)

    def test_the_readme_recipe_table_matches_the_recipes(self):
        """README.md's recipe table gives each recipe's base files, steps, learning rate and rank as they are here."""
        rows = []
        for line in (REPO / "README.md").read_text(encoding="utf-8").split("\n## Training a LoRA\n", 1)[1].splitlines():
            if line.startswith("|"):
                rows.append([c.strip() for c in line.strip().strip("|").split("|")])
            elif rows:
                break
        table = {r[0].strip("`"): r for r in rows[2:]}
        self.assertEqual(sorted(table), sorted(train.RECIPES))
        for name, r in train.RECIPES.items():
            size = sum(f["gb"] for f in train.family_files(r["family"]).values())
            steps, lr, rank = (train.arg_value(r["args"], k) for k in ("--max_train_steps", "--learning_rate",
                                                                       "--network_dim"))
            self.assertEqual(table[name][-2], "{} GB".format(round(size)), name)
            self.assertEqual(table[name][-1], "{:,} · {} · {}".format(int(steps), lr, rank), name)

    def test_the_frames_a_clip_gives(self):
        """The trainer's own count: frames dropped to reach the block's rate only beyond 1 fps of it; lengths of 8k + 1;
        head keeps a clip only when it is long enough, full keeps it whole up to max_frames."""
        self.assertEqual(train.frames_at(97, 25.0, 25.0), 97)
        self.assertEqual(train.frames_at(97, 24.0, 25.0), 97, "24 fps is within 1 of 25: no frame dropped")
        self.assertEqual(train.frames_at(97, 23.976, 25.0), 97)
        self.assertEqual(train.frames_at(97, 50.0, 25.0), 49)
        self.assertEqual(train.frames_at(121, 30.0, 25.0), 101)
        self.assertEqual(train.frames_at(60, 12.0, 25.0), 60, "a slower source keeps every frame")
        self.assertEqual(train.frames_at(300, 16.0, 25.0, 121), 78, "numbered at 25 fps, frame 121 comes after 78")
        self.assertEqual(train.frames_at(60, 12.0, 25.0, 121), 59)
        self.assertEqual(train.frames_at(150, 26.0, 25.0, 121), 150, "within 1 fps of the rate: read as it is")
        self.assertEqual(train.frames_at(300, 50.0, 25.0, 121), 150, "a faster source: its own count, cut by head later")
        head, full = dict(train.LTX_CLIPS), dict(frame_extraction="full", max_frames=121)
        self.assertEqual([train.frames_kept(head, n) for n in (97, 121, 300)], [0, 121, 121])
        self.assertEqual([train.frames_kept(full, n) for n in (5, 97, 100, 300)], [1, 97, 97, 121])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):                # a config of one's own that names no target_frames
            train.check_clips("t", dict(target_fps=25.0), [train.DATASETS / "fx" / "targets" / "s1.mp4"], False)
        self.assertIn("takes 1 frame of each clip, as a still", out.getvalue())

    def test_clip_lengths_are_checked_before_the_gpu(self):
        """A clip shorter than the recipe's 121 frames at 25 fps is named before any GPU time; no clip long enough, or a
        clip without its caption, stops the run; a run with sound names the clips without a sound track."""
        folder = train.DATASETS / "lengths"
        clip(folder / "a.mp4", 97, caption="w, a")
        clip(folder / "b.mp4", 97, fps=50.0, caption="w, b")
        clip(folder / "c.mp4", 150, caption="w, c")
        (folder / "d.mp4").write_bytes(b"\xff\xfe broken")
        (folder / "d.txt").write_text("w, d")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            infos = train.check_blocks("ltx", train.dataset_blocks(train.make_config("ltx", folder), "l"), "l")
        self.assertEqual(len(infos[0]["files"]), 4)
        self.assertIn("1 of 4 clips give the 121 frames", out.getvalue())
        self.assertIn("a.mp4 (97), b.mp4 (49)", out.getvalue())
        self.assertIn("could not be read as clips", out.getvalue())
        short = train.DATASETS / "allshort"
        clip(short / "a.mp4", 97, caption="w, a")
        err = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            train.check_blocks("ltx", train.dataset_blocks(train.make_config("ltx", short), "s"), "s")
        self.assertIn("no clip gives 121 frames at 25 fps", err.getvalue())
        clip(short / "e.mp4", 130)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            train.check_blocks("ltx", train.dataset_blocks(train.make_config("ltx", short), "s"), "s")
        self.assertIn("1 clip(s) without a caption, and the trainer stops on the first one", err.getvalue())
        sound = train.DATASETS / "withsound"
        clip(sound / "a.mp4", 130, sound=True, caption="w, a")
        clip(sound / "b.mp4", 130, caption="w, b")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            train.check_blocks("ltx_av", train.dataset_blocks(train.make_config("ltx_av", sound), "v"), "v")
        self.assertIn("1 clip(s) have no sound", out.getvalue())
        self.assertIn(": b.mp4", out.getvalue())
        slow = train.DATASETS / "slow"                      # 300 frames at 16 fps: head stops at the frame numbered 121
        clip(slow / "a.mp4", 300, fps=16.0, caption="w, a")
        clip(slow / "b.mp4", 150, caption="w, b")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            train.check_blocks("ltx", train.dataset_blocks(train.make_config("ltx", slow), "w"), "w")
        self.assertIn("a.mp4 (78)", out.getvalue())
        self.assertIn("1 of them run slower than 25 fps", out.getvalue())
        self.assertIn("a.mp4 (16 fps)", out.getvalue())
        mixed = train.DATASETS / "slowmix" / "clips"        # read whole, the same clip gives its 121 frames
        clip(mixed / "a.mp4", 300, fps=16.0, sound=True, caption="w, a")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            train.check_blocks("ltx_mix", train.dataset_blocks(train.make_config("ltx_mix", mixed.parent), "x"), "x")
        self.assertIn("1 clip(s) of 121 frames", out.getvalue())
        self.assertNotIn("run slower", out.getvalue())

    def test_a_mixed_dataset(self):
        """ltx_mix: one block per kind with its own cache; the clips read whole up to 121 frames; the blocks with sound
        repeated until 30% of the steps have sound; the settings of ltx_av with the two flags of a mixed run."""
        mix = train.DATASETS / "mix"
        (mix / "images").mkdir(parents=True)
        for k in range(12):
            (mix / "images" / "i{:02d}.png".format(k)).write_bytes(b"png")
            (mix / "images" / "i{:02d}.txt".format(k)).write_text("mxw, a lighthouse, still {}\n".format(k))
        clip(mix / "clips" / "c1.mp4", 121, sound=True, caption="mxw, waves break on the rocks, the sea roars")
        clip(mix / "clips" / "c2.mp4", 97, caption="mxw, gulls cross the frame")
        (mix / "audio").mkdir()
        (mix / "audio" / "s1.wav").write_bytes(b"wav")
        (mix / "audio" / "s1.txt").write_text("mxw, a foghorn in the distance\n")
        blocks = tomllib.loads(train.make_config("ltx_mix", mix))["datasets"]
        self.assertEqual([[k for k in b if k.endswith("_directory") and k != "cache_directory"] for b in blocks],
                         [["image_directory"], ["video_directory"], ["audio_directory"]])
        self.assertEqual((blocks[1]["frame_extraction"], blocks[1]["max_frames"]), ("full", 121))
        self.assertIsInstance(blocks[1]["target_fps"], float)
        self.assertEqual(len({b["cache_directory"] for b in blocks}), 3)
        self.assertEqual([b["num_repeats"] for b in blocks], [1, 4, 4], "8 of 24 steps with sound: 33%")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            infos = train.check_blocks("ltx_mix", train.dataset_blocks(train.make_config("ltx_mix", mix), "m"), "m")
        self.assertIn("8 of the 24 steps", out.getvalue())
        self.assertIn("1 clip(s) have no sound", out.getvalue())
        self.assertEqual([i["kind"] for i in infos], ["images", "clips", "sounds"])
        args = train.RECIPES["ltx_mix"]["args"]
        self.assertEqual(args[:len(train.LTX_AV)], train.LTX_AV)
        self.assertEqual((train.arg_value(args, "--audio_batch_probability"),
                          train.arg_value(args, "--audio_supervision_mode")), ("0.4", "warn"))
        self.assertEqual(train.RECIPES["ltx_mix"]["cache"], train.RECIPES["ltx_av"]["cache"])
        self.assertEqual(train.sound_repeats(0, 2, 0), 1)
        self.assertEqual(train.sound_repeats(12, 1, 99), 1, "out of reach even at 10: left at 1, and the check says so")

    def test_a_flat_mixed_folder(self):
        """Without sub-folders, one block per kind of file in the folder, each with its own cache; two files of one name
        would share a caption, which stops the run."""
        flat = train.DATASETS / "flat"
        flat.mkdir()
        (flat / "a.png").write_bytes(b"png")
        (flat / "a.txt").write_text("fw, a still\n")
        clip(flat / "b.mp4", 121, sound=True, caption="fw, a clip")
        (flat / "c.wav").write_bytes(b"wav")
        (flat / "c.txt").write_text("fw, a sound\n")
        blocks = tomllib.loads(train.make_config("ltx_mix", flat))["datasets"]
        self.assertEqual([b.get("image_directory") or b.get("video_directory") or b.get("audio_directory")
                          for b in blocks], [str(flat)] * 3)
        self.assertEqual(len({b["cache_directory"] for b in blocks}), 3)
        clip(flat / "a.mp4", 121, sound=True)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            train.make_config("ltx_mix", flat)

    def test_a_mixed_dataset_that_cannot_train_stops_before_the_gpu(self):
        """No sound at all, sound alone, a sound file without its caption; and a block of sound files given to a recipe
        that trains on clips."""
        mute = train.DATASETS / "mute"
        clip(mute / "clips" / "c.mp4", 121, caption="w, a silent shot")
        (mute / "images").mkdir()
        (mute / "images" / "i.png").write_bytes(b"png")
        (mute / "images" / "i.txt").write_text("w, a still")
        heard = train.DATASETS / "heard"
        (heard / "audio").mkdir(parents=True)
        (heard / "audio" / "s.wav").write_bytes(b"wav")
        (heard / "audio" / "s.txt").write_text("w, a voice")
        bare = train.DATASETS / "bare_sound"
        clip(bare / "clips" / "c.mp4", 121, sound=True, caption="w, a shot")
        (bare / "audio").mkdir()
        (bare / "audio" / "s.wav").write_bytes(b"wav")
        (bare / "audio" / "t.wav").write_bytes(b"wav")
        (bare / "audio" / "t.txt").write_text("w, a voice")
        for folder, words in ((mute, "no clip with a sound track"), (heard, "only sound files"),
                              (bare, "1 sound file(s) without a caption")):
            err = io.StringIO()
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                train.check_blocks("ltx_mix", train.dataset_blocks(train.make_config("ltx_mix", folder), "x"), "x")
            self.assertIn(words, err.getvalue())
        text = '[general]\nresolution = [960, 544]\n[[datasets]]\naudio_directory = "{}"\ncache_directory = "/c"\n'.format(
            heard / "audio")
        err = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            train.check_blocks("ltx", train.dataset_blocks(text, "t.toml"), "t.toml")
        self.assertIn("holds sounds, and the ltx recipe trains on clips", err.getvalue())
    def test_a_message_names_commands_as_its_reader_types_them(self):
        """Through `python pod.py train`, the pod's messages name every train command with those words; in the pod's own
        terminal they stay as they are."""
        msg = "No run on this pod yet. `train recipes` lists the recipes.\nNext:  train <recipe> (train <recipe> -h)"
        self.assertEqual(train.as_typed(msg), msg)
        old, train.VIA = train.VIA, "python pod.py "
        try:
            self.assertEqual(train.as_typed(msg), "No run on this pod yet. `python pod.py train recipes` lists the recipes."
                             "\nNext:  python pod.py train <recipe> (python pod.py train <recipe> -h)")
            self.assertEqual(train.as_typed("the trainer, a train of thought"), "the trainer, a train of thought")
        finally:
            train.VIA = old

    def test_the_next_line(self):
        """One format for the next command: two spaces after Next:, three before the reason. A command of the computer
        says so in the pod's terminal, and not when the words came from the computer (python pod.py train)."""
        def lines():
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                train.next_line("train log", "follows the run's log")
                train.next_line("python pod.py down", "terminates the pod", computer=True)
            return out.getvalue().splitlines()
        self.assertEqual(lines(), ["Next:  train log   (follows the run's log)",
                                   "Next:  on your computer, python pod.py down   (terminates the pod)"])
        old, train.VIA = train.VIA, "python pod.py "
        try:
            self.assertEqual(lines(), ["Next:  python pod.py train log   (follows the run's log)",
                                       "Next:  python pod.py down   (terminates the pod)"])
        finally:
            train.VIA = old
        self.assertEqual([train.dataset_name(r) for r in ("x/images", str(train.DATASETS / "y" / "clips"), "c.toml", "./z")],
                         ["x", "y", "c.toml", "z"])

    def test_avoid_is_an_instruction_to_the_caption_model(self):
        """--avoid goes to the model as an instruction, never as a deletion: a caption that uses a word (or its plural) is
        rewritten once; one that still uses it is written whole and returned, to be named. Without --avoid, nothing
        changes. Each caption shows whole, under its file's name."""
        folder = train.DATASETS / "avoid1"
        folder.mkdir()
        (folder / "a.png").write_bytes(b"png")
        target = train.caption_targets("avoid1", False)[0]
        todo, calls = [(target, folder / "a.png")], []

        def model(*answers):
            answers = list(answers)

            def ask(messages):
                calls.append(messages)
                return answers.pop(0)
            return ask

        def caption(avoid, *answers):
            calls.clear()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                kept = train.caption_files(todo, "trg", avoid, 60, ask=model(*answers))
            return kept, (folder / "a.txt").read_text(), out.getvalue()
        kept, text, out = caption([], "The scene is lit.")
        self.assertEqual((kept, text, len(calls)), ([], "trg, The scene is lit.\n", 1))
        self.assertEqual([m["role"] for m in calls[0]], ["user"])
        self.assertEqual(out, "[1/1] a.png\n  trg, The scene is lit.\n")
        kept, text, out = caption(["scene", "grain"], "The scenes are lit.", "The crater is lit.")
        self.assertEqual((kept, text, len(calls)), ([], "trg, The crater is lit.\n", 2))
        self.assertEqual(calls[0][0]["role"], "system")
        self.assertIn("scene, grain", calls[0][0]["content"][0]["text"])
        self.assertIn("The scenes are lit.", calls[1][-1]["content"][0]["text"])
        kept, text, out = caption(["scene"], "The scene is lit.", "Lit, the Scene.")
        self.assertEqual((kept, text), ([(folder / "a.txt", ["scene"])], "trg, Lit, the Scene.\n"))

        def refuses(messages):                              # the model answers for the picture, then fails on the rewrite
            calls.append(messages)
            if len(calls) > 1:
                raise RuntimeError("text-only request refused\nby the stand-in")
            return "The scene is lit."
        calls.clear()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            kept = train.caption_files(todo, "trg", ["scene"], 60, ask=refuses)
        self.assertEqual((kept, (folder / "a.txt").read_text()), ([(folder / "a.txt", ["scene"])], "trg, The scene is lit.\n"))
        self.assertEqual(out.getvalue(), "[1/1] a.png\n  trg, The scene is lit.\n  the request to rewrite this caption "
                         "failed (text-only request refused): it is kept as first written\n")
        self.assertEqual(train.clean_caption(' "the scene ,  is lit" '), "the scene , is lit")
        self.assertEqual(train.avoided_in("Grainy scenes, a scenery", ["scene", "grain"]), ["scene"])

    @unittest.skipIf(importlib.util.find_spec("huggingface_hub") is None, "huggingface_hub is in the trainer's environment")
    def test_the_caption_model_on_disk(self):
        """« downloaded this time » only when Hugging Face's cache lacks one of the model's weights files."""
        cache = TMP / "hfcache"
        snap = cache / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (snap.parent.parent / "refs").mkdir()
        (snap.parent.parent / "refs" / "main").write_text("abc123")
        self.assertFalse(train.model_on_disk(train.CAPTION_MODEL, cache_dir=str(cache)))
        (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "m-1.safetensors",
                                                                                     "b": "m-2.safetensors"}}))
        (snap / "m-1.safetensors").write_bytes(b"x")
        self.assertFalse(train.model_on_disk(train.CAPTION_MODEL, cache_dir=str(cache)))
        (snap / "m-2.safetensors").write_bytes(b"x")
        self.assertTrue(train.model_on_disk(train.CAPTION_MODEL, cache_dir=str(cache)))

    def test_the_caption_prompt_of_each_kind(self):
        """A mixed folder: a still as an image, a clip with a sound track with what is heard, a silent clip with what is
        seen; sound files get no prompt."""
        cap = train.DATASETS / "mixcap"
        (cap / "images").mkdir(parents=True)
        (cap / "images" / "i.png").write_bytes(b"png")
        clip(cap / "clips" / "loud.mp4", 121, sound=True)
        clip(cap / "clips" / "mute.mp4", 121)
        (cap / "audio").mkdir()
        (cap / "audio" / "s.wav").write_bytes(b"wav")
        targets = train.caption_targets("mixcap", False)
        self.assertEqual([t["mode"] for t in targets], ["image", "clips", "sound"])
        self.assertEqual([train.caption_prompt(targets[1], f) for f in targets[1]["files"]], ["audio", "video"])
        self.assertEqual([t["mode"] for t in train.caption_targets("chars", False)], ["image"])

@unittest.skipIf(train is None, "train.py runs on the pod, with Python 3.11 or newer")
class Runs(unittest.TestCase):
    def run_train(self, *args, env=None, timeout=120):
        e = dict(os.environ, **(env or {}))
        e.pop("HF_TOKEN", None)
        r = subprocess.run([sys.executable, str(REPO / "train.py")] + list(args), capture_output=True, text=True, env=e,
                           timeout=timeout, stdin=subprocess.DEVNULL)      # never the terminal the tests run in
        return r.stdout + r.stderr, r.returncode

    def test_a_zimage_run_end_to_end(self):
        """Caches, training, ComfyUI copies and links, the options on the trainer's command line, then status."""
        out, rc = self.run_train("zimage", "z1", "--dataset", "chars", "--steps", "300", "--rank", "64")
        self.assertEqual(rc, 0, out)
        self.assertIn("The run is done", out)
        run = train.RUNS / "z1"
        for name in ("z1.safetensors", "z1_comfy.safetensors", "z1-step00000250_comfy.safetensors", "dataset.toml",
                     "prompts.txt", "train.log"):
            self.assertTrue((run / name).is_file(), name + "\n" + out)
        link = train.LORAS / "z1" / "z1_comfy.safetensors"
        self.assertTrue(link.is_symlink() and link.resolve() == (run / "z1_comfy.safetensors").resolve(), out)
        args = json.loads((run / "args.json").read_text())
        self.assertEqual(train.arg_value(args, "--max_train_steps"), "300")
        self.assertEqual((train.arg_value(args, "--network_dim"), train.arg_value(args, "--network_alpha")), ("64", "32"))
        self.assertIn("mychr, a narrow street", (run / "prompts.txt").read_text())
        out, rc = self.run_train("zimage", "z1", "--dataset", "chars")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("a run named z1 exists", out)
        out, rc = self.run_train("status")
        self.assertIn("z1 · recipe zimage · stage 1 of 1 · done", out)

    def test_a_chain_of_two_stages(self):
        """The second stage starts from the first one's final weights, with its own steps and learning rate; the
        fork's ComfyUI copies are linked."""
        out, rc = self.run_train("ltx_iclora", "c1", "--dataset", "fx", "--then", "c2", "--from", "c1", "--steps", "250",
                                 "--lr", "5e-5")
        self.assertEqual(rc, 0, out)
        self.assertIn("2 stages", out)
        args = json.loads((train.RUNS / "c2" / "args.json").read_text())
        self.assertEqual(train.arg_value(args, "--network_weights"), str(train.RUNS / "c1" / "c1.safetensors"))
        self.assertEqual(train.arg_value(args, "--learning_rate"), "5e-05")
        self.assertTrue((train.LORAS / "c1" / "c1.comfy.safetensors").is_symlink(), out)
        self.assertFalse((train.LORAS / "c1" / "c1.safetensors").exists(), "only the ComfyUI copies are linked")

    def test_a_moss_run_through_sd_scripts(self):
        out, rc = self.run_train("moss", "m1", "--dataset", "chars")
        self.assertEqual(rc, 0, out)
        run = train.RUNS / "m1"
        args = json.loads((run / "args.json").read_text())
        self.assertTrue(train.arg_value(args, "--pretrained_model_name_or_path").endswith("moss_image_base_bf16.safetensors"))
        self.assertIn("[[datasets.subsets]]", (run / "dataset.toml").read_text())
        self.assertTrue((train.LORAS / "m1" / "m1.safetensors").is_symlink(), "sd-scripts' LoRA loads in ComfyUI as it is")
        self.assertIn(str(train.SDSCRIPTS / ".venv" / "bin" / "accelerate"), (run / "train.log").read_text())

    def test_a_failed_run_says_so(self):
        out, rc = self.run_train("zimage", "f1", "--dataset", "chars", env={"FAKE_FAIL": "1"})
        self.assertEqual(rc, 3, out)
        self.assertIn("FAILED · training ended with code 3", out)
        self.assertEqual(train.read_job()["state"], "failed")

    def test_stop(self):
        """A run in progress marks its record every BEAT seconds, its sign of life. Stopped from another terminal: its
        process group ends, the record says stopped, and the terminal that followed it returns."""
        e = dict(os.environ, FAKE_SLEEP="60", AINVFX_BEAT="1")
        e.pop("HF_TOKEN", None)
        follower = subprocess.Popen([sys.executable, str(REPO / "train.py"), "zimage", "st1", "--dataset", "chars"],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    env=e)
        try:
            for _ in range(100):
                job = train.read_job() or {}
                if job.get("stages", [{}])[0].get("name") == "st1" and job.get("state") == "running" and job.get("pid"):
                    break
                time.sleep(0.2)
            pid, beat = train.read_job()["pid"], train.read_job()["beat"]
            for _ in range(100):
                if train.read_job()["beat"] > beat:
                    break
                time.sleep(0.2)
            self.assertGreater(train.read_job()["beat"], beat, "the record of a run in progress keeps moving")
            out, rc = self.run_train("stop", "-y")
            self.assertEqual(rc, 0, out)
            self.assertEqual(train.read_job()["state"], "stopped")
            self.assertFalse(train.alive(pid), "the run's process group ends")
            follower.wait(timeout=30)
            self.assertIn("The run was stopped", follower.stdout.read())
        finally:
            if follower.poll() is None:
                follower.kill()
                follower.wait(timeout=30)
            follower.stdout.close()

    def test_each_command_ends_with_the_next_one(self):
        """The launch names `down --when-done` right after the run starts, and only there; `train log` names `down` when
        the run it followed is done; `train recipes` names a run. Through python pod.py train, no « on your computer »."""
        out, rc = self.run_train("zimage", "n1", "--dataset", "chars")
        self.assertEqual(rc, 0, out)
        lines = out.splitlines()
        i = next(k for k, line in enumerate(lines) if line.startswith("The run started in the background"))
        self.assertTrue(lines[i + 1].startswith("Next:  on your computer, python pod.py down --when-done   ("), out)
        self.assertEqual(sum(line.startswith("Next:") for line in lines), 1, out)
        out, rc = self.run_train("log")
        self.assertTrue(out.rstrip().splitlines()[-1].startswith("Next:  on your computer, python pod.py down   ("), out)
        out, rc = self.run_train("recipes")
        self.assertTrue(out.rstrip().splitlines()[-1].startswith("Next:  train "), out)
        self.assertEqual(self.run_train("recipe"), (out, rc), "train recipe lists them too")
        out, rc = self.run_train("zimage", "n2", "--dataset", "chars", env={"AINVFX_TRAIN_VIA": "python pod.py "})
        self.assertIn("\nNext:  python pod.py down --when-done   (", out)

    def test_caption_lists_what_it_would_write(self):
        out, rc = self.run_train("caption", "chars", "--trigger", "mychr", "--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertIn("3 file(s), 0 to caption", out)
        out, rc = self.run_train("caption", "chars", "--trigger", "two words")
        self.assertNotEqual(rc, 0, out)

    def test_caption_with_avoid_through_the_caption_model(self):
        """train caption --avoid through the caption code, with a stand-in model: the instruction holds the words, the
        captions written have none, each shows whole; a model that keeps a word gets the caption named at the end;
        without --avoid, no instruction."""
        folder = train.DATASETS / "avoid2"
        folder.mkdir()
        for stem in ("a", "b"):
            (folder / (stem + ".png")).write_bytes(b"png")
        log = TMP / "caption_chats.jsonl"
        out, rc = self.run_train("caption", "avoid2", "--trigger", "av", "--avoid", "scene",
                                 env={"FAKE_CAPTION_LOG": str(log)})
        self.assertEqual(rc, 0, out)
        self.assertIn("[1/2] a.png\n  av, A rocket stands in the crater, framed from above.\n", out)
        self.assertEqual((folder / "b.txt").read_text(), "av, A rocket stands in the crater, framed from above.\n")
        chats = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(chats), 4, "each caption: the first answer, then one rewrite")
        self.assertIn("scene", chats[0][0]["content"][0]["text"])
        out, rc = self.run_train("caption", "avoid2", "--trigger", "av", "--avoid", "scene", "--overwrite",
                                 env={"FAKE_CAPTION": "stubborn"})
        self.assertEqual(rc, 0, out)
        self.assertIn("2 caption(s) still use a word of --avoid: {} (scene)".format(folder / "a.txt"), out)
        self.assertEqual((folder / "a.txt").read_text(), "av, A rocket stands in the scene, framed from above.\n")
        log.unlink()
        out, rc = self.run_train("caption", "avoid2", "--trigger", "av", "--overwrite", env={"FAKE_CAPTION_LOG": str(log)})
        self.assertEqual(rc, 0, out)
        self.assertEqual([m["role"] for m in json.loads(log.read_text().splitlines()[0])], ["user"])
        self.assertEqual(out.rstrip().splitlines()[-1].split("   ")[0], "Next:  train prep <recipe> <run> --dataset avoid2")

    def test_prep_then_a_run_with_the_edited_files(self):
        """train prep checks, shows the settings and both files, writes them and starts nothing. An edited resolution and
        an edited prompt reach the trainer; each broken edit stops with its sentence; a run given other words than its
        prep stops; --prompts points to prep; a chain has one prompts file."""
        out, rc = self.run_train("prep", "zimage", "p1", "--dataset", "chars", "--steps", "300")
        self.assertEqual(rc, 0, out)
        run = train.RUNS / "p1"
        self.assertEqual(sorted(p.name for p in run.iterdir()), [".prep.json", "dataset.toml", "prompts.txt"], out)
        self.assertRegex(out, r"steps\s+300\s+from --steps")
        self.assertRegex(out, r"learning rate\s+1e-4\s+from the recipe")
        for said in ("settings of p1 (recipe zimage)", "dataset config: {} (written from the recipe)".format(
                run / "dataset.toml"), "sample prompts: {} (written from the recipe)".format(run / "prompts.txt"),
                "Nothing started"):
            self.assertIn(said, out)
        self.assertEqual(out.rstrip().splitlines()[-1].split("   ")[0], "Next:  train zimage p1 --dataset chars --steps 300")
        toml = (run / "dataset.toml").read_text()
        self.assertIn("resolution = [1024, 1024]", toml)
        (run / "dataset.toml").write_text(toml.replace("resolution = [1024, 1024]", "resolution = [512, 512]"))
        (run / "prompts.txt").write_text("# mine\nmychr, my own prompt --w 512 --h 512\n")
        out, rc = self.run_train("zimage", "p1", "--dataset", "chars", "--steps", "300")
        self.assertEqual(rc, 0, out)
        self.assertIn("dataset config: {} (prepared)".format(run / "dataset.toml"), out)
        self.assertIn("    mychr, my own prompt --w 512 --h 512", out)
        shown, _ = self.run_train("prep", "zimage", "p4", "--dataset", "chars", env={"AINVFX_TRAIN_VIA": "python pod.py "})
        self.assertIn("    # Written by train.py for the zimage recipe. To change a setting, edit this file between `train prep`",
                      shown, "a file shows as it is, through python pod.py train too")
        self.assertLess(out.index("sample prompts:"), out.index("The run started"))
        args = json.loads((run / "args.json").read_text())
        self.assertEqual(train.arg_value(args, "--dataset_config"), str(run / "dataset.toml"))
        self.assertIn("resolution = [512, 512]", (run / "dataset.toml").read_text())
        self.assertEqual((run / "prompts.txt").read_text(), "# mine\nmychr, my own prompt --w 512 --h 512\n")
        self.assertFalse((run / ".prep.json").exists())
        chars = str(train.DATASETS / "chars")
        for name, file, edit, said in (("b1", "dataset.toml", lambda t: "[general\n", "is not valid TOML"),
                                       ("b2", "dataset.toml", lambda t: t.replace(chars, chars + "_gone"), "no folder"),
                                       ("b3", "prompts.txt", lambda t: "# nothing\n", "has no prompt line")):
            out, rc = self.run_train("prep", "zimage", name, "--dataset", "chars")
            self.assertEqual(rc, 0, out)
            path = train.RUNS / name / file
            path.write_text(edit(path.read_text()))
            out, rc = self.run_train("zimage", name, "--dataset", "chars")
            self.assertNotEqual(rc, 0, out)
            self.assertIn(said, out)
            self.assertFalse((train.RUNS / name / "train.log").exists(), out)
        out, rc = self.run_train("prep", "zimage", "p2", "--dataset", "chars")
        out, rc = self.run_train("zimage", "p2", "--dataset", "chars", "--steps", "500")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("the run p2 was prepared with other words: `train prep zimage p2 --dataset chars`", out)
        out, rc = self.run_train("zimage", "p3", "--dataset", "chars", "--prompts", "x.txt")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("--prompts is gone: `train prep zimage p3 --dataset ...`", out)
        chain = ["pc1", "--dataset", "chars", "--then", "pc2", "--from", "pc1", "--steps", "100"]
        out, rc = self.run_train("prep", "zimage", *chain)
        self.assertEqual(rc, 0, out)
        self.assertIn("The chain has one prompts file", out)
        self.assertEqual(sorted(p.name for p in (train.RUNS / "pc2").iterdir()), ["dataset.toml"])
        (train.RUNS / "pc1" / "prompts.txt").write_text("mychr, the chain's prompt\n")
        out, rc = self.run_train("zimage", *chain)
        self.assertEqual(rc, 0, out)
        self.assertEqual((train.RUNS / "pc2" / "prompts.txt").read_text(), "mychr, the chain's prompt\n")
        out, rc = self.run_train("prep")
        self.assertEqual(rc, 2, out)
        self.assertIn("train prep <recipe> <run> --dataset <dataset> [options]", out)

    def test_a_mixed_run_end_to_end(self):
        """ltx_mix through the command: its config of three blocks, the settings of ltx_av with the two flags of a mixed
        run, a clip's caption as the first sample prompt, the fork's ComfyUI copies linked."""
        mix = train.DATASETS / "mixrun"
        (mix / "images").mkdir(parents=True)
        for stem in ("i1", "i2"):
            (mix / "images" / (stem + ".png")).write_bytes(b"png")
            (mix / "images" / (stem + ".txt")).write_text("mxr, a lighthouse at dusk\n")
        clip(mix / "clips" / "c1.mp4", 121, sound=True, caption="mxr, waves break on the rocks, the sea roars")
        (mix / "audio").mkdir()
        (mix / "audio" / "s1.wav").write_bytes(b"wav")
        (mix / "audio" / "s1.txt").write_text("mxr, a foghorn in the distance\n")
        out, rc = self.run_train("ltx_mix", "x1", "--dataset", "mixrun")
        self.assertEqual(rc, 0, out)
        run = train.RUNS / "x1"
        args = json.loads((run / "args.json").read_text())
        self.assertEqual(train.arg_value(args, "--ltx2_mode"), "av")
        self.assertEqual(train.arg_value(args, "--audio_batch_probability"), "0.4")
        self.assertEqual((run / "dataset.toml").read_text().count("[[datasets]]"), 3)
        self.assertTrue((run / "prompts.txt").read_text().splitlines()[2].startswith("mxr, waves break"), out)
        self.assertIn("2 of the 4 steps of each pass over the data have sound", out)
        self.assertTrue((train.LORAS / "x1" / "x1.comfy.safetensors").is_symlink(), out)
    def test_caption_a_mixed_folder(self):
        """The dry run lists the stills and clips to caption, and the sound files whose captions are written by hand."""
        cap = train.DATASETS / "mixcap2"
        (cap / "images").mkdir(parents=True)
        (cap / "images" / "i.png").write_bytes(b"png")
        clip(cap / "clips" / "c.mp4", 121, sound=True)
        (cap / "audio").mkdir()
        (cap / "audio" / "s.wav").write_bytes(b"wav")
        out, rc = self.run_train("caption", "mixcap2", "--trigger", "w", "--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertIn("2 file(s), 2 to caption", out)
        self.assertIn("1 sound file(s) without a caption", out)
        self.assertIn(": s.wav", out)

if __name__ == "__main__":
    unittest.main()
