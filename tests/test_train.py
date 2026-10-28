#!/usr/bin/env python3
"""Tests for train.py, the training runner of the pod. Standard library only, Python 3.11 or newer (train.py reads
TOML with tomllib, as on the pod); skipped on older Pythons.

    python -m unittest discover tests      # from the repository root

The pod's /workspace is a temporary folder (AINVFX_ROOT): fake trainer environments whose `python` is this one and
whose `accelerate` runs the script it is given, fake musubi-tuner and sd-scripts scripts that write what the real ones
write (checkpoints, ComfyUI copies, their arguments), the course's models.json with dummy base files, and datasets.
`Units` tests the pure functions; `Runs` runs the real commands through subprocess, as on the pod.
"""
import json, os, shutil, subprocess, sys, tempfile, time, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TMP = Path(tempfile.mkdtemp(prefix="ainvfx-train-"))
os.environ["AINVFX_ROOT"] = str(TMP)                  # before train.py is imported: its folders follow
os.environ["AINVFX_COMFY_URL"] = "http://127.0.0.1:9"  # no ComfyUI here: the request to free its memory fails at once
sys.path.insert(0, str(REPO))
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
        (train.DATASETS / "fx" / "targets" / (stem + ".mp4")).write_bytes(b"mp4")
        (train.DATASETS / "fx" / "targets" / (stem + ".txt")).write_text("fxw, smoke rises from a chimney\n")
        (train.DATASETS / "fx" / "control" / (stem + ".mp4")).write_bytes(b"mp4")


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
        stages, prompts, trigger = train.parse_stages("ltx_iclora", [
            "s1", "--dataset", "fx", "--trigger", "fxw", "--then", "s2", "--from", "s1", "--steps", "100", "--lr", "5e-5"])
        self.assertEqual([s["name"] for s in stages], ["s1", "s2"])
        self.assertEqual(stages[1]["dataset"], "fx", "a stage without --dataset takes the one before")
        self.assertEqual(stages[1]["start"], str(train.RUNS / "s1" / "s1.safetensors"))
        self.assertEqual((prompts, trigger), (None, "fxw"))
        for bad in (["x", "--dataset", "fx", "--from", "y"], ["x y", "--dataset", "fx"], ["x", "--dataset", "fx", "--then"],
                    ["x", "--dataset", "fx", "--then", "z", "--trigger", "t"], ["x", "--dataset", "fx", "--then", "x"],
                    ["x", "--dataset", "fx", "--from", "y_comfy.safetensors", "--steps", "5"]):
            with self.assertRaises(SystemExit, msg=bad):
                train.parse_stages("ltx_iclora", bad)


@unittest.skipIf(train is None, "train.py runs on the pod, with Python 3.11 or newer")
class Runs(unittest.TestCase):
    def run_train(self, *args, env=None, timeout=120):
        e = dict(os.environ, **(env or {}))
        e.pop("HF_TOKEN", None)
        r = subprocess.run([sys.executable, str(REPO / "train.py")] + list(args), capture_output=True, text=True, env=e,
                           timeout=timeout)
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
        """A run stopped from another terminal: its process group ends, the record says stopped, and the terminal
        that followed it returns."""
        e = dict(os.environ, FAKE_SLEEP="60")
        e.pop("HF_TOKEN", None)
        follower = subprocess.Popen([sys.executable, str(REPO / "train.py"), "zimage", "st1", "--dataset", "chars"],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=e)
        try:
            for _ in range(100):
                job = train.read_job() or {}
                if job.get("stages", [{}])[0].get("name") == "st1" and job.get("state") == "running" and job.get("pid"):
                    break
                time.sleep(0.2)
            pid = train.read_job()["pid"]
            out, rc = self.run_train("stop", "-y")
            self.assertEqual(rc, 0, out)
            self.assertEqual(train.read_job()["state"], "stopped")
            self.assertFalse(train.alive(pid), "the run's process group ends")
            follower.wait(timeout=30)
            self.assertIn("The run was stopped", follower.stdout.read())
        finally:
            if follower.poll() is None:
                follower.kill()

    def test_caption_lists_what_it_would_write(self):
        out, rc = self.run_train("caption", "chars", "--trigger", "mychr", "--dry-run")
        self.assertEqual(rc, 0, out)
        self.assertIn("3 file(s), 0 to caption", out)
        out, rc = self.run_train("caption", "chars", "--trigger", "two words")
        self.assertNotEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main()
