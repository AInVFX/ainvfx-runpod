#!/usr/bin/env python3
"""ainvfx-runpod · train.py: LoRA training ON THE POD, with musubi-tuner and sd-scripts.

The `train` profile installs it (bootstrap.sh): the two trainers, each in its own environment, and the `train`
command, which runs this file with musubi-tuner's Python. In a terminal on the pod (JupyterLab: File, New, Terminal; or `python pod.py ssh <pod>`):

  train recipes                                 every recipe: what it learns, its dataset, its base files
  train caption <dataset> --trigger WORD        writes a caption (.txt) next to every image or clip that has none
  train <recipe> <run> --dataset <dataset>      trains a LoRA in the background; its log follows on screen
  train log                                     follows the log of the run in progress (or of the last one) again
  train status                                  the run in progress: stage, last lines, checkpoints, GPU
  train stop                                    stops the run in progress

<dataset> is a folder of /workspace/dataset (`python pod.py push --dataset NAME FOLDER` puts one there), or a dataset
config (.toml) in musubi-tuner's format, for several folders or other settings. A run writes everything into
ComfyUI/output/train/<run>/ (checkpoints, samples, logs, its dataset config and prompts), so `pod.py pull` brings it
home, and links every checkpoint into ComfyUI's LoRA list, under train/<run>/.

The recipes repeat the settings of the AInVFX LoRA kit, proven on RTX PRO 6000 pods in September 2026, on the
musubi-tuner commit that bootstrap.sh pins; `moss` follows the MOSS-Image guide of sd-scripts, the trainer that
bootstrap.sh pins for it. README.md, "Training a LoRA", is the guide.
"""
import argparse
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("AINVFX_ROOT", "/workspace"))          # the tests point it at a temporary folder
COMFY = ROOT / "ComfyUI"
MUSUBI = ROOT / "musubi-tuner"
SDSCRIPTS = ROOT / "sd-scripts"
DATASETS = ROOT / "dataset"
CACHE = ROOT / "cache"
RUNS = COMFY / "output" / "train"
LORAS = COMFY / "models" / "loras" / "train"
MODELS_JSON = ROOT / "models.json"                                  # the models list bootstrap.sh fetched
JOB = RUNS / ".job.json"                                            # the run in progress, or the last one
STAGING = ROOT / ".hfdl-train"
TRAINERS = {"musubi-tuner": (MUSUBI, "src/musubi_tuner/"), "sd-scripts": (SDSCRIPTS, "")}   # folder, scripts there
COMFY_URL = os.environ.get("AINVFX_COMFY_URL", "http://127.0.0.1:8188")
CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"                         # Apache 2.0, 17.6 GB, downloaded at first use
IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif")        # the fork's IMAGE_EXTENSIONS and VIDEO_EXTENSIONS
VIDEOS = (".mp4", ".webm", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg")
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
ACTIVE = ("starting", "running")
SAVE_EVERY = 250          # steps between two checkpoints, and two sets of samples
STABLE_SECONDS = 15       # a checkpoint untouched this long is complete: it is then converted for ComfyUI

# ----------------------------------------------------------------------------- recipes
# A recipe is one kind of LoRA: its family of base files (the `training` part of models.json names them), the data it
# reads, its trainer (musubi-tuner unless named), the settings of the dataset config written for a folder, the
# trainer's caching steps, the training script and its arguments, how its checkpoints load in ComfyUI, and its
# sample prompts. {name} placeholders are filled per run: the base files by role, and config, prompts, run (the
# run's folder) and name.
COMMON = ["--dataset_config", "{config}", "--max_data_loader_n_workers", "2", "--persistent_data_loader_workers",
          "--save_every_n_steps", str(SAVE_EVERY), "--sample_at_first", "--sample_every_n_steps", str(SAVE_EVERY),
          "--sample_prompts", "{prompts}", "--seed", "42", "--log_with", "tensorboard", "--logging_dir", "{run}/logs",
          "--output_dir", "{run}", "--output_name", "{name}"]
LTX_LATENTS = ["ltx2_cache_latents.py", "--ltx2_checkpoint", "{checkpoint}", "--vae", "{vae}", "--ltx_version", "2.5",
               "--device", "cuda", "--vae_dtype", "bf16"]
# The sample prompts are encoded with the captions and read back by the training (--use_precached_sample_prompts),
# which then loads no text encoder. The samples stay small and are decoded in one piece: LTX 2.5's video decoder
# imposes its own tiling and refuses --sample_tiled_vae.
LTX_TEXT = ["ltx2_cache_text_encoder_outputs.py", "--ltx2_checkpoint", "{checkpoint}", "--ltx2_text_encoder_checkpoint",
            "{text_encoder}", "--ltx_version", "2.5", "--device", "cuda", "--mixed_precision", "bf16", "--batch_size", "4",
            "--precache_sample_prompts", "--sample_prompts", "{prompts}"]
LTX_TRAIN = ["--mixed_precision", "bf16", "--ltx2_checkpoint", "{checkpoint}", "--ltx2_video_vae", "{vae}",
             "--ltx_version", "2.5", "--ltx_version_check_mode", "warn", "--sdpa", "--fp8_base", "--fp8_scaled",
             "--gradient_checkpointing", "--network_module", "networks.lora_ltx2", "--network_dim", "32",
             "--network_alpha", "32", "--learning_rate", "1e-4", "--optimizer_type", "AdamW8bit",
             "--lr_scheduler", "constant_with_warmup", "--lr_warmup_steps", "10", "--use_precached_sample_prompts",
             "--sample_sampling_preset", "ltx25_full", "--sample_with_offloading", "--save_checkpoint_metadata"]
LTX_CLIPS = dict(target_frames=[121], frame_extraction="head", target_fps=25.0)   # 121 frames from the first; fps a decimal
LTX_SAMPLE = "--n blurry, low quality, static, watermark --w 768 --h 448 --f 65 --s 30 --l 4.0 --d 42"
LTX_STYLE = ["--ltx2_mode", "video", "--lora_target_preset", "video_sa_ca_ff", "--timestep_sampling",
             "shifted_logit_normal", "--shifted_logit_uniform_prob", "0.30", "--caption_dropout_rate", "0.1",
             "--max_train_steps", "1500"]
RECIPES = {
    "zimage": dict(
        about="Z-Image LoRA on Z-Image Base: a style, a character, an object",
        family="zimage", data="images", general=dict(resolution=[1024, 1024], batch_size=2), block={},
        cache=[["zimage_cache_latents.py", "--vae", "{vae}"],
               ["zimage_cache_text_encoder_outputs.py", "--text_encoder", "{text_encoder}", "--batch_size", "16"]],
        script="zimage_train_network.py",
        args=["--dit", "{dit}", "--vae", "{vae}", "--text_encoder", "{text_encoder}", "--sdpa", "--mixed_precision",
              "bf16", "--timestep_sampling", "logsnr", "--logit_mean", "-6.0", "--logit_std", "2.0",
              "--optimizer_type", "optimi.AdamW", "--optimizer_args", "betas=[0.9,0.99]", "weight_decay=0.01",
              "eps=1e-8", "--learning_rate", "1e-4", "--network_module", "networks.lora_zimage", "--network_dim", "32",
              "--network_alpha", "16", "--max_train_steps", "1500"],
        comfy="convert",
        sample="a narrow street at dusk, wet cobblestones, a woman in a red coat walking away, warm shop windows",
        options="--n blurry, low quality, watermark, text --w 1024 --h 1024 --fs 2.0 --s 25 --l 4 --d 42", control=None),
    "qwen_edit": dict(
        about="Qwen-Image-Edit-2511 LoRA: one edit, learned from pairs (an image to edit, its result)",
        family="qwen_edit", data="image pairs", general=dict(resolution=[1024, 1024], batch_size=1),
        block=dict(control_resolution=[1024, 1024]),
        cache=[["qwen_image_cache_latents.py", "--vae", "{vae}", "--model_version", "edit-2511"],
               ["qwen_image_cache_text_encoder_outputs.py", "--text_encoder", "{text_encoder}", "--batch_size", "1",
                "--model_version", "edit-2511"]],
        script="qwen_image_train_network.py",
        args=["--dit", "{dit}", "--vae", "{vae}", "--text_encoder", "{text_encoder}", "--model_version", "edit-2511",
              "--sdpa", "--mixed_precision", "bf16", "--timestep_sampling", "shift", "--weighting_scheme", "none",
              "--discrete_flow_shift", "2.2", "--optimizer_type", "adamw8bit", "--learning_rate", "5e-5",
              "--gradient_checkpointing", "--network_module", "networks.lora_qwen_image", "--network_dim", "16",
              "--network_alpha", "1", "--max_train_steps", "1250"],
        comfy="same", sample="make it night, the street lit by sodium lamps",
        options="--n blurry, low quality, watermark --w 1024 --h 1024 --s 20 --l 4.0 --d 42", control="--ci"),
    # sd-scripts trains MOSS-Image, which musubi-tuner does not: its own dataset format, its caches made by the training
    # itself (next to the images), the settings of its MOSS-Image guide (the timestep distribution of the pre-training).
    "moss": dict(
        about="MOSS-Image LoRA on MOSS-Image Base: a style, a character, an object (trained by sd-scripts)",
        family="moss", data="images", trainer="sd-scripts", general=dict(resolution=[1024, 1024], batch_size=1),
        block={}, cache=[], script="moss_image_train_network.py",
        args=["--pretrained_model_name_or_path", "{dit}", "--text_encoder", "{text_encoder}", "--vae", "{vae}",
              "--save_model_as", "safetensors", "--network_module", "networks.lora_generic", "--network_dim", "16",
              "--network_alpha", "1", "--network_train_unet_only", "--learning_rate", "1e-4", "--optimizer_type",
              "AdamW8bit", "--lr_scheduler", "constant", "--timestep_sampling", "truncated_logit_normal",
              "--logit_mean", "-1.09", "--logit_std", "0.95", "--discrete_flow_shift", "1.0", "--mixed_precision", "bf16",
              "--gradient_checkpointing", "--cache_latents", "--cache_latents_to_disk", "--cache_text_encoder_outputs",
              "--cache_text_encoder_outputs_to_disk", "--max_train_steps", "1500"],
        comfy="same",
        sample="a narrow street at dusk, wet cobblestones, a woman in a red coat walking away, warm shop windows",
        options="--w 1024 --h 1024 --s 30 --l 3.0 --d 42", control=None),
    "ltx": dict(
        about="LTX 2.5 LoRA from clips: a style or a motion",
        family="ltx", data="clips", general=dict(resolution=[960, 544], batch_size=1), block=LTX_CLIPS,
        cache=[LTX_LATENTS, LTX_TEXT], script="ltx2_train_network.py", args=LTX_TRAIN + LTX_STYLE, comfy="fork",
        sample="a slow dolly in on a lighthouse at dusk, waves breaking on the rocks, gulls crossing the frame",
        options=LTX_SAMPLE, control=None),
    "ltx_images": dict(
        about="LTX 2.5 LoRA from still images: a look (colours, lines, materials), never a motion",
        family="ltx", data="images", general=dict(resolution=[1024, 576], batch_size=1), block={},
        cache=[LTX_LATENTS, LTX_TEXT], script="ltx2_train_network.py", args=LTX_TRAIN + LTX_STYLE, comfy="fork",
        sample="a slow dolly in on a lighthouse at dusk, waves breaking on the rocks, gulls crossing the frame",
        options=LTX_SAMPLE, control=None),
    # The control video is encoded at half size (--reference_downscale 2): the LoRA file then carries that factor, which
    # the LTX IC-LoRA Loader Model Only node reads in ComfyUI, and the trainer aligns buckets on 64 pixels, hence
    # 1024 x 576. The samples need 121 frames, the length the control video is encoded at.
    "ltx_iclora": dict(
        about="LTX 2.5 IC-LoRA: a control video turned into a result, learned from pairs",
        family="ltx", data="clip pairs", general=dict(resolution=[1024, 576], batch_size=1), block=LTX_CLIPS,
        cache=[LTX_LATENTS + ["--reference_frames", "121", "--reference_downscale", "2", "--precache_sample_latents",
                              "--sample_prompts", "{prompts}"], LTX_TEXT],
        script="ltx2_train_network.py",
        args=LTX_TRAIN + ["--ltx2_mode", "video", "--lora_target_preset", "v2v", "--reference_frames", "121",
                          "--reference_downscale", "2", "--ltx2_first_frame_conditioning_p", "0.2",
                          "--timestep_sampling", "shifted_logit_normal", "--max_train_steps", "2000",
                          "--sample_include_reference"],
        comfy="fork", sample="a woman walks along a harbour at sunset, warm rim light, gentle camera pan",
        options="--n blurry, low quality, watermark --w 576 --h 320 --f 121 --s 30 --l 4.0 --d 42", control="--v"),
    "ltx_av": dict(
        about="LTX 2.5 LoRA from clips with their sound: the picture and the audio together",
        family="ltx", data="clips", general=dict(resolution=[960, 544], batch_size=1), block=LTX_CLIPS,
        cache=[LTX_LATENTS + ["--ltx2_audio_vae", "{audio_vae}", "--ltx2_mode", "av", "--ltx2_audio_source", "video"],
               LTX_TEXT + ["--ltx2_mode", "av"]],
        script="ltx2_train_network.py",
        args=LTX_TRAIN + ["--ltx2_audio_vae", "{audio_vae}", "--ltx2_mode", "av", "--separate_audio_buckets",
                          "--lora_target_preset", "t2v", "--audio_dim", "16", "--audio_alpha", "16", "--audio_lr", "5e-5",
                          "--timestep_sampling", "shifted_logit_normal", "--shifted_logit_uniform_prob", "0.30",
                          "--caption_dropout_rate", "0.1", "--max_train_steps", "1500", "--sample_merge_audio"],
        comfy="fork", sample="a control room, rows of consoles under a large screen, slow pan; a voice on the radio "
                             "counts down, the room is quiet", options=LTX_SAMPLE, control=None),
}
DATA = {
    "images": "images, each with its caption: a .txt of the same name",
    "clips": "video clips of 121 frames or more, each with its caption",
    "image pairs": "targets/ (the results, with their captions) and control/ (the images to edit, same names)",
    "clip pairs": "targets/ (the result clips, with their captions) and control/ (the control clips, same names)",
}
CAPTION_PROMPTS = {
    "image": "Describe this image in two sentences: subject, action, framing, lighting, setting. Present tense, "
             "no opinions, no artistic style words.",
    "video": "Describe this short video clip in two sentences: the subject and its action, the camera movement, the "
             "framing, the lighting and the setting. Present tense, no opinions.",
    "audio": "Describe this short video clip in three sentences: the subject and its action, the camera movement, the "
             "framing, the lighting and the setting, then WHAT is heard (a voice, a radio, an ambience, a noise), "
             "without describing how it sounds: no words about texture, quality, distortion or age of the sound. "
             "Present tense, no opinions.",
    "edit": "You see two images. The first is the SOURCE, the second is the RESULT after an edit. Write the single "
            "instruction, in the imperative, one sentence, that turns the first image into the second: what to change "
            "(style, shading, colours, rendering, objects) and what to keep (subject, pose, framing). Do not describe "
            "the images, do not add commentary, output only the instruction.",
}
STOP_WORDS = set("""a an the and or of in on at to with from by for is are was were be been this that these those it its
as into over under near behind against while during their his her they them he she we you very soft hard light dark""".split())


# ----------------------------------------------------------------------------- small helpers
def say(msg=""):
    print(msg, flush=True)


def die(msg, code=2):
    print("\nERROR: " + msg, file=sys.stderr, flush=True)
    sys.exit(code)


def duration(secs):
    secs = int(secs)
    if secs >= 3600:
        return "{} h {:02d} min".format(secs // 3600, secs % 3600 // 60)
    return "{} min {:02d} s".format(secs // 60, secs % 60) if secs >= 60 else "{} s".format(secs)


def resolve_path(ref, base):
    """A path as given: absolute, or relative to this folder when it exists here, else under `base`."""
    p = Path(ref).expanduser()
    if p.is_absolute() or p.exists():
        return p.resolve()
    return base / ref


def arg_value(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def set_flag(args, flag, value):
    """The arguments with `flag value` replaced, or added when the flag is absent."""
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def with_rank(args, rank):
    """Rank and alpha change by the same factor: the trainer scales every LoRA update by alpha / rank, so changing
    the rank alone would change the learning speed. The learning rate is the speed setting."""
    dim, alpha = int(arg_value(args, "--network_dim")), float(arg_value(args, "--network_alpha"))
    return set_flag(set_flag(args, "--network_dim", str(rank)), "--network_alpha", "{:g}".format(alpha * rank / dim))


def fill(args, values):
    return [re.sub(r"\{(\w+)\}", lambda m: values[m.group(1)], a) for a in args]


def alive(pid):
    """True while the process runs; a zombie (ended, not yet collected) counts as ended."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path("/proc/{}/stat".format(pid)).read_text().rsplit(") ", 1)[-1][:1] != "Z"
    except OSError:
        return True


# ----------------------------------------------------------------------------- the run's record
def read_job():
    try:
        return json.loads(JOB.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_job(job):
    tmp = JOB.with_name(".job.json.tmp")
    tmp.write_text(json.dumps(job, indent=1), encoding="utf-8")
    os.replace(tmp, JOB)


def update_job(**changes):
    job = read_job() or {}
    job.update(changes)
    write_job(job)
    return job


def active_job():
    """The run in progress, or None. A run whose process is gone (a pod restart) is not in progress."""
    job = read_job()
    if not job or job.get("state") not in ACTIVE:
        return None
    if job.get("pid"):
        return job if alive(job["pid"]) else None
    return job if time.time() - job.get("created", 0) < 60 else None


# ----------------------------------------------------------------------------- base files
def family_files(family):
    """{role: dict(key, repo, path, gb, gated, dest)} for one family of base files, from the pod's models list."""
    try:
        spec = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        die("no readable models list at {}: bootstrap.sh writes it at start (`bash /tmp/ainvfx-bootstrap.sh models` "
            "fetches it again).".format(MODELS_JSON))
    fam = (spec.get("training") or {}).get(family)
    if not fam:
        die("the models list has no training entry for {}: a models list of your own needs the `training` part of the "
            "course's models.json.".format(family))
    out = {}
    for role, key in fam.items():
        f = (spec.get("files") or {}).get(key)
        if not f:
            die("the models list names {} for {}, and has no such file.".format(key, family))
        out[role] = dict(key=key, repo=f["repo"], path=f["path"], gb=float(f.get("gb") or 0), gated=bool(f.get("gated")),
                         dest=COMFY / "models" / f["dir"] / Path(f["path"]).name)
    return out


def present(f):
    return f["dest"].is_file() and f["dest"].stat().st_size > 0


def hf_token():
    """The pod's Hugging Face token: HF_TOKEN, else the container's own variables (a terminal can lack them).
    Runpod's placeholder for a missing secret counts as no token."""
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        try:
            for item in Path("/proc/1/environ").read_bytes().split(b"\0"):
                if item.startswith(b"HF_TOKEN="):
                    token = item[len(b"HF_TOKEN="):].decode("utf-8", "replace")
                    break
        except OSError:
            pass
    return "" if "{{" in token else token


def ensure_models(family):
    """The paths of a family's base files by role, downloaded first when missing: once per pod."""
    files = family_files(family)
    todo = [f for f in files.values() if not present(f)]
    if todo:
        need, free = sum(f["gb"] for f in todo), shutil.disk_usage(ROOT).free / 1e9
        if need * 1.05 > free:
            die("{:.0f} GB to download for this recipe and {:.0f} GB free on the disk: create the pod with more disk "
                "(`python pod.py up train` sizes it for the largest recipe).".format(need, free))
        download(todo)
    return {role: str(f["dest"]) for role, f in files.items()}


def download(todo):
    from huggingface_hub import hf_hub_download          # part of the trainer's environment
    token, waiting = hf_token(), []
    say("models: {} base file(s) for this recipe, {:.1f} GB, downloaded once and kept for the next runs".format(
        len(todo), sum(f["gb"] for f in todo)))
    for k, f in enumerate(todo, 1):
        say("models {}/{} · {} ({:.2f} GB) from {}".format(k, len(todo), f["dest"].name, f["gb"], f["repo"]))
        t0 = time.time()
        try:
            got = hf_hub_download(f["repo"], f["path"], local_dir=str(STAGING), token=token or None)
        except Exception as e:                            # huggingface_hub's GatedRepoError, or the network
            if f["gated"] and ("gated" in type(e).__name__.lower() or " 401" in str(e) or " 403" in str(e)):
                waiting.append(f)
                continue
            die("{} could not be downloaded from {}: {}: {}. Run the same command again: the files already here are "
                "kept.".format(f["dest"].name, f["repo"], type(e).__name__, str(e)[:300]))
        f["dest"].parent.mkdir(parents=True, exist_ok=True)
        os.replace(got, f["dest"])
        say("models {}/{} · {} in {}".format(k, len(todo), f["dest"].name, duration(time.time() - t0)))
    shutil.rmtree(STAGING, ignore_errors=True)
    if waiting:
        pages = ", ".join("https://huggingface.co/" + r for r in sorted({f["repo"] for f in waiting}))
        if token:
            die("ACTION NEEDED · the Hugging Face account of this pod's token has not accepted the licence of {}. Logged "
                "in to huggingface.co with that account, open each page and click « Agree and access repository », "
                "then run the same command again.".format(pages))
        die("ACTION NEEDED · {} is gated and this pod has no Hugging Face token. On your computer: python pod.py setup "
            "--hf-token, accept the licence on each page, then create a new pod.".format(pages))


# ----------------------------------------------------------------------------- datasets
def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    return "[" + ", ".join(toml_value(x) for x in v) + "]"


def cache_name(folder):
    try:
        rel = str(folder.relative_to(DATASETS))
    except ValueError:
        rel = folder.name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", rel)


def make_config(recipe, folder):
    """The dataset config of one folder, in its trainer's format, with the recipe's settings."""
    r = RECIPES[recipe]
    head = ["# Written by train.py for the {} recipe. To change a setting, copy this file, edit the copy, then "
            "--dataset <the copy>.toml.".format(recipe), ""]
    if r.get("trainer") == "sd-scripts":           # one dataset of one subset; sd-scripts caches next to the images
        lines = head + ["[general]", 'caption_extension = ".txt"', "enable_bucket = true", "", "[[datasets]]"]
        lines += ["{} = {}".format(k, toml_value(v)) for k, v in r["general"].items()]
        lines += ["", "[[datasets.subsets]]", "image_dir = {}".format(toml_value(str(folder))), "num_repeats = 1"]
        return "\n".join(lines) + "\n"
    general = dict(dict(caption_extension=".txt", enable_bucket=True, bucket_no_upscale=False), **r["general"])
    cache = CACHE / recipe / cache_name(folder)
    block = {}
    if r["data"] == "images":
        block["image_directory"] = str(folder)
    elif r["data"] == "clips":
        block["video_directory"] = str(folder)
    elif r["data"] == "image pairs":
        block.update(image_directory=str(folder / "targets"), control_directory=str(folder / "control"))
    else:
        block.update(video_directory=str(folder / "targets"), reference_directory=str(folder / "control"),
                     reference_cache_directory=str(cache) + "_control")
    block.update(r["block"])
    block.update(cache_directory=str(cache), num_repeats=1)
    lines = head + ["[general]"]
    lines += ["{} = {}".format(k, toml_value(v)) for k, v in general.items()]
    lines += ["", "[[datasets]]"] + ["{} = {}".format(k, toml_value(v)) for k, v in block.items()]
    return "\n".join(lines) + "\n"


def dataset_config(recipe, ref):
    """(the dataset config's text, where it comes from): the .toml named, or one written for the folder named."""
    if ref.endswith(".toml"):
        path = resolve_path(ref, DATASETS)
        try:
            return path.read_text(encoding="utf-8"), str(path)
        except OSError:
            die("no dataset config {}".format(path))
    folder = resolve_path(ref, DATASETS)
    if not folder.is_dir():
        die("no dataset folder {} (python pod.py push --dataset {} <your folder> sends one)".format(folder, ref))
    return make_config(recipe, folder), str(folder)


def dataset_blocks(text, source):
    """Every folder block of a config, with the settings it inherits and the format it came in: musubi-tuner's
    (the folder in the [[datasets]] block) or sd-scripts' (one folder per [[datasets.subsets]], image_dir)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        die("{} is not valid TOML: {}".format(source, e))
    general, blocks = data.get("general") or {}, []
    for b in data.get("datasets") or []:
        if "subsets" not in b:
            blocks.append(dict(general, _format="musubi-tuner", **b))
            continue
        for s in b["subsets"]:
            m = dict(general, _format="sd-scripts")
            m.update((k, v) for k, v in b.items() if k != "subsets")
            m.update(s)
            if "image_dir" in m:
                m["image_directory"] = m.pop("image_dir")
            blocks.append(m)
    if not blocks:
        die("{} has no [[datasets]] block".format(source))
    return blocks


def media_files(folder, exts):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."))


def caption_of(f):
    try:
        return " ".join(f.with_suffix(".txt").read_text(encoding="utf-8").split())
    except OSError:
        return ""


def control_index(folder):
    """{file name without extension: file} of a control folder: a pair is matched by name, whatever the extension."""
    return {p.stem: p for p in media_files(folder, IMAGES + VIDEOS)}


def check_blocks(recipe, blocks, source):
    """Every block of a dataset config, checked against the recipe before any GPU time is spent: its folder, its files,
    their captions and partners, and two settings the trainer refuses. Returns one summary per block (None for a
    JSONL dataset, which is not checked here)."""
    want, infos, caches, problems = RECIPES[recipe]["data"], [], [], []
    for n, b in enumerate(blocks, 1):
        where = "{} (block {})".format(source, n) if len(blocks) > 1 else source
        trainer = RECIPES[recipe].get("trainer", "musubi-tuner")
        if b.get("_format") != trainer:
            problems.append("{} is in {}'s dataset format, and the {} recipe trains with {}".format(
                where, b.get("_format"), recipe, trainer))
            continue
        if "image_jsonl_file" in b or "video_jsonl_file" in b:
            say("{}: a JSONL dataset, not checked here".format(where))
            infos.append(None)
            continue
        video = "video_directory" in b
        folder, control = b.get("video_directory") or b.get("image_directory"), b.get("reference_directory") or b.get(
            "control_directory")
        kind = ("clip" if video else "image") + (" pairs" if control else "s")
        if kind != want:
            problems.append("{} holds {}, and the {} recipe trains on {}".format(where, kind, recipe, want))
            continue
        if not folder or not Path(folder).is_dir():
            problems.append("{}: no folder {}{}".format(where, folder, " (a pair dataset holds targets/ and control/)"
                                                        if control else ""))
            continue
        files = media_files(folder, VIDEOS if video else IMAGES)
        captioned = [f for f in files if caption_of(f)]
        index = control_index(control) if control and Path(control).is_dir() else {}
        if not files:
            problems.append("{}: no {} in {}".format(where, "clips" if video else "images", folder))
        elif not captioned:
            problems.append("{}: no captions in {}: `train caption <dataset> --trigger <word>` writes them".format(
                where, folder))
        elif control and not Path(control).is_dir():
            problems.append("{}: no folder {} (the files the model is given, same names as the results)".format(
                where, control))
        elif control and any(f.stem not in index for f in files):
            lonely = [f.name for f in files if f.stem not in index]
            problems.append("{}: {} file(s) without a partner of the same name in {}: {}".format(
                where, len(lonely), control, ", ".join(lonely[:3])))
        fps = b.get("target_fps")
        if fps is not None and not isinstance(fps, float):
            problems.append("{}: write target_fps as a decimal ({:.1f}): the trainer refuses a whole number".format(
                where, fps))
        if trainer == "musubi-tuner":                    # sd-scripts keeps its caches next to the images
            cache = b.get("cache_directory")
            if not cache:
                problems.append("{}: no cache_directory".format(where))
            elif cache in caches:
                problems.append("{}: cache_directory {} is another block's too: the trainer needs one per "
                                "block".format(where, cache))
            caches.append(cache)
        if files and captioned:
            say("{}: {} {}, {} with a caption{}".format(where, len(files), "clips" if video else "images", len(captioned),
                                                        ", each with its partner" if control else ""))
            if len(captioned) < len(files):
                say("  {} without a caption: write theirs, or take them out".format(len(files) - len(captioned)))
        infos.append(dict(files=captioned, control=control, index=index))
    if problems:
        die("the dataset is not ready:\n  " + "\n  ".join(problems))
    return infos


# ----------------------------------------------------------------------------- sample prompts
def sample_line(r, text, partner):
    text = " ".join(text.replace(" --", " - ").split())        # a caption must not open an option of the prompt line
    ctrl = "{} {} ".format(r["control"], partner) if r["control"] and partner else ""
    return "{} {}{}".format(text, ctrl, r["options"])


def guess_trigger(caption):
    """The trigger word of a caption written by `train caption`: the word before its first comma."""
    m = re.match(r"([^\s,]+),\s", caption)
    return m.group(1) if m else ""


def make_prompts(recipe, info, trigger):
    """(the text of the sample prompts, the trigger used): two captions of the dataset (the first and the middle one),
    to see early that the LoRA learns what it sees, then the recipe's own subject with the trigger, to see whether
    the LoRA carries over to something new."""
    r, files = RECIPES[recipe], info["files"]
    picks = [files[0]] + ([files[len(files) // 2]] if len(files) > 1 else [])

    def partner(f):
        return info["index"].get(f.stem) if r["control"] else None
    lines = [sample_line(r, caption_of(f), partner(f)) for f in picks]
    trigger = trigger or guess_trigger(caption_of(files[0]))
    if trigger:
        lines.append(sample_line(r, "{}, {}".format(trigger, r["sample"]), partner(files[0])))
    head = ("# The sample prompts of this run, one per line, with their options after the text (--w, --h, --s, --d...).\n"
            "# The trainer renders them at the start and every {} steps, into the run's sample folder.\n".format(SAVE_EVERY))
    return head + "\n".join(lines) + "\n", trigger


# ----------------------------------------------------------------------------- a run
def resolve_from(ref, earlier):
    """The weights a stage starts from: a run's final LoRA by the run's name, a checkpoint by its file name, or a path."""
    if ref.endswith((".comfy.safetensors", "_comfy.safetensors")):
        die("--from takes a checkpoint in musubi-tuner's own format ({}), not its ComfyUI copy".format(
            ref.replace(".comfy.safetensors", ".safetensors").replace("_comfy.safetensors", ".safetensors")))
    if not ref.endswith(".safetensors"):                          # a run's name: its final weights
        path = RUNS / ref / (ref + ".safetensors")
        if ref not in earlier and not path.is_file():
            die("--from {}: no finished run of that name ({} is missing)".format(ref, path))
        return str(path)
    if "/" in ref or os.sep in ref:
        path = Path(ref).expanduser()
        if not path.is_file():
            die("--from: no file {}".format(path))
        return str(path.resolve())
    hits = sorted(RUNS.glob("*/" + ref))
    if len(hits) != 1:
        die("--from {}: {} in {}".format(ref, "no such checkpoint" if not hits else "several runs hold that name", RUNS))
    return str(hits[0])


def parse_stages(recipe, argv):
    """(stages, prompts file, trigger) from `<run> [options] --then <run> [options] ...`. A stage without --dataset
    takes the one before."""
    groups, cur = [], []
    for a in argv:
        if a == "--then":
            groups.append(cur)
            cur = []
        else:
            cur.append(a)
    groups.append(cur)
    p = argparse.ArgumentParser(prog="train {} <run>".format(recipe), description=RECIPES[recipe]["about"] + ". Dataset: "
                                + DATA[RECIPES[recipe]["data"]] + ". Several stages: --then <run> [options], repeated.")
    p.add_argument("run", help="the run's name: its folder in ComfyUI/output/train and the name of its files")
    p.add_argument("--dataset", help="a folder of /workspace/dataset, or a dataset config (.toml)")
    p.add_argument("--steps", type=int, help="the number of training steps (the recipe's own otherwise)")
    p.add_argument("--lr", type=float, help="the learning rate (the recipe's own otherwise)")
    p.add_argument("--rank", type=int, help="the LoRA's rank; alpha changes by the same factor")
    p.add_argument("--from", dest="start", metavar="RUN_OR_CHECKPOINT",
                   help="start from these weights: a run's name (its final LoRA) or a checkpoint's file name; with --steps")
    p.add_argument("--prompts", help="your own sample prompts file (first stage)")
    p.add_argument("--trigger", help="the trigger word of the third sample prompt (read from the captions otherwise)")
    stages, earlier, first = [], [], None
    for i, group in enumerate(groups):
        if not group:
            die("--then must be followed by a run name and its options")
        a = p.parse_args(group)
        first = first or a
        if not RUN_NAME.fullmatch(a.run):
            die("a run name takes letters, digits, _ and -, up to 64 (got \"{}\")".format(a.run))
        if a.run in earlier:
            die("two stages are named {}".format(a.run))
        if i and (a.prompts or a.trigger):
            die("--prompts and --trigger go with the first run")
        if (a.steps is not None and a.steps < 1) or (a.rank is not None and a.rank < 1) or (a.lr is not None and a.lr <= 0):
            die("--steps, --rank and --lr take positive numbers")
        dataset = a.dataset or (stages[-1]["dataset"] if stages else None)
        if not dataset:
            die("which data? --dataset NAME (a folder of /workspace/dataset) or --dataset FILE.toml")
        if a.start and not a.steps:
            die("--from needs --steps N: the steps of the new run, whose count starts at 0")
        stages.append(dict(name=a.run, dataset=dataset, steps=a.steps, lr=a.lr, rank=a.rank,
                           start=resolve_from(a.start, earlier) if a.start else None))
        earlier.append(a.run)
    return stages, first.prompts, first.trigger


def stage_commands(recipe, stage, values):
    """[title, argv] of every step of one stage: the trainer's caches (musubi-tuner makes them first, sd-scripts
    within the training), then the training, each with the trainer's own Python."""
    r = RECIPES[recipe]
    home, scripts = TRAINERS[r.get("trainer", "musubi-tuner")]
    python, accelerate = str(home / ".venv" / "bin" / "python"), str(home / ".venv" / "bin" / "accelerate")
    cmds = [["caching ({})".format(c[0].split("_cache_")[1][:-3].replace("_", " ")),
             [python, scripts + c[0], "--dataset_config", values["config"]] + fill(c[1:], values)]
            for c in r["cache"]]
    args = fill(r["args"] + COMMON, values)
    if stage["steps"]:
        args = set_flag(args, "--max_train_steps", str(stage["steps"]))
    if stage["lr"]:
        args = set_flag(args, "--learning_rate", "{:g}".format(stage["lr"]))
    if stage["rank"]:
        args = with_rank(args, stage["rank"])
    if stage["start"]:
        args = set_flag(args, "--network_weights", stage["start"])
    cmds.append(["training", [accelerate, "launch", "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
                              scripts + r["script"]] + args])
    return cmds


def publish(recipe, run, name, final=False):
    """Every checkpoint of a run, usable in ComfyUI: Z-Image's converted (musubi-tuner's own format does not load
    there), LTX's as the fork writes them (.comfy.safetensors), Qwen-Image-Edit's and MOSS-Image's as they are; each linked into ComfyUI's LoRA
    list, under train/<run>/."""
    kind, now = RECIPES[recipe]["comfy"], time.time()
    weights = [f for f in sorted(run.glob(name + "*.safetensors"))
               if not f.name.endswith((".comfy.safetensors", "_comfy.safetensors"))]
    if kind == "convert":
        for f in weights:
            out = f.with_name(f.stem + "_comfy.safetensors")
            if out.exists() or (not final and now - f.stat().st_mtime < STABLE_SECONDS):
                continue
            tmp = out.with_name("." + out.name)
            r = subprocess.run([str(MUSUBI / ".venv" / "bin" / "python"), "src/musubi_tuner/convert_lora.py",
                                "--input", str(f), "--output", str(tmp), "--target", "other"],
                               cwd=str(MUSUBI), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if r.returncode == 0 and tmp.is_file():
                os.replace(tmp, out)
            else:
                tmp.unlink(missing_ok=True)
                print("conversion of {} for ComfyUI failed: {}".format(f.name, r.stderr.decode("utf-8", "replace")[-300:]),
                      file=sys.stderr, flush=True)
    ending = {"convert": "_comfy.safetensors", "fork": ".comfy.safetensors"}.get(kind)
    usable = [f for f in sorted(run.glob(name + "*.safetensors")) if (f.name.endswith(ending) if ending else f in weights)]
    if usable:
        folder = LORAS / name
        folder.mkdir(parents=True, exist_ok=True)
        for f in usable:
            link = folder / f.name
            if not link.is_symlink() and not link.exists():
                link.symlink_to(f)


def free_comfy_memory():
    """ComfyUI, when it runs on this pod, unloads its models: a training run needs the GPU's memory."""
    req = urllib.request.Request(COMFY_URL + "/free", data=b'{"unload_models": true, "free_memory": true}',
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=10).close()
        say("ComfyUI unloaded its models from the GPU for the run (run no workflow until it ends)")
    except OSError:
        pass


def cmd_train(recipe, argv):
    trainer = RECIPES[recipe].get("trainer", "musubi-tuner")
    if not (TRAINERS[trainer][0] / ".venv" / "bin" / "python").is_file():
        die("{} is not installed on this pod: create the pod with the train profile, or install the trainers "
            "here: bash /tmp/ainvfx-bootstrap.sh trainer".format(trainer))
    stages, prompts_file, trigger = parse_stages(recipe, argv)
    job = active_job()
    if job:
        die("a run is in progress: {}. `train status` shows it, `train stop` stops it.".format(
            job["stages"][job.get("stage", 0)]["name"]))
    for st in stages:
        run = RUNS / st["name"]
        if run.exists() and any(run.iterdir()):
            die("a run named {} exists ({}): choose another name, or delete that folder first".format(st["name"], run))
    plans = []
    for st in stages:
        text, source = dataset_config(recipe, st["dataset"])
        say("dataset of {}: {}".format(st["name"], source))
        plans.append((text, check_blocks(recipe, dataset_blocks(text, source), source)))
    if prompts_file:
        try:
            prompts = resolve_path(prompts_file, ROOT).read_text(encoding="utf-8")
        except OSError:
            die("no prompts file {}".format(prompts_file))
    else:
        info = next((i for i in plans[0][1] if i), None)
        if not info:
            die("--prompts FILE is needed: this dataset's files could not be read here")
        prompts, trigger = make_prompts(recipe, info, trigger)
        if not trigger:
            say("no trigger word found (`train caption` starts every caption with it): the third sample prompt is left "
                "out; --trigger WORD adds it")
    say("sample prompts:")
    for line in prompts.splitlines():
        if line.strip() and not line.startswith("#"):
            say("  " + line)
    paths = ensure_models(RECIPES[recipe]["family"])
    job = dict(recipe=recipe, state="starting", stage=0, created=time.time(), stages=[])
    for st, (text, _) in zip(stages, plans):
        run = RUNS / st["name"]
        (run / "logs").mkdir(parents=True, exist_ok=True)
        (run / "dataset.toml").write_text(text, encoding="utf-8")
        (run / "prompts.txt").write_text(prompts, encoding="utf-8")
        values = dict(paths, config=str(run / "dataset.toml"), prompts=str(run / "prompts.txt"), run=str(run),
                      name=st["name"])
        job["stages"].append(dict(name=st["name"], run=str(run), cwd=str(TRAINERS[trainer][0]),
                                  commands=stage_commands(recipe, st, values)))
    free_comfy_memory()
    write_job(job)
    with open(RUNS / ".job.log", "ab") as out:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_run"], cwd=str(MUSUBI),
                                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    say("The run started in the background{}. Ctrl+C stops following the log, never the run; `train log` follows it "
        "again.\n".format(" ({} stages)".format(len(stages)) if len(stages) > 1 else ""))
    return follow(proc)


def run_job(argv=None):
    """The background part of a run (`train _run`): every stage in order; the first failure ends the run."""
    job = update_job(pid=os.getpid(), state="running", started=time.time())
    for i, st in enumerate(job["stages"]):
        update_job(stage=i)
        rc = run_stage(job["recipe"], st, i, len(job["stages"]))
        if rc:
            update_job(state="failed", rc=rc, ended=time.time())
            return rc
    update_job(state="done", rc=0, ended=time.time())
    return 0


def run_stage(recipe, st, i, n):
    run = Path(st["run"])
    with open(run / "train.log", "ab", buffering=0) as log:
        def note(text):
            log.write("[AINVFX] {}\n".format(text).encode("utf-8"))
        note("{} · recipe {} · stage {} of {} · {}".format(st["name"], recipe, i + 1, n, time.strftime("%Y-%m-%d %H:%M %Z")))
        for title, cmd in st["commands"]:
            note("{}: {}".format(title, " ".join(shlex.quote(a) for a in cmd)))
            t0 = last = time.time()
            p = subprocess.Popen(cmd, cwd=st["cwd"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            while p.poll() is None:
                time.sleep(2)
                if title == "training" and time.time() - last >= 30:
                    publish(recipe, run, st["name"])
                    last = time.time()
            if p.returncode:
                note("FAILED · {} ended with code {} after {}: the lines above say why".format(
                    title, p.returncode, duration(time.time() - t0)))
                return p.returncode
            note("{} done in {}".format(title, duration(time.time() - t0)))
        publish(recipe, run, st["name"], final=True)
        final = run / (st["name"] + ".safetensors")
        if not final.is_file():
            note("FAILED · the trainer ended without writing {}".format(final.name))
            return 1
        note("DONE · {} · in ComfyUI's LoRA list under train/{}/ (press r to refresh it)".format(final, st["name"]))
    return 0


def follow(proc=None, tail=None):
    """Prints the run's log as it grows, stage after stage, until the run ends; returns its code. Ctrl+C stops the
    printing only: the run goes on in the background."""
    stage, pos = None, None
    try:
        while True:
            job = read_job()
            if not job or not job.get("stages"):
                die("no run on this pod yet")
            current = job.get("stage", 0)
            stage = current if stage is None else stage
            log = Path(job["stages"][stage]["run"]) / "train.log"
            if pos is None:
                pos = max(0, log.stat().st_size - tail) if tail and log.exists() else 0
            chunk = b""
            if log.exists():
                with open(log, "rb") as f:
                    f.seek(pos)
                    chunk = f.read()
                pos += len(chunk)
            if chunk:
                sys.stdout.write(chunk.decode("utf-8", "replace"))
                sys.stdout.flush()
                continue
            if stage < current:                                  # the next stage has started: its own log
                stage, pos = stage + 1, 0
                continue
            if job.get("state") not in ACTIVE:
                return report_end(job)
            gone = proc.poll() is not None if proc else bool(job.get("pid")) and not alive(job["pid"])
            late = job.get("state") == "starting" and not job.get("pid") and time.time() - job.get("created", 0) > 60
            if gone or late:
                time.sleep(1)                                    # it may have written its end just before exiting
                job = read_job()
                if job.get("state") in ACTIVE:
                    job = update_job(state="failed", rc=1, ended=time.time())
                    say("\nThe run's process ended without a word. Its own log: {}".format(RUNS / ".job.log"))
                continue
            time.sleep(1)
    except KeyboardInterrupt:
        say("\nStopped following. The run goes on: `train log` follows it again, `train stop` stops it.")
        return 130


def report_end(job):
    state, last = job.get("state"), job["stages"][job.get("stage", 0)]
    if state == "done":
        say("\nThe run is done{}. Its folder: {} (`python pod.py pull` brings it home).".format(
            ": {} stages".format(len(job["stages"])) if len(job["stages"]) > 1 else "", last["run"]))
        return 0
    if state == "stopped":
        say("\nThe run was stopped. Its checkpoints stay in {}; `--from <checkpoint> --steps N` continues from one of "
            "them.".format(last["run"]))
    else:
        say("\nThe run FAILED at {} (code {}): the lines above say why. Its log: {}".format(
            last["name"], job.get("rc"), Path(last["run"]) / "train.log"))
    return job.get("rc") or 1


# ----------------------------------------------------------------------------- the other commands
def cmd_log(argv):
    argparse.ArgumentParser(prog="train log", description="Follows the log of the run in progress, or prints the end "
                                                          "of the last one.").parse_args(argv)
    if not read_job():
        say("No run on this pod yet. `train recipes` lists the recipes.")
        return 0
    return follow(tail=6000)


def cmd_status(argv):
    argparse.ArgumentParser(prog="train status", description="Where the run stands.").parse_args(argv)
    job = read_job()
    if not job or not job.get("stages"):
        say("No run on this pod yet. `train recipes` lists the recipes.")
    else:
        st, state = job["stages"][job.get("stage", 0)], job.get("state")
        if state in ACTIVE and not active_job():
            state = "ended without a word (the pod restarted?)"
        since = time.time() - job.get("started", job.get("created", time.time()))
        say("{} · recipe {} · stage {} of {} · {} · {} since its start".format(
            st["name"], job["recipe"], job.get("stage", 0) + 1, len(job["stages"]), state, duration(since)))
        log = Path(st["run"]) / "train.log"
        if log.exists():
            lines = [line.rsplit("\r", 1)[-1] for line in log.read_text(encoding="utf-8", errors="replace").splitlines()]
            for line in [x for x in lines if x.strip()][-6:]:
                say("  " + line[:200])
        weights = sorted(Path(st["run"]).glob("*.safetensors"))
        say("checkpoints: " + (", ".join(f.name for f in weights[-4:]) if weights else "none yet (one every {} steps)".format(
            SAVE_EVERY)))
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        say("GPU: " + gpu)
    except (OSError, subprocess.SubprocessError):
        pass
    say("disk free: {:.0f} GB".format(shutil.disk_usage(ROOT).free / 1e9))
    return 0


def cmd_stop(argv):
    p = argparse.ArgumentParser(prog="train stop", description="Stops the run in progress. Its checkpoints stay.")
    p.add_argument("-y", "--yes", action="store_true", help="no question")
    a = p.parse_args(argv)
    job = active_job()
    if not job:
        say("No run in progress.")
        return 0
    name = job["stages"][job.get("stage", 0)]["name"]
    if not a.yes and input("Stop the run {}? Its checkpoints stay. [y/N] ".format(name)).strip().lower() not in ("y", "yes"):
        say("The run goes on.")
        return 0
    pid = job.get("pid")
    if pid:
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                break
            for _ in range(wait):
                if not alive(pid):
                    break
                time.sleep(1)
            if not alive(pid):
                break
    update_job(state="stopped", rc=143, ended=time.time())
    say("Stopped: {}. Its checkpoints stay in {}.".format(name, job["stages"][job.get("stage", 0)]["run"]))
    return 0


def cmd_recipes(argv):
    argparse.ArgumentParser(prog="train recipes", description="Every recipe and its base files.").parse_args(argv)
    for name, r in RECIPES.items():
        files = family_files(r["family"])
        on_pod = all(present(f) for f in files.values())
        say("{:<11} {}".format(name, r["about"]))
        say("{:<11} dataset: {}".format("", DATA[r["data"]]))
        say("{:<11} base files: {:.0f} GB, {} · {} steps, learning rate {}, rank {}".format(
            "", sum(f["gb"] for f in files.values()), "on this pod" if on_pod else "downloaded by its first run",
            arg_value(r["args"], "--max_train_steps"), arg_value(r["args"], "--learning_rate"),
            arg_value(r["args"], "--network_dim")))
    say("\nA run:  train <recipe> <run name> --dataset <a folder of /workspace/dataset, or a .toml>   "
        "(train <recipe> -h lists the options)")
    return 0


# ----------------------------------------------------------------------------- captions
def caption_targets(ref, audio):
    """[dict(files, control, index, mode)] for one dataset: its folder (with targets/ and control/ for pairs), or
    each block of a dataset config. Images with an image to edit get an edit instruction; clips are described."""
    if ref.endswith(".toml"):
        path = resolve_path(ref, DATASETS)
        try:
            blocks = dataset_blocks(path.read_text(encoding="utf-8"), str(path))
        except OSError:
            die("no dataset config {}".format(path))
        pairs = [(b.get("video_directory") or b.get("image_directory"), b.get("control_directory")) for b in blocks]
    else:
        folder = resolve_path(ref, DATASETS)
        if (folder / "targets").is_dir():
            pairs = [(str(folder / "targets"), str(folder / "control") if (folder / "control").is_dir() else None)]
        else:
            pairs = [(str(folder), None)]
    targets = []
    for media, control in pairs:
        if not media or not Path(media).is_dir():
            die("no folder {} (from {})".format(media, ref))
        videos = media_files(media, VIDEOS)
        files = videos or media_files(media, IMAGES)
        if not files:
            die("no images or clips in {}".format(media))
        edit = bool(control) and not videos
        mode = ("audio" if audio else "video") if videos else ("edit" if edit else "image")
        targets.append(dict(files=files, control=control, index=control_index(control) if edit else {}, mode=mode))
    return targets


def clean_caption(text, avoid):
    text = " ".join(text.replace("\r", " ").split()).strip().strip("\"'")
    for w in avoid:
        text = re.sub(r"\b" + re.escape(w) + r"\b", "", text, flags=re.I)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,;")


def word_report(files, trigger):
    """The 15 words found in the most captions: a word in nearly all of them often describes the style, which the
    LoRA should learn from the pictures, not from the words: it goes in --avoid."""
    caps = [c for c in (caption_of(f) for f in files) if c]
    if not caps:
        return
    count = Counter()
    for cap in caps:
        words = set(re.findall(r"[a-z']{3,}", cap.lower())) - STOP_WORDS - {trigger.lower()}
        count.update(words)
    say("\nWords in the most captions (share of captions): " + ", ".join(
        "{} ({}%)".format(w, 100 * k // len(caps)) for w, k in count.most_common(15)))
    say("A word in most captions that describes the style: add it to --avoid, then run again with --overwrite.")


def caption_files(todo, trigger, avoid, max_words):
    import torch                                          # the trainer's environment holds these
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    say("loading {} (17.6 GB, downloaded the first time)".format(CAPTION_MODEL))
    model = Qwen3VLForConditionalGeneration.from_pretrained(CAPTION_MODEL, dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL)
    for k, (t, f) in enumerate(todo, 1):
        if t["mode"] in ("video", "audio"):
            content = [{"type": "video", "video": str(f), "fps": 2.0, "max_pixels": 360 * 640}]
        elif t["mode"] == "edit":
            source = t["index"].get(f.stem)
            if not source:
                say("[{}/{}] {}: no image of the same name in {}: skipped".format(k, len(todo), f.name, t["control"]))
                continue
            content = [{"type": "image", "image": str(source), "max_pixels": 1024 * 1024},
                       {"type": "image", "image": str(f), "max_pixels": 1024 * 1024}]
        else:
            content = [{"type": "image", "image": str(f), "max_pixels": 1024 * 1024}]
        prompt = CAPTION_PROMPTS[t["mode"]] + " Maximum {} words.".format(max_words)
        messages = [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        caption = "{}, {}".format(trigger, clean_caption(processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0], avoid))
        f.with_suffix(".txt").write_text(caption + "\n", encoding="utf-8", newline="\n")
        say("[{}/{}] {} -> {}".format(k, len(todo), f.name, caption[:110]))


def cmd_caption(argv):
    p = argparse.ArgumentParser(prog="train caption", description="Writes a caption next to every image or clip that has "
                                "none, with {}. Each caption starts with the trigger word.".format(CAPTION_MODEL))
    p.add_argument("datasets", nargs="+", help="folders of /workspace/dataset, or dataset configs (.toml)")
    p.add_argument("--trigger", required=True, help="the trigger word, at the start of every caption")
    p.add_argument("--avoid", default="", help="words to take out, separated by commas: those that describe the style")
    p.add_argument("--overwrite", action="store_true", help="write every caption again")
    p.add_argument("--audio", action="store_true", help="clips: say also what is heard (the ltx_av recipe)")
    p.add_argument("--max-words", type=int, default=60)
    p.add_argument("--dry-run", action="store_true", help="list the files to caption, write nothing")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[^\s,]+", a.trigger):
        die("the trigger is one word, without spaces or commas")
    avoid = [w.strip() for w in a.avoid.split(",") if w.strip()]
    targets = [t for ref in a.datasets for t in caption_targets(ref, a.audio)]
    todo = [(t, f) for t in targets for f in t["files"] if a.overwrite or not caption_of(f)]
    say("{} file(s), {} to caption{}".format(sum(len(t["files"]) for t in targets), len(todo),
                                              "" if a.overwrite else " (the captions already written are kept; "
                                                                     "--overwrite writes them again)"))
    if a.dry_run:
        for _, f in todo:
            say("  " + str(f))
        return 0
    if todo:
        if active_job():
            say("A training run is in progress: captioning shares the GPU with it.")
        caption_files(todo, a.trigger, avoid, a.max_words)
    word_report([f for t in targets for f in t["files"]], a.trigger)
    say("Read every caption before training: the model makes mistakes, and a wrong word is learned like the others.")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "{{" in os.environ.get("HF_TOKEN", ""):     # Runpod's placeholder for a missing secret: Hugging Face would
        del os.environ["HF_TOKEN"]                  # refuse even the public files with it
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if not argv or argv[0] in ("-h", "--help", "help"):
        say(__doc__.split("\n\n")[1])
        say("\nRecipes: " + ", ".join(RECIPES) + ". README.md, \"Training a LoRA\", is the guide.")
        return 0
    if argv[0] in RECIPES:
        return cmd_train(argv[0], argv[1:])
    commands = {"recipes": cmd_recipes, "caption": cmd_caption, "log": cmd_log, "status": cmd_status, "stop": cmd_stop,
                "_run": run_job}
    if argv[0] not in commands:
        die("no command or recipe named {}. Commands: recipes, caption, log, status, stop. Recipes: {}.".format(
            argv[0], ", ".join(RECIPES)))
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())
