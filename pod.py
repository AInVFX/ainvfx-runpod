#!/usr/bin/env python3
"""ainvfx-runpod · pod.py : create, use and terminate a Runpod GPU pod running ComfyUI.

One file, no dependency beyond Python 3.8 or newer. Works on Windows, macOS and Linux.
Nothing but the Runpod API key is stored on this machine, in a file only your account can read.

  python pod.py setup              once: your Runpod API key, your region, your SSH key (optional)
  python pod.py setup --hf-token   optional: store your Hugging Face token as a Runpod Secret
  python pod.py up image           create a pod for the image sessions (RTX 5090; the image models, 105 GB)
  python pod.py up image+3d        the image models plus the 3D block (RTX 5090)
  python pod.py up image+video     image plus LTX 2.5 for the video sessions (RTX PRO 6000)
  python pod.py up video+train     LTX 2.5 plus the bf16 files of the training kit (RTX PRO 6000)
  python pod.py up all             every profile (image, 3d, video, train), about 290 GB of models
  python pod.py up image --gpu "RTX PRO 4500 SE"   another GPU from the list `up` shows
  python pod.py up video --gpu B200 --gpu B300 --retry   wait until a B200 or a B300 is free, then create it
  python pod.py up image --community       also consider Community Cloud (off by default since 0.8.0)
  python pod.py up image --models-url URL  your own models.json for this pod (a raw link, or a Google
                                           Drive, Dropbox or GitHub share link)
  python pod.py up train --name jar-lora   a second training pod, known here as `jar-lora`
  python pod.py status [pod]       status, GPU, data center, cost so far, ComfyUI address, last log lines
  python pod.py logs [pod]         follow the pod's log until READY (or Ctrl+C)
  python pod.py open [pod]         open ComfyUI in your browser
  python pod.py push [pod] FILES...   copy images or videos into the pod's input folder
  python pod.py pull [pod]         download every file of the pod's output folder into outputs/<pod name>/
  python pod.py down [pod]         pull, terminate (billing stops), then what it cost and the credit left; `--all`
  python pod.py ssh [pod]          a terminal on the pod
  python pod.py list               every pod of your account, with its hourly price
  python pod.py doctor             check Python, ssh, the API key, the configuration

[pod] is the profile, the name given to `up --name`, the pod's name or its id; it can be left out
when one pod is recorded. A pod created from the console or another machine is attached by its
first command: `python pod.py status <name or id>`.

The pod runs bootstrap.sh from this repository at start: SSH and JupyterLab first, a health check
(driver, disk speed), then ComfyUI at a pinned tag, then the models of the profile (each line says
file k of n, GB done, percent and the time left), then one test image. settings.env, next to this
script, holds every choice the pod makes (the ComfyUI tag, Python, PyTorch, the models list,
custom nodes, the checks): edit it, then `up`; its values travel with the pod as environment variables.
Your own values go in settings.local.env, next to it (ignored by git, so `git pull` never touches it):
each line there replaces the same line of settings.env. For one pod only, `up --set KEY=VALUE`
(repeatable) and `up --models-url URL` replace a value on the command line.
Its log lines start with [AINVFX]; `up`, `logs` and `status` read them for you, from the API
log stream and from the copy the pod serves through its proxy. Before the first [AINVFX] line,
Runpod's own system log tells what the host is doing (fetching the image, starting the container):
`up` shows it as [RUNPOD] lines, with one summary of the image download (layers done, extracting,
downloading) so a long pull never looks like a hang. Measured on two RTX 5090 pods (4 and 5 Oct 2026):
READY 5 to 15 minutes after creation. A pod with no container 8 minutes after creation gets a warning:
the usual answer is `down`, then `up` again, usually in another data center. A line starting with
FAILED (bootstrap.sh 5.4: PyTorch out of reach, no GPU from PyTorch, no ComfyUI) ends the wait at
once: that pod cannot work, and the answer is the same, `down` then `up`.

Where the pod goes. `up` creates a pod only where Runpod's catalog shows stock, on Secure Cloud, and
names the data center. When the profile's GPU has no stock, `up` shows every GPU in stock that fits
the profile (32 GB of VRAM or more for image, 96 GB for video and train, a Blackwell chip or newer,
not a 1g MIG slice), cheapest first, proposes the cheapest (never above 2.50 USD per hour on its own;
--max-price changes that), and prints the command to take another one. With --retry, `up` waits instead of
stopping: it reads the catalog again every 15 seconds (AINVFX_RETRY_EVERY) and creates the first GPU free,
by the same rules, then stops (one pod). Its question comes once, before the wait. Community Cloud (machines owned
by third parties, cheaper, the host picked by Runpod) is left out unless `--community` is given: on
9 October 2026 the first pod of the session, a Community RTX 5090, never started. Right after creation
`up` reads the pod's host: a host with maintenance under way or starting within 24 hours is terminated
at once and the next GPU is tried.
Money: before its question, `up` shows your Runpod credit and the hours it buys on the card it proposes,
and the question names the price. `down` says what each pod cost (its hours at its price, an estimate),
lists any pod still running with its price per hour, and gives the credit left.

Rule of the course: create at the start of the session, pull your results, terminate at the end.
A terminated pod costs nothing. A stopped pod keeps a dead entry and, with a volume disk, keeps
billing it.
"""
import argparse
import getpass
import http.client
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

VERSION = "0.9.3"
API = os.environ.get("AINVFX_API_BASE", "https://api.runpod.io/v2")   # the test harness points this at a fake
REPO_RAW = "https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main"
# The image of Runpod's own "Runpod Pytorch 2.8.0" template (id runpod-torch-v280). Runpod keeps the images
# of its own templates on its hosts, so a pod on it starts within seconds; any other tag has to be
# fetched from Docker Hub first (about 9 GB, 19 layers, 5 to 15 minutes on 5 Oct 2026 with the
# 1.0.7-cu1300 tag). The image's own PyTorch and CUDA toolkit are not used: bootstrap.sh builds its
# environment with the cu130 PyTorch wheels, which carry their own CUDA 13 libraries.
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


def read_env_file(path):
    """The AINVFX_* variables of one KEY=value file: # comments, empty values dropped."""
    values = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key.startswith("AINVFX_") and value:
                values[key] = value
    except OSError:
        pass
    return values


def load_settings():
    """settings.env (next to this script, versioned), then settings.local.env (yours, ignored by git):
    a key in the local file replaces the same key of settings.env. The result is sent with every pod
    and read by bootstrap.sh on the pod."""
    values = read_env_file(SETTINGS)
    values.update(read_env_file(SETTINGS_LOCAL))
    return values


def comfy_tag():
    return load_settings().get("AINVFX_COMFY_TAG", "v0.41.2")
MIN_CUDA = "13.0"            # host driver 580 or newer: the int8 kernels of the course models need it
COMFY_PORT = 8188
JUPYTER_PORT = 8888
TEMPLATE_ID = "4i789znkrd"   # the public AInVFX template "AInVFX bootcamp ComfyUI"; `setup --template ID` overrides
HF_SECRET = "huggingface_token"                   # the Runpod Secret holding your Hugging Face token
HF_SECRET_REF = "{{ RUNPOD_SECRET_%s }}" % HF_SECRET   # Runpod substitutes the value when the pod boots
WINDOWS = platform.system() == "Windows"
CONFIG_DIR = Path.home() / ".ainvfx-runpod"
CONFIG = CONFIG_DIR / "config.json"
HERE = Path(__file__).resolve().parent
OUTPUTS = Path(os.environ.get("AINVFX_OUTPUTS") or HERE / "outputs")   # where `pull` puts the files
SETTINGS = Path(os.environ.get("AINVFX_SETTINGS") or HERE / "settings.env")   # the pod's choices, one per line
SETTINGS_LOCAL = Path(os.environ.get("AINVFX_SETTINGS_LOCAL") or SETTINGS.with_name("settings.local.env"))  # yours, not versioned
DEFAULT_MODELS_URL = REPO_RAW + "/models.json"
FINAL = ("EXITED", "ERROR", "TERMINATED")
# The API log stream can stay open and silent while the pod writes lines that the stored log holds:
# every STREAM_MAX_AGE seconds the stream is reopened a few seconds back, and every PROXY_POLL
# seconds the copy of the log that the pod serves itself is read. Both overridable for the tests.
STREAM_MAX_AGE = int(os.environ.get("AINVFX_STREAM_MAX_AGE", "60"))
PROXY_POLL = int(os.environ.get("AINVFX_PROXY_POLL", "15"))
# While the host fetches the image, the container has not started and nothing of ours is in the log:
# `up` prints a summary of the download every PULL_SUMMARY seconds, a heartbeat after HEARTBEAT
# seconds of silence, and a warning when no container has started STALL_MINUTES after creation.
PULL_SUMMARY = int(os.environ.get("AINVFX_PULL_SUMMARY", "15"))
HEARTBEAT = int(os.environ.get("AINVFX_HEARTBEAT", "60"))
STALL_MINUTES = float(os.environ.get("AINVFX_STALL_MINUTES", "8"))
STATUS_POLL = int(os.environ.get("AINVFX_STATUS_POLL", "45"))      # seconds between two reads of the pod's status
STREAM_IDLE = int(os.environ.get("AINVFX_STREAM_IDLE", "30"))      # seconds without a byte before the stream is reopened

# Each profile names its preferred GPU and the least VRAM it needs. When the preferred GPU has no stock
# on Secure Cloud, `up` proposes the cheapest GPU in stock with at least that VRAM and a Blackwell chip
# or newer (see "Choosing the GPU" below).
# Since 0.8.0 (10 Oct 2026) the profiles are atomic and combine: `up image+3d`, `up image+video`,
# `up video+train`, `up all`. Since 0.8.1 (13 Oct 2026) the size of the models comes from the list the
# pod downloads (models.json next to this script, or your own list): the container disk is 115% of it
# plus 30 GB for the environment, rounded up to 10, never under 100. The GPU is the one of the profile
# that needs the most VRAM.
PROFILES = {
    "image": dict(vram=32, gpu="NVIDIA GeForce RTX 5090"),
    "3d":    dict(vram=32, gpu="NVIDIA GeForce RTX 5090"),
    "video": dict(vram=96, gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition"),
    "train": dict(vram=96, gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition"),
}
PROFILE_ORDER = ["image", "3d", "video", "train"]


def parse_profiles(text):
    """`image`, `image+3d`, `image,video,train` or `all` -> the profiles in PROFILE_ORDER, no duplicate."""
    parts = [p.strip().lower() for p in re.split(r"[+,\s]+", (text or "image").strip()) if p.strip()]
    if "all" in parts:
        return list(PROFILE_ORDER)
    unknown = [p for p in parts if p not in PROFILES]
    if unknown:
        die("unknown profile(s): {}. Known: {}, joined by + (image+3d), or all.".format(
            ", ".join(unknown), ", ".join(PROFILE_ORDER)))
    return [p for p in PROFILE_ORDER if p in parts]


def direct_url(url):
    """A share link turned into a link to the file itself: Google Drive, Dropbox, a file page on GitHub."""
    m = re.match(r"https?://drive\.google\.com/(?:file/d/|open\?id=)([\w-]+)", url)
    if m:
        return "https://drive.google.com/uc?export=download&id=" + m.group(1)
    url = re.sub(r"^https?://github\.com/([^/]+)/([^/]+)/blob/", r"https://raw.githubusercontent.com/\1/\2/", url)
    return re.sub(r"(dropbox\.com/.*[?&])dl=0\b", r"\1dl=1", url)


def read_models(url=DEFAULT_MODELS_URL):
    """The list the pod will download: the copy next to this script for the course's list, else fetched."""
    try:
        local = HERE / "models.json"
        if url == DEFAULT_MODELS_URL and local.is_file():
            data = json.loads(local.read_text(encoding="utf-8"))
        else:
            req = urllib.request.Request(direct_url(url), headers={"User-Agent": "ainvfx-runpod/" + VERSION})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
        if not (isinstance(data, dict) and isinstance(data.get("profiles"), dict) and isinstance(data.get("files"), dict)):
            raise ValueError("no profiles and files")
        return data
    except ValueError:                        # a web page, or JSON of another shape
        why = "the link gives a web page or another file, not the JSON itself"
    except OSError as e:                      # unreachable, 404, refused
        why = str(e)
    die("the models list at {} could not be read: {}. Give a raw link, or a Google Drive, Dropbox or GitHub "
        "share link to the file.".format(url, why))


def profile_spec(profiles, models=None):
    """disk, vram, gpu, the size of the models (GB) and the label (`image+3d`) for one or several profiles."""
    models = models or read_models()
    missing = [p for p in profiles if p not in models["profiles"]]
    if missing:
        die("the models list has no profile {} (it has: {}).".format(", ".join(missing), ", ".join(models["profiles"])))
    keys = []
    def walk(name, seen=()):                  # `@image` inside a profile includes the image profile
        for item in models["profiles"].get(name, []):
            if item.startswith("@"):
                if item[1:] not in seen:
                    walk(item[1:], seen + (name,))
            elif item not in keys:
                keys.append(item)
    for p in profiles:
        walk(p)
    unknown = [k for k in keys if k not in models["files"]]
    if unknown:
        die("the models list names files that are not in its files list: {}.".format(", ".join(unknown)))
    gb = sum(models["files"][k].get("gb") or 0 for k in keys)
    vram = max(PROFILES[p]["vram"] for p in profiles)
    gpu = next(PROFILES[p]["gpu"] for p in profiles if PROFILES[p]["vram"] == vram)
    disk = max(100, int(-(-(gb * 1.15 + 30) // 10) * 10))
    return dict(disk=disk, vram=vram, gpu=gpu, gb=round(gb), label="+".join(profiles))
# Choosing the GPU. Two rules, learned on 5 and 7 October 2026, when a pod created with no data center
# named landed on a host scheduled for removal (console notice "This server will be removed from the
# platform"; the container never started), although the catalog showed no stock for that GPU:
# 1. `up` never creates a pod where the catalog shows no stock, and never leaves the Secure Cloud data
#    center to Runpod. Community stock carries no data center in the catalog, so a Community pod is the
#    one case where Runpod picks the host, and only when the catalog shows Community stock.
# 2. After each creation, `up` reads the pod's host through the GraphQL API (the REST API v2 does not
#    report it). A host with maintenance under way, or starting within MAINT_HOURS, is terminated at
#    once (a few seconds billed) and the next candidate is tried.
NEWER_GPU = re.compile(r"blackwell|rubin|\brtx [5-9]0[5-9]0\b|\bg?b[1-9]00\b|\bv?r[1-9]00\b", re.I)
# A 1g MIG slice (Multi-Instance GPU: a fixed share of a bigger GPU, with its own memory) is the smallest
# share: one seventh of a B300 (the catalog offers 56 on an 8-GPU machine). On 7 Oct 2026 the B300 MIG
# 1g.34gb ran the image self-test in 111 s, against 16.1 s on an RTX 5090. `up` leaves 1g slices out of
# its list and never picks one on its own; `--gpu "B300 MIG 34GB"` still takes one.
SMALL_SLICE = re.compile(r"\bMIG 1g\.", re.I)
MAX_PRICE = float(os.environ.get("AINVFX_MAX_PRICE", "2.5"))     # USD per hour: the most `up` picks on its own
MAINT_HOURS = float(os.environ.get("AINVFX_MAINT_HOURS", "24"))  # maintenance starting sooner than this is refused
HOST_WAIT = int(os.environ.get("AINVFX_HOST_WAIT", "40"))        # seconds to wait for the API to name the host
# `up --retry` (0.9.0, 19 Oct 2026: each B200 and B300 freed on Secure Cloud was taken within seconds, before
# a person could run `up` again). Instead of stopping when nothing is free, `up` reads the catalog again every
# RETRY_EVERY seconds and creates the first GPU free, by the same rules as without --retry, then stops: one
# pod, never more. One read is one request per cloud: 240 an hour at 15 s, against the 3,600 an hour that
# Runpod allows from 27 Oct 2026; never under 1 s. A "still waiting" line comes every RETRY_NOTE seconds.
RETRY_EVERY = max(1, int(os.environ.get("AINVFX_RETRY_EVERY", "15")))
RETRY_NOTE = int(os.environ.get("AINVFX_RETRY_NOTE", "60"))
GRAPHQL = os.environ.get("AINVFX_GRAPHQL", "https://api.runpod.io/graphql")   # the test harness points this at a fake
# The same command as the public template. If GitHub cannot be reached, the pod falls back to
# Runpod's own /start.sh (SSH and JupyterLab), so the error can be read instead of a restart loop.
START_CMD = ('bash -c "curl -fsSL --retry 5 --retry-delay 3 {raw}/bootstrap.sh -o /tmp/ainvfx-bootstrap.sh '
             '&& bash /tmp/ainvfx-bootstrap.sh || /start.sh"').format(raw=REPO_RAW)


# ----------------------------------------------------------------------------- small helpers
def die(msg, code=1):
    print("\nERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def say(msg=""):
    print(msg, flush=True)


def load_config():
    if CONFIG.exists():
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            die("{} is unreadable: delete it and run `setup` again".format(CONFIG))
    return {"api_key": "", "region": "", "template_id": "", "ssh_key": "", "hf_secret": False, "pods": {}}


def save_config(cfg):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    restrict(CONFIG)


def remember_pod(cfg, tag, rec):
    """Write one pod record, or remove it (rec None). The file is read again just before the write, so
    two commands running at once (two terminals, two `up`) keep each other's records. Before 0.6.1 each
    command wrote back the whole file it had read at its start, and the last one to finish erased the
    other's pod (7 Oct 2026: `down test10` no longer knew test10)."""
    fresh = load_config()
    pods = fresh.setdefault("pods", {})
    if rec is None:
        pods.pop(tag, None)
    else:
        pods[tag] = rec
    save_config(fresh)
    cfg["pods"] = pods


def restrict(path):
    """Only this account may read the file (the API key is in it)."""
    try:
        if WINDOWS:
            who = subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip() or getpass.getuser()
            subprocess.run(["icacls", str(path), "/inheritance:r"], capture_output=True)
            subprocess.run(["icacls", str(path), "/grant:r", "{}:(R,W)".format(who)], capture_output=True)
        else:
            os.chmod(path, 0o600)
    except Exception:
        pass


def api_key(cfg):
    key = os.environ.get("RUNPOD_API_KEY") or cfg.get("api_key")
    if not key:
        die("no Runpod API key. First:  python pod.py setup")
    return key


class ApiError(Exception):
    def __init__(self, code, text, url):
        self.code, self.text, self.url = code, text, url
        self.detail, self.errors, self.retry_after = text, None, None
        try:                       # Runpod errors are application/problem+json: title, status, detail, errors
            prob = json.loads(text)
            self.detail = prob.get("detail") or prob.get("title") or text
            self.errors = prob.get("errors")
        except (ValueError, AttributeError):
            pass
        super().__init__("HTTP {} on {}: {}".format(code, url, self.detail))

    def hint(self):
        if self.code == 401:
            return "The API key is missing, wrong or expired. Runpod console > Account > Credentials > API Keys. Then:  python pod.py setup"
        if self.code == 403:
            return "The API key does not allow this call. Console > Account > Credentials > API Keys: edit the key and set api.runpod.io/graphql to Read / Write, or create a new key with that permission."
        if self.code == 402:
            return "Insufficient balance on the Runpod account: add credit, then try again."
        if self.code == 429:
            return "Too many requests: the Runpod API allows 180 a minute and 3,600 an hour per account; wait a minute, then try again."
        if self.code == 0 and self.timed_out:
            return "The server accepted the request but did not answer in time: a host that is not responding, or a slow link. Try again; `python pod.py status` shows the pod."
        if self.code == 0:
            return "No network, or api.runpod.io unreachable from this machine."
        return ""

    @property
    def timed_out(self):
        return "timed out" in (self.detail or "").lower()


RATE_RETRIES = 5    # 0.8.2: on a 429 from api.runpod.io, wait Retry-After or 2, 4, 8, 16, 32 s, then give up


def request(method, url, key=None, body=None, timeout=60, raw=False, headers=None, stream=False, data=None):
    """One HTTP request with urllib. Returns parsed JSON, bytes (raw=True), or the open response (stream=True).
    `body` is sent as JSON; `data` as given (bytes), with the Content-Type of `headers`.
    Every network failure, including a timeout while the server prepares its answer, is an ApiError
    with code 0: nothing here raises a bare socket error (a pod on a host that does not respond
    once made `up` stop with a traceback).
    Since 0.8.2: a 429 from the Runpod API (api.runpod.io: 180 requests a minute, 3,600 an hour,
    43,200 a day from 27 Oct 2026) is retried RATE_RETRIES times, after the Retry-After the server gives
    or an exponential wait; a pod's proxy and Hugging Face are not retried here."""
    runpod_host = url.startswith(API) or url.startswith(GRAPHQL)
    for attempt in range(RATE_RETRIES + 1):
        try:
            return _request_once(method, url, key, body, timeout, raw, headers, stream, data)
        except ApiError as e:
            if e.code != 429 or not runpod_host or attempt == RATE_RETRIES:
                raise
            wait = int(e.retry_after) if (e.retry_after or "").isdigit() else 2 ** (attempt + 1)
            print("   Runpod rate limit (429): waiting {} s, try {} of {}".format(wait, attempt + 2, RATE_RETRIES + 1), flush=True)
            time.sleep(wait)


def _request_once(method, url, key=None, body=None, timeout=60, raw=False, headers=None, stream=False, data=None):
    hdrs = {"Accept": "application/json", "User-Agent": "ainvfx-runpod/" + VERSION}
    if key:
        hdrs["Authorization"] = "Bearer " + key
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:1000]
        err = ApiError(e.code, text, url)
        err.retry_after = e.headers.get("Retry-After")
        raise err
    except urllib.error.URLError as e:
        reason = e.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ApiError(0, "timed out after {} s (connecting)".format(timeout), url)
        raise ApiError(0, str(reason), url)
    except (socket.timeout, TimeoutError):
        raise ApiError(0, "timed out after {} s (no answer)".format(timeout), url)
    except (OSError, http.client.HTTPException) as e:
        raise ApiError(0, "{}: {}".format(type(e).__name__, e), url)
    if stream:
        return r
    try:
        with r:
            payload = r.read()
    except (socket.timeout, TimeoutError):
        raise ApiError(0, "timed out after {} s (reading the answer)".format(timeout), url)
    except (OSError, http.client.HTTPException) as e:
        raise ApiError(0, "{}: {}".format(type(e).__name__, e), url)
    if raw:
        return payload
    if not payload:
        return {}
    try:
        return json.loads(payload.decode("utf-8"))
    except ValueError:
        return {"_text": payload.decode("utf-8", "replace")}


def rp(method, path, cfg, body=None, timeout=60, raw=False, headers=None, stream=False):
    return request(method, API + path, api_key(cfg), body, timeout, raw, headers, stream)


def pod_url(pod_id, port=COMFY_PORT):
    fmt = os.environ.get("AINVFX_PROXY_FMT", "https://{id}-{port}.proxy.runpod.net")   # the test harness overrides it
    return fmt.format(id=pod_id, port=port)


# The bootstrap writes its log inside ComfyUI's input folder, so the log is also readable through the
# proxy once ComfyUI answers: a second way to see READY when the API log stream misses it.
LOG_VIEW = "/view?filename=bootstrap.log&subfolder=ainvfx&type=input"


def proxy_log(pod_id):
    """The [AINVFX] lines of the bootstrap log, read through the pod's own ComfyUI; [] when not readable."""
    try:
        data = request("GET", pod_url(pod_id) + LOG_VIEW, timeout=10, raw=True)
    except ApiError:
        return []
    return [l for l in data.decode("utf-8", "replace").splitlines() if "[AINVFX]" in l]


REGION_NAMES = {"EU": "Europe", "NA": "North America"}
# Runpod data center ids start with the country: CA-MTL-1, US-TX-3, EU-FR-1, EUR-IS-2, EU-CZ-1.
CANADA_ZONES = ("Toronto", "Montreal", "Vancouver", "Edmonton", "Winnipeg", "Halifax", "Regina", "St_Johns",
                "Moncton", "Whitehorse", "Yellowknife", "Iqaluit", "Calgary")
EUROPE_CITY_COUNTRY = {"Paris": "FR", "Prague": "CZ", "Bucharest": "RO", "Amsterdam": "NL", "Stockholm": "SE",
                       "Reykjavik": "IS", "Oslo": "NO"}


def region_of(dc_id):
    p = (dc_id or "").upper()
    if p.startswith(("EU", "EUR")):
        return "EU"
    if p.startswith(("US", "CA")):
        return "NA"
    return "OTHER"


def country_of(dc_id):
    """'CA' for CA-MTL-1, 'FR' for EU-FR-1, 'IS' for EUR-IS-2, 'US' for US-TX-3."""
    parts = (dc_id or "").upper().split("-")
    if len(parts) >= 3 and parts[0] in ("EU", "EUR"):
        return parts[1]
    return parts[0] if parts else ""


def local_zone():
    """This computer's time zone name (America/Toronto, Europe/Paris, or a Windows name), or ''."""
    tz = os.environ.get("TZ", "")
    if tz:
        return tz
    try:
        if WINDOWS:
            return subprocess.run(["tzutil", "/g"], capture_output=True, text=True, timeout=5).stdout.strip()
        for cand in ("/etc/localtime", "/var/db/timezone/localtime"):
            if os.path.islink(cand):
                target = os.readlink(cand)
                if "zoneinfo/" in target:
                    return target.split("zoneinfo/", 1)[1]
        if os.path.exists("/etc/timezone"):
            return open("/etc/timezone", encoding="utf-8").read().strip()
    except Exception:
        pass
    return ""


def local_place():
    """(region, country) guessed from this computer's clock: ('NA', 'CA') for America/Toronto,
    ('EU', 'FR') for Europe/Paris, ('EU', '') for Europe/Berlin, ('NA', 'US') for America/Chicago."""
    zone = local_zone()
    city = zone.split("/")[-1] if "/" in zone else ""
    if zone.startswith("Canada/") or (zone.startswith("America/") and city in CANADA_ZONES):
        return "NA", "CA"
    if zone.startswith(("America/", "US/")) or zone.endswith("Standard Time") and any(
            k in zone for k in ("Eastern", "Central", "Mountain", "Pacific", "Atlantic", "Alaskan", "Hawaiian")):
        return "NA", "US"
    if zone.startswith("Europe/"):
        return "EU", EUROPE_CITY_COUNTRY.get(city, "")
    offset_h = (time.localtime().tm_gmtoff or 0) / 3600          # no zone name: the UTC offset decides
    return ("NA", "") if offset_h <= -2 else ("EU", "")


def local_region():
    return local_place()[0]


def dc_order(dc_id, region, country):
    """Sort key: the person's country first, then the rest of their region, then the other main region."""
    same_region = region_of(dc_id) == region
    return (0 if country and same_region and country_of(dc_id) == country else
            1 if same_region else 2 if region_of(dc_id) in REGION_NAMES else 3)


def forget_gone(cfg, account):
    """Remove the records of pods that are no longer in the account (terminated from the Runpod console,
    from another machine, or by Runpod itself): only `down` removed a record before 0.9.3, so such pods
    stayed recorded here and every command without a name asked which one (Adrien, 22 Oct 2026: seven
    records, one pod). A stopped pod (EXITED) is still in the account and keeps its record. Returns the
    names forgotten."""
    alive = {str(p.get("id")) for p in account}
    gone = [(t, r) for t, r in list((cfg.get("pods") or {}).items()) if str(r.get("id")) not in alive]
    for t, _ in gone:
        remember_pod(cfg, t, None)
    return [r.get("name") or t for t, r in gone]


def which_pod(cfg, ref):
    """The pod a command targets: (tag, record). `ref` is a tag (a profile name, or the `--name`
    given to `up`), a pod name, a pod id, or nothing when one pod is recorded. A pod that exists in
    the account but not here (created from the console, or from another machine) is attached: its
    record is written to the configuration under its name, so the next commands can use it."""
    pods = cfg.setdefault("pods", {})
    if ref:
        ref = ref.strip().replace("+", "-")   # `down image+3d` and `down image-3d` name the same pod
    if ref is None:
        if len(pods) > 1:                   # 0.9.3: first forget the pods that no longer exist
            try:
                gone = forget_gone(cfg, list_pods(cfg, quiet=True))
            except ApiError:
                gone = []
            if gone:
                say("forgot {} pod(s) no longer on Runpod: {}".format(len(gone), ", ".join(gone)))
            pods = cfg.setdefault("pods", {})
        if len(pods) == 1:
            tag = next(iter(pods))
            return tag, pods[tag]
        if not pods:
            die("no pod recorded here. `python pod.py up image` creates one; "
                "`python pod.py list` shows the account's pods, then `python pod.py status <name or id>` attaches to one.")
        die("several pods recorded, name one: {}".format(
            "  ".join("{} ({})".format(t, r.get("name")) for t, r in pods.items())))
    if ref in pods:
        return ref, pods[ref]
    for tag, rec in pods.items():
        if ref in (rec.get("id"), rec.get("name")):
            return tag, rec
    account = list_pods(cfg, quiet=True)
    short = {"ainvfx-{}-{}".format(k, ref) for k in PROFILES}   # `up image --name test10` names its pod ainvfx-image-test10

    def attach(p, tag):
        pid, pname = str(p.get("id") or ""), str(p.get("name") or "")
        rec = {"id": pid, "name": pname or pid, "profile": next((k for k in PROFILES if "-{}-".format(k) in pname), ""),
               "created": p.get("createdAt") or ""}
        remember_pod(cfg, tag, rec)
        say("attached to {} ({}) as `{}`".format(rec["name"], pid, tag))
        return tag, rec
    for p in account:
        pid, pname = str(p.get("id") or ""), str(p.get("name") or "")
        if pname in short:
            return attach(p, ref)
        if ref in (pid, pname) or (pname and pname.startswith(ref)):
            return attach(p, pname or pid)
    if ref in PROFILES:                     # `up image` names its pod ainvfx-image-<month><day>-<time>
        hits = [p for p in account if str(p.get("name") or "").startswith("ainvfx-{}-".format(ref))
                and pod_status(p) not in FINAL]
        if len(hits) == 1:
            return attach(hits[0], ref)
        if hits:
            die("several {} pods in your account: {}. Name one: `python pod.py {} <name or id>`.".format(
                ref, "  ".join("{} ({})".format(p.get("name"), p.get("id")) for p in hits), "status"))
    die("no pod '{}' recorded here or in your account. `python pod.py list` shows them.".format(ref))


def fmt_money(x):
    return "{:.2f} USD".format(x)


def parse_time(s):
    """An RFC 3339 timestamp from the API ('2026-10-07T14:50:00Z') as an aware datetime."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


# ----------------------------------------------------------------------------- Runpod calls
# Field names below are those of the Runpod REST API v2 (https://api.runpod.io/v2/openapi.json):
# status, cost (USD per hour), gpu.id, dataCenterId, cudaVersion, ssh.direct and ssh.proxy,
# runtime.ports, createdAt. The logs endpoint is a Server-Sent Events stream.
def get_pod(cfg, pod_id):
    return rp("GET", "/pods/" + pod_id, cfg)


def pod_status(pod):
    return str(pod.get("status") or "?").upper()


def pod_price(pod):
    try:
        return float(pod.get("cost"))
    except (TypeError, ValueError):
        return None


def pod_spent(pod, rec=None):
    """(hours, USD, price per hour) since the pod was created, at its hourly price: an estimate, since Runpod
    bills from creation. None when the price or the creation time is unknown."""
    price = pod_price(pod)
    created = parse_time(pod.get("createdAt") or "") or parse_time((rec or {}).get("created", ""))
    if not price or not created:
        return None
    hours = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 3600)
    return hours, hours * price, price


def fmt_duration(hours):
    minutes = int(round(hours * 60))
    if minutes < 1:
        return "under a minute"
    h, m = divmod(minutes, 60)
    return "{} h {:02d} min".format(h, m) if h else "{} min".format(m)


def pod_summary(pod):
    gpu = (pod.get("gpu") or {}).get("id") or "?"
    dc = pod.get("dataCenterId") or "?"
    return str(gpu), str(dc), pod_price(pod)


def pod_ssh(pod):
    """(kind, host, port, user): 'direct' (full ssh, scp, rsync) if 22/tcp got a public port, else 'proxy' (shell only)."""
    ssh = pod.get("ssh") or {}
    for kind in ("direct", "proxy"):
        c = ssh.get(kind)
        if c and c.get("host") and c.get("port") and c.get("username"):
            return kind, c["host"], int(c["port"]), c["username"]
    return None, None, None, None


def read_catalog(cfg, cloud):
    """Every GPU type with its pod stock on `cloud` (SECURE or COMMUNITY) for hosts on CUDA MIN_CUDA or
    newer, in one call: id, name, memory (VRAM in GB), price per cloud, availability, and for Secure the
    data centers with stock (GET /catalog/gpus, REST API v2)."""
    q = urllib.parse.urlencode({"include": "AVAILABILITY", "product": "POD", "minCudaVersion": MIN_CUDA, "cloud": cloud})
    data = rp("GET", "/catalog/gpus?" + q, cfg)
    return [g for g in (data.get("gpus") or []) if isinstance(g, dict) and g.get("id")]


def in_stock(level):
    return str(level or "").upper() not in ("", "NONE")


def newer_gpu(gpu):
    """True for a Blackwell GPU (RTX 50 series, RTX PRO Blackwell, B200, B300 and their MIG slices) or a newer one."""
    return bool(NEWER_GPU.search("{} {}".format(gpu.get("id", ""), gpu.get("name", ""))))


def make_offers(catalogs, region, country):
    """One offer per GPU type and cloud with stock, cheapest first (Secure first at the same price):
    dict(id, name, vram, cloud, price, dcs, newer, slice). A Secure offer lists its data centers with stock,
    the person's country first, and a create names one of them: Secure stock with no data center named
    is skipped. A Community offer has no data center (the catalog gives none for Community)."""
    offers = []
    for cloud, gpus in catalogs.items():
        for g in gpus:
            price = (g.get("price") or {}).get(cloud.lower())
            if not in_stock(g.get("availability")) or not price:
                continue
            dcs = [d.get("id") for d in (g.get("dataCenters") or []) if d.get("id") and in_stock(d.get("availability"))]
            if cloud == "SECURE" and not dcs:
                continue
            dcs.sort(key=lambda d: dc_order(d, region, country))
            offers.append({"id": g["id"], "name": str(g.get("name") or g["id"]), "vram": int(g.get("memory") or 0),
                           "cloud": cloud, "price": float(price), "dcs": dcs, "newer": newer_gpu(g),
                           "slice": bool(SMALL_SLICE.search(g["id"]))})
    offers.sort(key=lambda o: (o["price"], o["cloud"] != "SECURE", -o["vram"], o["name"]))
    return offers


def fits_profile(offer, vram):
    """True when `up` may pick the offer on its own for a profile needing `vram` GB: enough VRAM,
    a Blackwell chip or newer, and not a 1g MIG slice."""
    return offer["vram"] >= vram and offer["newer"] and not offer["slice"]


def find_gpu(catalog, ref):
    """The catalog entry that `ref` names: its full id or its short name ("RTX PRO 4500 SE"), in any case,
    or a part of either that matches one GPU type only."""
    r = ref.strip().lower()
    for key in ("id", "name"):
        hits = [g for g in catalog if str(g.get(key, "")).lower() == r]
        if hits:
            return hits[0]
    hits, seen = [], set()                     # the same GPU type comes once per cloud: keep the first
    for g in catalog:
        if (r in str(g.get("id", "")).lower() or r in str(g.get("name", "")).lower()) and g["id"] not in seen:
            seen.add(g["id"])
            hits.append(g)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        die("no GPU type named \"{}\" in the Runpod catalog. `python pod.py up` lists the GPUs in stock. "
            "For several GPUs, repeat --gpu: --gpu B200 --gpu B300.".format(ref))
    die("\"{}\" matches several GPU types: {}. Give one of these names to --gpu.".format(
        ref, ", ".join('"{}"'.format(g.get("name") or g["id"]) for g in hits[:12])))


MACHINE_QUERY = ("query {{ pod(input: {{podId: {} }}) {{ id machine {{ podHostId dataCenterId location "
                 "maintenanceStart maintenanceEnd maintenanceNote }} }} }}")


def pod_machine(cfg, pod_id):
    """The pod's host as the GraphQL API reports it (the REST API v2 has no maintenance field): a dict,
    empty while no host is assigned; None when GraphQL cannot be read (the check is then skipped)."""
    try:
        data = request("POST", GRAPHQL, api_key(cfg), body={"query": MACHINE_QUERY.format(json.dumps(pod_id))}, timeout=20)
    except ApiError:
        return None
    if not isinstance(data, dict) or data.get("errors") or "data" not in data:
        return None
    machine = ((data.get("data") or {}).get("pod") or {}).get("machine") or {}
    return machine if any(machine.values()) else {}


MONEY_QUERY = "query { myself { clientBalance currentSpendPerHr } }"


def account_money(cfg):
    """(credit left in USD, what the account spends now in USD per hour), read through the GraphQL API (the
    REST API v2 has no balance); (None, None) when it cannot be read, and the money lines are left out."""
    try:
        data = request("POST", GRAPHQL, api_key(cfg), body={"query": MONEY_QUERY}, timeout=20)
    except ApiError:
        return None, None
    if not isinstance(data, dict) or data.get("errors"):
        return None, None
    me = (data.get("data") or {}).get("myself")
    if not isinstance(me, dict):
        return None, None

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return num(me.get("clientBalance")), num(me.get("currentSpendPerHr"))


def to_time(value):
    """An API time (RFC 3339 text, or epoch seconds or milliseconds) as an aware datetime, or None."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) or re.fullmatch(r"\d{9,13}(\.\d+)?", str(value)):
        x = float(value)
        return datetime.fromtimestamp(x / 1000 if x > 1e11 else x, tz=timezone.utc)
    return parse_time(str(value))


def maintenance_of(machine, now=None):
    """A sentence when the host has maintenance under way or starting within MAINT_HOURS, else None.
    A window that is over, or that starts later, does not matter for a pod that lives one session.
    A time that cannot be read counts as under way: refusing a host costs less than a dead pod."""
    machine = machine or {}
    start_raw, end_raw = machine.get("maintenanceStart"), machine.get("maintenanceEnd")
    if not start_raw and not end_raw:
        return None
    now = now or datetime.now(timezone.utc)
    start, end = to_time(start_raw), to_time(end_raw)
    if end and end <= now:
        return None
    if start and start > now + timedelta(hours=MAINT_HOURS):
        return None

    def when(t, raw):
        return t.strftime("%d %b %Y %H:%M UTC") if t else str(raw)
    if start_raw:
        text = "maintenance {} {}".format("from" if start and start > now else "since", when(start, start_raw))
    else:
        text = "maintenance under way"
    if end_raw:
        text += ", until {}".format(when(end, end_raw))
    note = " ".join(str(machine.get("maintenanceNote") or "").split())
    if note:
        text += ' ("{}")'.format(note if len(note) <= 140 else note[:140] + "...")
    return text


def host_check(cfg, pod_id):
    """(machine, problem) for a pod just created. machine is None when GraphQL cannot be read, {} when
    no host is reported within HOST_WAIT seconds; problem is the maintenance sentence, or None."""
    deadline = time.time() + HOST_WAIT
    while True:
        machine = pod_machine(cfg, pod_id)
        if machine is None or machine:
            return machine, (maintenance_of(machine) if machine else None)
        if time.time() >= deadline:
            return {}, None
        time.sleep(3)


def offer_where(o, limit=4):
    if not o["dcs"]:
        return "a host Runpod picks"
    more = len(o["dcs"]) - limit
    return ", ".join(o["dcs"][:limit]) + (" and {} more".format(more) if more > 0 else "")


def offer_line(o):
    return "{} ({} GB of VRAM, {} per hour, {} Cloud, {})".format(
        o["name"], o["vram"], fmt_money(o["price"]), o["cloud"].title(), offer_where(o))


def show_offers(offers, pick, max_price):
    """The table of GPUs in stock that fit the profile, cheapest first; `>` marks the one `up` will create."""
    say("     {:<18} {:>6}  {:>9}  {:<9}  {}".format("GPU", "VRAM", "per hour", "cloud", "in stock in"))
    for o in offers:
        note = "   (above {}: only with --gpu)".format(fmt_money(max_price)) if o["price"] > max_price else ""
        say("   {} {:<18} {:>3} GB  {:>9}  {:<9}  {}{}".format(
            ">" if o is pick else " ", o["name"][:18], o["vram"], fmt_money(o["price"]), o["cloud"].title(),
            offer_where(o), note))


def list_pods(cfg, quiet=False):
    pods, cursor = [], None
    while True:
        q = "?" + urllib.parse.urlencode({"cursor": cursor}) if cursor else ""
        data = rp("GET", "/pods" + q, cfg)
        pods += data.get("pods") or []
        cursor = (data.get("pagination") or {}).get("nextCursor")
        if not cursor:
            break
    if not quiet:
        if not pods:
            say("No pod in your account.")
        tags = {r.get("id"): t for t, r in (cfg.get("pods") or {}).items()}
        for p in pods:
            gpu, dc, price = pod_summary(p)
            say("{:<28} {:<12} {:<40} {:<10} {}".format(
                str(p.get("name") or "?")[:28], pod_status(p)[:12], gpu[:40], dc[:10],
                (fmt_money(price) + "/h") if price else ""))
            tag = tags.get(p.get("id"))
            say("    id {}   {}   ComfyUI {}".format(
                p.get("id"), "here: `{}`".format(tag) if tag else "not recorded here", pod_url(str(p.get("id")))))
        if pods:
            say("Commands take the name in backquotes, the pod name or the id: `python pod.py status <that>`. "
                "A pod not recorded here is attached by its first command.")
    return pods


def secrets_named(cfg):
    data = rp("GET", "/account/secrets?" + urllib.parse.urlencode({"name": HF_SECRET}), cfg)
    items = data if isinstance(data, list) else (data.get("secrets") or data.get("items") or [])
    return [s for s in items if isinstance(s, dict) and str(s.get("name", "")).lower() == HF_SECRET]


def secret_exists(cfg):
    """True when the account holds the Hugging Face secret; None when the key cannot list secrets."""
    try:
        return bool(secrets_named(cfg))
    except ApiError:
        return None


def log_stream(cfg, pod_id, tail=200, since=None, source=None, idle=20, max_age=None):
    """Yield (event_id, source, line) from GET /pods/{id}/logs, a Server-Sent Events stream
    (one `id:` line and one `data:` JSON line per event, with ts, source and line; the id is the ts).
    `tail` lines are sent first, then live lines; with `since` (an RFC 3339 time) the stream resumes
    from that time instead. Stops after `idle` seconds without any data, or after `max_age` seconds
    in all cases. Yields (None, None, None) on keepalive lines, so the caller can do periodic work."""
    q = {}
    opened = time.time()
    hdrs = {"Accept": "text/event-stream"}
    if since:
        q["since"] = since
    else:
        q["tail"] = tail
    if source:
        q["source"] = source
    url = "/pods/{}/logs".format(pod_id) + ("?" + urllib.parse.urlencode(q) if q else "")
    r = rp("GET", url, cfg, timeout=idle, headers=hdrs, stream=True)
    eid = None
    try:
        while True:
            if max_age and time.time() - opened > max_age:
                return
            try:
                raw = r.readline()
            except (socket.timeout, TimeoutError, OSError):
                return
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("id:"):
                eid = line[3:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
                try:
                    ev = json.loads(payload)
                except ValueError:
                    ev = {"line": payload}
                if isinstance(ev, dict):
                    yield eid, str(ev.get("source") or ""), str(ev.get("line") or "")
            else:
                yield None, None, None          # a blank or comment line: a keepalive
    finally:
        try:
            r.close()
        except Exception:
            pass


def pod_logs(cfg, pod_id, tail=400, seconds=10, source="container"):
    """The last `tail` log lines of one source (container, or system), as a list of strings (the
    stream is left after `seconds`)."""
    lines, t0 = [], time.time()
    try:
        for _, _, line in log_stream(cfg, pod_id, tail=tail, source=source, idle=3, max_age=seconds):
            if line:
                lines.append(line)
    except ApiError as e:
        return None, e
    return lines, None


def ainvfx_lines(lines):
    return [l for l in lines if "[AINVFX]" in l]


def comfy_alive(pod_id):
    try:
        request("GET", pod_url(pod_id) + "/system_stats", timeout=8)
        return True
    except ApiError:
        return False


def resume_point(event_id, seconds=5):
    """An RFC 3339 time a few seconds before the given event id (itself a time), so that a reconnect
    overlaps the lines already seen instead of skipping those written in the same second."""
    t = parse_time(event_id or "")
    if not t:
        return None
    return (t - timedelta(seconds=seconds)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Runpod's system log narrates the image download one Docker layer at a time ("38e4c3f3c358 Extracting",
# "7f1a... Pull complete"), many lines per layer. They are counted into one summary line instead.
LAYER_DONE = ("Pull complete", "Already exists")
LAYER_RE = re.compile(r"^([0-9a-f]{12})\s+(Pulling fs layer|Waiting|Downloading|Verifying Checksum|"
                      r"Download complete|Extracting|Pull complete|Already exists)\b")


class ImagePull:
    """The state of the image download, from the layer lines of the system log."""
    def __init__(self):
        self.layers, self.started = {}, None

    def feed(self, line):
        """True when the line is a layer line (and was counted)."""
        m = LAYER_RE.match(line.strip())
        if not m:
            return False
        self.layers[m.group(1)] = m.group(2)
        self.started = self.started or time.time()
        return True

    def counts(self):
        done = sum(1 for s in self.layers.values() if s in LAYER_DONE)
        extracting = sum(1 for s in self.layers.values() if s == "Extracting")
        return len(self.layers), done, extracting, len(self.layers) - done - extracting

    def summary(self):
        n, done, extracting, downloading = self.counts()
        elapsed = int(time.time() - (self.started or time.time()))
        return "image download: {} layers · {} done · {} extracting · {} downloading ({} min {:02d} s)".format(
            n, done, extracting, downloading, elapsed // 60, elapsed % 60)


def fmt_since(t):
    """'3 min' from an aware datetime to now."""
    if not t:
        return "?"
    secs = int((datetime.now(timezone.utc) - t).total_seconds())
    return "{} min".format(secs // 60) if secs >= 60 else "{} s".format(secs)


def follow_logs(cfg, pod_id, minutes=25, pod=None, created=None):
    """Print the pod's log as it appears, until READY, a dead pod, or the time runs out.

    Two kinds of lines. [AINVFX] lines come from bootstrap.sh, through two sources, because the API
    stream has been seen staying open and silent while the pod wrote lines that only the stored log
    held: the stream is reopened every STREAM_MAX_AGE seconds a few seconds back (duplicates are
    dropped), and every PROXY_POLL seconds the copy of the log that the pod serves through its own
    ComfyUI is read. [RUNPOD] lines are Runpod's system log, what the host does before our script
    runs (fetching the image, starting the container): the image download is one summary line every
    PULL_SUMMARY seconds, the rest is printed once. A heartbeat shows after HEARTBEAT seconds without
    anything to print, and a warning when no container has started STALL_MINUTES after creation.
    Every API error, a timeout included, is a message, never a traceback: the pod exists and bills."""
    deadline = time.time() + minutes * 60
    seen, printed, last_id = set(), set(), None
    last_poll = time.time()
    created = created or parse_time((pod or {}).get("createdAt") or "")   # when this machine saw the pod created
    pull, pull_shown, pull_done = ImagePull(), 0.0, False
    state = {"container": False, "stall_told": False, "stream_told": False, "status": pod_status(pod) if pod else "?"}

    def out(text):
        say("  " + text)
        state["last_output"] = time.time()

    state["last_output"] = time.time()

    def show(line, key):
        """An [AINVFX] line, once. True on READY, and on FAILED (state["failed"] then says which)."""
        if key in seen or line in printed:
            return False
        seen.add(key); printed.add(line)
        state["container"] = True
        out(line[line.index("[AINVFX]"):])
        if "[AINVFX] FAILED" in line:
            state["failed"] = True
            return True
        return "READY" in line

    def done():
        """The end of the wait: True on READY; False on FAILED, with what to do."""
        if state.get("failed"):
            say("\nThe pod reported FAILED (the line above): it cannot work on that machine, and it bills until "
                "it is terminated. Terminate it and create another, which lands on another machine:\n"
                "    python pod.py down    then    python pod.py up")
            return False
        return True

    def system(line):
        """A line of Runpod's system log."""
        text = line.strip()
        if not text:
            return
        if pull.feed(text):
            pull_summary(False)
            return
        if text in printed:
            return
        printed.add(text)
        out("[RUNPOD] " + text[:200])

    def pull_summary(final):
        nonlocal pull_shown, pull_done
        if not pull.layers or pull_done:
            return
        if not pull_shown and not final and time.time() - pull.started < 3:
            return                          # let the backlog of layer lines arrive before judging
        n, done, extracting, downloading = pull.counts()
        finished = n and done == n
        if not (final or finished or time.time() - pull_shown >= PULL_SUMMARY):
            return
        if not pull_shown and finished:
            out("[RUNPOD] the image is already on the host ({} layers present): the container starts now".format(n))
            pull_shown, pull_done = time.time(), True
            return
        if not pull_shown:
            out("[RUNPOD] the host is fetching the pod's image from Docker Hub (several minutes; a host "
                "that already has it starts the container within seconds)")
        pull_shown = time.time()
        out("[RUNPOD] " + pull.summary())
        if finished:
            pull_done = True

    def poll_proxy():
        for line in proxy_log(pod_id):
            if show(line, ("proxy", line)):
                return True
        return False

    def check_pod():
        """Every 45 s: the pod's status, and the warning when the container has not started."""
        if time.time() - last_check_box[0] < STATUS_POLL:
            return None
        last_check_box[0] = time.time()
        try:
            current = get_pod(cfg, pod_id)
        except ApiError:
            return None
        st = pod_status(current)
        state["status"] = st
        if st in FINAL:
            out("the pod is {}. Read its log in the console, then `python pod.py down` and `up` again.".format(st))
            return False
        if not state["container"] and current.get("runtime") is None and created \
                and (datetime.now(timezone.utc) - created).total_seconds() > STALL_MINUTES * 60 \
                and not state["stall_told"]:
            state["stall_told"] = True
            gpu, dc, _ = pod_summary(current)
            out("WARNING: NO CONTAINER {} AFTER CREATION (status {}, {}). The host is still fetching the image, or "
                "does not respond; the console (Pods > your pod) shows its log and any maintenance notice. "
                "The pod bills meanwhile. Usual answer:  python pod.py down <pod>  then  python pod.py up  again, "
                "usually in another data center.".format(fmt_since(created), st, dc if dc != "?" else "data center not reported"))
        return None

    last_check_box = [time.time()]

    def heartbeat():
        if time.time() - state["last_output"] >= HEARTBEAT:
            out("[RUNPOD] waiting: pod {}, {}{} since creation".format(
                state["status"], "no container yet, " if not state["container"] else "", fmt_since(created)))

    while time.time() < deadline:
        try:
            for eid, source, line in log_stream(cfg, pod_id, tail=500, since=resume_point(last_id), idle=STREAM_IDLE,
                                                max_age=STREAM_MAX_AGE):
                if line and "[AINVFX]" in line:
                    if show(line, (eid, line)):
                        pull_summary(True)
                        return done()
                elif line and source == "system":
                    system(line)
                last_id = eid or last_id
                if time.time() - last_poll > PROXY_POLL:
                    last_poll = time.time()
                    if poll_proxy():
                        return done()
                if check_pod() is False:
                    return False
                pull_summary(False)
                heartbeat()
                if time.time() > deadline:
                    break
        except ApiError as e:
            if e.code == 0 and e.timed_out:
                if not state["stream_told"]:
                    state["stream_told"] = True
                    out("[RUNPOD] the log stream did not answer ({}): the host may not be ready; retrying".format(e.detail))
            elif not state["stream_told"]:
                state["stream_told"] = True
                out("(the log cannot be read through the API from here: {}. Follow it in the console, "
                    "Pods > your pod > Logs, or wait: the pod's own copy is read as well.)".format(e.detail))
            time.sleep(min(PROXY_POLL, 10))
        last_poll = time.time()
        if poll_proxy():
            return done()
        if check_pod() is False:
            return False
        pull_summary(False)
        heartbeat()
    say("Still working after {} minutes: `python pod.py status` shows where it is.".format(minutes))
    return False


# ----------------------------------------------------------------------------- commands
def cmd_setup(args):
    cfg = load_config()
    say("ainvfx-runpod {} · setup".format(VERSION))
    key = args.key or os.environ.get("RUNPOD_API_KEY", "") or cfg.get("api_key", "")
    if not key:
        say("\n1. Your Runpod API key. Console > Account > Credentials > API Keys > Create API Key.")
        say("   Name: ainvfx-runpod (any name works; this one says what the key is for).")
        say("   Permission: Restricted. Then two lines appear:")
        say("     api.runpod.io/graphql  ->  Read / Write   (the API this script uses: pods, catalog, secrets, SSH keys)")
        say("     api.runpod.ai          ->  None           (Serverless endpoints, not used here)")
        say("   Paste it below (nothing shows while you type), then Enter.")
        key = getpass.getpass("   API key: ").strip()
    if not key:
        die("empty key")
    cfg["api_key"] = key
    say("   checking the key...")
    try:
        pods = list_pods(cfg, quiet=True)
    except ApiError as e:
        die("the key does not work: {}\n{}".format(e, e.hint()))
    say("   ok ({} pod(s) in the account right now)".format(len(pods)))

    guess_region, guess_country = local_place()
    region = args.region or cfg.get("region") or guess_region
    country = (args.country or cfg.get("country") or (guess_country if region == guess_region else "")).upper()
    cfg["region"], cfg["country"] = region, country
    zone = local_zone() or "UTC offset only"
    say("\n2. Nearest data centers first: {}{} (from this computer's clock: {}).".format(
        REGION_NAMES.get(region, region), ", " + country + " first" if country else "", zone))
    say("   Runpod names them by country: CA-MTL-1, US-TX-3, EU-FR-1, EUR-IS-2... `up` tries yours first,")
    say("   then the rest of the region, then the other region. Change with `setup --region EU --country FR`.")
    if args.template:
        cfg["template_id"] = args.template
    elif not cfg.get("template_id"):
        cfg["template_id"] = TEMPLATE_ID
    say("   template: {}".format(cfg["template_id"] or "none (the pod is described by this script itself)"))

    say("\n3. Hugging Face token (for the gated files of models.json: FLUX 3 Image Klein in the image profile, LTX in video and train).")
    if args.hf_token:
        say("   The token is stored on Runpod as the secret '{}' and never on this machine. Pods created".format(HF_SECRET))
        say("   by `up` receive it as HF_TOKEN. A read token from https://huggingface.co/settings/tokens.")
        token = getpass.getpass("   Hugging Face token: ").strip()
        if token:
            try:
                rp("POST", "/account/secrets", cfg,
                   body={"name": HF_SECRET, "value": token, "description": "Hugging Face read token (ainvfx-runpod)"})
                say("   secret '{}' created".format(HF_SECRET))
                cfg["hf_secret"] = True
            except ApiError as e:
                if e.code == 409:              # the name exists: rotate its value
                    try:
                        sid = secrets_named(cfg)[0]["id"]
                        rp("PATCH", "/account/secrets/" + sid, cfg, body={"value": token})
                        say("   secret '{}' updated".format(HF_SECRET))
                        cfg["hf_secret"] = True
                    except (ApiError, IndexError, KeyError) as e2:
                        say("   could not update the secret: {}".format(e2))
                else:
                    say("   could not store the secret ({}): {}".format(e.code, e.hint() or e.detail))
                    say("   Console > Account > Credentials > Secrets > name '{}', value: your token.".format(HF_SECRET))
    else:
        found = secret_exists(cfg)
        cfg["hf_secret"] = bool(found)
        say("   secret '{}': {}. To store a token:  python pod.py setup --hf-token".format(
            HF_SECRET, "present" if found else "absent" if found is False else "unknown (the key cannot list secrets)"))

    say("\n4. SSH key (optional: only for `pod.py ssh`; everything else goes through the browser).")
    if shutil.which("ssh") and shutil.which("ssh-keygen"):
        keyfile = Path(cfg.get("ssh_key") or Path.home() / ".ssh" / "id_ed25519")
        pub = keyfile.with_name(keyfile.name + ".pub")
        if not keyfile.exists():
            ans = "y" if args.yes else input("   No SSH key found. Create one now? [Y/n] ").strip().lower()
            if ans in ("", "y", "yes"):
                keyfile.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(keyfile), "-q",
                                "-C", "ainvfx-runpod {}".format(getpass.getuser())])
        if pub.exists():
            pubkey = pub.read_text(encoding="utf-8").strip()
            try:
                keys = [k for k in (rp("GET", "/account/ssh-keys", cfg).get("keys") or []) if isinstance(k, str)]
                if not any(pubkey.split()[1] in k for k in keys if len(k.split()) > 1):
                    rp("PUT", "/account/ssh-keys", cfg, body={"keys": keys + [pubkey]})   # PUT replaces the full set
                    say("   public key registered on your Runpod account")
                else:
                    say("   public key already registered")
                cfg["ssh_key"] = str(keyfile)
                if WINDOWS:
                    restrict(keyfile)
            except ApiError as e:
                say("   could not register the key ({}): `pod.py ssh` will not work, the rest will. "
                    "Console > Account > Credentials > SSH Public Keys.".format(e.code))
    else:
        say("   no `ssh` on this machine: skipped. JupyterLab's terminal replaces it on the pod.")
    cfg.setdefault("pods", {})
    save_config(cfg)
    say("\nSaved in {} (readable by your account only).".format(CONFIG))
    say("Next:  python pod.py up image")


def create_pod(cfg, body, where):
    """POST /pods for one candidate. Returns the pod, or None to try the next candidate.
    Retry rules from the API reference: 422 fix and stop, 402 stop, 400 next candidate,
    403 skip, 429 wait then retry, 5xx retry once."""
    for attempt in range(3):
        try:
            return rp("POST", "/pods", cfg, body=body, timeout=120)
        except ApiError as e:
            if e.code in (401, 402, 404, 422):
                detail = e.detail if not e.errors else "{} {}".format(e.detail, json.dumps(e.errors)[:400])
                die("{}: {}\n{}".format(e.code, detail, e.hint()))
            if e.code == 429:
                wait = int(e.retry_after or 10)
                say("   rate limited: waiting {} s".format(wait)); time.sleep(wait); continue
            if e.code >= 500 and attempt < 2:
                say("   {}: Runpod error {}, retrying".format(where, e.code)); time.sleep(5); continue
            if e.code == 403:
                say("   {}: not accessible with this account (403), skipped".format(where)); return None
            say("   {}: no machine ({}: {})".format(where, e.code, e.detail[:160].replace("\n", " ")))
            return None
    return None


def read_catalogs(cfg, community, fatal=True):
    """The catalog of each cloud `up` considers, as {cloud: GPU types}. When the Secure catalog cannot be read,
    `up` stops (fatal), or the ApiError goes up to the --retry wait, which reads again at its next try."""
    catalogs = {}
    for cloud in (("SECURE", "COMMUNITY") if community else ("SECURE",)):
        try:
            catalogs[cloud] = read_catalog(cfg, cloud)
        except ApiError as e:
            if cloud == "SECURE":
                if not fatal:
                    raise
                die("the Runpod catalog could not be read ({}: {}).\n{}".format(e.code, e.detail[:200], e.hint()))
            if fatal:
                say("Community Cloud catalog not readable ({}): Secure Cloud only.".format(e.code))
    return catalogs


def pick_candidates(catalogs, wants, named, spec, region, country, max_price):
    """What `up` may create now, in the order it tries them. With --gpu: the GPUs named, in the order given.
    Without: the profile's GPU on Secure Cloud, then every GPU that fits the profile at max_price or less,
    cheapest first. Returns (offers, fits, preferred, candidates)."""
    offers = make_offers(catalogs, region, country)
    fits = [o for o in offers if fits_profile(o, spec["vram"])]
    if named:
        return offers, fits, None, [o for w in wants for o in offers if o["id"] == w["id"]]
    preferred = next((o for o in offers if o["id"] == wants[0]["id"] and o["cloud"] == "SECURE"), None)
    rest = [o for o in fits if o is not preferred and o["price"] <= max_price]
    return offers, fits, preferred, ([preferred] + rest) if preferred else rest


def gpu_report(want, offers, region, community):
    """Four lines on one GPU type: its availability on Secure Cloud, where it is in stock, its prices."""
    wname = str(want.get("name") or want["id"])
    secure = next((o for o in offers if o["id"] == want["id"] and o["cloud"] == "SECURE"), None)
    dcs = secure["dcs"] if secure else []
    price = want.get("price") or {}
    p_secure, p_comm = price.get("secure"), price.get("community")
    say("\n{}: availability {} on Secure Cloud with hosts on CUDA {} or newer".format(
        wname, str(want.get("availability", "?")).upper(), MIN_CUDA))
    say("   in stock in {}: {}".format(REGION_NAMES.get(region, region),
                                       ", ".join(d for d in dcs if region_of(d) == region) or "none right now"))
    say("   elsewhere: {}".format(", ".join(d for d in dcs if region_of(d) != region) or "none"))
    say("   price per hour: {} Secure, {} Community{}".format(
        fmt_money(p_secure) if p_secure else "?", fmt_money(p_comm) if p_comm else "?",
        "" if community else " (not considered: add --community)"))


def retry_cap(wants, named, community, max_price):
    """The highest price per hour a --retry wait may create at: the dearest GPU named with --gpu, on the clouds
    considered; without --gpu, max_price or the profile's GPU, whichever costs more. None when unknown."""
    keys = ("secure", "community") if community else ("secure",)
    prices = []
    for w in wants:
        for k in keys:
            try:
                prices.append(float((w.get("price") or {}).get(k)))
            except (TypeError, ValueError):
                pass
    if not named:
        prices.append(max_price)
    return max(prices) if prices else None


def try_create(cfg, candidates, name, disk, env):
    """Create one pod from the candidates, in order: for a Secure offer, every data center the catalog lists
    with stock; then the next offer. A pod whose host has maintenance under way, or starting within MAINT_HOURS,
    is terminated at once and the next machine tried. Returns (pod, host, stray): pod is None when no pod could be created; stray
    lists the refused pods that could not be terminated."""
    stray = []
    for n, offer in enumerate(candidates):
        if n:
            say("   Next: {}.".format(offer_line(offer)))
        # every data center the catalog lists with stock for this GPU (0.6.0 stopped at 6: with ten pods
        # created in the same minutes, the seventh data center can be the one with a free GPU)
        attempts = [[dc] for dc in offer["dcs"]] if offer["cloud"] == "SECURE" else [[]]
        for dc_list in attempts:
            body = {"name": name, "cloud": offer["cloud"], "disk": disk, "env": env,
                    "startSsh": True, "startJupyter": True,
                    "gpu": {"id": offer["id"], "count": 1, "minCudaVersion": MIN_CUDA}}
            if dc_list:
                body["dataCenterIds"] = dc_list
            if cfg.get("template_id") or TEMPLATE_ID:
                body["templateId"] = cfg.get("template_id") or TEMPLATE_ID   # image, command, ports come from the template
            else:
                body["image"] = IMAGE
                body["args"] = START_CMD
                body["ports"] = ["{}/http".format(COMFY_PORT), "{}/http".format(JUPYTER_PORT), "22/tcp"]
            where = "{} {}".format(offer["cloud"], dc_list[0] if dc_list else "(host picked by Runpod)")
            pod = create_pod(cfg, body, where)
            if not (pod and pod.get("id")):
                continue
            say("   created on {}: pod {} ({})".format(where, pod["id"], offer["name"]))
            machine, problem = host_check(cfg, pod["id"])
            if problem:
                say("   REFUSED: its host has {}.".format(problem))
                try:
                    rp("DELETE", "/pods/" + pod["id"], cfg)
                    say("   pod {} terminated at once (a few seconds billed). Trying the next machine.".format(pod["id"]))
                except ApiError as e:
                    say("   pod {} COULD NOT BE TERMINATED ({}): run `python pod.py down {}` once this ends."
                        .format(pod["id"], e.detail[:100], pod["id"]))
                    stray.append(pod["id"])
                continue
            if machine is None:
                say("   host check skipped: the GraphQL API did not answer. The log below shows whether the host works.")
            elif not machine:
                say("   host check: no host reported after {} s. The log below shows whether it works.".format(HOST_WAIT))
            else:
                say("   host checked: no maintenance planned{}.".format(
                    " ({})".format(machine["dataCenterId"]) if machine.get("dataCenterId") else ""))
            return pod, (machine or {}), stray
    return None, None, stray


def stop_on_stray(stray):
    """A --retry wait never goes on while a refused pod still runs: it would bill next to the next pod."""
    if stray:
        die("pod {} could not be terminated, and it bills: `python pod.py down {}` terminates it. "
            "The wait stops here.".format(", ".join(stray), stray[0]))


def wait_for_stock(cfg, args, tag, wants, named, spec, region, country, max_price, make_name, disk, env):
    """`up --retry`: read the catalog every RETRY_EVERY seconds until a candidate is free, then create it: one
    pod, never more. Returns (pod, host, name). Ctrl+C ends the wait with a message, exit code 130.
    A wait can last hours, so before each creation the record of `tag` is read again: if another `up` has
    created a pod under the same name in the meantime, the wait stops instead of creating a second one."""
    if named:
        what = " or ".join(str(w.get("name") or w["id"]) for w in wants)
    else:
        what = "{}, or any GPU that fits the {} profile at up to {} per hour".format(
            wants[0].get("name") or wants[0]["id"], spec["label"], fmt_money(max_price))
    say("\nWaiting on {} for {}. Every {} s, `up` reads the catalog and creates the first one free.".format(
        "Secure and Community Cloud" if args.community else "Secure Cloud", what, RETRY_EVERY))
    say("Ctrl+C stops the wait. No pod bills before one is created.")

    def reads(n):
        return "{} read{}".format(n, "" if n == 1 else "s")
    started = last_note = time.time()
    tries, creating, name = 0, False, None
    try:
        while True:
            time.sleep(RETRY_EVERY)
            tries += 1
            now = datetime.now().strftime("%H:%M:%S")
            try:
                catalogs = read_catalogs(cfg, args.community, fatal=False)
            except ApiError as e:
                if e.code in (401, 403):
                    die("the Runpod catalog could not be read ({}: {}).\n{}".format(e.code, e.detail[:200], e.hint()))
                if time.time() - last_note >= RETRY_NOTE:
                    say("   {}  the catalog could not be read ({}): still waiting".format(now, e.code or e.detail[:80]))
                    last_note = time.time()
                continue
            candidates = pick_candidates(catalogs, wants, named, spec, region, country, max_price)[3]
            if not candidates:
                if time.time() - last_note >= RETRY_NOTE:
                    say("   {}  still waiting: {} in {}, none free".format(
                        now, reads(tries), fmt_duration((time.time() - started) / 3600)))
                    last_note = time.time()
                continue
            first = candidates[0]
            say("   {}  {} free on {} Cloud ({}): creating it".format(now, first["name"], first["cloud"].title(), offer_where(first)))
            rec = (load_config().get("pods") or {}).get(tag)
            if rec:
                die("a pod was recorded as `{}` while this wait ran, by another command on this machine: {} ({}). "
                    "The wait stops here, so that two pods do not bill. `python pod.py list` shows your pods.".format(
                        tag, rec.get("name"), rec.get("id")))
            name = make_name()
            creating = True
            pod, host, stray = try_create(cfg, candidates, name, disk, env)
            creating = False
            if pod:
                say("   The wait took {} ({}).".format(fmt_duration((time.time() - started) / 3600), reads(tries)))
                return pod, host, name
            stop_on_stray(stray)
            say("   {}  taken in the meantime: still waiting".format(datetime.now().strftime("%H:%M:%S")))
            last_note = time.time()
    except KeyboardInterrupt:
        if creating:
            say("\nStopped during a creation request: `python pod.py list` shows whether pod {0} exists, and "
                "`python pod.py down {0}` terminates it.".format(name))
        else:
            say("\nStopped waiting after {}: this wait created no pod.".format(reads(tries)))
        sys.exit(130)


def cmd_up(args):
    cfg = load_config()
    profiles = parse_profiles(args.profile)
    profile = "+".join(profiles)              # `image+3d`: the value AINVFX_PROFILE carries
    api_key(cfg)
    tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in (args.name or "")).strip("-") or profile.replace("+", "-")
    cfg.setdefault("pods", {})
    old = cfg["pods"].get(tag)
    if old:
        try:
            pod = get_pod(cfg, old["id"])
            st = pod_status(pod)
            if st not in FINAL and st != "?":
                say("A pod already exists as `{}`: {} ({}). Address: {}".format(tag, old.get("name"), st, pod_url(old["id"])))
                say("Use it, terminate it first (python pod.py down {}), or create another one with `up {} --name <name>`."
                    .format(tag, profile))
                return
        except ApiError as e:
            if e.code != 404:
                die("{}\n{}".format(e, e.hint()))
        remember_pod(cfg, tag, None)

    region = args.region or cfg.get("region") or local_region()
    country = (cfg.get("country") or "").upper() if not args.region or args.region == cfg.get("region") else ""
    def make_name():                          # called at creation: after a --retry wait, the name has the creation's time
        return "ainvfx-{}-{}".format(profile.replace("+", "-"), tag if tag != profile.replace("+", "-")
                                     else datetime.now().strftime("%m%d-%H%M"))
    env = dict(load_settings())              # settings.env, then settings.local.env: tag, Python, PyTorch, models...
    for item in (args.set or []):            # `--set KEY=VALUE`, for this pod only
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.startswith("AINVFX_") or not value:
            die("--set takes AINVFX_KEY=value (got \"{}\"). The keys are listed in settings.env.".format(item))
        env[key] = value
    if args.models_url:                       # `--models-url URL`, the same as --set AINVFX_MODELS_URL=URL
        env["AINVFX_MODELS_URL"] = args.models_url.strip()
    env["AINVFX_PROFILE"] = profile           # the profile named on the command line wins
    if cfg.get("hf_secret") or secret_exists(cfg):
        env["HF_TOKEN"] = HF_SECRET_REF          # the reference, never the token itself
    if not args.selftest:
        env["AINVFX_SELFTEST"] = "0"
    spec = profile_spec(profiles, read_models(env.get("AINVFX_MODELS_URL", DEFAULT_MODELS_URL)))  # before any spend
    disk = args.disk or spec["disk"]

    # The catalog, read once per cloud: every GPU type with its stock for hosts on CUDA MIN_CUDA or newer.
    catalogs = read_catalogs(cfg, args.community)
    secure = catalogs.get("SECURE") or []
    max_price = MAX_PRICE if args.max_price is None else args.max_price
    named = bool(args.gpu)
    if named:                                 # `--gpu B200 --gpu B300`: those GPUs only, in that order
        wants = []
        for ref in args.gpu:
            g = find_gpu(secure + (catalogs.get("COMMUNITY") or []), ref)
            if all(g["id"] != w["id"] for w in wants):
                wants.append(g)
    else:
        wants = [next((g for g in secure if g["id"] == spec["gpu"]), {"id": spec["gpu"], "name": spec["gpu"]})]
    offers, fits, preferred, candidates = pick_candidates(catalogs, wants, named, spec, region, country, max_price)
    wname = " or ".join(str(w.get("name") or w["id"]) for w in wants)

    for want in wants:
        gpu_report(want, offers, region, args.community)
    models_url = env.get("AINVFX_MODELS_URL", DEFAULT_MODELS_URL)
    if models_url != DEFAULT_MODELS_URL:
        say("   models list: {} (your own, not the course's)".format(models_url))
    for key in sorted(k for k in env if k.startswith("AINVFX_") and k not in ("AINVFX_PROFILE", "AINVFX_MODELS_URL", "AINVFX_SELFTEST")
                      and read_env_file(SETTINGS).get(k) != env[k]):
        say("   {}: {} (yours)".format(key, env[key]))

    profile_rule = "{} GB of VRAM or more, Blackwell or newer, not a 1g MIG slice, host on CUDA {} or newer".format(
        spec["vram"], MIN_CUDA)
    say("   profile {}: about {} GB of models, a {} GB container disk".format(profile, spec["gb"], disk))
    if named:                                 # the GPUs named on the command line, and nothing else
        if not candidates:
            say("   No {} in stock right now{}.".format(wname, " on Secure Cloud" if not args.community else ""))
            others = [o for o in fits if all(o["id"] != w["id"] for w in wants)]
            if others:
                say("\nIn stock now for the {} profile ({}), cheapest first:".format(profile, profile_rule))
                show_offers(others, None, max_price)
                say('   To take one:  python pod.py up {} --gpu "{}"'.format(profile, others[0]["name"]))
            if not args.retry:
                die("nothing created.")
    elif preferred:
        say("   tried first: {}".format(preferred["dcs"][0]))
    else:
        say("   No {} is free on Secure Cloud. `up` never creates a pod where the catalog shows no stock:".format(wname))
        say("   Runpod would then pick any host, and twice that was a host being removed from the platform.")
        if candidates:
            say("   Replacement: {}.".format(offer_line(candidates[0])))
        say("\nIn stock now for the {} profile ({}), cheapest first:".format(profile, profile_rule))
        if fits:
            show_offers(fits, candidates[0] if candidates else None, max_price)
        else:
            say("   none right now.")
        if not candidates:
            if fits:
                say('   To take one above {} per hour:  python pod.py up {} --gpu "{}"'.format(
                    fmt_money(max_price), profile, fits[0]["name"]))
            if not args.retry:
                die("nothing in stock fits the {} profile at {} per hour or less. Try again in a few minutes.".format(
                    profile, fmt_money(max_price)))
        else:
            other = next((o for o in fits if o["id"] != candidates[0]["id"]), None)
            if other:
                say('   To choose another one:  python pod.py up {} --gpu "{}"   (any name from the GPU column)'.format(
                    profile, other["name"]))
        if not args.community:
            say("   Community Cloud is not considered: add --community to include it (cheaper; the host is"
                " picked by Runpod and can fail to start, as on 9 October 2026).")

    balance, spend = account_money(cfg)
    cap = retry_cap(wants, named, args.community, max_price) if args.retry else None
    created, host = None, None
    if candidates:
        first = candidates[0]
        if balance is not None:
            line = "   Your Runpod credit: {}, about {} hours of {} at {} per hour".format(
                fmt_money(balance), int(balance // first["price"]), first["name"], fmt_money(first["price"]))
            if spend:
                line += "; your account already spends {} per hour".format(fmt_money(spend))
            say(line + ".")
            if balance < 3 * first["price"]:
                say("   That is under 3 hours: add credit in the Runpod console (Billing), or pick a cheaper card.")
        if not args.yes:
            if len(candidates) > 1:
                say("   If it is taken in the meantime, `up` tries {}, up to {} per hour.".format(
                    "the next GPU named, in the order given" if len(wants) > 1 else "the next GPUs that fit, cheapest first",
                    fmt_money(max(o["price"] for o in candidates))))
            if args.retry:
                say("   With --retry: if none can be created now, `up` reads the catalog every {} s and creates the first "
                    "one free{}, without asking again.".format(RETRY_EVERY, ", at up to {} per hour".format(fmt_money(cap)) if cap else ""))
            ans = input("   Create a {} at {} per hour, with a {} GB disk, billed from creation? [Y/n] ".format(
                first["name"], fmt_money(first["price"]), disk)).strip().lower()
            if ans not in ("", "y", "yes"):
                say('Nothing created. To choose another GPU:  python pod.py up {} --gpu "<name from the list>"'.format(profile))
                return
        name = make_name()
        created, host, stray = try_create(cfg, candidates, name, disk, env)
        if not created:
            if not args.retry:
                die("no pod could be created: the machines in stock were taken in the meantime, or under maintenance.\n"
                    "Try again in a minute: `python pod.py up {}` reads the stock again.".format(profile))
            stop_on_stray(stray)
            say("   None could be created now: waiting, as --retry asks.")
    else:                                     # --retry and nothing free: one question, before the wait
        if balance is not None and cap:
            line = "   Your Runpod credit: {}, about {} hours at {} per hour, the highest price this wait accepts".format(
                fmt_money(balance), int(balance // cap), fmt_money(cap))
            if spend:
                line += "; your account already spends {} per hour".format(fmt_money(spend))
            say(line + ".")
            if balance < 3 * cap:
                say("   That is under 3 hours: add credit in the Runpod console (Billing), or pick a cheaper card.")
        if not args.yes:
            ans = input("   Wait, then create the first {} free ({}a {} GB disk, billed from creation)? [Y/n] ".format(
                wname if named else "GPU that fits", "up to {} per hour, ".format(fmt_money(cap)) if cap else "",
                disk)).strip().lower()
            if ans not in ("", "y", "yes"):
                say("Nothing created.")
                return
    if not created:
        created, host, name = wait_for_stock(cfg, args, tag, wants, named, spec, region, country, max_price, make_name,
                                             disk, env)

    pod_id = created["id"]
    created_at = datetime.now(timezone.utc)
    remember_pod(cfg, tag, {"id": pod_id, "name": name, "profile": profile, "created": created_at.isoformat()})
    say("\nWaiting for the machine (PROVISIONING, STARTING, then RUNNING)...")
    wait_until = time.time() + 15 * 60
    pod = created
    while time.time() < wait_until:
        try:
            pod = get_pod(cfg, pod_id)
        except ApiError:
            pass
        st = pod_status(pod)
        if st == "RUNNING":
            break
        if st in FINAL:
            die("the pod is {} before it ran. Read its log in the console, then `python pod.py down {}` and try again.".format(st, profile))
        time.sleep(8)
    for _ in range(4):                     # the data center and the price arrive a few seconds after RUNNING
        gpu, dc, price = pod_summary(pod)
        if dc != "?" and price:
            break
        time.sleep(5)
        try:
            pod = get_pod(cfg, pod_id)
        except ApiError:
            pass
    gpu, dc, price = pod_summary(pod)
    if dc == "?" and host.get("dataCenterId"):
        dc = host["dataCenterId"]
    say("Pod {} · {} · {} · CUDA {} · {}".format(
        name, gpu, dc if dc != "?" else "data center not reported yet (the console shows it)",
        pod.get("cudaVersion") or "?", (fmt_money(price) + " per hour") if price else "price in the console"))
    say("ComfyUI address (ready once the log says COMFYUI UP): {}".format(pod_url(pod_id)))
    say("JupyterLab: {}  (the READY line gives the link with its token)".format(pod_url(pod_id, JUPYTER_PORT)))
    say("The pod bills from now on; `python pod.py down {}` terminates it at any time.\n".format(tag))
    try:
        ready = follow_logs(cfg, pod_id, minutes=args.wait, pod=pod, created=created_at)
    except KeyboardInterrupt:
        say("\nStopped following the log. The pod keeps running: `python pod.py logs {0}` to follow again, "
            "`python pod.py down {0}` to terminate.".format(tag))
        sys.exit(130)
    except ApiError as e:
        say("\nThe log could not be followed ({}). {}".format(e.detail, e.hint()))
        say("The pod keeps running: `python pod.py status {0}` shows it, `python pod.py down {0}` terminates it.".format(tag))
        sys.exit(1)
    say("\nWhen you are done:  python pod.py down {}".format(tag))
    if not ready:
        sys.exit(1)


def cmd_logs(args):
    cfg = load_config()
    tag, rec = which_pod(cfg, args.profile)
    try:
        pod = get_pod(cfg, rec["id"])
    except ApiError:
        pod = None
    try:
        follow_logs(cfg, rec["id"], minutes=args.wait, pod=pod, created=parse_time(rec.get("created") or ""))
    except KeyboardInterrupt:
        say("\nStopped following the log. The pod keeps running: `python pod.py down {}` terminates it.".format(tag))
        sys.exit(130)


def cmd_status(args):
    cfg = load_config()
    tag, rec = which_pod(cfg, args.profile)
    try:
        pod = get_pod(cfg, rec["id"])
    except ApiError as e:
        if e.code == 404:
            die("the pod no longer exists on Runpod (terminated from the console?). Run `up` for a new one.")
        die("{}\n{}".format(e, e.hint()))
    gpu, dc, price = pod_summary(pod)
    st = pod_status(pod)
    say("`{}` · {} · {} · {} · CUDA {} · {}".format(tag, rec.get("name"), gpu, dc, pod.get("cudaVersion") or "?", st))
    spent = pod_spent(pod, rec)
    if spent:
        say("running for {} · about {} spent so far at {} per hour".format(fmt_duration(spent[0]), fmt_money(spent[1]), fmt_money(spent[2])))
        balance, _ = account_money(cfg)
        if balance is not None:
            say("credit left on Runpod: {}, about {} more hours of this pod".format(fmt_money(balance), int(balance // spent[2])))
    rt = pod.get("runtime") or {}
    if rt.get("gpus"):
        g = rt["gpus"][0]
        say("GPU in use: {}% · VRAM {}%".format(g.get("util", "?"), g.get("memoryUtil", "?")))
    elif st == "RUNNING" and pod.get("runtime") is None:
        say("container: not started yet (the host is fetching the image, or has a problem): Runpod's system log below says which")
    say("ComfyUI: {}  ({})".format(pod_url(rec["id"]), "answers" if comfy_alive(rec["id"]) else "not answering yet"))
    say("JupyterLab: {}".format(pod_url(rec["id"], JUPYTER_PORT)))
    kind, host, port, user = pod_ssh(pod)
    if kind:
        say("SSH ({}): ssh -p {} {}@{}".format(kind, port, user, host))
    lines = proxy_log(rec["id"])              # the pod's own copy of the log is complete; the API stream is the fallback
    err = None
    if not lines:
        lines, err = pod_logs(cfg, rec["id"])
        lines = ainvfx_lines(lines or [])
    if not lines:
        say("log: nothing from the pod's script yet{}; open it in the console, Pods > your pod > Logs".format(
            " (the API said: {})".format(err.detail) if err is not None else ""))
        system, err2 = pod_logs(cfg, rec["id"], tail=40, source="system")
        if system:
            pull = ImagePull()
            others = [l for l in system if not pull.feed(l)]
            if pull.layers:
                say("  [RUNPOD] " + pull.summary())
            for l in others[-5:]:
                say("  [RUNPOD] " + l.strip()[:200])
    for l in lines[-15:]:
        say("  " + l[l.index("[AINVFX]"):])


def cmd_open(args):
    cfg = load_config()
    _, rec = which_pod(cfg, args.profile)
    url = pod_url(rec["id"])
    say(url)
    webbrowser.open(url)


def multipart(fields, filefield, path):
    boundary = "----ainvfx" + uuid.uuid4().hex
    body = bytearray()
    for k, v in fields.items():
        body += ("--{}\r\nContent-Disposition: form-data; name=\"{}\"\r\n\r\n{}\r\n".format(boundary, k, v)).encode()
    body += ("--{}\r\nContent-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\n"
             "Content-Type: application/octet-stream\r\n\r\n".format(boundary, filefield, path.name)).encode()
    body += path.read_bytes()
    body += ("\r\n--{}--\r\n".format(boundary)).encode()
    return bytes(body), "multipart/form-data; boundary=" + boundary


def cmd_push(args):
    cfg = load_config()
    _, rec = which_pod(cfg, args.profile)
    base = pod_url(rec["id"])
    files = [Path(f).expanduser() for f in args.files]
    for f in files:
        if not f.exists():
            die("file not found: {}".format(f))
    if not comfy_alive(rec["id"]):
        die("ComfyUI does not answer on the pod yet. `python pod.py status` shows the log.")
    for f in files:
        body, ctype = multipart({"overwrite": "true", "type": "input"}, "image", f)
        try:
            request("POST", base + "/upload/image", data=body, headers={"Content-Type": ctype}, timeout=600, raw=True)
            say("  {}  ({:.1f} MB) -> input/".format(f.name, f.stat().st_size / 1e6))
        except ApiError as e:
            die("upload of {} failed: {}\n{}".format(f.name, e.detail[:200], e.hint()))
    say("Done. The files are in the lists of the Load Image and Load Video nodes.")


def cmd_pull(args):
    cfg = load_config()
    _, rec = which_pod(cfg, args.profile)
    n = pull(cfg, rec)
    say("{} file(s) new in {}".format(n, OUTPUTS / rec.get("name", rec["id"])))


# 0.9.2 (22 Oct 2026, after BTC2 session 6): `pull`, and so `down`, brought back only the files that ComfyUI's history
# reports under five keys (images, gifs, videos, audio, files). ComfyUI reports every 3D save under "3d" (Save 3D
# Model, Save 3D (Advanced), Save Splat, Save Point Cloud: comfy_api/latest/_ui.py, Saved3DModels, v0.42.1), so the
# session's GLB files were terminated with their pods. Two sources now, joined:
#  1. the whole output folder, walked through JupyterLab's contents API with the token of the bootstrap's READY line
#     (Runpod starts JupyterLab with --ServerApp.root_dir=/ and JUPYTER_PASSWORD as its token: runpod/containers,
#     container-template/start.sh, 4456421): it also finds what a node saves without reporting it (Save Checkpoint, a
#     trained LoRA) and what was made before a ComfyUI restart, since the history lives in memory;
#  2. ComfyUI's history, every key, through /view: the fallback when JupyterLab is off.
# ComfyUI itself has no recursive listing: /internal/files/output lists the top folder only and is meant for its own
# frontend (api_server/routes/internal/internal_routes.py).
OUTPUT_DIRS = ("workspace/ComfyUI/output", "ComfyUI/output")   # the output folder seen from JupyterLab's root (/, or /workspace)
SKIP_NAMES = {"_output_images_will_be_put_here"}                # the empty placeholder ComfyUI creates


def history_files(history):
    """(subfolder, filename) of every file ComfyUI's history reports in the output folder, whatever the key: images,
    gifs, videos, audio, 3d, files, or a key a custom node invents. Entries of the temp and input folders are left out."""
    found = []

    def walk(v):
        if isinstance(v, dict):
            if isinstance(v.get("filename"), str) and v.get("type") in (None, "output"):
                found.append((v.get("subfolder") or "", v["filename"]))
                return
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    for entry in (history or {}).values():
        if isinstance(entry, dict):
            walk(entry.get("outputs") or {})
    seen, out = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def jupyter_token(lines):
    """The JupyterLab token of the bootstrap's READY line (`.../lab?token=...`), or None."""
    for line in reversed(lines or []):
        m = re.search(r"/lab\?token=([^\s·]+)", line)
        if m:
            return m.group(1)
    return None


def safe_rel(rel):
    """A relative path inside the output folder, with / separators, or None for anything that could leave it."""
    parts = [x for x in (rel or "").replace("\\", "/").split("/") if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts) or rel.startswith("/") or ":" in parts[0]:
        return None
    return "/".join(parts)


def jupyter_files(pod_id, token):
    """(root, [(relative path, size)]) for every file under the pod's output folder, through JupyterLab's contents
    API; (None, []) when JupyterLab does not answer or has no output folder."""
    base = pod_url(pod_id, JUPYTER_PORT)
    hdrs = {"Authorization": "token " + token}

    def listing(path):
        return request("GET", base + "/api/contents/" + urllib.parse.quote(path) + "?content=1", headers=hdrs, timeout=60)

    for root in OUTPUT_DIRS:
        try:
            top = listing(root)
        except ApiError:
            continue
        if top.get("type") != "directory":
            continue
        files, todo = [], [top]
        while todo:
            folder = todo.pop()
            for e in folder.get("content") or []:
                name, path = e.get("name") or "", e.get("path") or ""
                if name.startswith(".") or name in SKIP_NAMES or not path.startswith(root + "/"):
                    continue
                if e.get("type") == "directory":
                    try:
                        todo.append(listing(path))
                    except ApiError as err:
                        say("  could not list {} ({})".format(path[len(root) + 1:], err.code or err.detail[:60]))
                elif e.get("type") in ("file", "notebook"):
                    rel = safe_rel(path[len(root) + 1:])
                    if rel:
                        files.append((rel, e.get("size")))
        return root, sorted(files)
    return None, []


def fetch_to(url, target, headers=None, timeout=600):
    """Download url into target through a .part file, so an interrupted download never leaves a half file. Returns
    the size in bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    hdrs = {"Accept": "*/*"}
    hdrs.update(headers or {})
    r = request("GET", url, headers=hdrs, timeout=timeout, stream=True)
    size = 0
    try:
        with r, open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                size += len(chunk)
    except (socket.timeout, TimeoutError, OSError, http.client.HTTPException) as e:
        try:
            part.unlink()
        except OSError:
            pass
        raise ApiError(0, "{}: {}".format(type(e).__name__, e), url)
    os.replace(str(part), str(target))
    return size


def pull(cfg, rec):
    """Every file of the pod's output folder that is not here yet, into outputs/<pod name>/, subfolders kept: the whole
    folder through JupyterLab, then whatever ComfyUI's history lists that JupyterLab did not (JupyterLab off)."""
    pod_id = rec["id"]
    dest = OUTPUTS / rec.get("name", pod_id)
    comfy = comfy_alive(pod_id)
    token = jupyter_token(proxy_log(pod_id)) if comfy else None
    count, listed = 0, set()

    def got(target, size):
        say("  {}  ({:.1f} MB)".format(target.relative_to(OUTPUTS), size / 1e6))

    root = None
    if token:
        root, files = jupyter_files(pod_id, token)
        if root:
            base = pod_url(pod_id, JUPYTER_PORT)
            for rel, size in files:
                listed.add(rel)
                target = dest.joinpath(*rel.split("/"))
                if target.exists() and (size is None or target.stat().st_size == size):
                    continue
                try:
                    n = fetch_to(base + "/files/" + urllib.parse.quote(root + "/" + rel), target,
                                 headers={"Authorization": "token " + token})
                except ApiError as e:
                    say("  skipped {} ({})".format(rel, e.code or e.detail[:60]))
                    continue
                count += 1
                got(target, n)
    if not comfy:
        say("ComfyUI does not answer on the pod: nothing to pull through it.")
        return count
    base = pod_url(pod_id)
    try:
        history = request("GET", base + "/history", timeout=60)
    except ApiError as e:
        say("could not read the history: {}".format(e))
        history = None
    for sub, name in history_files(history):
        rel = safe_rel((sub + "/" + name) if sub else name)
        if not rel or rel in listed:
            continue
        target = dest.joinpath(*rel.split("/"))
        if target.exists():
            continue
        q = urllib.parse.urlencode({"filename": name, "subfolder": sub, "type": "output"})
        try:
            n = fetch_to(base + "/view?" + q, target)
        except ApiError as e:
            say("  skipped {} ({})".format(name, e.code))
            continue
        count += 1
        got(target, n)
    if not root:
        say("JupyterLab did not answer: only the files in ComfyUI's history came back. A file saved before a ComfyUI "
            "restart, or by a node that reports nothing, stays on the pod: fetch it from JupyterLab's file browser.")
    return count


def cmd_down(args):
    cfg = load_config()
    if args.all:
        targets = list((cfg.get("pods") or {}).items())
        if not targets:
            die("no pod recorded here. `python pod.py list` shows the account's pods.")
    else:
        targets = [which_pod(cfg, args.profile)]
    if not args.yes:
        names = ", ".join(r.get("name") or r["id"] for _, r in targets)
        ans = input("Terminate {} now? The disk is erased, billing stops. [Y/n] ".format(names)).strip().lower()
        if ans not in ("", "y", "yes"):
            say("Kept running. Remember: a pod bills until terminated.")
            return
    for tag, rec in targets:
        if not args.no_pull:
            n = pull(cfg, rec)
            say("{} file(s) pulled into {}".format(n, OUTPUTS / rec.get("name", rec["id"])))
        try:
            spent = pod_spent(get_pod(cfg, rec["id"]), rec)
        except ApiError:
            spent = None
        try:
            rp("DELETE", "/pods/" + rec["id"], cfg)          # DELETE /pods/{id}: terminate, 204 no body
            if spent:
                say("Terminated: {}. It ran {} at {} per hour: about {} spent.".format(
                    rec.get("name"), fmt_duration(spent[0]), fmt_money(spent[2]), fmt_money(spent[1])))
            else:
                say("Terminated: {}".format(rec.get("name")))
        except ApiError as e:
            if e.code == 404:
                say("{}: already gone from Runpod.".format(rec.get("name")))
            else:
                die("{}\n{}".format(e, e.hint()))
        remember_pod(cfg, tag, None)
    try:
        remaining = list_pods(cfg, quiet=True)
    except ApiError:
        remaining = None
    if remaining is not None:
        running = [p for p in remaining if pod_status(p) not in FINAL]
        if running:
            total = sum(pod_price(p) or 0 for p in running)
            say("Attention: {} other pod(s) still running, {} per hour in total:".format(len(running), fmt_money(total)))
            for p in running:
                price = pod_price(p)
                say("  - {} ({}{})".format(p.get("name") or p.get("id"), pod_status(p),
                                           ", {} per hour".format(fmt_money(price)) if price else ""))
            say("`python pod.py down <name>` terminates one of them.")
        else:
            say("No pod running in your account. Nothing bills.")
    balance, _ = account_money(cfg)
    if balance is not None:
        say("Credit left on Runpod: {} (the last minutes may still be deducted).".format(fmt_money(balance)))


def cmd_list(args):
    cfg = load_config()
    gone = forget_gone(cfg, list_pods(cfg))
    if gone:
        say("Forgot {} record(s) of pods no longer on Runpod: {}.".format(len(gone), ", ".join(gone)))


def cmd_ssh(args):
    cfg = load_config()
    _, rec = which_pod(cfg, args.profile)
    pod = get_pod(cfg, rec["id"])
    kind, host, port, user = pod_ssh(pod)
    if not kind:
        die("no SSH address yet (the pod is {}). Use JupyterLab's terminal: {}".format(
            pod_status(pod), pod_url(rec["id"], JUPYTER_PORT)))
    if kind == "proxy":
        say("(through Runpod's SSH proxy: a shell only, no scp or rsync; the pod has no public port for 22/tcp)")
    cmd = ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new"]
    if cfg.get("ssh_key"):
        cmd += ["-i", cfg["ssh_key"]]
    cmd.append("{}@{}".format(user, host))
    if args.command:
        cmd += args.command
    say(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


def cmd_doctor(args):
    cfg = load_config()
    say("ainvfx-runpod {} · Python {} on {}".format(VERSION, sys.version.split()[0], platform.platform()))
    say("  ssh          {}".format(shutil.which("ssh") or "absent (JupyterLab replaces it)"))
    say("  config       {} ({})".format(CONFIG, "present" if CONFIG.exists() else "absent: run `setup`"))
    region, country = cfg.get("region") or local_region(), cfg.get("country") or ""
    say("  nearest      {}{}{}".format(REGION_NAMES.get(region, region), ", " + country + " first" if country else "",
                                        "" if cfg.get("region") else " (guessed from the clock: {})".format(local_zone() or "offset")))
    say("  template     {}".format(cfg.get("template_id") or TEMPLATE_ID or "none: the script describes the pod itself"))
    say("  settings     {}{}".format(SETTINGS, "" if SETTINGS.exists() else " (absent: the bootstrap's defaults apply)"))
    say("  yours        {}{}".format(SETTINGS_LOCAL, "" if SETTINGS_LOCAL.exists() else " (absent: nothing replaced)"))
    local = read_env_file(SETTINGS_LOCAL)
    for k, v in load_settings().items():
        say("    {:<22} {}{}".format(k, v, "   (settings.local.env)" if k in local else ""))
    say("  image        {} (used when no template is set)".format(IMAGE))
    if cfg.get("api_key") or os.environ.get("RUNPOD_API_KEY"):
        try:
            pods = list_pods(cfg, quiet=True)
            say("  API          ok, {} pod(s) in the account".format(len(pods)))
        except ApiError as e:
            say("  API          FAILED: {} · {}".format(e, e.hint()))
        found = secret_exists(cfg)
        say("  HF secret    {}".format("present" if found else "absent (setup --hf-token)" if found is False
                                       else "unknown: the key cannot list secrets"))
    for tag, rec in (cfg.get("pods") or {}).items():
        say("  pod `{}`: {} ({})".format(tag, rec.get("name"), rec.get("id")))


def main():
    ap = argparse.ArgumentParser(description="Runpod pod remote for the AInVFX bootcamp.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="API key, region, Hugging Face secret, SSH key")
    p.add_argument("--key", help="the Runpod API key (otherwise asked, hidden)")
    p.add_argument("--region", choices=["EU", "NA"], help="EU or NA: the data centers to try first")
    p.add_argument("--country", help="two letters (CA, US, FR, CZ, RO, NL, SE, IS, NO): tried before the rest of the region")
    p.add_argument("--template", help="a Runpod template id to create pods from")
    p.add_argument("--hf-token", action="store_true", help="store a Hugging Face token as the Runpod secret")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("up", help="create a pod")
    p.add_argument("profile", nargs="?", default="image",
                   help="which models the pod downloads: image, 3d, video, train, several joined by + (image+3d, "
                        "image+video, video+train), or all (default image)")
    p.add_argument("--name", help="a name for this pod, to run several pods of one profile (`up train --name jar-lora`); "
                                  "the other commands then take that name")
    p.add_argument("--gpu", action="append", metavar="NAME",
                   help='a GPU from the list `up` shows, by its name ("RTX PRO 4500 SE") or its full id; only that GPU '
                        "is tried. Repeat it to accept several, tried in the order given: --gpu B200 --gpu B300")
    p.add_argument("--max-price", type=float, help="the most per hour `up` takes on its own when the profile's GPU "
                                                   "has no stock (default {:.2f} USD)".format(MAX_PRICE))
    p.add_argument("--disk", type=int, help="container disk in GB")
    p.add_argument("--region", choices=["EU", "NA"])
    p.add_argument("--community", action="store_true",
                   help="also consider Community Cloud (machines owned by third parties, cheaper, the host picked by "
                        "Runpod); off by default since 0.8.0")
    p.add_argument("--set", action="append", metavar="AINVFX_KEY=VALUE",
                   help="replace one settings.env value for this pod (repeatable), for example "
                        "--set AINVFX_COMFY_TAG=master; lasting changes go in settings.local.env")
    p.add_argument("--models-url", metavar="URL",
                   help="your own models.json for this pod: a raw link, or a Google Drive, Dropbox or GitHub share link "
                        "(the same as --set AINVFX_MODELS_URL=URL)")
    p.add_argument("--no-selftest", dest="selftest", action="store_false", help="skip the test image at the end of the install")
    p.add_argument("--retry", action="store_true",
                   help="when nothing is free, read the catalog again every {} s (AINVFX_RETRY_EVERY) and create the "
                        "first GPU free, by the same rules: one pod. The question comes once, before the wait (-y skips "
                        "it); Ctrl+C stops the wait".format(RETRY_EVERY))
    p.add_argument("--wait", type=int, default=25, help="minutes to follow the log (default 25)")
    p.add_argument("-y", "--yes", action="store_true", help="no confirmation prompt")
    p.set_defaults(func=cmd_up)

    POD_REF = "which pod: the profile, the name given to `up --name`, the pod's name or its id (optional with one pod)"
    p = sub.add_parser("logs", help="follow the log until READY")
    p.add_argument("profile", nargs="?", metavar="pod", help=POD_REF)
    p.add_argument("--wait", type=int, default=25, help="minutes (default 25)")
    p.set_defaults(func=cmd_logs)

    for name, fn, hlp in (("status", cmd_status, "status, GPU, cost, address, log"),
                          ("open", cmd_open, "open ComfyUI in the browser"),
                          ("pull", cmd_pull, "download the outputs")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("profile", nargs="?", metavar="pod", help=POD_REF)
        p.set_defaults(func=fn)

    p = sub.add_parser("push", help="upload files into the pod's input folder")
    p.add_argument("profile", nargs="?", metavar="pod", help=POD_REF)
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("down", help="pull, then terminate")
    p.add_argument("profile", nargs="?", metavar="pod", help=POD_REF)
    p.add_argument("--all", action="store_true", help="every pod recorded here")
    p.add_argument("--no-pull", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(func=cmd_down)

    p = sub.add_parser("ssh", help="a terminal on the pod")
    p.add_argument("profile", nargs="?", metavar="pod", help=POD_REF)
    p.add_argument("command", nargs="*")
    p.set_defaults(func=cmd_ssh)

    sub.add_parser("list", help="every pod of the account, and which ones are recorded here").set_defaults(func=cmd_list)
    sub.add_parser("doctor", help="check this machine and the configuration").set_defaults(func=cmd_doctor)

    args = ap.parse_args()
    # `push image file.png` and `push file.png` both work: a first argument that is a file is a file
    if args.cmd == "push" and args.profile and Path(args.profile).expanduser().exists():
        args.files.insert(0, args.profile)
        args.profile = None
    try:
        args.func(args)
    except ApiError as e:
        die("{}\n{}".format(e, e.hint()))
    except KeyboardInterrupt:
        say("\nStopped.")
        sys.exit(130)
    except Exception as e:                    # never a bare traceback: the pods keep running and billing
        if os.environ.get("AINVFX_DEBUG"):
            raise
        die("unexpected error: {}: {}\nYour pods keep running: `python pod.py list` shows them, `python pod.py down` "
            "terminates one. Run the command again with AINVFX_DEBUG=1 for the full trace, and report it at "
            "https://github.com/AInVFX/ainvfx-runpod/issues".format(type(e).__name__, str(e)[:300]))


if __name__ == "__main__":
    main()
