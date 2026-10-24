#!/usr/bin/env python3
"""ainvfx-runpod · pod.py: create, use and terminate a Runpod GPU pod running ComfyUI.

One file, Python 3.8 or newer, standard library only. Windows, macOS and Linux.

  python pod.py setup              once: Runpod API key, nearest data centers, SSH key (optional)
  python pod.py setup --hf-token   store your Hugging Face token as a Runpod secret (gated models)
  python pod.py up image           create a pod; profiles: image, 3d, video, train, joined by + (image+3d), or all
  python pod.py status [pod]       GPU, data center, cost so far, ComfyUI address, last log lines
  python pod.py logs [pod]         follow the pod's log until READY
  python pod.py open [pod]         open ComfyUI in your browser
  python pod.py push [pod] FILES   copy files into the pod's input folder
  python pod.py pull [pod]         download the pod's output folder into outputs/<pod name>/
  python pod.py down [pod]         pull, then terminate (billing stops); --all for every pod recorded here
  python pod.py ssh [pod]          a terminal on the pod
  python pod.py list               every pod of your account
  python pod.py doctor             check this machine and the configuration

Useful `up` options: --gpu NAME (repeatable, tried in order), --retry (wait until a GPU is free),
--community (also Community Cloud), --name NAME (a second pod), --models-url URL, --set AINVFX_KEY=VALUE.

[pod] is the profile, the --name given to `up`, the pod's name or its id, matched exactly (`image`
never selects an `image+3d` pod); it can be left out when one pod is recorded. Every pod choice lives
in settings.env; your own values go in settings.local.env.
Full guide: README.md, or https://github.com/AInVFX/ainvfx-runpod
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

VERSION = "1.1.2"
REPO_RAW = "https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main"
DEFAULT_MODELS_URL = REPO_RAW + "/models.json"
TEMPLATE_ID = "4i789znkrd"   # the public template "AInVFX bootcamp ComfyUI": image, ports, start command
MIN_CUDA = "13.0"            # host driver 580 or newer: the int8 kernels of the course models need it
COMFY_PORT, JUPYTER_PORT = 8188, 8888
HF_SECRET = "huggingface_token"                        # the Runpod secret holding the Hugging Face token
HF_SECRET_REF = "{{ RUNPOD_SECRET_%s }}" % HF_SECRET   # Runpod puts the value in when the pod boots
FINAL = ("EXITED", "ERROR", "TERMINATED")
WINDOWS = platform.system() == "Windows"

HERE = Path(__file__).resolve().parent
CONFIG_DIR = Path.home() / ".ainvfx-runpod"
CONFIG = CONFIG_DIR / "config.json"


def env_num(name, default, kind=int):
    return kind(os.environ.get(name, default))


# Overridable through the environment (the tests point them at a fake API and shorten every wait).
API = os.environ.get("AINVFX_API_BASE", "https://api.runpod.io/v2")
GRAPHQL = os.environ.get("AINVFX_GRAPHQL", "https://api.runpod.io/graphql")
PROXY_FMT = os.environ.get("AINVFX_PROXY_FMT", "https://{id}-{port}.proxy.runpod.net")
OUTPUTS = Path(os.environ.get("AINVFX_OUTPUTS") or HERE / "outputs")
SETTINGS = Path(os.environ.get("AINVFX_SETTINGS") or HERE / "settings.env")
SETTINGS_LOCAL = Path(os.environ.get("AINVFX_SETTINGS_LOCAL") or SETTINGS.with_name("settings.local.env"))
MAX_PRICE = env_num("AINVFX_MAX_PRICE", "2.5", float)      # USD per hour: the most `up` picks on its own
MAINT_HOURS = env_num("AINVFX_MAINT_HOURS", "24", float)   # a host with maintenance this close is refused
HOST_WAIT = env_num("AINVFX_HOST_WAIT", "40")              # seconds to wait for the API to name the host
RETRY_EVERY = max(1, env_num("AINVFX_RETRY_EVERY", "15"))  # `up --retry`: seconds between catalog reads
RETRY_NOTE = env_num("AINVFX_RETRY_NOTE", "60")            # seconds between two "still waiting" lines
# Runpod's live log stream has been seen staying open and silent while the pod wrote its last lines, so
# the stream is reopened every STREAM_MAX_AGE seconds and the pod's own copy of the log is read every PROXY_POLL.
STREAM_MAX_AGE = env_num("AINVFX_STREAM_MAX_AGE", "60")
STREAM_IDLE = env_num("AINVFX_STREAM_IDLE", "30")          # seconds without a byte before the stream is reopened
PROXY_POLL = env_num("AINVFX_PROXY_POLL", "15")
STATUS_POLL = env_num("AINVFX_STATUS_POLL", "45")          # seconds between two reads of the pod's status
PULL_SUMMARY = env_num("AINVFX_PULL_SUMMARY", "15")        # seconds between two image-download summaries
HEARTBEAT = env_num("AINVFX_HEARTBEAT", "60")              # seconds of silence before a "waiting" line
STALL_MINUTES = env_num("AINVFX_STALL_MINUTES", "8", float)  # no container this long after creation: warn
RATE_RETRIES = 5   # a 429 from the Runpod API is retried after Retry-After, or 2, 4, 8, 16, 32 s

# Each profile names its preferred GPU and the least VRAM it needs; a combination takes the largest.
PROFILES = {
    "image": dict(vram=32, gpu="NVIDIA GeForce RTX 5090"),
    "3d":    dict(vram=32, gpu="NVIDIA GeForce RTX 5090"),
    "video": dict(vram=96, gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition"),
    "train": dict(vram=96, gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition"),
}
PROFILE_ORDER = ["image", "3d", "video", "train"]
# `up` picks on its own only a Blackwell GPU or newer, and never a 1g MIG slice (the smallest share of a
# bigger GPU: one ran the self-test in 111 s against 16 s on an RTX 5090). `--gpu` still takes any GPU.
NEWER_GPU = re.compile(r"blackwell|rubin|\brtx [5-9]0[5-9]0\b|\bg?b[1-9]00\b|\bv?r[1-9]00\b", re.I)
SMALL_SLICE = re.compile(r"\bMIG 1g\.", re.I)
REGION_NAMES = {"EU": "Europe", "NA": "North America"}
CANADA_ZONES = ("Toronto", "Montreal", "Vancouver", "Edmonton", "Winnipeg", "Halifax", "Regina", "St_Johns",
                "Moncton", "Whitehorse", "Yellowknife", "Iqaluit", "Calgary")
EUROPE_CITY_COUNTRY = {"Paris": "FR", "Prague": "CZ", "Bucharest": "RO", "Amsterdam": "NL", "Stockholm": "SE",
                       "Reykjavik": "IS", "Oslo": "NO"}
# bootstrap.sh writes its log inside ComfyUI's input folder, so the pod also serves it through its proxy.
LOG_VIEW = "/view?filename=bootstrap.log&subfolder=ainvfx&type=input"
OUTPUT_DIRS = ("workspace/ComfyUI/output", "ComfyUI/output")   # the output folder seen from JupyterLab's root
SKIP_NAMES = {"_output_images_will_be_put_here"}               # the empty placeholder ComfyUI creates


# ----------------------------------------------------------------------------- small helpers
def die(msg, code=1):
    print("\nERROR: " + msg, file=sys.stderr)
    sys.exit(code)


def say(msg=""):
    print(msg, flush=True)


def confirm(question, yes=False):
    return yes or input(question).strip().lower() in ("", "y", "yes")


def fmt_money(x):
    return "{:.2f} USD".format(x)


def fmt_duration(hours):
    minutes = int(round(hours * 60))
    if minutes < 1:
        return "under a minute"
    h, m = divmod(minutes, 60)
    return "{} h {:02d} min".format(h, m) if h else "{} min".format(m)


def fmt_since(t):
    """'3 min' or '40 s' from an aware datetime to now."""
    if not t:
        return "?"
    secs = int((datetime.now(timezone.utc) - t).total_seconds())
    return "{} min".format(secs // 60) if secs >= 60 else "{} s".format(secs)


def parse_time(s):
    """An RFC 3339 timestamp ('2026-10-07T14:50:00Z') as an aware datetime, or None."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def to_time(value):
    """An API time (RFC 3339 text, or epoch seconds or milliseconds) as an aware datetime, or None."""
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)) or re.fullmatch(r"\d{9,13}(\.\d+)?", str(value)):
        x = float(value)
        return datetime.fromtimestamp(x / 1000 if x > 1e11 else x, tz=timezone.utc)
    return parse_time(str(value))


def ainvfx_lines(lines):
    return [l for l in lines if "[AINVFX]" in l]


# ----------------------------------------------------------------------------- settings and models list
def read_env_file(path):
    """The AINVFX_* variables of one KEY=value file: # comments and empty values dropped, quotes removed."""
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        key, sep, value = raw.strip().partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key.startswith("#"):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.startswith("AINVFX_") and value:
            values[key] = value
    return values


def load_settings():
    """settings.env, then settings.local.env: a key in the local file replaces the same key."""
    values = read_env_file(SETTINGS)
    values.update(read_env_file(SETTINGS_LOCAL))
    return values


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
    """A share link turned into a link to the file itself: Google Drive, Dropbox, a file page on GitHub.
    bootstrap.sh does the same conversion on the pod."""
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
    """disk, vram, gpu, the size of the models (GB) and the label (`image+3d`) for one or several profiles.
    The disk is 115% of the models plus 30 GB for the environment, rounded up to 10, never under 100."""
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


# ----------------------------------------------------------------------------- configuration
def load_config():
    if CONFIG.exists():
        try:
            cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            die("{} is unreadable: delete it and run `setup` again".format(CONFIG))
        first = {}                            # one record per pod: 1.1.0 could record a pod twice; the oldest stays
        for tag, rec in (cfg.get("pods") or {}).items():
            first.setdefault(str(rec.get("id")), (tag, rec))
        cfg["pods"] = dict(first.values())
        return cfg
    return {"api_key": "", "region": "", "template_id": "", "ssh_key": "", "hf_secret": False, "pods": {}}


def save_config(cfg):
    """Written to a private temporary file, then moved into place: never readable by others, never half written."""
    CONFIG_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = CONFIG.with_name("{}.{}.tmp".format(CONFIG.name, os.getpid()))
    try:
        with os.fdopen(os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as f:
            f.write(json.dumps(cfg, indent=2) + "\n")
        os.replace(str(tmp), str(CONFIG))
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    restrict(CONFIG)


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


def remember_pod(cfg, tag, rec):
    """Write one pod record, or remove it (rec None). The file is read again just before the write, so two
    commands running at once (two terminals, two `up`) keep each other's records."""
    fresh = load_config()
    pods = fresh.setdefault("pods", {})
    if rec is None:
        pods.pop(tag, None)
    else:
        pods[tag] = rec
    save_config(fresh)
    cfg["pods"] = pods


def api_key(cfg):
    key = os.environ.get("RUNPOD_API_KEY") or cfg.get("api_key")
    if not key:
        die("no Runpod API key. First:  python pod.py setup")
    return key


# ----------------------------------------------------------------------------- HTTP
class ApiError(Exception):
    """Every HTTP or network failure. Code 0 means no HTTP answer (no network, refused, timed out)."""
    def __init__(self, code, text, url):
        self.code, self.detail, self.errors, self.retry_after = code, text, None, None
        try:                       # Runpod errors are application/problem+json: title, status, detail, errors
            prob = json.loads(text)
            self.detail = prob.get("detail") or prob.get("title") or text
            self.errors = prob.get("errors")
        except (ValueError, AttributeError):
            pass
        super().__init__("HTTP {} on {}: {}".format(code, url, self.detail))

    @property
    def timed_out(self):
        return "timed out" in (self.detail or "").lower()

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


def request(method, url, key=None, body=None, timeout=60, raw=False, headers=None, stream=False, data=None):
    """One HTTP request. Returns parsed JSON, bytes (raw=True), or the open response (stream=True).
    `body` is sent as JSON; `data` as given, with the Content-Type of `headers`. A 429 from the Runpod
    API is retried RATE_RETRIES times; a pod's proxy and other hosts are not retried."""
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


def _request_once(method, url, key, body, timeout, raw, headers, stream, data):
    hdrs = {"Accept": "application/json", "User-Agent": "ainvfx-runpod/" + VERSION}
    if key:
        hdrs["Authorization"] = "Bearer " + key
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    hdrs.update(headers or {})
    req = urllib.request.Request(url, data=data, method=method, headers=hdrs)
    try:
        r = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        err = ApiError(e.code, e.read().decode("utf-8", "replace")[:1000], url)
        err.retry_after = e.headers.get("Retry-After")
        raise err
    except urllib.error.URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            raise ApiError(0, "timed out after {} s (connecting)".format(timeout), url)
        raise ApiError(0, str(e.reason), url)
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


def rp(method, path, cfg, body=None, timeout=60, headers=None, stream=False):
    """A call to the Runpod REST API v2 (https://api.runpod.io/v2/openapi.json)."""
    return request(method, API + path, api_key(cfg), body, timeout, headers=headers, stream=stream)


def graphql(cfg, query):
    """The `data` of a GraphQL answer, or None when it cannot be read. Used only for what REST v2 lacks:
    the pod's host (maintenance) and the account's credit."""
    try:
        data = request("POST", GRAPHQL, api_key(cfg), body={"query": query}, timeout=20)
    except ApiError:
        return None
    if not isinstance(data, dict) or data.get("errors") or "data" not in data:
        return None
    return data.get("data") or {}


def pod_url(pod_id, port=COMFY_PORT):
    return PROXY_FMT.format(id=pod_id, port=port)


def proxy_log(pod_id):
    """The [AINVFX] lines of the bootstrap log, read through the pod's own ComfyUI; [] when not readable."""
    try:
        data = request("GET", pod_url(pod_id) + LOG_VIEW, timeout=10, raw=True)
    except ApiError:
        return []
    return ainvfx_lines(data.decode("utf-8", "replace").splitlines())


def comfy_alive(pod_id):
    try:
        request("GET", pod_url(pod_id) + "/system_stats", timeout=8)
        return True
    except ApiError:
        return False


# ----------------------------------------------------------------------------- where the person is
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
        return Path("/etc/timezone").read_text(encoding="utf-8").strip()
    except Exception:
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


def dc_order(dc_id, region, country):
    """Sort key: the person's country first, then the rest of their region, then the other main region."""
    same_region = region_of(dc_id) == region
    return (0 if country and same_region and country_of(dc_id) == country else
            1 if same_region else 2 if region_of(dc_id) in REGION_NAMES else 3)


# ----------------------------------------------------------------------------- pods
def get_pod(cfg, pod_id):
    return rp("GET", "/pods/" + pod_id, cfg)


def list_pods(cfg):
    pods, cursor = [], None
    while True:
        q = "?" + urllib.parse.urlencode({"cursor": cursor}) if cursor else ""
        data = rp("GET", "/pods" + q, cfg)
        pods += data.get("pods") or []
        cursor = (data.get("pagination") or {}).get("nextCursor")
        if not cursor:
            return pods


def pod_status(pod):
    return str(pod.get("status") or "?").upper()


def pod_price(pod):
    try:
        return float(pod.get("cost"))
    except (TypeError, ValueError):
        return None


def pod_summary(pod):
    """(GPU, data center, price per hour) of a pod, '?' when not reported."""
    return str((pod.get("gpu") or {}).get("id") or "?"), str(pod.get("dataCenterId") or "?"), pod_price(pod)


def pod_spent(pod, rec=None):
    """(hours, USD, price per hour) since the pod was created: an estimate, since Runpod bills from creation.
    None when the price or the creation time is unknown."""
    price = pod_price(pod)
    created = parse_time(pod.get("createdAt") or "") or parse_time((rec or {}).get("created", ""))
    if not price or not created:
        return None
    hours = max(0.0, (datetime.now(timezone.utc) - created).total_seconds() / 3600)
    return hours, hours * price, price


def pod_ssh(pod):
    """(kind, host, port, user): 'direct' (full ssh, scp, rsync) if 22/tcp got a public port, else 'proxy' (shell only)."""
    ssh = pod.get("ssh") or {}
    for kind in ("direct", "proxy"):
        c = ssh.get(kind)
        if c and c.get("host") and c.get("port") and c.get("username"):
            return kind, c["host"], int(c["port"]), c["username"]
    return None, None, None, None


def outputs_of(rec):
    """outputs/<pod name>/, the pod name made safe as a folder name."""
    folder = re.sub(r"[\\/:]+", "-", rec.get("name") or "").strip(". ")
    return OUTPUTS / (folder or rec["id"])


def forget_gone(cfg, account):
    """Remove the records of pods no longer in the account (terminated from the console, from another machine,
    or by Runpod). A stopped pod is still in the account and keeps its record. Returns the names forgotten."""
    alive = {str(p.get("id")) for p in account}
    gone = [(t, r) for t, r in list((cfg.get("pods") or {}).items()) if str(r.get("id")) not in alive]
    for t, _ in gone:
        remember_pod(cfg, t, None)
    return [r.get("name") or t for t, r in gone]


def is_slug(text):
    """True for the profiles part of a pod name, as `up` writes it: image, image-3d, image-3d-video..."""
    parts = text.split("-")
    return parts == [p for p in PROFILE_ORDER if p in parts]


def named_for(name, tag):
    """True when `up` names a pod `name` and records it as `tag`: ainvfx-<profiles>-<MMDD-HHMM> when the tag is
    the profiles, ainvfx-<profiles>-<tag> for a --name. Exact: `image` never matches the pod of image+3d+video."""
    if is_slug(tag) and re.fullmatch(r"ainvfx-{}-\d{{4}}-\d{{4}}".format(re.escape(tag)), name):
        return True
    head, tail = "ainvfx-", "-" + tag
    return name.startswith(head) and name.endswith(tail) and is_slug(name[len(head):-len(tail)])


def which_pod(cfg, ref):
    """The pod a command targets: (tag, record). `ref` names it exactly: its tag here (the profiles given to `up`,
    or its --name), the pod's name or its id; nothing when one pod is recorded. Records of pods gone from the
    account are forgotten first. A pod found in the account but not here (created from the console or another
    machine) is attached: recorded here. A pod already recorded here is never recorded a second time."""
    try:
        account = list_pods(cfg)
    except ApiError:
        account = None                        # Runpod unreachable: the records as they are
    if account is not None:                   # a record of a gone pod would end in an HTTP 404
        gone = forget_gone(cfg, account)
        if gone:
            say("forgot {} pod(s) no longer on Runpod: {}".format(len(gone), ", ".join(gone)))
    pods = cfg.setdefault("pods", {})
    if not ref:
        if len(pods) == 1:
            return next(iter(pods.items()))
        if not pods:
            die("no pod recorded here. `python pod.py up image` creates one; "
                "`python pod.py list` shows the account's pods, then `python pod.py status <name or id>` attaches to one.")
        die("several pods recorded, name one: {}".format(
            "  ".join("{} ({})".format(t, r.get("name")) for t, r in pods.items())))
    ref = ref.strip().replace("+", "-")       # `down image+3d` and `down image-3d` name the same pod
    if ref in pods:
        return ref, pods[ref]
    for tag, rec in pods.items():
        if ref in (rec.get("id"), rec.get("name")):
            return tag, rec
    if account is None:
        account = list_pods(cfg)              # asked again: an error now is the command's error
    hits = []
    for p in account:
        pid, pname = str(p.get("id") or ""), str(p.get("name") or "")
        if ref in (pid, pname) or named_for(pname, ref):
            hits.append((pid, pname, p))
    if len(hits) > 1:
        die("several pods in your account match '{}': {}. Name one: `python pod.py status <name or id>`.".format(
            ref, "  ".join("{} ({})".format(pname, pid) for pid, pname, _ in hits)))
    if not hits:
        die("no pod '{}' recorded here or in your account. `python pod.py list` shows them.".format(ref))
    pid, pname, p = hits[0]
    for tag, rec in pods.items():             # already recorded under another tag: that record, never a second one
        if str(rec.get("id")) == pid:
            say("using {} ({}), recorded here as `{}`".format(rec.get("name") or pid, pid, tag))
            return tag, rec
    tag = ref if named_for(pname, ref) else (pname or pid)
    rec = {"id": pid, "name": pname or pid, "profile": next((k for k in PROFILES if "-{}-".format(k) in pname), ""),
           "created": p.get("createdAt") or ""}
    remember_pod(cfg, tag, rec)
    say("attached to {} ({}) as `{}`".format(rec["name"], pid, tag))
    return tag, rec


# ----------------------------------------------------------------------------- the GPU catalog
def read_catalog(cfg, cloud):
    """Every GPU type with its pod stock on `cloud` (SECURE or COMMUNITY) for hosts on CUDA MIN_CUDA or newer:
    id, name, memory (GB), price per cloud, availability, and for Secure the data centers with stock."""
    q = urllib.parse.urlencode({"include": "AVAILABILITY", "product": "POD", "minCudaVersion": MIN_CUDA, "cloud": cloud})
    data = rp("GET", "/catalog/gpus?" + q, cfg)
    return [g for g in (data.get("gpus") or []) if isinstance(g, dict) and g.get("id")]


def read_catalogs(cfg, community, fatal=True):
    """{cloud: GPU types} for each cloud considered. When the Secure catalog cannot be read, `up` stops (fatal),
    or the ApiError goes up to the --retry wait, which reads again at its next try."""
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


def in_stock(level):
    return str(level or "").upper() not in ("", "NONE")


def newer_gpu(gpu):
    """True for a Blackwell GPU (RTX 50 series, RTX PRO Blackwell, B200, B300 and their MIG slices) or a newer one."""
    return bool(NEWER_GPU.search("{} {}".format(gpu.get("id", ""), gpu.get("name", ""))))


def make_offers(catalogs, region, country):
    """One offer per GPU type and cloud with stock, cheapest first (Secure first at the same price):
    dict(id, name, vram, cloud, price, dcs, newer, slice). A Secure offer lists its data centers with stock,
    the person's country first; Secure stock with no data center named is skipped, so a Secure pod always
    names its data center. A Community offer has none: Runpod picks the host."""
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
    """True when `up` may pick the offer on its own: enough VRAM, Blackwell or newer, not a 1g MIG slice."""
    return offer["vram"] >= vram and offer["newer"] and not offer["slice"]


def find_gpu(catalog, ref):
    """The catalog entry `ref` names: its full id or short name ("RTX PRO 4500 SE"), in any case, or a part of
    either that matches one GPU type only."""
    r = ref.strip().lower()
    for key in ("id", "name"):
        hits = [g for g in catalog if str(g.get(key, "")).lower() == r]
        if hits:
            return hits[0]
    hits = {}                                  # the same GPU type comes once per cloud: keep the first
    for g in catalog:
        if r in str(g.get("id", "")).lower() or r in str(g.get("name", "")).lower():
            hits.setdefault(g["id"], g)
    if len(hits) == 1:
        return next(iter(hits.values()))
    if not hits:
        die("no GPU type named \"{}\" in the Runpod catalog. `python pod.py up` lists the GPUs in stock. "
            "For several GPUs, repeat --gpu: --gpu B200 --gpu B300.".format(ref))
    die("\"{}\" matches several GPU types: {}. Give one of these names to --gpu.".format(
        ref, ", ".join('"{}"'.format(g.get("name") or g["id"]) for g in list(hits.values())[:12])))


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


def gpu_report(want, offers, region, community):
    """Four lines on one GPU type: its availability on Secure Cloud, where it is in stock, its prices."""
    secure = next((o for o in offers if o["id"] == want["id"] and o["cloud"] == "SECURE"), None)
    dcs = secure["dcs"] if secure else []
    price = want.get("price") or {}
    p_secure, p_comm = price.get("secure"), price.get("community")
    say("\n{}: availability {} on Secure Cloud with hosts on CUDA {} or newer".format(
        want.get("name") or want["id"], str(want.get("availability", "?")).upper(), MIN_CUDA))
    say("   in stock in {}: {}".format(REGION_NAMES.get(region, region),
                                       ", ".join(d for d in dcs if region_of(d) == region) or "none right now"))
    say("   elsewhere: {}".format(", ".join(d for d in dcs if region_of(d) != region) or "none"))
    say("   price per hour: {} Secure, {} Community{}".format(
        fmt_money(p_secure) if p_secure else "?", fmt_money(p_comm) if p_comm else "?",
        "" if community else " (not considered: add --community)"))


# ----------------------------------------------------------------------------- host, money and secrets
MACHINE_QUERY = ("query {{ pod(input: {{podId: {} }}) {{ id machine {{ podHostId dataCenterId location "
                 "maintenanceStart maintenanceEnd maintenanceNote }} }} }}")
MONEY_QUERY = "query { myself { clientBalance currentSpendPerHr } }"


def pod_machine(cfg, pod_id):
    """The pod's host: a dict, empty while no host is assigned; None when GraphQL cannot be read."""
    data = graphql(cfg, MACHINE_QUERY.format(json.dumps(pod_id)))
    if data is None:
        return None
    machine = (data.get("pod") or {}).get("machine") or {}
    return machine if any(machine.values()) else {}


def maintenance_of(machine, now=None):
    """A sentence when the host has maintenance under way or starting within MAINT_HOURS, else None.
    A window that is over, or starts later, does not matter for a pod that lives one session. A time that
    cannot be read counts as under way: refusing a host costs less than a dead pod."""
    machine = machine or {}
    start_raw, end_raw = machine.get("maintenanceStart"), machine.get("maintenanceEnd")
    if not start_raw and not end_raw:
        return None
    now = now or datetime.now(timezone.utc)
    start, end = to_time(start_raw), to_time(end_raw)
    if (end and end <= now) or (start and start > now + timedelta(hours=MAINT_HOURS)):
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
    """(machine, problem) for a pod just created. machine is None when GraphQL cannot be read, {} when no host
    is reported within HOST_WAIT seconds; problem is the maintenance sentence, or None."""
    deadline = time.time() + HOST_WAIT
    while True:
        machine = pod_machine(cfg, pod_id)
        if machine is None or machine:
            return machine, (maintenance_of(machine) if machine else None)
        if time.time() >= deadline:
            return {}, None
        time.sleep(3)


def account_money(cfg):
    """(credit left in USD, what the account spends now in USD per hour); (None, None) when unreadable."""
    me = (graphql(cfg, MONEY_QUERY) or {}).get("myself")
    if not isinstance(me, dict):
        return None, None

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return num(me.get("clientBalance")), num(me.get("currentSpendPerHr"))


def say_credit(balance, spend, price, middle, tail=""):
    """The credit line `up` prints before its question, and the warning under 3 hours."""
    line = "   Your Runpod credit: {}, about {} hours {}{} per hour{}".format(
        fmt_money(balance), int(balance // price), middle, fmt_money(price), tail)
    if spend:
        line += "; your account already spends {} per hour".format(fmt_money(spend))
    say(line + ".")
    if balance < 3 * price:
        say("   That is under 3 hours: add credit in the Runpod console (Billing), or pick a cheaper card.")


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


# ----------------------------------------------------------------------------- logs
def log_stream(cfg, pod_id, tail=200, since=None, source=None, idle=20, max_age=None):
    """Yield (event_id, source, line) from GET /pods/{id}/logs, a Server-Sent Events stream (an `id:` line, the
    event's time, and a `data:` JSON line per event). `tail` lines come first, or with `since` (RFC 3339) the
    stream resumes from that time. Stops after `idle` seconds without data, or `max_age` seconds in all cases.
    Yields (None, None, None) on keepalive lines, so the caller can do periodic work."""
    q = {"since": since} if since else {"tail": tail}
    if source:
        q["source"] = source
    opened = time.time()
    r = rp("GET", "/pods/{}/logs?{}".format(pod_id, urllib.parse.urlencode(q)), cfg, timeout=idle,
           headers={"Accept": "text/event-stream"}, stream=True)
    eid = None
    try:
        while not (max_age and time.time() - opened > max_age):
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
    """(the last `tail` lines of one source, None) or (None, ApiError); the stream is left after `seconds`."""
    try:
        return [l for _, _, l in log_stream(cfg, pod_id, tail=tail, source=source, idle=3, max_age=seconds) if l], None
    except ApiError as e:
        return None, e


def resume_point(event_id, seconds=5):
    """An RFC 3339 time a few seconds before the given event id (itself a time): a reconnect overlaps the lines
    already seen instead of skipping those written in the same second."""
    t = parse_time(event_id or "")
    if not t:
        return None
    return (t - timedelta(seconds=seconds)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Runpod's system log narrates the image download one Docker layer at a time, many lines per layer.
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


def follow_logs(cfg, pod_id, minutes=25, pod=None, created=None):
    """Print the pod's log as it appears, until READY (True), FAILED, a dead pod or the time limit (False).

    [AINVFX] lines come from bootstrap.sh through two sources: the API stream, reopened every STREAM_MAX_AGE
    seconds a few seconds back (duplicates dropped), and every PROXY_POLL seconds the copy of the log that the
    pod serves through its own ComfyUI. [RUNPOD] lines are Runpod's system log (the image download, summarised
    every PULL_SUMMARY seconds; the container start). A heartbeat follows HEARTBEAT seconds of silence, and a
    warning when no container has started STALL_MINUTES after creation. An API error is a message, never a
    traceback: the pod exists and bills."""
    deadline = time.time() + minutes * 60
    seen, printed, last_id = set(), set(), None
    created = created or parse_time((pod or {}).get("createdAt") or "")
    pull = ImagePull()
    st = {"container": False, "failed": False, "stall_told": False, "stream_told": False,
          "status": pod_status(pod) if pod else "?", "last_output": time.time(), "last_check": time.time(),
          "last_poll": time.time(), "pull_shown": 0.0, "pull_done": False}

    def out(text):
        say("  " + text)
        st["last_output"] = time.time()

    def show(line, key):
        """An [AINVFX] line, once. True when it ends the wait: READY, or FAILED."""
        if key in seen or line in printed:
            return False
        seen.add(key)
        printed.add(line)
        st["container"] = True
        out(line[line.index("[AINVFX]"):])
        st["failed"] = "[AINVFX] FAILED" in line
        return st["failed"] or "READY" in line

    def done():
        if st["failed"]:
            say("\nThe pod reported FAILED (the line above): it cannot work on that machine, and it bills until "
                "it is terminated. Terminate it and create another, which lands on another machine:\n"
                "    python pod.py down    then    python pod.py up")
            return False
        return True

    def system(line):
        text = line.strip()
        if not text:
            return
        if pull.feed(text):
            pull_summary(False)
        elif text not in printed:
            printed.add(text)
            out("[RUNPOD] " + text[:200])

    def pull_summary(final):
        if not pull.layers or st["pull_done"]:
            return
        if not st["pull_shown"] and not final and time.time() - pull.started < 3:
            return                          # let the backlog of layer lines arrive before judging
        n, finished_layers, _, _ = pull.counts()
        finished = n and finished_layers == n
        if not (final or finished or time.time() - st["pull_shown"] >= PULL_SUMMARY):
            return
        if not st["pull_shown"] and finished:
            out("[RUNPOD] the image is already on the host ({} layers present): the container starts now".format(n))
            st["pull_shown"], st["pull_done"] = time.time(), True
            return
        if not st["pull_shown"]:
            out("[RUNPOD] the host is fetching the pod's image from Docker Hub (several minutes; a host "
                "that already has it starts the container within seconds)")
        st["pull_shown"] = time.time()
        out("[RUNPOD] " + pull.summary())
        st["pull_done"] = bool(finished)

    def poll_proxy():
        st["last_poll"] = time.time()
        return any(show(line, ("proxy", line)) for line in proxy_log(pod_id))

    def pod_gone():
        """Every STATUS_POLL seconds: True when the pod has ended; warns once when no container has started."""
        if time.time() - st["last_check"] < STATUS_POLL:
            return False
        st["last_check"] = time.time()
        try:
            current = get_pod(cfg, pod_id)
        except ApiError:
            return False
        st["status"] = pod_status(current)
        if st["status"] in FINAL:
            out("the pod is {}. Read its log in the console, then `python pod.py down` and `up` again.".format(st["status"]))
            return True
        if (not st["container"] and current.get("runtime") is None and created and not st["stall_told"]
                and (datetime.now(timezone.utc) - created).total_seconds() > STALL_MINUTES * 60):
            st["stall_told"] = True
            dc = pod_summary(current)[1]
            out("WARNING: NO CONTAINER {} AFTER CREATION (status {}, {}). The host is still fetching the image, or "
                "does not respond; the console (Pods > your pod) shows its log and any maintenance notice. "
                "The pod bills meanwhile. Usual answer:  python pod.py down <pod>  then  python pod.py up  again, "
                "usually in another data center.".format(fmt_since(created), st["status"],
                                                         dc if dc != "?" else "data center not reported"))
        return False

    def tick():
        """The periodic work between two lines. None to go on, else the result of the wait."""
        if pod_gone():
            return False
        pull_summary(False)
        if time.time() - st["last_output"] >= HEARTBEAT:
            out("[RUNPOD] waiting: pod {}, {}{} since creation".format(
                st["status"], "" if st["container"] else "no container yet, ", fmt_since(created)))
        return None

    while time.time() < deadline:
        try:
            for eid, source, line in log_stream(cfg, pod_id, tail=500, since=resume_point(last_id),
                                                idle=STREAM_IDLE, max_age=STREAM_MAX_AGE):
                if line and "[AINVFX]" in line:
                    if show(line, (eid, line)):
                        pull_summary(True)
                        return done()
                elif line and source == "system":
                    system(line)
                last_id = eid or last_id
                if time.time() - st["last_poll"] > PROXY_POLL and poll_proxy():
                    return done()
                result = tick()
                if result is not None:
                    return result
                if time.time() > deadline:
                    break
        except ApiError as e:
            if not st["stream_told"]:
                st["stream_told"] = True
                if e.code == 0 and e.timed_out:
                    out("[RUNPOD] the log stream did not answer ({}): the host may not be ready; retrying".format(e.detail))
                else:
                    out("(the log cannot be read through the API from here: {}. Follow it in the console, "
                        "Pods > your pod > Logs, or wait: the pod's own copy is read as well.)".format(e.detail))
            time.sleep(min(PROXY_POLL, 10))
        if poll_proxy():
            return done()
        result = tick()
        if result is not None:
            return result
    say("Still working after {} minutes: `python pod.py status` shows where it is.".format(minutes))
    return False


# ----------------------------------------------------------------------------- creating a pod
def create_pod(cfg, body, where):
    """POST /pods for one candidate. Returns the pod, or None to try the next candidate: 401, 402, 404 and 422
    stop, 403 and 400 move on, a 5xx is retried twice (a 429 is already retried by `request`)."""
    for attempt in range(3):
        try:
            return rp("POST", "/pods", cfg, body=body, timeout=120)
        except ApiError as e:
            if e.code in (401, 402, 404, 422):
                detail = e.detail if not e.errors else "{} {}".format(e.detail, json.dumps(e.errors)[:400])
                die("{}: {}\n{}".format(e.code, detail, e.hint()))
            if e.code >= 500 and attempt < 2:
                say("   {}: Runpod error {}, retrying".format(where, e.code))
                time.sleep(5)
                continue
            if e.code == 403:
                say("   {}: not accessible with this account (403), skipped".format(where))
            else:
                say("   {}: no machine ({}: {})".format(where, e.code, e.detail[:160].replace("\n", " ")))
            return None


def try_create(cfg, candidates, name, disk, env):
    """Create one pod from the candidates, in order: for a Secure offer, every data center with stock; then the
    next offer. A pod whose host has maintenance under way, or starting within MAINT_HOURS, is terminated at
    once and the next machine tried. Returns (pod, host, stray): pod None when nothing could be created; stray
    lists the refused pods that could not be terminated."""
    stray = []
    template = cfg.get("template_id") or TEMPLATE_ID   # image, start command and ports come from the template
    for n, offer in enumerate(candidates):
        if n:
            say("   Next: {}.".format(offer_line(offer)))
        for dc in (offer["dcs"] if offer["cloud"] == "SECURE" else [None]):
            body = {"name": name, "cloud": offer["cloud"], "disk": disk, "env": env, "templateId": template,
                    "startSsh": True, "startJupyter": True,
                    "gpu": {"id": offer["id"], "count": 1, "minCudaVersion": MIN_CUDA}}
            if dc:
                body["dataCenterIds"] = [dc]
            where = "{} {}".format(offer["cloud"], dc or "(host picked by Runpod)")
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
    pod, never more. Returns (pod, host, name). Before each creation the record of `tag` is read again: if
    another `up` has created a pod under that name meanwhile, the wait stops. Ctrl+C exits with code 130."""
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
        pods = list_pods(cfg)
    except ApiError as e:
        die("the key does not work: {}\n{}".format(e, e.hint()))
    say("   ok ({} pod(s) in the account right now)".format(len(pods)))

    guess_region, guess_country = local_place()
    region = args.region or cfg.get("region") or guess_region
    country = (args.country or cfg.get("country") or (guess_country if region == guess_region else "")).upper()
    cfg["region"], cfg["country"] = region, country
    say("\n2. Nearest data centers first: {}{} (from this computer's clock: {}).".format(
        REGION_NAMES.get(region, region), ", " + country + " first" if country else "", local_zone() or "UTC offset only"))
    say("   Runpod names them by country: CA-MTL-1, US-TX-3, EU-FR-1, EUR-IS-2... `up` tries yours first,")
    say("   then the rest of the region, then the other region. Change with `setup --region EU --country FR`.")
    cfg["template_id"] = args.template or cfg.get("template_id") or TEMPLATE_ID
    say("   template: {}".format(cfg["template_id"]))

    say("\n3. Hugging Face token (for the gated files of models.json: FLUX 3 Image Klein in the image profile, LTX in video and train).")
    if args.hf_token:
        say("   The token is stored on Runpod as the secret '{}' and never on this machine. Pods created".format(HF_SECRET))
        say("   by `up` receive it as HF_TOKEN. A read token from https://huggingface.co/settings/tokens.")
        token = getpass.getpass("   Hugging Face token: ").strip()
        if token:
            store_hf_secret(cfg, token)
    else:
        found = secret_exists(cfg)
        cfg["hf_secret"] = bool(found)
        say("   secret '{}': {}. To store a token:  python pod.py setup --hf-token".format(
            HF_SECRET, "present" if found else "absent" if found is False else "unknown (the key cannot list secrets)"))

    say("\n4. SSH key (optional: only for `pod.py ssh`; everything else goes through the browser).")
    if shutil.which("ssh") and shutil.which("ssh-keygen"):
        register_ssh_key(cfg, args.yes)
    else:
        say("   no `ssh` on this machine: skipped. JupyterLab's terminal replaces it on the pod.")
    cfg.setdefault("pods", {})
    save_config(cfg)
    say("\nSaved in {} (readable by your account only).".format(CONFIG))
    say("Next:  python pod.py up image")


def store_hf_secret(cfg, token):
    """Create the Runpod secret holding the Hugging Face token, or rotate its value when it exists."""
    try:
        rp("POST", "/account/secrets", cfg,
           body={"name": HF_SECRET, "value": token, "description": "Hugging Face read token (ainvfx-runpod)"})
        say("   secret '{}' created".format(HF_SECRET))
        cfg["hf_secret"] = True
    except ApiError as e:
        if e.code != 409:
            say("   could not store the secret ({}): {}".format(e.code, e.hint() or e.detail))
            say("   Console > Account > Credentials > Secrets > name '{}', value: your token.".format(HF_SECRET))
            return
        try:                                  # 409: the name exists
            rp("PATCH", "/account/secrets/" + secrets_named(cfg)[0]["id"], cfg, body={"value": token})
            say("   secret '{}' updated".format(HF_SECRET))
            cfg["hf_secret"] = True
        except (ApiError, IndexError, KeyError) as e2:
            say("   could not update the secret: {}".format(e2))


def register_ssh_key(cfg, yes):
    """Create ~/.ssh/id_ed25519 if asked, and add its public half to the Runpod account."""
    keyfile = Path(cfg.get("ssh_key") or Path.home() / ".ssh" / "id_ed25519")
    pub = keyfile.with_name(keyfile.name + ".pub")
    if not keyfile.exists() and confirm("   No SSH key found. Create one now? [Y/n] ", yes):
        keyfile.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(keyfile), "-q",
                        "-C", "ainvfx-runpod {}".format(getpass.getuser())])
    if not pub.exists():
        return
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


def cmd_up(args):
    cfg = load_config()
    profiles = parse_profiles(args.profile)
    profile = "+".join(profiles)              # `image+3d`: the value AINVFX_PROFILE carries
    slug = profile.replace("+", "-")
    api_key(cfg)
    tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in (args.name or "")).strip("-") or slug
    old = cfg.setdefault("pods", {}).get(tag)
    if old:
        try:
            st = pod_status(get_pod(cfg, old["id"]))
            if st not in FINAL and st != "?":
                say("A pod already exists as `{}`: {} ({}). Address: {}".format(tag, old.get("name"), st, pod_url(old["id"])))
                say("Use it, terminate it first (python pod.py down {}), or create another one with `up {} --name <name>`."
                    .format(tag, profile))
                return
        except ApiError as e:
            if e.code != 404:
                die("{}\n{}".format(e, e.hint()))
        remember_pod(cfg, tag, None)

    region = args.region or cfg.get("region") or local_place()[0]
    country = (cfg.get("country") or "").upper() if not args.region or args.region == cfg.get("region") else ""

    def make_name():                          # called at creation: after a --retry wait, the name has the creation's time
        return "ainvfx-{}-{}".format(slug, tag if tag != slug else datetime.now().strftime("%m%d-%H%M"))

    env = load_settings()                     # settings.env, then settings.local.env
    for item in args.set or []:               # `--set KEY=VALUE`, for this pod only
        key, sep, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.startswith("AINVFX_") or not value:
            die("--set takes AINVFX_KEY=value (got \"{}\"). The keys are listed in settings.env.".format(item))
        env[key] = value
    if args.models_url:
        env["AINVFX_MODELS_URL"] = args.models_url.strip()
    env["AINVFX_PROFILE"] = profile
    if cfg.get("hf_secret") or secret_exists(cfg):
        env["HF_TOKEN"] = HF_SECRET_REF       # the reference, never the token itself
    if not args.selftest:
        env["AINVFX_SELFTEST"] = "0"
    models_url = env.get("AINVFX_MODELS_URL", DEFAULT_MODELS_URL)
    spec = profile_spec(profiles, read_models(models_url))   # before anything is spent
    disk = args.disk or spec["disk"]

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
    if models_url != DEFAULT_MODELS_URL:
        say("   models list: {} (your own, not the course's)".format(models_url))
    course = read_env_file(SETTINGS)
    for key in sorted(k for k in env if k.startswith("AINVFX_") and course.get(k) != env[k]
                      and k not in ("AINVFX_PROFILE", "AINVFX_MODELS_URL", "AINVFX_SELFTEST")):
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
                " picked by Runpod and can fail to start).")

    balance, spend = account_money(cfg)
    created, host = None, None
    if candidates:
        first = candidates[0]
        if balance is not None:
            say_credit(balance, spend, first["price"], "of {} at ".format(first["name"]))
        if not args.yes:
            if len(candidates) > 1:
                say("   If it is taken in the meantime, `up` tries {}, up to {} per hour.".format(
                    "the next GPU named, in the order given" if len(wants) > 1 else "the next GPUs that fit, cheapest first",
                    fmt_money(max(o["price"] for o in candidates))))
            if args.retry:
                cap = retry_cap(wants, named, args.community, max_price)
                say("   With --retry: if none can be created now, `up` reads the catalog every {} s and creates the first "
                    "one free{}, without asking again.".format(RETRY_EVERY, ", at up to {} per hour".format(fmt_money(cap)) if cap else ""))
            if not confirm("   Create a {} at {} per hour, with a {} GB disk, billed from creation? [Y/n] ".format(
                    first["name"], fmt_money(first["price"]), disk)):
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
        cap = retry_cap(wants, named, args.community, max_price)
        if balance is not None and cap:
            say_credit(balance, spend, cap, "at ", ", the highest price this wait accepts")
        if not confirm("   Wait, then create the first {} free ({}a {} GB disk, billed from creation)? [Y/n] ".format(
                wname if named else "GPU that fits", "up to {} per hour, ".format(fmt_money(cap)) if cap else "", disk),
                args.yes):
            say("Nothing created.")
            return
    if not created:
        created, host, name = wait_for_stock(cfg, args, tag, wants, named, spec, region, country, max_price, make_name,
                                             disk, env)

    pod_id = created["id"]
    created_at = datetime.now(timezone.utc)
    remember_pod(cfg, tag, {"id": pod_id, "name": name, "profile": profile, "created": created_at.isoformat()})
    say("\nWaiting for the machine (PROVISIONING, STARTING, then RUNNING)...")
    pod = wait_running(cfg, pod_id, created, profile)
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


def wait_running(cfg, pod_id, pod, profile):
    """The pod once RUNNING (15 minutes at most), with its data center and price when the API reports them."""
    def reread(p):
        try:
            return get_pod(cfg, pod_id)
        except ApiError:
            return p
    wait_until = time.time() + 15 * 60
    while time.time() < wait_until:
        pod = reread(pod)
        st = pod_status(pod)
        if st == "RUNNING":
            break
        if st in FINAL:
            die("the pod is {} before it ran. Read its log in the console, then `python pod.py down {}` and try again.".format(st, profile))
        time.sleep(8)
    for _ in range(4):                        # the data center and the price arrive a few seconds after RUNNING
        _, dc, price = pod_summary(pod)
        if dc != "?" and price:
            break
        time.sleep(5)
        pod = reread(pod)
    return pod


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
    gpu, dc, _ = pod_summary(pod)
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
    lines = proxy_log(rec["id"])              # the pod's own copy is complete; the API stream is the fallback
    err = None
    if not lines:
        lines, err = pod_logs(cfg, rec["id"])
        lines = ainvfx_lines(lines or [])
    if not lines:
        say("log: nothing from the pod's script yet{}; open it in the console, Pods > your pod > Logs".format(
            " (the API said: {})".format(err.detail) if err is not None else ""))
        system, _ = pod_logs(cfg, rec["id"], tail=40, source="system")
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
    _, rec = which_pod(load_config(), args.profile)
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
    files = [Path(f).expanduser() for f in args.files]
    for f in files:
        if not f.exists():
            die("file not found: {}".format(f))
    _, rec = which_pod(load_config(), args.profile)   # after the files: a wrong path stops before any lookup
    if not comfy_alive(rec["id"]):
        die("ComfyUI does not answer on the pod yet. `python pod.py status` shows the log.")
    for f in files:
        body, ctype = multipart({"overwrite": "true", "type": "input"}, "image", f)
        try:
            request("POST", pod_url(rec["id"]) + "/upload/image", data=body, headers={"Content-Type": ctype}, timeout=600, raw=True)
            say("  {}  ({:.1f} MB) -> input/".format(f.name, f.stat().st_size / 1e6))
        except ApiError as e:
            die("upload of {} failed: {}\n{}".format(f.name, e.detail[:200], e.hint()))
    say("Done. The files are in the lists of the Load Image and Load Video nodes.")


def cmd_pull(args):
    _, rec = which_pod(load_config(), args.profile)
    n = pull(rec)
    say("{} file(s) new in {}".format(n, outputs_of(rec)))


# Pull reads two sources. (1) The whole output folder, walked through JupyterLab's contents API with the token
# of the READY line: it also finds files a node saves without reporting them (a trained LoRA) and files made
# before a ComfyUI restart. (2) ComfyUI's history, every key ("images", "3d"...), through /view: the fallback
# when JupyterLab is off. ComfyUI itself has no recursive listing of its output folder.
def history_files(history):
    """(subfolder, filename) of every output file ComfyUI's history reports, whatever the key; temp and input
    entries left out; each file once."""
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
    return list(dict.fromkeys(found))


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
    if not parts or ".." in parts or rel.startswith("/") or ":" in parts[0]:
        return None
    return "/".join(parts)


def jupyter_files(pod_id, token):
    """(root, [(relative path, size)]) for every file under the pod's output folder, through JupyterLab's
    contents API; (None, []) when JupyterLab does not answer or has no output folder."""
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
            for e in todo.pop().get("content") or []:
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
    """Download url into target through a .part file, so an interrupted download never leaves a half file.
    Returns the size in bytes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    hdrs = {"Accept": "*/*"}
    hdrs.update(headers or {})
    r = request("GET", url, headers=hdrs, timeout=timeout, stream=True)
    size = 0
    try:
        with r, open(part, "wb") as f:
            for chunk in iter(lambda: r.read(1 << 20), b""):
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


def pull(rec):
    """Every file of the pod's output folder not here yet, into outputs/<pod name>/, subfolders kept. Returns
    the number of files downloaded."""
    pod_id, dest = rec["id"], outputs_of(rec)
    comfy = comfy_alive(pod_id)
    token = jupyter_token(proxy_log(pod_id)) if comfy else None
    count, listed, root = 0, set(), None

    def get(rel, url, size=None, headers=None):
        nonlocal count
        target = dest.joinpath(*rel.split("/"))
        if target.exists() and (size is None or target.stat().st_size == size):
            return
        try:
            n = fetch_to(url, target, headers=headers)
        except ApiError as e:
            say("  skipped {} ({})".format(rel, e.code or e.detail[:60]))
            return
        count += 1
        say("  {}  ({:.1f} MB)".format(target.relative_to(OUTPUTS), n / 1e6))

    if token:
        root, files = jupyter_files(pod_id, token)
        for rel, size in files:
            listed.add(rel)
            get(rel, pod_url(pod_id, JUPYTER_PORT) + "/files/" + urllib.parse.quote(root + "/" + rel), size,
                {"Authorization": "token " + token})
    if not comfy:
        say("ComfyUI does not answer on the pod: nothing to pull through it.")
        return count
    try:
        history = request("GET", pod_url(pod_id) + "/history", timeout=60)
    except ApiError as e:
        say("could not read the history: {}".format(e))
        history = None
    for sub, name in history_files(history):
        rel = safe_rel((sub + "/" + name) if sub else name)
        if rel and rel not in listed:
            q = urllib.parse.urlencode({"filename": name, "subfolder": sub, "type": "output"})
            get(rel, pod_url(pod_id) + "/view?" + q)
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
    names = ", ".join(r.get("name") or r["id"] for _, r in targets)
    if not confirm("Terminate {} now? The disk is erased, billing stops. [Y/n] ".format(names), args.yes):
        say("Kept running. Remember: a pod bills until terminated.")
        return
    for tag, rec in targets:
        if not args.no_pull:
            say("{} file(s) pulled into {}".format(pull(rec), outputs_of(rec)))
        try:
            spent = pod_spent(get_pod(cfg, rec["id"]), rec)
        except ApiError:
            spent = None
        try:
            rp("DELETE", "/pods/" + rec["id"], cfg)          # terminate: 204, no body
            if spent:
                say("Terminated: {}. It ran {} at {} per hour: about {} spent.".format(
                    rec.get("name"), fmt_duration(spent[0]), fmt_money(spent[2]), fmt_money(spent[1])))
            else:
                say("Terminated: {}".format(rec.get("name")))
        except ApiError as e:
            if e.code != 404:
                die("{}\n{}".format(e, e.hint()))
            say("{}: already gone from Runpod.".format(rec.get("name")))
        remember_pod(cfg, tag, None)
    try:
        running = [p for p in list_pods(cfg) if pod_status(p) not in FINAL]
    except ApiError:
        running = None
    if running:
        say("Attention: {} other pod(s) still running, {} per hour in total:".format(
            len(running), fmt_money(sum(pod_price(p) or 0 for p in running))))
        for p in running:
            price = pod_price(p)
            say("  - {} ({}{})".format(p.get("name") or p.get("id"), pod_status(p),
                                       ", {} per hour".format(fmt_money(price)) if price else ""))
        say("`python pod.py down <name>` terminates one of them.")
    elif running is not None:
        say("No pod running in your account. Nothing bills.")
    balance, _ = account_money(cfg)
    if balance is not None:
        say("Credit left on Runpod: {} (the last minutes may still be deducted).".format(fmt_money(balance)))


def cmd_list(args):
    cfg = load_config()
    pods = list_pods(cfg)
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
    gone = forget_gone(cfg, pods)
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
    cmd += ["{}@{}".format(user, host)] + (args.command or [])
    say(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


def cmd_doctor(args):
    cfg = load_config()
    say("ainvfx-runpod {} · Python {} on {}".format(VERSION, sys.version.split()[0], platform.platform()))
    say("  ssh          {}".format(shutil.which("ssh") or "absent (JupyterLab replaces it)"))
    say("  config       {} ({})".format(CONFIG, "present" if CONFIG.exists() else "absent: run `setup`"))
    region, country = cfg.get("region") or local_place()[0], cfg.get("country") or ""
    say("  nearest      {}{}{}".format(REGION_NAMES.get(region, region), ", " + country + " first" if country else "",
                                        "" if cfg.get("region") else " (guessed from the clock: {})".format(local_zone() or "offset")))
    say("  template     {}".format(cfg.get("template_id") or TEMPLATE_ID))
    say("  settings     {}{}".format(SETTINGS, "" if SETTINGS.exists() else " (absent: the bootstrap's defaults apply)"))
    say("  yours        {}{}".format(SETTINGS_LOCAL, "" if SETTINGS_LOCAL.exists() else " (absent: nothing replaced)"))
    local = read_env_file(SETTINGS_LOCAL)
    for k, v in load_settings().items():
        say("    {:<22} {}{}".format(k, v, "   (settings.local.env)" if k in local else ""))
    if cfg.get("api_key") or os.environ.get("RUNPOD_API_KEY"):
        try:
            say("  API          ok, {} pod(s) in the account".format(len(list_pods(cfg))))
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
    pod_ref = "which pod: the profile, the name given to `up --name`, the pod's name or its id (optional with one pod)"

    def command(name, func, hlp, pod=True):
        p = sub.add_parser(name, help=hlp)
        if pod:
            p.add_argument("profile", nargs="?", metavar="pod", help=pod_ref)
        p.set_defaults(func=func)
        return p

    p = command("setup", cmd_setup, "API key, region, Hugging Face secret, SSH key", pod=False)
    p.add_argument("--key", help="the Runpod API key (otherwise asked, hidden)")
    p.add_argument("--region", choices=["EU", "NA"], help="EU or NA: the data centers to try first")
    p.add_argument("--country", help="two letters (CA, US, FR, CZ, RO, NL, SE, IS, NO): tried before the rest of the region")
    p.add_argument("--template", help="a Runpod template id to create pods from")
    p.add_argument("--hf-token", action="store_true", help="store a Hugging Face token as the Runpod secret")
    p.add_argument("-y", "--yes", action="store_true")

    p = command("up", cmd_up, "create a pod", pod=False)
    p.add_argument("profile", nargs="?", default="image",
                   help="which models the pod downloads: image, 3d, video, train, several joined by + (image+3d), "
                        "or all (default image)")
    p.add_argument("--name", help="a name for this pod, to run several pods of one profile (`up train --name jar-lora`)")
    p.add_argument("--gpu", action="append", metavar="NAME",
                   help='a GPU by its name ("RTX PRO 4500 SE") or full id; only that GPU is tried. '
                        "Repeat it to accept several, tried in the order given: --gpu B200 --gpu B300")
    p.add_argument("--max-price", type=float, help="the most per hour `up` takes on its own when the profile's GPU "
                                                   "has no stock (default {:.2f} USD)".format(MAX_PRICE))
    p.add_argument("--disk", type=int, help="container disk in GB (default: from the models list)")
    p.add_argument("--region", choices=["EU", "NA"])
    p.add_argument("--community", action="store_true",
                   help="also consider Community Cloud (third-party machines, cheaper, the host picked by Runpod)")
    p.add_argument("--set", action="append", metavar="AINVFX_KEY=VALUE",
                   help="replace one settings.env value for this pod (repeatable); lasting changes go in settings.local.env")
    p.add_argument("--models-url", metavar="URL",
                   help="your own models.json for this pod: a raw link, or a Google Drive, Dropbox or GitHub share link")
    p.add_argument("--no-selftest", dest="selftest", action="store_false", help="skip the test image at the end of the install")
    p.add_argument("--retry", action="store_true",
                   help="when nothing is free, read the catalog every {} s and create the first GPU free (one pod); "
                        "Ctrl+C stops the wait".format(RETRY_EVERY))
    p.add_argument("--wait", type=int, default=25, help="minutes to follow the log (default 25)")
    p.add_argument("-y", "--yes", action="store_true", help="no confirmation prompt")

    command("logs", cmd_logs, "follow the log until READY").add_argument(
        "--wait", type=int, default=25, help="minutes (default 25)")
    command("status", cmd_status, "status, GPU, cost, address, log")
    command("open", cmd_open, "open ComfyUI in the browser")
    command("pull", cmd_pull, "download the outputs")
    command("push", cmd_push, "upload files into the pod's input folder").add_argument("files", nargs="+")
    p = command("down", cmd_down, "pull, then terminate")
    p.add_argument("--all", action="store_true", help="every pod recorded here")
    p.add_argument("--no-pull", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    command("ssh", cmd_ssh, "a terminal on the pod").add_argument("command", nargs="*")
    command("list", cmd_list, "every pod of the account, and which ones are recorded here", pod=False)
    command("doctor", cmd_doctor, "check this machine and the configuration", pod=False)

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
