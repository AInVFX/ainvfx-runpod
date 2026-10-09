#!/usr/bin/env python3
"""ainvfx-runpod · pod.py : create, use and terminate a Runpod GPU pod running ComfyUI.

One file, no dependency beyond Python 3.8 or newer. Works on Windows, macOS and Linux.
Nothing but the Runpod API key is stored on this machine, in a file only your account can read.

  python pod.py setup              once: your Runpod API key, your region, your SSH key (optional)
  python pod.py setup --hf-token   optional: store your Hugging Face token as a Runpod Secret
  python pod.py up image           create a pod for the image sessions (RTX 5090, 100 GB disk)
  python pod.py up video           create a pod for video (RTX PRO 6000, 200 GB disk)
  python pod.py up train           create a pod for LoRA training (RTX PRO 6000, 250 GB disk)
  python pod.py up image --gpu "RTX PRO 4500 SE"   another GPU from the list `up` shows
  python pod.py up train --name jar-lora   a second training pod, known here as `jar-lora`
  python pod.py status [pod]       status, GPU, data center, cost so far, ComfyUI address, last log lines
  python pod.py logs [pod]         follow the pod's log until READY (or Ctrl+C)
  python pod.py open [pod]         open ComfyUI in your browser
  python pod.py push [pod] FILES...   copy images or videos into the pod's input folder
  python pod.py pull [pod]         download the pod's outputs into outputs/<pod name>/
  python pod.py down [pod]         pull, then terminate the pod (billing stops); `down --all` for every pod
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
Its log lines start with [AINVFX]; `up`, `logs` and `status` read them for you, from the API
log stream and from the copy the pod serves through its proxy. Before the first [AINVFX] line,
Runpod's own system log tells what the host is doing (fetching the image, starting the container):
`up` shows it as [RUNPOD] lines, with one summary of the image download (layers done, extracting,
downloading) so a long pull never looks like a hang. Measured on two RTX 5090 pods (4 and 5 Oct 2026):
READY 5 to 15 minutes after creation. A pod with no container 8 minutes after creation gets a warning:
the usual answer is `down`, then `up` again, usually in another data center. A line starting with
FAILED (bootstrap.sh 5.4: PyTorch out of reach, no GPU from PyTorch, no ComfyUI) ends the wait at
once: that pod cannot work, and the answer is the same, `down` then `up`.

Where the pod goes. `up` creates a pod only where Runpod's catalog shows stock, and names the data
center on Secure Cloud. When the profile's GPU has no stock, `up` shows every GPU in stock that fits
the profile (32 GB of VRAM or more for image, 96 GB for video and train, a Blackwell chip or newer,
not a 1g MIG slice),
cheapest first, proposes the cheapest (never above 2.50 USD per hour on its own; --max-price changes
that), and prints the command to take another one. Right after creation it reads the pod's host: a host
with maintenance under way or starting within 24 hours is terminated at once and the next GPU is tried.

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

VERSION = "0.6.2"
API = os.environ.get("AINVFX_API_BASE", "https://api.runpod.io/v2")   # the test harness points this at a fake
REPO_RAW = "https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main"
# The image of Runpod's own "Runpod Pytorch 2.8.0" template (id runpod-torch-v280). Runpod keeps the images
# of its own templates on its hosts, so a pod on it starts within seconds; any other tag has to be
# fetched from Docker Hub first (about 9 GB, 19 layers, 5 to 15 minutes on 5 Oct 2026 with the
# 1.0.7-cu1300 tag). The image's own PyTorch and CUDA toolkit are not used: bootstrap.sh builds its
# environment with the cu130 PyTorch wheels, which carry their own CUDA 13 libraries.
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"


def load_settings():
    """The AINVFX_* variables of settings.env (next to this script): KEY=value lines, # comments,
    empty values dropped. They are sent with every pod and read by bootstrap.sh on the pod."""
    values = {}
    try:
        for raw in SETTINGS.read_text(encoding="utf-8").splitlines():
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


def comfy_tag():
    return load_settings().get("AINVFX_COMFY_TAG", "v0.39.0")
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
PROFILES = {
    "image": dict(disk=100, vram=32, gpu="NVIDIA GeForce RTX 5090"),
    "video": dict(disk=200, vram=96, gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition"),
    "train": dict(disk=250, vram=96, gpu="NVIDIA RTX PRO 6000 Blackwell Server Edition"),
}
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
            return "Too many requests: wait a minute, then try again."
        if self.code == 0 and self.timed_out:
            return "The server accepted the request but did not answer in time: a host that is not responding, or a slow link. Try again; `python pod.py status` shows the pod."
        if self.code == 0:
            return "No network, or api.runpod.io unreachable from this machine."
        return ""

    @property
    def timed_out(self):
        return "timed out" in (self.detail or "").lower()


def request(method, url, key=None, body=None, timeout=60, raw=False, headers=None, stream=False, data=None):
    """One HTTP request with urllib. Returns parsed JSON, bytes (raw=True), or the open response (stream=True).
    `body` is sent as JSON; `data` as given (bytes), with the Content-Type of `headers`.
    Every network failure, including a timeout while the server prepares its answer, is an ApiError
    with code 0: nothing here raises a bare socket error (a pod on a host that does not respond
    once made `up` stop with a traceback)."""
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


def which_pod(cfg, ref):
    """The pod a command targets: (tag, record). `ref` is a tag (a profile name, or the `--name`
    given to `up`), a pod name, a pod id, or nothing when one pod is recorded. A pod that exists in
    the account but not here (created from the console, or from another machine) is attached: its
    record is written to the configuration under its name, so the next commands can use it."""
    pods = cfg.setdefault("pods", {})
    if ref is None:
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
        die("no GPU type named \"{}\" in the Runpod catalog. `python pod.py up` lists the GPUs in stock.".format(ref))
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

    say("\n3. Hugging Face token (optional: only the gated LTX files need it, for the video and LoRA sessions).")
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


def cmd_up(args):
    cfg = load_config()
    profile = args.profile
    spec = PROFILES[profile]
    api_key(cfg)
    tag = "".join(c if c.isalnum() or c in "-_" else "-" for c in (args.name or "")).strip("-") or profile
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
    disk = args.disk or spec["disk"]
    name = "ainvfx-{}-{}".format(profile, tag if tag != profile else datetime.now().strftime("%m%d-%H%M"))
    env = dict(load_settings())              # settings.env: ComfyUI tag, Python, PyTorch, models, custom nodes...
    env["AINVFX_PROFILE"] = profile           # the profile named on the command line wins
    if cfg.get("hf_secret") or secret_exists(cfg):
        env["HF_TOKEN"] = HF_SECRET_REF          # the reference, never the token itself
    if not args.selftest:
        env["AINVFX_SELFTEST"] = "0"

    # The catalog, read once per cloud: every GPU type with its stock for hosts on CUDA MIN_CUDA or newer.
    catalogs = {}
    for cloud in (("SECURE",) if args.secure_only else ("SECURE", "COMMUNITY")):
        try:
            catalogs[cloud] = read_catalog(cfg, cloud)
        except ApiError as e:
            if cloud == "SECURE":
                die("the Runpod catalog could not be read ({}: {}).\n{}".format(e.code, e.detail[:200], e.hint()))
            say("Community Cloud catalog not readable ({}): Secure Cloud only.".format(e.code))
    secure = catalogs.get("SECURE") or []
    max_price = MAX_PRICE if args.max_price is None else args.max_price
    if args.gpu:
        want = find_gpu(secure + (catalogs.get("COMMUNITY") or []), args.gpu)
    else:
        want = next((g for g in secure if g["id"] == spec["gpu"]), {"id": spec["gpu"], "name": spec["gpu"]})
    offers = make_offers(catalogs, region, country)
    mine = [o for o in offers if o["id"] == want["id"]]
    preferred = next((o for o in mine if o["cloud"] == "SECURE"), None)
    fits = [o for o in offers if fits_profile(o, spec["vram"])]
    wname = str(want.get("name") or want["id"])

    price = want.get("price") or {}
    p_secure, p_comm = price.get("secure"), price.get("community")
    dcs = preferred["dcs"] if preferred else []
    say("\n{}: availability {} on Secure Cloud with hosts on CUDA {} or newer".format(
        wname, str(want.get("availability", "?")).upper(), MIN_CUDA))
    say("   in stock in {}: {}".format(REGION_NAMES.get(region, region),
                                       ", ".join(d for d in dcs if region_of(d) == region) or "none right now"))
    say("   elsewhere: {}".format(", ".join(d for d in dcs if region_of(d) != region) or "none"))
    say("   price per hour: {} Secure, {} Community".format(
        fmt_money(p_secure) if p_secure else "?", fmt_money(p_comm) if p_comm else "?"))

    profile_rule = "{} GB of VRAM or more, Blackwell or newer, not a 1g MIG slice, host on CUDA {} or newer".format(
        spec["vram"], MIN_CUDA)
    if args.gpu:
        candidates = mine                     # the GPU named on the command line, and nothing else
        if not candidates:
            say("   No {} in stock right now{}.".format(wname, " on Secure Cloud" if args.secure_only else ""))
            others = [o for o in fits if o["id"] != want["id"]]
            if others:
                say("\nIn stock now for the {} profile ({}), cheapest first:".format(profile, profile_rule))
                show_offers(others, None, max_price)
                say('   To take one:  python pod.py up {} --gpu "{}"'.format(profile, others[0]["name"]))
            die("nothing created.")
    elif preferred:
        say("   tried first: {}".format(dcs[0]))
        candidates = [preferred] + [o for o in fits if o is not preferred and o["price"] <= max_price]
    else:
        say("   No {} is free on Secure Cloud. `up` never creates a pod where the catalog shows no stock:".format(wname))
        say("   Runpod would then pick any host, and twice that was a host being removed from the platform.")
        candidates = [o for o in fits if o["price"] <= max_price]
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
            die("nothing in stock fits the {} profile at {} per hour or less. Try again in a few minutes.".format(
                profile, fmt_money(max_price)))
        other = next((o for o in fits if o["id"] != candidates[0]["id"]), None)
        if other:
            say('   To choose another one:  python pod.py up {} --gpu "{}"   (any name from the GPU column)'.format(
                profile, other["name"]))
        if any(o["cloud"] == "COMMUNITY" for o in fits):
            say("   Add --secure-only to leave out Community Cloud.")

    first = candidates[0]
    if not args.yes:
        if len(candidates) > 1:
            say("   If it is taken in the meantime, `up` tries the next GPUs that fit, cheapest first, up to {} per hour."
                .format(fmt_money(max(o["price"] for o in candidates))))
        ans = input("   Create a {} with a {} GB disk, billed from creation? [Y/n] ".format(first["name"], disk)).strip().lower()
        if ans not in ("", "y", "yes"):
            say('Nothing created. To choose another GPU:  python pod.py up {} --gpu "<name from the list>"'.format(profile))
            return

    created, host = None, None
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
                continue
            if machine is None:
                say("   host check skipped: the GraphQL API did not answer. The log below shows whether the host works.")
            elif not machine:
                say("   host check: no host reported after {} s. The log below shows whether it works.".format(HOST_WAIT))
            else:
                say("   host checked: no maintenance planned{}.".format(
                    " ({})".format(machine["dataCenterId"]) if machine.get("dataCenterId") else ""))
            created, host = pod, (machine or {})
            break
        if created:
            break
    if not created:
        die("no pod could be created: the machines in stock were taken in the meantime, or under maintenance.\n"
            "Try again in a minute: `python pod.py up {}` reads the stock again.".format(profile))

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
    created = parse_time(pod.get("createdAt") or "") or parse_time(rec.get("created", ""))
    if created and price:
        hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        say("running for {:.1f} h · about {} spent so far at {} per hour".format(hours, fmt_money(hours * price), fmt_money(price)))
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


def pull(cfg, rec):
    """Every output file listed in ComfyUI's history, through the pod's own HTTP endpoints."""
    base = pod_url(rec["id"])
    if not comfy_alive(rec["id"]):
        say("ComfyUI does not answer on the pod: nothing to pull through it.")
        return 0
    try:
        history = request("GET", base + "/history", timeout=60)
    except ApiError as e:
        say("could not read the history: {}".format(e))
        return 0
    dest = OUTPUTS / rec.get("name", rec["id"])
    dest.mkdir(parents=True, exist_ok=True)
    seen, count = set(), 0
    for _, entry in (history or {}).items():
        for _, out in (entry.get("outputs") or {}).items():
            for kind in ("images", "gifs", "videos", "audio", "files"):
                for item in out.get(kind, []) or []:
                    if item.get("type") not in (None, "output"):
                        continue
                    name, sub = item.get("filename"), item.get("subfolder", "")
                    if not name or (name, sub) in seen:
                        continue
                    seen.add((name, sub))
                    target = dest / sub / name if sub else dest / name
                    if target.exists():
                        continue
                    q = urllib.parse.urlencode({"filename": name, "subfolder": sub, "type": "output"})
                    try:
                        data = request("GET", base + "/view?" + q, timeout=600, raw=True)
                    except ApiError as e:
                        say("  skipped {} ({})".format(name, e.code))
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    count += 1
                    say("  {}  ({:.1f} MB)".format(target.relative_to(OUTPUTS), len(data) / 1e6))
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
            rp("DELETE", "/pods/" + rec["id"], cfg)          # DELETE /pods/{id}: terminate, 204 no body
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
        return
    running = [p for p in remaining if pod_status(p) not in FINAL]
    if running:
        say("Attention: {} other pod(s) still in your account:".format(len(running)))
        for p in running:
            say("  - {} ({})".format(p.get("name") or p.get("id"), pod_status(p)))
    else:
        say("No pod running in your account. Nothing bills.")


def cmd_list(args):
    cfg = load_config()
    list_pods(cfg)


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
    for k, v in load_settings().items():
        say("    {:<22} {}".format(k, v))
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
    p.add_argument("profile", nargs="?", default="image", choices=sorted(PROFILES))
    p.add_argument("--name", help="a name for this pod, to run several pods of one profile (`up train --name jar-lora`); "
                                  "the other commands then take that name")
    p.add_argument("--gpu", help='a GPU from the list `up` shows, by its name ("RTX PRO 4500 SE") or its full id; '
                                 "only that GPU is tried")
    p.add_argument("--max-price", type=float, help="the most per hour `up` takes on its own when the profile's GPU "
                                                   "has no stock (default {:.2f} USD)".format(MAX_PRICE))
    p.add_argument("--disk", type=int, help="container disk in GB")
    p.add_argument("--region", choices=["EU", "NA"])
    p.add_argument("--secure-only", action="store_true", help="leave out Community Cloud")
    p.add_argument("--no-selftest", dest="selftest", action="store_false", help="skip the test image at the end of the install")
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
