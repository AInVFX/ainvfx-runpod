#!/usr/bin/env python3
"""ainvfx-runpod · pod.py : create, use and terminate a Runpod GPU pod running ComfyUI.

One file, no dependency beyond Python 3.8 or newer. Works on Windows, macOS and Linux.
Nothing but the Runpod API key is stored on this machine, in a file only your account can read.

  python pod.py setup              once: your Runpod API key, your region, your SSH key (optional)
  python pod.py setup --hf-token   optional: store your Hugging Face token as a Runpod Secret
  python pod.py up image           create a pod for the image sessions (RTX 5090, 100 GB disk)
  python pod.py up video           create a pod for video (RTX PRO 6000, 200 GB disk)
  python pod.py up train           create a pod for LoRA training (RTX PRO 6000, 250 GB disk)
  python pod.py status [profile]   status, GPU, data center, cost so far, ComfyUI address, last log lines
  python pod.py logs [profile]     follow the pod's log until READY (or Ctrl+C)
  python pod.py open [profile]     open ComfyUI in your browser
  python pod.py push [profile] FILES...   copy images or videos into the pod's input folder
  python pod.py pull [profile]     download the pod's outputs into outputs/<pod name>/
  python pod.py down [profile]     pull, then terminate the pod (billing stops)
  python pod.py ssh [profile]      a terminal on the pod
  python pod.py list               every pod of your account, with its hourly price
  python pod.py doctor             check Python, ssh, the API key, the configuration

The pod runs bootstrap.sh from this repository at start: SSH and JupyterLab first, a health check
(driver, disk speed, download speed), then ComfyUI at a pinned tag, then the models of the profile,
then one test image. Its log lines start with [AINVFX]; `up`, `logs` and `status` read them for you.
On the test pod (RTX 5090, 4 Oct 2026) the log said READY about 15 minutes after creation.

Rule of the course: create at the start of the session, pull your results, terminate at the end.
A terminated pod costs nothing. A stopped pod keeps a dead entry and, with a volume disk, keeps
billing it.
"""
import argparse
import getpass
import json
import os
import platform
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
from datetime import datetime, timezone
from pathlib import Path

VERSION = "0.2.3"
API = "https://api.runpod.io/v2"
REPO_RAW = "https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main"
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"      # Ubuntu 24.04, official Runpod image
COMFY_TAG = "v0.38.2"
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
OUTPUTS = HERE / "outputs"
FINAL = ("EXITED", "ERROR", "TERMINATED")

PROFILES = {
    "image": dict(disk=100, gpus=["NVIDIA GeForce RTX 5090",
                                 "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb",
                                 "NVIDIA RTX PRO 6000 Blackwell Server Edition"]),
    "video": dict(disk=200, gpus=["NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                  "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                                  "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb"]),
    "train": dict(disk=250, gpus=["NVIDIA RTX PRO 6000 Blackwell Server Edition",
                                  "NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]),
}
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
        if self.code == 0:
            return "No network, or api.runpod.io unreachable from this machine."
        return ""


def request(method, url, key=None, body=None, timeout=60, raw=False, headers=None, stream=False):
    """One HTTP request with urllib. Returns parsed JSON, bytes (raw=True), or the open response (stream=True)."""
    data = None
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
        raise ApiError(0, str(e.reason), url)
    if stream:
        return r
    with r:
        payload = r.read()
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
    return "https://{}-{}.proxy.runpod.net".format(pod_id, port)


def region_of(dc_id):
    p = (dc_id or "").upper()
    if p.startswith(("EU", "EUR")):
        return "EU"
    if p.startswith(("US", "CA")):
        return "NA"
    return "OTHER"


def local_region():
    """EU or NA from this machine's UTC offset; a guess the person can change in `setup`."""
    offset_h = (time.localtime().tm_gmtoff or 0) / 3600
    return "NA" if offset_h <= -2 else "EU"


def profile_of(cfg, name):
    if name is None:
        pods = cfg.get("pods", {})
        if len(pods) == 1:
            return next(iter(pods))
        if not pods:
            die("no pod recorded. First:  python pod.py up image")
        die("several pods recorded ({}): name the profile".format(", ".join(pods)))
    if name not in PROFILES:
        die("unknown profile '{}'. Choose: {}".format(name, ", ".join(PROFILES)))
    return name


def recorded_pod(cfg, profile):
    pod = cfg.get("pods", {}).get(profile)
    if not pod:
        die("no pod recorded for '{}'. First:  python pod.py up {}".format(profile, profile))
    return pod


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


def gpu_catalog(cfg, gpu_id):
    q = urllib.parse.urlencode({"include": "AVAILABILITY", "product": "POD", "minCudaVersion": MIN_CUDA})
    return rp("GET", "/catalog/gpus/{}?{}".format(urllib.parse.quote(gpu_id, safe=""), q), cfg)


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
        for p in pods:
            gpu, dc, price = pod_summary(p)
            say("{:<28} {:<12} {:<40} {:<10} {}".format(
                str(p.get("name") or "?")[:28], pod_status(p)[:12], gpu[:40], dc[:10],
                (fmt_money(price) + "/h") if price else ""))
            say("    id {}   ComfyUI {}".format(p.get("id"), pod_url(str(p.get("id")))))
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


def log_stream(cfg, pod_id, tail=200, last_id=None, source=None, idle=20):
    """Yield (event_id, source, line) from GET /pods/{id}/logs, a Server-Sent Events stream
    (one `id:` line and one `data:` JSON line per event, with ts, source and line).
    `tail` lines are sent first, then live lines. Stops after `idle` seconds without data."""
    q = {}
    hdrs = {"Accept": "text/event-stream"}
    if last_id:
        hdrs["Last-Event-ID"] = last_id      # resume where the previous read stopped
    else:
        q["tail"] = tail
    if source:
        q["source"] = source
    url = "/pods/{}/logs".format(pod_id) + ("?" + urllib.parse.urlencode(q) if q else "")
    r = rp("GET", url, cfg, timeout=idle, headers=hdrs, stream=True)
    eid = None
    try:
        while True:
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
    finally:
        try:
            r.close()
        except Exception:
            pass


def pod_logs(cfg, pod_id, tail=400, seconds=10):
    """The last `tail` container log lines, as a list of strings (the stream is left after `seconds`)."""
    lines, t0 = [], time.time()
    try:
        for _, _, line in log_stream(cfg, pod_id, tail=tail, source="container", idle=3):
            if line:
                lines.append(line)
            if time.time() - t0 > seconds:
                break
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


def follow_logs(cfg, pod_id, minutes=25):
    """Print the [AINVFX] lines as they appear, until READY, a dead pod, or the time runs out."""
    deadline = time.time() + minutes * 60
    seen, last_id, told, last_check = set(), None, False, time.time()
    while time.time() < deadline:
        try:
            for eid, _, line in log_stream(cfg, pod_id, tail=500 if last_id is None else 0,
                                           last_id=last_id, idle=30):
                last_id = eid or last_id
                if "[AINVFX]" in line and line not in seen:
                    seen.add(line)
                    say("  " + line[line.index("[AINVFX]"):])
                    if "READY" in line:
                        return True
                if time.time() > deadline:
                    break
        except ApiError as e:
            if not told:
                say("(the log cannot be read through the API from here: {}. Follow it in the console, "
                    "Pods > your pod > Logs. The ComfyUI address answers once the log says COMFYUI UP.)".format(e.detail))
                told = True
            if comfy_alive(pod_id):
                say("ComfyUI answers: {}".format(pod_url(pod_id)))
                return True
            time.sleep(15)
        # the stream went quiet: make sure the pod is still alive
        if time.time() - last_check > 45:
            last_check = time.time()
            try:
                st = pod_status(get_pod(cfg, pod_id))
                if st in FINAL:
                    say("The pod is {}. Read its log in the console, then `python pod.py down` and `up` again.".format(st))
                    return False
            except ApiError:
                pass
    say("Still working after {} minutes: check `python pod.py status`.".format(minutes))
    return False


# ----------------------------------------------------------------------------- commands
def cmd_setup(args):
    cfg = load_config()
    say("ainvfx-runpod {} · setup".format(VERSION))
    key = args.key or os.environ.get("RUNPOD_API_KEY", "") or cfg.get("api_key", "")
    if not key:
        say("\n1. Your Runpod API key. Console > Account > Credentials > API Keys > Create API Key.")
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

    region = args.region or cfg.get("region") or local_region()
    say("\n2. Your region, for the nearest data centers: {} (EU or NA; `setup --region NA` to change)".format(region))
    cfg["region"] = region
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
    profile = profile_of(cfg, args.profile)
    spec = PROFILES[profile]
    api_key(cfg)
    old = cfg.get("pods", {}).get(profile)
    if old:
        try:
            pod = get_pod(cfg, old["id"])
            st = pod_status(pod)
            if st not in FINAL and st != "?":
                say("A pod already exists for '{}': {} ({}). Address: {}".format(
                    profile, old.get("name"), st, pod_url(old["id"])))
                say("Use it, or terminate it first:  python pod.py down {}".format(profile))
                return
        except ApiError as e:
            if e.code != 404:
                die("{}\n{}".format(e, e.hint()))
        cfg["pods"].pop(profile, None)
        save_config(cfg)

    region = args.region or cfg.get("region") or local_region()
    disk = args.disk or spec["disk"]
    gpus = [args.gpu] if args.gpu else spec["gpus"]
    name = "ainvfx-{}-{}".format(profile, datetime.now().strftime("%m%d-%H%M"))
    env = {"AINVFX_PROFILE": profile, "AINVFX_COMFY_TAG": COMFY_TAG}
    if cfg.get("hf_secret") or secret_exists(cfg):
        env["HF_TOKEN"] = HF_SECRET_REF          # the reference, never the token itself
    if not args.selftest:
        env["AINVFX_SELFTEST"] = "0"

    created = None
    for gpu in gpus:
        try:
            cat = gpu_catalog(cfg, gpu)
        except ApiError as e:
            say("catalog: {} ({}: {})".format(gpu, e.code, e.detail[:100]))
            continue
        price = cat.get("price") or {}
        p_secure, p_comm = price.get("secure"), price.get("community")
        dcs = [d.get("id") for d in (cat.get("dataCenters") or []) if d.get("id")
               and str(d.get("availability", "")).upper() not in ("NONE", "")]
        dcs.sort(key=lambda d: (0 if region_of(d) == region else 1 if region_of(d) in ("EU", "NA") else 2))
        avail = str(cat.get("availability", "?")).upper()
        say("\n{}: availability {} on Secure Cloud with CUDA {}+; data centers in stock: {}".format(
            cat.get("name", gpu), avail, MIN_CUDA, ", ".join(dcs) or "none"))
        say("   price per hour: {} Secure, {} Community".format(
            fmt_money(p_secure) if p_secure else "?", fmt_money(p_comm) if p_comm else "?"))
        if not dcs and args.secure_only:
            continue
        if not args.yes:
            ans = input("   Create a {} with a {} GB disk, billed from creation? [Y/n] ".format(cat.get("name", gpu), disk)).strip().lower()
            if ans not in ("", "y", "yes"):
                continue
        attempts = [("SECURE", [dc]) for dc in dcs[:6]] + [("SECURE", [])]
        if not args.secure_only:
            attempts.append(("COMMUNITY", []))
        for cloud, dc_list in attempts:
            body = {"name": name, "cloud": cloud, "disk": disk, "env": env,
                    "startSsh": True, "startJupyter": True,
                    "gpu": {"id": gpu, "count": 1, "minCudaVersion": MIN_CUDA}}
            if dc_list:
                body["dataCenterIds"] = dc_list
            if cfg.get("template_id") or TEMPLATE_ID:
                body["templateId"] = cfg.get("template_id") or TEMPLATE_ID   # image, command, ports come from the template
            else:
                body["image"] = IMAGE
                body["args"] = START_CMD
                body["ports"] = ["{}/http".format(COMFY_PORT), "{}/http".format(JUPYTER_PORT), "22/tcp"]
            where = "{} {}".format(cloud, dc_list[0] if dc_list else "(any data center)")
            created = create_pod(cfg, body, where)
            if created and created.get("id"):
                say("   created on {}: pod {}".format(where, created["id"]))
                break
            created = None
        if created:
            break
    if not created:
        die("no pod could be created. Try again in a few minutes, another profile, or the Runpod console.")

    pod_id = created["id"]
    cfg["pods"][profile] = {"id": pod_id, "name": name, "created": datetime.now(timezone.utc).isoformat()}
    save_config(cfg)
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
    gpu, dc, price = pod_summary(pod)
    say("Pod {} · {} · {} · CUDA {} · {}".format(name, gpu, dc, pod.get("cudaVersion") or "?",
                                                   (fmt_money(price) + " per hour") if price else "price in the console"))
    say("ComfyUI address (ready once the log says COMFYUI UP): {}".format(pod_url(pod_id)))
    say("JupyterLab: {}  (token under Connect in the console)\n".format(pod_url(pod_id, JUPYTER_PORT)))
    follow_logs(cfg, pod_id, minutes=args.wait)
    say("\nWhen you are done:  python pod.py down {}".format(profile))


def cmd_logs(args):
    cfg = load_config()
    rec = recorded_pod(cfg, profile_of(cfg, args.profile))
    follow_logs(cfg, rec["id"], minutes=args.wait)


def cmd_status(args):
    cfg = load_config()
    profile = profile_of(cfg, args.profile)
    rec = recorded_pod(cfg, profile)
    try:
        pod = get_pod(cfg, rec["id"])
    except ApiError as e:
        if e.code == 404:
            die("the pod no longer exists on Runpod (terminated from the console?). Run `up` for a new one.")
        die("{}\n{}".format(e, e.hint()))
    gpu, dc, price = pod_summary(pod)
    st = pod_status(pod)
    say("{} · {} · {} · CUDA {} · {}".format(rec.get("name"), gpu, dc, pod.get("cudaVersion") or "?", st))
    created = parse_time(pod.get("createdAt") or "") or parse_time(rec.get("created", ""))
    if created and price:
        hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
        say("running for {:.1f} h · about {} spent so far at {} per hour".format(hours, fmt_money(hours * price), fmt_money(price)))
    rt = pod.get("runtime") or {}
    if rt.get("gpus"):
        g = rt["gpus"][0]
        say("GPU in use: {}% · VRAM {}%".format(g.get("util", "?"), g.get("memoryUtil", "?")))
    say("ComfyUI: {}  ({})".format(pod_url(rec["id"]), "answers" if comfy_alive(rec["id"]) else "not answering yet"))
    say("JupyterLab: {}".format(pod_url(rec["id"], JUPYTER_PORT)))
    kind, host, port, user = pod_ssh(pod)
    if kind:
        say("SSH ({}): ssh -p {} {}@{}".format(kind, port, user, host))
    lines, err = pod_logs(cfg, rec["id"])
    if err is not None:
        say("log: not readable through the API ({}); open it in the console".format(err.detail))
    else:
        for l in ainvfx_lines(lines)[-15:]:
            say("  " + l[l.index("[AINVFX]"):])


def cmd_open(args):
    cfg = load_config()
    rec = recorded_pod(cfg, profile_of(cfg, args.profile))
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
    rec = recorded_pod(cfg, profile_of(cfg, args.profile))
    base = pod_url(rec["id"])
    files = [Path(f).expanduser() for f in args.files]
    for f in files:
        if not f.exists():
            die("file not found: {}".format(f))
    if not comfy_alive(rec["id"]):
        die("ComfyUI does not answer on the pod yet. `python pod.py status` shows the log.")
    for f in files:
        body, ctype = multipart({"overwrite": "true", "type": "input"}, "image", f)
        req = urllib.request.Request(base + "/upload/image", data=body, method="POST",
                                     headers={"Content-Type": ctype, "User-Agent": "ainvfx-runpod/" + VERSION})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                r.read()
            say("  {}  ({:.1f} MB) -> input/".format(f.name, f.stat().st_size / 1e6))
        except urllib.error.HTTPError as e:
            die("upload of {} failed: HTTP {} {}".format(f.name, e.code, e.read()[:200]))
    say("Done. The files are in the lists of the Load Image and Load Video nodes.")


def cmd_pull(args):
    cfg = load_config()
    profile = profile_of(cfg, args.profile)
    rec = recorded_pod(cfg, profile)
    n = pull(cfg, rec)
    say("{} file(s) new in {}".format(n, OUTPUTS / rec.get("name", profile)))


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
    profile = profile_of(cfg, args.profile)
    rec = recorded_pod(cfg, profile)
    if not args.no_pull:
        n = pull(cfg, rec)
        say("{} file(s) pulled into {}".format(n, OUTPUTS / rec.get("name", profile)))
    if not args.yes:
        ans = input("Terminate pod {} now? Its disk is erased, billing stops. [Y/n] ".format(rec.get("name"))).strip().lower()
        if ans not in ("", "y", "yes"):
            say("Kept running. Remember: it bills until terminated.")
            return
    try:
        rp("DELETE", "/pods/" + rec["id"], cfg)          # DELETE /pods/{id}: terminate, 204 no body
        say("Terminated: {}".format(rec.get("name")))
    except ApiError as e:
        if e.code == 404:
            say("Already gone from Runpod.")
        else:
            die("{}\n{}".format(e, e.hint()))
    cfg["pods"].pop(profile, None)
    save_config(cfg)
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
    rec = recorded_pod(cfg, profile_of(cfg, args.profile))
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
    say("  region       {}".format(cfg.get("region") or local_region() + " (guessed)"))
    say("  template     {}".format(cfg.get("template_id") or TEMPLATE_ID or "none: the script describes the pod itself"))
    say("  ComfyUI tag  {} · image {}".format(COMFY_TAG, IMAGE))
    if cfg.get("api_key") or os.environ.get("RUNPOD_API_KEY"):
        try:
            pods = list_pods(cfg, quiet=True)
            say("  API          ok, {} pod(s) in the account".format(len(pods)))
        except ApiError as e:
            say("  API          FAILED: {} · {}".format(e, e.hint()))
        found = secret_exists(cfg)
        say("  HF secret    {}".format("present" if found else "absent (setup --hf-token)" if found is False
                                       else "unknown: the key cannot list secrets"))
    for prof, rec in (cfg.get("pods") or {}).items():
        say("  pod {:<8} {} ({})".format(prof, rec.get("name"), rec.get("id")))


def main():
    ap = argparse.ArgumentParser(description="Runpod pod remote for the AInVFX bootcamp.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--version", action="version", version=VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="API key, region, Hugging Face secret, SSH key")
    p.add_argument("--key", help="the Runpod API key (otherwise asked, hidden)")
    p.add_argument("--region", choices=["EU", "NA"], help="nearest data centers first")
    p.add_argument("--template", help="a Runpod template id to create pods from")
    p.add_argument("--hf-token", action="store_true", help="store a Hugging Face token as the Runpod secret")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("up", help="create a pod")
    p.add_argument("profile", nargs="?", default="image", choices=sorted(PROFILES))
    p.add_argument("--gpu", help="exact GPU type id, instead of the profile's list")
    p.add_argument("--disk", type=int, help="container disk in GB")
    p.add_argument("--region", choices=["EU", "NA"])
    p.add_argument("--secure-only", action="store_true", help="never fall back to Community Cloud")
    p.add_argument("--no-selftest", dest="selftest", action="store_false", help="skip the test image at the end of the install")
    p.add_argument("--wait", type=int, default=25, help="minutes to follow the log (default 25)")
    p.add_argument("-y", "--yes", action="store_true", help="no confirmation prompt")
    p.set_defaults(func=cmd_up)

    p = sub.add_parser("logs", help="follow the log until READY")
    p.add_argument("profile", nargs="?")
    p.add_argument("--wait", type=int, default=25, help="minutes (default 25)")
    p.set_defaults(func=cmd_logs)

    for name, fn, hlp in (("status", cmd_status, "status, GPU, cost, address, log"),
                          ("open", cmd_open, "open ComfyUI in the browser"),
                          ("pull", cmd_pull, "download the outputs")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("profile", nargs="?")
        p.set_defaults(func=fn)

    p = sub.add_parser("push", help="upload files into the pod's input folder")
    p.add_argument("profile", nargs="?")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("down", help="pull, then terminate")
    p.add_argument("profile", nargs="?")
    p.add_argument("--no-pull", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    p.set_defaults(func=cmd_down)

    p = sub.add_parser("ssh", help="a terminal on the pod")
    p.add_argument("profile", nargs="?")
    p.add_argument("command", nargs="*")
    p.set_defaults(func=cmd_ssh)

    sub.add_parser("list", help="every pod of the account").set_defaults(func=cmd_list)
    sub.add_parser("doctor", help="check this machine and the configuration").set_defaults(func=cmd_doctor)

    args = ap.parse_args()
    # `push image file.png` and `push file.png` both work
    if args.cmd == "push" and args.profile and args.profile not in PROFILES:
        args.files.insert(0, args.profile)
        args.profile = None
    try:
        args.func(args)
    except ApiError as e:
        die("{}\n{}".format(e, e.hint()))
    except KeyboardInterrupt:
        say("\nStopped.")
        sys.exit(130)


if __name__ == "__main__":
    main()
