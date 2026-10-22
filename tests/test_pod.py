#!/usr/bin/env python3
"""Tests for pod.py. Standard library only: nothing reaches Runpod, nothing is billed, and the real
~/.ainvfx-runpod is untouched (the end-to-end tests run with HOME in a temporary folder).

    python -m unittest discover tests      # from the repository root

`Helpers` are unit tests of the pure functions. `EndToEnd` starts a fake Runpod API (REST v2, the GraphQL
host and credit queries, the GPU catalog), a fake ComfyUI and a fake JupyterLab on 127.0.0.1, and runs the real
commands through subprocess, as a user would. The fake log stream behaves as the real one was seen to: events
carry their time as id, a `since` cursor is exclusive at one-second resolution, the stream stays open with
keepalives, and the lines after SELFTEST OK never arrive on a live connection (only a fresh request serves
them). Each test's docstring names the incident it reproduces.
"""
import json, os, re, shutil, signal, subprocess, sys, tempfile, threading, time, unittest, urllib.parse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POD_PY = os.path.join(ROOT, "pod.py")
sys.path.insert(0, ROOT)
import pod  # noqa: E402  (the module under test; importing it has no side effect)
BOOT = [  # the scripted bootstrap log: (seconds after creation, line)
    (0, "[AINVFX] bootstrap start · profile image · ComfyUI v0.39.0"),
    (0, "[AINVFX] no HF_TOKEN: the gated files (LTX, video sessions) will be skipped; the image models need none"),
    (1, "[AINVFX] step 0/5 starting Runpod's /start.sh in the background (SSH, JupyterLab on port 8888)"),
    (1, "[AINVFX] step 1/5 health check"),
    (1, "[AINVFX] GPU: NVIDIA GeForce RTX 5090, 32607 MiB, 610.43.02"),
    (2, "[AINVFX] disk: write 4284 MB/s · read 4306 MB/s"),
    (2, "[AINVFX] step 2/5 install (uv, Python 3.13, PyTorch cu130, ComfyUI v0.39.0)"),
    (3, "[AINVFX] ComfyUI v0.39.0 in /workspace/ComfyUI · environment /workspace/venv"),
    (3, "[AINVFX] step 3/5 start ComfyUI on port 8188"),
    (4, "[AINVFX] COMFYUI UP · https://fakepod1-8188.proxy.runpod.net"),
    (4, "[AINVFX] PROXY OK · https://fakepod1-8188.proxy.runpod.net answers from outside"),
    (4, "[AINVFX] step 4/5 models of profile image"),
    (5, "[AINVFX] models: 14 files in profile image · 14 to download (62.9 GB) · 0 already present or skipped"),
    (5, "[AINVFX] models 1/14 · 0.0 of 62.9 GB done · downloading z_image_turbo_int8_convrot.safetensors (6.2 GB) from Comfy-Org/z_image_turbo"),
    (5, "[AINVFX] models 1/14 · z_image_turbo_int8_convrot.safetensors in 10 s · 600 MB/s · 6.2 of 62.9 GB (9%) · about 1 min 34 s left"),
    (6, "[AINVFX] MODELS DONE 14/14 present · 59G on disk · 59 GB downloaded in 91 s (650 MB/s)"),
    (6, "[AINVFX] step 5/5 self-test"),
    (7, "[AINVFX] SELFTEST OK · Z-Image Turbo 1024 x 1024, 8 steps, in 16.0 s (models loaded from disk) · output/ainvfx_selftest_00001_.png"),
    (7, "[AINVFX] READY · ComfyUI https://fakepod1-8188.proxy.runpod.net · JupyterLab https://fakepod1-8888.proxy.runpod.net/lab?token=faketoken · log /workspace/ComfyUI/input/ainvfx/bootstrap.log"),
    (7, "[AINVFX] remember: terminate the pod when you are done"),
]
# bootstrap.sh 5.4 on a host whose network times out on every PyTorch route (seen on 6 Oct 2026, before 5.4):
# the log stops at a FAILED line and the pod stays up; READY never comes.
BOOT_FAILED = BOOT[:7] + [
    (3, "[AINVFX] WARNING: PyTorch download from https://download.pytorch.org/whl/cu130 failed (the lines above name the file)"),
    (3, "[AINVFX] PyTorch: trying PyPI (attempt 1 of 2): the same CUDA 13.0 build, with the NVIDIA libraries from PyPI"),
    (4, "[AINVFX] PyTorch: trying PyPI (attempt 2 of 2): the same CUDA 13.0 build, with the NVIDIA libraries from PyPI"),
    (4, "[AINVFX] FAILED: PYTORCH COULD NOT BE DOWNLOADED on this machine: the network of this host timed out (the lines above)."),
    (4, "[AINVFX] the pod stays up so this log can be read, and it bills until you terminate it"),
]
# Runpod's own system log, as seen on a real pod on 5 October 2026: the host fetches the image one
# Docker layer at a time (many "<layer> Extracting" lines), then starts the container. PULL_START
# lines are scripted from creation; PULL_END lines from the container start (STATE["pull_seconds"]).
PULL_START = [
    (0, "create container runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"),
    (0, "1.0.2-cu1281-torch280-ubuntu2404 Pulling from runpod/pytorch"),
    (0, "38e4c3f3c358 Pulling fs layer"), (0, "a1b2c3d4e5f6 Pulling fs layer"), (0, "0123456789ab Pulling fs layer"),
    (1, "0123456789ab Already exists"), (1, "38e4c3f3c358 Downloading"), (1, "a1b2c3d4e5f6 Downloading"),
    (2, "38e4c3f3c358 Extracting"), (3, "38e4c3f3c358 Extracting"), (4, "38e4c3f3c358 Extracting"),
    (5, "a1b2c3d4e5f6 Download complete"), (6, "38e4c3f3c358 Extracting"),
]
PULL_END = [
    (-2, "38e4c3f3c358 Pull complete"), (-1, "a1b2c3d4e5f6 Extracting"), (-1, "a1b2c3d4e5f6 Pull complete"),
    (-1, "Status: Downloaded newer image for runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"),
    (0, "start container for runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404: begin"),
]
STEP = 0.4            # one scripted second of pod time = 0.4 real seconds
LIVE_LIMIT = 17       # BOOT lines from this index on (SELFTEST OK, READY) are never sent on a live connection
NEVER_LIVE = {l for _, l in BOOT[LIVE_LIMIT:]}
STATE = {"pods": {}, "posts": [], "logs_403": False, "no_proxy_log": False, "uploads": [], "count": 0,
         "no_5090": False, "maint_dcs": set(), "maint_community": False, "graphql_down": False,
         "full_dcs": set(), "pro6000_dcs": None, "balance": 212.40,
         "catalog_reads": 0,      # GET /v2/catalog/gpus on Secure Cloud so far (--retry)
         "schedule": {},          # GPU id -> [(first read, data centers with stock), ...]: see stock_now()
         "pull_seconds": 0,       # scripted seconds the image pull lasts before the container starts
         "history_3d": False,     # the history also reports a GLB under "3d", and a temp preview
         "outfiles": None,        # the pod's output folder as {relative path: bytes}; None = JupyterLab off
         "logs_hang_for": 0}      # real seconds after creation during which the logs route sends nothing, not even headers


def stock_now(gid, default):
    """The Secure data centers with stock for one GPU type at this catalog read. STATE["schedule"] maps a GPU id
    to [(first read, data centers), ...]: the last step reached wins, so a GPU can come into stock, or move,
    while `up --retry` waits. A GPU type not in it keeps its usual stock."""
    plan = STATE.get("schedule", {}).get(gid)
    if plan is None:
        return default
    dcs = ()
    for first_read, step in plan:
        if STATE["catalog_reads"] >= first_read:
            dcs = step
    return tuple((d, "LOW") for d in dcs)


def fake_catalog(cloud):
    """GET /v2/catalog/gpus for one cloud, in the shape read from Runpod on 7 October 2026: Secure stock
    lists its data centers, Community stock comes with none. STATE["no_5090"] reproduces that morning:
    no RTX 5090 on Secure Cloud anywhere, some on Community. The B200 (prices of 19 October 2026) has no
    stock unless STATE["schedule"] gives it some."""
    def gpu(gid, name, mem, p_secure, p_comm, secure_dcs=(), community=False):
        secure_dcs = stock_now(gid, secure_dcs)
        e = {"id": gid, "name": name, "memory": mem, "manufacturer": "NVIDIA",
             "price": {"secure": p_secure, "community": p_comm}}
        if cloud == "SECURE":
            e["availability"] = "LOW" if any(lvl != "NONE" for _, lvl in secure_dcs) else "NONE"
            if secure_dcs:
                e["dataCenters"] = [{"id": d, "name": d, "availability": lvl} for d, lvl in secure_dcs]
        else:
            e["availability"] = "LOW" if community else "NONE"
        return e
    rtx5090_dcs = () if STATE["no_5090"] else (("EU-CZ-1", "LOW"), ("US-TX-3", "HIGH"), ("CA-MTL-1", "LOW"), ("EUR-NO-1", "NONE"))
    return [
        gpu("NVIDIA GeForce RTX 5090", "RTX 5090", 32, 1.19, 0.69, rtx5090_dcs, community=True),
        gpu("NVIDIA GeForce RTX 4090", "RTX 4090", 24, 0.74, 0.34, (("CA-MTL-1", "HIGH"),), community=True),
        gpu("NVIDIA H100 80GB HBM3", "H100 SXM", 80, 3.49, 2.69, (("CA-MTL-1", "LOW"),)),
        gpu("NVIDIA RTX PRO 4500 Blackwell Server Edition", "RTX PRO 4500 SE", 32, 0.72, 0.5, (("US-KS-2", "LOW"),)),
        gpu("NVIDIA B300 SXM6 AC MIG 1g.34gb", "B300 MIG 34GB", 34, 0.5, 0.5, (("US-WA-2", "LOW"),)),
        gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb", "PRO 6000 MIG 24GB", 24, 0.59, 0.5, (("US-PA-1", "LOW"),)),
        gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition", "RTX PRO 6000", 96, 2.49, 1.69,
            STATE.get("pro6000_dcs") or (("US-NC-2", "LOW"), ("CA-MTL-1", "LOW"), ("EU-CZ-1", "LOW"))),
        gpu("NVIDIA B300 SXM6 AC", "B300", 288, 7.89, 6.94, (("EU-NL-1", "LOW"),)),
        gpu("NVIDIA B200", "B200", 180, 7.99, 5.98),
    ]


REMOVED_NOTE = ("This server will be removed from the platform to make room for more hardware. "
                "Please note that ALL DATA WILL BE LOST.")


def pod_seconds(pod):
    return (time.time() - pod["t0"]) / STEP


def container_started(pod):
    return pod_seconds(pod) >= STATE["pull_seconds"]


def all_events(pod):
    """Every scripted (second, source, line) of this pod, in order."""
    pull = STATE["pull_seconds"]
    ev = [(s, "system", l) for s, l in PULL_START if pull] + [(pull + s, "system", l) for s, l in PULL_END if pull]
    ev += [(pull + s, "container", l) for s, l in (STATE.get("boot") or BOOT)]
    return sorted(ev, key=lambda e: e[0])


def visible_events(pod):
    return [e for e in all_events(pod) if e[0] <= pod_seconds(pod)]


def visible_lines(pod):
    """The container lines written so far, as (second, line): what the pod's own log copy holds."""
    return [(s, l) for s, src, l in visible_events(pod) if src == "container"]


def pod_of(path):
    for pid, p in STATE["pods"].items():
        if "/" + pid in path:
            return pid, p
    return None, None


class QuietServer(ThreadingHTTPServer):
    """The fake server. socketserver prints a traceback for any error in a request thread; a connection that
    the client closed while the fake still wrote (BrokenPipeError, or ConnectionResetError when the client's
    side has already reset it: seen with Python 3.14 on Ubuntu, 16 and 19 Oct 2026) is normal here, not a
    failure, so it prints nothing. Every other error still prints its traceback."""
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def ts_of(sec):
    base = datetime(2026, 10, 5, 3, 36, 57, tzinfo=timezone.utc)
    return (base + timedelta(seconds=sec)).strftime("%Y-%m-%dT%H:%M:%SZ")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def problem(self, code, detail):
        self.send_json(code, {"title": "error", "status": code, "detail": detail})

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # ------------------------------------------------------------- fake Runpod API v2
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        p = u.path
        if p.startswith("/proxy/"):
            return self.proxy_get(p, q)
        if p.startswith("/files/"):          # a models list served as a raw file
            body = STATE["files"].get(p)
            if body is None:
                return self.problem(404, "no such file")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if p == "/v2/slow":                 # answers after 3 s: a host that does not respond
            time.sleep(3)
            return self.send_json(200, {"slow": True})
        if self.headers.get("Authorization") != "Bearer fake-key":
            return self.problem(401, "bad key")
        if p == "/v2/pods":
            pods = [self.view_of(pod) for pod in STATE["pods"].values() if not pod["deleted"]]
            return self.send_json(200, {"pods": pods, "pagination": {"nextCursor": None}})
        if p.startswith("/v2/pods/") and p.endswith("/logs"):
            pid, pod = pod_of(p)
            return self.logs(q, pod) if pod and not pod["deleted"] else self.problem(404, "pod not found")
        if p.startswith("/v2/pods/"):
            pid, pod = pod_of(p)
            if not pod or pod["deleted"]:
                return self.problem(404, "pod not found")
            return self.send_json(200, self.view_of(pod))
        if p == "/v2/catalog/gpus":
            cloud = q.get("cloud", ["SECURE"])[0]
            if cloud == "SECURE":
                STATE["catalog_reads"] += 1
            return self.send_json(200, {"gpus": fake_catalog(cloud)})
        if p == "/v2/account/secrets":
            return self.send_json(200, {"secrets": []})
        if p == "/v2/account/ssh-keys":
            return self.send_json(200, {"keys": []})
        return self.problem(404, "no route " + p)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/graphql":
            return self.graphql()
        if u.path.startswith("/proxy/") and u.path.endswith("/upload/image"):
            STATE["uploads"].append(len(self.body()))
            return self.send_json(200, {"name": "x.png", "subfolder": "", "type": "input"})
        if self.headers.get("Authorization") != "Bearer fake-key":
            return self.problem(401, "bad key")
        if u.path == "/v2/pods":
            body = json.loads(self.body())
            STATE["posts"].append(body)
            if body.get("dataCenterIds") == ["EU-CZ-1"]:
                return self.problem(400, "no capacity in EU-CZ-1")   # the first candidate fails: the loop must go on
            if (body.get("dataCenterIds") or [None])[0] in STATE.get("full_dcs", ()):
                return self.problem(400, "There are no instances currently available")
            pod = self.new_pod(body["name"], body["gpu"]["id"], (body.get("dataCenterIds") or ["?"])[0])
            pod["cloud"] = body.get("cloud") or "SECURE"
            return self.send_json(201, self.view_of(pod))
        return self.problem(404, "no route")

    def graphql(self):
        """The one GraphQL query pod.py sends: the pod's host and its maintenance window. A host in a data
        center of STATE["maint_dcs"] (or any Community host with STATE["maint_community"]) is the host of
        5 and 7 October 2026: maintenance started days ago, ends in months, the server is being removed."""
        query = json.loads(self.body() or b"{}").get("query", "")
        if self.headers.get("Authorization") != "Bearer fake-key":
            return self.send_json(200, {"errors": [{"message": "Unauthorized"}], "data": {"pod": None}})
        if STATE["graphql_down"]:
            return self.problem(503, "unavailable")
        if "myself" in query:            # the credit and the spend rate
            running = [p for p in STATE["pods"].values() if not p.get("deleted")]
            return self.send_json(200, {"data": {"myself": {"clientBalance": STATE["balance"],
                                                            "currentSpendPerHr": round(sum(p["cost"] for p in running), 2)}}})
        m = re.search(r'podId:\s*"([^"]+)"', query)
        pod = STATE["pods"].get(m.group(1)) if m else None
        if not pod:
            return self.send_json(200, {"data": {"pod": None}})
        machine = {"podHostId": pod["id"] + "-host", "dataCenterId": None if pod["dataCenterId"] == "?" else pod["dataCenterId"],
                   "location": "somewhere", "maintenanceStart": None, "maintenanceEnd": None, "maintenanceNote": None}
        if pod["dataCenterId"] in STATE["maint_dcs"] or (pod.get("cloud") == "COMMUNITY" and STATE["maint_community"]):
            now = datetime.now(timezone.utc)
            machine.update(maintenanceStart=(now - timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                           maintenanceEnd=(now + timedelta(days=172)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                           maintenanceNote=REMOVED_NOTE)
        return self.send_json(200, {"data": {"pod": {"id": pod["id"], "machine": machine}}})

    @staticmethod
    def new_pod(name, gpu, dc):
        STATE["count"] += 1
        pid = "fakepod{}".format(STATE["count"])
        cost = {"NVIDIA GeForce RTX 5090": 1.19, "NVIDIA RTX PRO 6000 Blackwell Server Edition": 2.49}.get(gpu, 0.72)
        pod = {"id": pid, "name": name, "cost": cost, "gpu": {"id": gpu, "count": 1}, "dataCenterId": dc,
               "cudaVersion": "13.3", "createdAt": ts_of(0), "template": None,
               "ssh": {"direct": {"host": "81.27.69.177", "port": 32554, "username": "root"}},
               "runtime": {"gpus": [{"util": 0, "memoryUtil": 0}]}, "t0": time.time(), "deleted": False}
        STATE["pods"][pid] = pod
        return pod

    @staticmethod
    def view_of(pod):
        v = {k: val for k, val in pod.items() if k not in ("t0", "deleted", "cloud")}
        v["status"] = "RUNNING" if pod_seconds(pod) >= 1 else "STARTING"
        if not container_started(pod):
            v["runtime"] = None              # no container reported yet: image pull, create or boot
        if pod_seconds(pod) < 2:
            v["dataCenterId"] = None         # the scheduler reports it a few seconds after RUNNING
        return v

    def do_PUT(self):
        self.body()
        return self.send_json(200, {"keys": []})

    def do_DELETE(self):
        if self.headers.get("Authorization") != "Bearer fake-key":
            return self.problem(401, "bad key")
        pid, pod = pod_of(self.path)
        if not pod:
            return self.problem(404, "pod not found")
        pod["deleted"] = True
        self.send_response(204)
        self.end_headers()

    def logs(self, q, pod):
        if STATE["logs_403"]:
            return self.problem(403, "this key cannot read logs")
        while time.time() - pod["t0"] < STATE["logs_hang_for"]:
            time.sleep(0.1)                        # a host that does not answer: no headers at all
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        since = q.get("since", [None])[0]
        tail = int(q.get("tail", ["100"])[0])
        source = q.get("source", [None])[0]
        events = [e for e in all_events(pod) if not source or e[1] == source]
        if since:
            # exclusive at second resolution, like the real cursor
            start = [i for i, e in enumerate(events) if ts_of(e[0]) > since]
            idx = start[0] if start else len([e for e in events if e[0] <= pod_seconds(pod)])
        else:
            idx = max(0, len([e for e in events if e[0] <= pod_seconds(pod)]) - tail)
        backlog = len([e for e in events if e[0] <= pod_seconds(pod)])   # what the stored log holds at connection time
        deadline, last_keepalive = time.time() + 60, time.time()
        while time.time() < deadline:
            vis = [e for e in events if e[0] <= pod_seconds(pod)]
            while idx < len(vis):
                s, src, line = vis[idx]
                if line in NEVER_LIVE and idx >= backlog:
                    break                          # written after SELFTEST OK: never delivered live
                ev = "id: {}\ndata: {}\n\n".format(ts_of(s), json.dumps({"ts": ts_of(s), "source": src, "line": line}))
                try:
                    self.wfile.write(ev.encode()); self.wfile.flush()
                except ConnectionError:            # the client closed the stream: pod.py does it on purpose
                    return
                idx += 1
            if time.time() - last_keepalive > 1:   # keepalives defeat an idle timeout
                last_keepalive = time.time()
                try:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                except ConnectionError:
                    return
            time.sleep(0.1)

    # ------------------------------------------------------------- fake ComfyUI behind the proxy
    def proxy_get(self, p, q):
        pid, pod = pod_of(p)
        up = pod and not pod["deleted"] and pod_seconds(pod) >= STATE["pull_seconds"] + 4
        if not up:
            return self.problem(502, "no comfy yet")
        if "/%d/" % 8888 in p:
            return self.jupyter_get(p)
        if p.endswith("/system_stats"):
            return self.send_json(200, {"system": {"comfyui_version": "0.39.0"}})
        if p.endswith("/history"):
            done = pod_seconds(pod) >= STATE["pull_seconds"] + 7
            hist = {"p1": {"outputs": {"9": {"images": [{"filename": "ainvfx_selftest_00001_.png",
                                        "subfolder": "", "type": "output"}]}}, "status": {"completed": True}}}
            if STATE["history_3d"]:      # the shape of ComfyUI 0.42.1's Save 3D (Advanced) and a temp preview
                hist["p2"] = {"outputs": {"12": {"3d": [{"filename": "D6_E2_mesh_00001_.glb", "subfolder": "BTC2",
                                                         "type": "output"}],
                                                 "camera_info": [{"position": [0, 1, 2]}], "model_3d_info": []},
                                          "7": {"images": [{"filename": "ComfyUI_temp_00001_.png", "subfolder": "",
                                                            "type": "temp"}]}},
                              "status": {"completed": True}}
            return self.send_json(200, hist if done else {})
        if p.endswith("/view"):
            if q.get("type") == ["input"] and q.get("filename") == ["bootstrap.log"] and q.get("subfolder") == ["ainvfx"]:
                if STATE["no_proxy_log"]:
                    return self.problem(404, "no such file")
                data = "\n".join(l for _, l in visible_lines(pod)).encode()
            else:
                data = b"\x89PNG fake image bytes"
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data); return
        return self.problem(404, "no route")

    def jupyter_get(self, p):
        """A fake JupyterLab with its root at /, as Runpod starts it: the contents API (one folder per request) and
        the raw files handler, behind the token of the READY line."""
        files = STATE["outfiles"]
        if files is None:
            return self.problem(404, "no jupyter")
        if self.headers.get("Authorization") != "token faketoken":
            return self.problem(403, "Forbidden")
        root = "workspace/ComfyUI/output"
        rest = urllib.parse.unquote(p.split("/8888/", 1)[1])
        if rest.startswith("api/contents/"):
            path = rest[len("api/contents/"):].strip("/")
            if not (path == root or path.startswith(root + "/")):
                return self.problem(404, "No such file or directory: " + path)
            sub = path[len(root):].strip("/")
            prefix = sub + "/" if sub else ""
            entries, dirs = [], set()
            for rel, data in files.items():
                if not rel.startswith(prefix):
                    continue
                tail = rel[len(prefix):]
                if "/" in tail:
                    dirs.add(tail.split("/")[0])
                else:
                    entries.append({"name": tail, "path": root + "/" + rel, "type": "file", "size": len(data)})
            entries += [{"name": n, "path": root + "/" + prefix + n, "type": "directory", "size": None} for n in sorted(dirs)]
            if sub and not entries:
                return self.problem(404, "No such file or directory: " + path)
            return self.send_json(200, {"name": path.split("/")[-1], "path": path, "type": "directory", "content": entries})
        if rest.startswith("files/"):
            path = rest[len("files/"):]
            rel = path[len(root) + 1:] if path.startswith(root + "/") else None
            if rel not in files:
                return self.problem(404, "no such file")
            data = files[rel]
            self.send_response(200); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data); return
        return self.problem(404, "no route")



# ----------------------------------------------------------------------------- unit tests
class Helpers(unittest.TestCase):
    def test_country_and_region_of_data_center_ids(self):
        self.assertEqual((pod.region_of("CA-MTL-1"), pod.country_of("CA-MTL-1")), ("NA", "CA"))
        self.assertEqual((pod.region_of("US-TX-3"), pod.country_of("US-TX-3")), ("NA", "US"))
        self.assertEqual((pod.region_of("EU-FR-1"), pod.country_of("EU-FR-1")), ("EU", "FR"))
        self.assertEqual((pod.region_of("EUR-IS-2"), pod.country_of("EUR-IS-2")), ("EU", "IS"))
        self.assertEqual(pod.region_of("AP-JP-1"), "OTHER")

    def test_data_centers_sorted_country_then_region(self):
        dcs = ["EU-CZ-1", "US-TX-3", "CA-MTL-1", "EUR-IS-2", "CA-MTL-3", "US-CA-2", "AP-JP-1"]
        self.assertEqual(sorted(dcs, key=lambda d: pod.dc_order(d, "NA", "CA")),
                         ["CA-MTL-1", "CA-MTL-3", "US-TX-3", "US-CA-2", "EU-CZ-1", "EUR-IS-2", "AP-JP-1"])
        self.assertEqual(sorted(dcs, key=lambda d: pod.dc_order(d, "EU", "IS"))[:2], ["EUR-IS-2", "EU-CZ-1"])
        self.assertEqual(sorted(dcs, key=lambda d: pod.dc_order(d, "EU", ""))[:2], ["EU-CZ-1", "EUR-IS-2"])

    def test_place_guessed_from_the_time_zone(self):
        for zone, expected in (("America/Toronto", ("NA", "CA")), ("America/Montreal", ("NA", "CA")),
                               ("America/Chicago", ("NA", "US")), ("Europe/Paris", ("EU", "FR")),
                               ("Europe/Berlin", ("EU", "")), ("Eastern Standard Time", ("NA", "US"))):
            os.environ["TZ"] = zone
            self.assertEqual(pod.local_place(), expected, zone)
        os.environ.pop("TZ", None)

    def test_resume_point_is_a_few_seconds_before_the_event(self):
        self.assertEqual(pod.resume_point("2026-10-05T03:40:12.123Z"), "2026-10-05T03:40:07Z")
        self.assertEqual(pod.resume_point("2026-10-05T03:40:12Z", seconds=60), "2026-10-05T03:39:12Z")
        self.assertIsNone(pod.resume_point(None))
        self.assertIsNone(pod.resume_point("not a time"))

    def test_parse_time(self):
        t = pod.parse_time("2026-10-05T01:56:56.833Z")
        self.assertEqual((t.year, t.hour, t.tzinfo is not None), (2026, 1, True))
        self.assertIsNone(pod.parse_time("?"))

    def test_api_error_reads_problem_json_and_hints(self):
        e = pod.ApiError(403, json.dumps({"title": "Forbidden", "status": 403, "detail": "no access"}), "u")
        self.assertEqual(e.detail, "no access")
        self.assertIn("api.runpod.io/graphql", e.hint())
        self.assertIn("Credentials", pod.ApiError(401, "nope", "u").hint())
        self.assertIn("balance", pod.ApiError(402, "", "u").hint())
        self.assertEqual(pod.ApiError(418, "teapot", "u").hint(), "")

    def test_pod_json_parsing(self):
        p = {"status": "RUNNING", "cost": 0.99, "gpu": {"id": "NVIDIA GeForce RTX 5090"}, "dataCenterId": "EUR-IS-2",
             "ssh": {"direct": {"host": "81.27.69.177", "port": 32554, "username": "root"}, "proxy": {}}}
        self.assertEqual(pod.pod_status(p), "RUNNING")
        self.assertEqual(pod.pod_summary(p), ("NVIDIA GeForce RTX 5090", "EUR-IS-2", 0.99))
        self.assertEqual(pod.pod_ssh(p), ("direct", "81.27.69.177", 32554, "root"))
        self.assertEqual(pod.pod_ssh({})[0], None)

    def test_settings_file_parsing(self):
        d = tempfile.mkdtemp(prefix="ainvfx-settings-")
        f = os.path.join(d, "settings.env")
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("# comment\nAINVFX_COMFY_TAG=master\nAINVFX_TORCH=\"--pre torch torchvision\"\n"
                     "AINVFX_SELFTEST=\nOTHER=1\n  AINVFX_PYTHON = 3.14 \n")
        old, old_local = pod.SETTINGS, pod.SETTINGS_LOCAL
        pod.SETTINGS, pod.SETTINGS_LOCAL = pod.Path(f), pod.Path(d) / "settings.local.env"
        try:
            self.assertEqual(pod.load_settings(), {"AINVFX_COMFY_TAG": "master", "AINVFX_TORCH": "--pre torch torchvision",
                                                   "AINVFX_PYTHON": "3.14"})
            with open(pod.SETTINGS_LOCAL, "w", encoding="utf-8") as fh:   # the local file wins, key by key
                fh.write("AINVFX_COMFY_TAG=v0.38.2\nAINVFX_MODELS_URL=https://example.org/mine/models.json\n")
            self.assertEqual(pod.load_settings(), {"AINVFX_COMFY_TAG": "v0.38.2", "AINVFX_TORCH": "--pre torch torchvision",
                                                   "AINVFX_PYTHON": "3.14",
                                                   "AINVFX_MODELS_URL": "https://example.org/mine/models.json"})
            pod.SETTINGS = pod.Path(d) / "missing.env"
            self.assertEqual(pod.load_settings(), {"AINVFX_COMFY_TAG": "v0.38.2",
                                                   "AINVFX_MODELS_URL": "https://example.org/mine/models.json"})
            pod.SETTINGS_LOCAL = pod.Path(d) / "missing.local.env"
            self.assertEqual(pod.load_settings(), {})
        finally:
            pod.SETTINGS, pod.SETTINGS_LOCAL = old, old_local
            shutil.rmtree(d, ignore_errors=True)

    def test_image_pull_counts_layer_lines(self):
        pull = pod.ImagePull()
        self.assertTrue(pull.feed("38e4c3f3c358 Extracting"))
        self.assertTrue(pull.feed("38e4c3f3c358 Extracting"))        # repeats of the same layer count once
        self.assertTrue(pull.feed("a1b2c3d4e5f6 Pull complete"))
        self.assertTrue(pull.feed("0123456789ab Already exists"))
        self.assertTrue(pull.feed("fedcba987654 Downloading"))
        self.assertFalse(pull.feed("start container for runpod/pytorch: begin"))
        self.assertFalse(pull.feed("[AINVFX] step 1/5 health check"))
        self.assertEqual(pull.counts(), (4, 2, 1, 1))
        self.assertIn("4 layers · 2 done · 1 extracting · 1 downloading", pull.summary())

    def test_api_error_timeout_hint(self):
        e = pod.ApiError(0, "timed out after 30 s (no answer)", "u")
        self.assertTrue(e.timed_out)
        self.assertIn("did not answer in time", e.hint())
        self.assertFalse(pod.ApiError(0, "[Errno -2] Name or service not known", "u").timed_out)

    def test_newer_gpu_rule(self):
        """Blackwell or newer: the RTX 50 series, RTX PRO Blackwell, B200, B300 and their MIG slices."""
        for g in ("NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 4500 Blackwell",
                  "NVIDIA B300 SXM6 AC MIG 1g.34gb", "NVIDIA B200", "NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb"):
            self.assertTrue(pod.newer_gpu({"id": g}), g)
        for g in ("NVIDIA H100 80GB HBM3", "NVIDIA H200", "NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX 5000 Ada Generation",
                  "NVIDIA GeForce RTX 4090", "NVIDIA A100-SXM4-80GB", "AMD Instinct MI300X OAM", "Tesla V100-PCIE-16GB",
                  "NVIDIA L40S", "NVIDIA RTX A6000"):
            self.assertFalse(pod.newer_gpu({"id": g}), g)

    def test_duration_and_spent(self):
        self.assertEqual(pod.fmt_duration(0.004), "under a minute")
        self.assertEqual(pod.fmt_duration(0.75), "45 min")
        self.assertEqual(pod.fmt_duration(2.5), "2 h 30 min")
        from datetime import timedelta as td
        p = {"cost": 1.19, "createdAt": (datetime.now(timezone.utc) - td(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")}
        hours, usd, price = pod.pod_spent(p)
        self.assertAlmostEqual(hours, 2.0, places=2)
        self.assertAlmostEqual(usd, 2.38, places=2)
        self.assertIsNone(pod.pod_spent({"cost": None, "createdAt": p["createdAt"]}))

    def test_maintenance_window(self):
        """The host of 7 Oct 2026: maintenance from 29 Sep 2026 to 28 Mar 2027, server being removed."""
        now = datetime(2026, 10, 7, 13, 0, tzinfo=timezone.utc)
        removed = {"maintenanceStart": "2026-09-29T15:17:00.000Z", "maintenanceEnd": "2027-03-28T15:17:00.000Z",
                   "maintenanceNote": REMOVED_NOTE}
        text = pod.maintenance_of(removed, now)
        self.assertIn("since 29 Sep 2026 15:17 UTC, until 28 Mar 2027 15:17 UTC", text)
        self.assertIn("removed from the platform", text)
        self.assertIsNone(pod.maintenance_of({"maintenanceStart": None, "maintenanceEnd": None}, now))
        self.assertIsNone(pod.maintenance_of({}, now))
        self.assertIsNone(pod.maintenance_of({"maintenanceStart": "2026-09-01T00:00:00Z",
                                              "maintenanceEnd": "2026-09-02T00:00:00Z"}, now), "a window that is over")
        self.assertIsNone(pod.maintenance_of({"maintenanceStart": "2026-10-10T00:00:00Z"}, now), "three days ahead")
        self.assertIn("from 07 Oct 2026 15:00 UTC", pod.maintenance_of({"maintenanceStart": "2026-10-07T15:00:00Z"}, now))
        ms = int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp() * 1000)
        self.assertIn("since 01 Oct 2026", pod.maintenance_of({"maintenanceStart": ms}, now), "epoch milliseconds")
        self.assertIsNotNone(pod.maintenance_of({"maintenanceStart": "next week"}, now), "unreadable: refused")

    def test_offers_and_gpu_names(self):
        cat = {"SECURE": [
            {"id": "NVIDIA GeForce RTX 5090", "name": "RTX 5090", "memory": 32, "price": {"secure": 0.99, "community": 0.69},
             "availability": "NONE"},
            {"id": "NVIDIA RTX PRO 4500 Blackwell Server Edition", "name": "RTX PRO 4500 SE", "memory": 32,
             "price": {"secure": 0.72}, "availability": "LOW", "dataCenters": [{"id": "US-KS-2", "availability": "LOW"}]},
            {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "name": "RTX PRO 6000", "memory": 96,
             "price": {"secure": 2.09}, "availability": "LOW"}],          # Secure stock with no data center: skipped
            "COMMUNITY": [{"id": "NVIDIA GeForce RTX 5090", "name": "RTX 5090", "memory": 32,
                           "price": {"secure": 0.99, "community": 0.69}, "availability": "LOW"}]}
        cat["SECURE"].append({"id": "NVIDIA B300 SXM6 AC MIG 1g.34gb", "name": "B300 MIG 34GB", "memory": 34,
                              "price": {"secure": 0.5}, "availability": "LOW",
                              "dataCenters": [{"id": "US-WA-2", "availability": "LOW"}]})
        offers = pod.make_offers(cat, "NA", "CA")
        self.assertEqual([(o["name"], o["cloud"], o["dcs"], o["slice"]) for o in offers],
                         [("B300 MIG 34GB", "SECURE", ["US-WA-2"], True), ("RTX 5090", "COMMUNITY", [], False),
                          ("RTX PRO 4500 SE", "SECURE", ["US-KS-2"], False)])
        self.assertEqual([o["name"] for o in offers if pod.fits_profile(o, 32)], ["RTX 5090", "RTX PRO 4500 SE"])
        self.assertEqual([o["name"] for o in offers if pod.fits_profile(o, 96)], [])
        cat["SECURE"].pop()
        both = cat["SECURE"] + cat["COMMUNITY"]
        self.assertEqual(pod.find_gpu(both, "NVIDIA GeForce RTX 5090")["id"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(pod.find_gpu(both, "rtx pro 4500 se")["id"], "NVIDIA RTX PRO 4500 Blackwell Server Edition")
        self.assertIs(pod.find_gpu(both, "5090"), cat["SECURE"][0], "one GPU type in two clouds is one match")
        with self.assertRaises(SystemExit):
            pod.find_gpu(both, "RTX PRO")            # two GPU types
        with self.assertRaises(SystemExit):
            pod.find_gpu(both, "GTX 1080")           # none

    def test_several_gpus_named_and_the_retry_ceiling(self):
        """`--gpu B300 --gpu B200` tries the B300 first, although the B200 costs less: the order given wins.
        Without --gpu, nothing above max_price follows the profile's GPU. The price ceiling of a --retry wait is
        the dearest GPU named, or max_price without --gpu."""
        cat = {"SECURE": [
            {"id": "NVIDIA B200", "name": "B200", "memory": 180, "price": {"secure": 7.99, "community": 5.98},
             "availability": "LOW", "dataCenters": [{"id": "US-TX-3", "availability": "LOW"}]},
            {"id": "NVIDIA B300 SXM6 AC", "name": "B300", "memory": 288, "price": {"secure": 8.99, "community": 6.94},
             "availability": "LOW", "dataCenters": [{"id": "EU-NL-1", "availability": "LOW"}]},
            {"id": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "name": "RTX PRO 6000", "memory": 96,
             "price": {"secure": 2.49, "community": 1.69}, "availability": "LOW",
             "dataCenters": [{"id": "US-NC-1", "availability": "LOW"}]}]}
        b200, b300, pro = cat["SECURE"]
        spec = {"vram": 96, "gpu": pro["id"]}
        cands = pod.pick_candidates(cat, [b300, b200], True, spec, "NA", "CA", 2.5)[3]
        self.assertEqual([o["name"] for o in cands], ["B300", "B200"])
        offers, fits, preferred, cands = pod.pick_candidates(cat, [pro], False, spec, "NA", "CA", 2.5)
        self.assertEqual(preferred["name"], "RTX PRO 6000")
        self.assertEqual([o["name"] for o in cands], ["RTX PRO 6000"], "the B200 and the B300 cost more than 2.50 USD")
        self.assertEqual([o["name"] for o in fits], ["RTX PRO 6000", "B200", "B300"])
        self.assertEqual(pod.retry_cap([b200, b300], True, False, 2.5), 8.99)
        self.assertEqual(pod.retry_cap([b200], True, True, 2.5), 7.99, "Community is cheaper: the ceiling stays the Secure price")
        self.assertEqual(pod.retry_cap([pro], False, False, 2.5), 2.5)
        self.assertEqual(pod.retry_cap([pro], False, False, 2.0), 2.49, "the profile's GPU is taken above max_price")
        self.assertIsNone(pod.retry_cap([{"id": "x", "name": "x"}], True, False, 2.5))

    def test_two_commands_keep_each_others_pods(self):
        """Two `up` started at once from one machine (7 Oct 2026): both records must survive."""
        d = tempfile.mkdtemp(prefix="ainvfx-cfg-")
        old = (pod.CONFIG_DIR, pod.CONFIG)
        pod.CONFIG_DIR, pod.CONFIG = pod.Path(d), pod.Path(d) / "config.json"
        try:
            a, b = pod.load_config(), pod.load_config()        # both read the file before either writes
            pod.remember_pod(a, "test10", {"id": "p10"})
            pod.remember_pod(b, "test11", {"id": "p11"})
            self.assertEqual(sorted(pod.load_config()["pods"]), ["test10", "test11"])
            pod.remember_pod(a, "test10", None)
            self.assertEqual(sorted(pod.load_config()["pods"]), ["test11"])
        finally:
            pod.CONFIG_DIR, pod.CONFIG = old
            shutil.rmtree(d, ignore_errors=True)

    def test_combined_profiles(self):
        """Atomic profiles that combine. The disk follows the models, the GPU the
        profile that needs the most VRAM; `all` is every profile; an unknown profile is refused."""
        self.assertEqual(pod.parse_profiles("image"), ["image"])
        self.assertEqual(pod.parse_profiles("3d+image"), ["image", "3d"])
        self.assertEqual(pod.parse_profiles("image, video,train"), ["image", "video", "train"])
        self.assertEqual(pod.parse_profiles("all"), ["image", "3d", "video", "train"])
        self.assertEqual(pod.profile_spec(["image"])["disk"], 170)   # follows models.json
        self.assertEqual(pod.profile_spec(["image"])["gpu"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(pod.profile_spec(["3d"])["disk"], 100, "never under 100 GB")
        self.assertEqual(pod.profile_spec(["image", "3d"])["gpu"], "NVIDIA GeForce RTX 5090")
        self.assertEqual(pod.profile_spec(["image", "video"])["gpu"], "NVIDIA RTX PRO 6000 Blackwell Server Edition")
        self.assertEqual(pod.profile_spec(["image", "video"])["vram"], 96)
        self.assertEqual(pod.profile_spec(["video", "train"])["disk"], 210)
        spec = pod.profile_spec(["image", "3d", "video", "train"])
        self.assertEqual((spec["label"], spec["disk"]), ("image+3d+video+train", 380))   # and MoGe 3 joins 3d
        with self.assertRaises(SystemExit):
            pod.parse_profiles("nope+3d")

    def test_history_files_takes_every_key_but_temp(self):
        """The 3D saves report under "3d"; temp previews and non-file values are left out; one entry per file."""
        h = {"p1": {"outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
                                "12": {"3d": [{"filename": "m.glb", "subfolder": "BTC2", "type": "output"}],
                                       "camera_info": [{"position": [0, 1, 2]}]},
                                "7": {"images": [{"filename": "t.png", "subfolder": "", "type": "temp"}]},
                                "8": {"text": ["Vertices: 12"]}}},
             "p2": {"outputs": {"9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
                                "5": {"latents": [{"filename": "l.latent", "subfolder": "latents"}]}}}}
        self.assertEqual(pod.history_files(h), [("", "a.png"), ("BTC2", "m.glb"), ("latents", "l.latent")])
        self.assertEqual(pod.history_files({}), [])
        self.assertEqual(pod.history_files(None), [])

    def test_jupyter_token_and_safe_paths(self):
        ready = ("[AINVFX] READY · ComfyUI https://x-8188.proxy.runpod.net · JupyterLab "
                 "https://x-8888.proxy.runpod.net/lab?token=abc123 · log /workspace/ComfyUI/input/ainvfx/bootstrap.log")
        self.assertEqual(pod.jupyter_token(["[AINVFX] step 5/5", ready]), "abc123")
        self.assertIsNone(pod.jupyter_token(["[AINVFX] READY · ComfyUI https://x · JupyterLab off"]))
        self.assertIsNone(pod.jupyter_token([]))
        self.assertEqual(pod.safe_rel("BTC2/D6_E2_mesh_00001_.glb"), "BTC2/D6_E2_mesh_00001_.glb")
        self.assertEqual(pod.safe_rel("a\\b.png"), "a/b.png")
        for bad in ("../x.png", "BTC2/../../x", "/etc/passwd", "C:/x.png", ""):
            self.assertIsNone(pod.safe_rel(bad), bad)

    def test_profiles(self):
        self.assertEqual(sorted(pod.PROFILES), ["3d", "image", "train", "video"])


# ----------------------------------------------------------------------------- end to end
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = QuietServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        STATE.update({"pods": {}, "posts": [], "logs_403": False, "no_proxy_log": False, "uploads": [], "count": 0,
                      "pull_seconds": 0, "logs_hang_for": 0, "boot": None, "no_5090": False, "maint_dcs": set(),
                      "maint_community": False, "graphql_down": False, "full_dcs": set(), "pro6000_dcs": None,
                      "balance": 212.40, "files": {}, "catalog_reads": 0, "schedule": {},
                      "history_3d": False, "outfiles": None})
        self.home = tempfile.mkdtemp(prefix="ainvfx-test-")
        self.env = dict(os.environ, HOME=self.home, USERPROFILE=self.home, TZ="America/Toronto", RUNPOD_API_KEY="fake-key",
                        AINVFX_OUTPUTS=os.path.join(self.home, "outputs"),
                        AINVFX_API_BASE="http://127.0.0.1:{}/v2".format(self.port),
                        AINVFX_PROXY_FMT="http://127.0.0.1:%d/proxy/{id}/{port}" % self.port,
                        AINVFX_STREAM_MAX_AGE="4", AINVFX_PROXY_POLL="3", AINVFX_STATUS_POLL="3",
                        AINVFX_PULL_SUMMARY="2", AINVFX_HEARTBEAT="4", AINVFX_STALL_MINUTES="0.1",
                        AINVFX_STREAM_IDLE="2", AINVFX_HOST_WAIT="3",
                        AINVFX_GRAPHQL="http://127.0.0.1:{}/graphql".format(self.port),
                        AINVFX_SETTINGS=os.path.join(self.home, "settings.env"))
        with open(self.env["AINVFX_SETTINGS"], "w", encoding="utf-8") as f:
            f.write("AINVFX_COMFY_TAG=v0.39.0\nAINVFX_SELFTEST=0\n")

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def run_pod(self, *cmd, timeout=120, input=None):
        r = subprocess.run([sys.executable, POD_PY] + list(cmd), capture_output=True, text=True, env=self.env, timeout=timeout,
                           input=input)
        return r.stdout + r.stderr, r.returncode

    def whole_session(self):
        out, rc = self.run_pod("setup", "-y")
        self.assertEqual(rc, 0, out)
        self.assertIn("North America, CA first", out, out)
        out, rc = self.run_pod("doctor")
        self.assertIn("North America, CA first", out, out)
        self.assertIn("4i789znkrd", out, out)
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=90)
        self.assertIn("in stock in North America: CA-MTL-1, US-TX-3", out, out)
        self.assertIn("elsewhere: EU-CZ-1", out, out)
        self.assertEqual(STATE["posts"][0].get("templateId"), "4i789znkrd")
        self.assertEqual(STATE["posts"][0].get("dataCenterIds"), ["CA-MTL-1"], "the country comes first")
        sent = STATE["posts"][0].get("env") or {}
        self.assertEqual(sent.get("AINVFX_PROFILE"), "image")
        self.assertEqual(sent.get("AINVFX_SELFTEST"), "0", "settings.env values travel with the pod")
        self.assertEqual(sent.get("AINVFX_COMFY_TAG"), "v0.39.0")
        self.assertIn("SELFTEST OK", out, out)
        self.assertIn("READY", out, "READY must be printed although the live stream never delivers it")
        self.assertNotIn("Still working", out, "`up` must return on READY, not on its deadline")
        self.assertEqual(rc, 0, out)
        out, rc = self.run_pod("status")
        self.assertIn("answers", out, out)
        self.assertIn("READY", out, "status reads the pod's own copy of the log")
        out, rc = self.run_pod("pull")
        self.assertIn("1 file(s) new", out, out)
        img = os.path.join(self.home, "in.png")
        with open(img, "wb") as f:
            f.write(b"\x89PNG test")
        out, rc = self.run_pod("push", img)
        self.assertTrue(STATE["uploads"], out)
        self.assertIn("-> input/", out, out)
        out, rc = self.run_pod("list")
        self.assertIn("fakepod1", out, out)
        self.assertIn("here: `image`", out, out)
        out, rc = self.run_pod("down", "-y")
        self.assertTrue(STATE["pods"]["fakepod1"]["deleted"], out)
        self.assertIn("Nothing bills", out, out)

    def test_pull_takes_every_file_of_the_output_folder(self):
        """The GLB that ComfyUI reports under "3d", and a file that only the folder holds (a trained LoRA), come
        back; the placeholder does not; a second pull fetches nothing; JupyterLab off, the history still brings the GLB."""
        out, rc = self.run_pod("setup", "-y")
        self.assertEqual(rc, 0, out)
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=90)
        self.assertEqual(rc, 0, out)
        STATE["history_3d"] = True
        STATE["outfiles"] = {"ainvfx_selftest_00001_.png": b"\x89PNG fake image bytes",
                             "BTC2/D6_E2_mesh_00001_.glb": b"glTF fake mesh",
                             "loras/jar/jar_lora_000250.safetensors": b"fake lora weights",
                             "_output_images_will_be_put_here": b""}
        out, rc = self.run_pod("pull")
        self.assertEqual(rc, 0, out)
        self.assertIn("3 file(s) new", out, out)
        self.assertNotIn("JupyterLab did not answer", out, out)
        got = {os.path.relpath(os.path.join(r, f), os.path.join(self.home, "outputs")).replace(os.sep, "/").split("/", 1)[1]
               for r, _, fs in os.walk(os.path.join(self.home, "outputs")) for f in fs}
        self.assertEqual(got, {"ainvfx_selftest_00001_.png", "BTC2/D6_E2_mesh_00001_.glb",
                               "loras/jar/jar_lora_000250.safetensors"}, out)
        out, rc = self.run_pod("pull")
        self.assertIn("0 file(s) new", out, out)
        STATE["outfiles"] = None                                    # JupyterLab off: the history alone
        shutil.rmtree(os.path.join(self.home, "outputs"))
        out, rc = self.run_pod("pull")
        self.assertIn("2 file(s) new", out, out)
        self.assertIn("D6_E2_mesh_00001_.glb", out, out)
        self.assertNotIn("ComfyUI_temp", out, out)
        self.assertIn("JupyterLab did not answer", out, out)
        out, rc = self.run_pod("down", "-y")
        self.assertTrue(STATE["pods"]["fakepod1"]["deleted"], out)

    def test_whole_session_with_both_log_sources(self):
        """The live stream keeps its keepalives and never delivers READY; the pod's copy does."""
        self.whole_session()

    def test_whole_session_when_the_pods_log_copy_is_unavailable(self):
        """Only the API stream: READY must come from a reopened connection, a few seconds back."""
        STATE["no_proxy_log"] = True
        self.whole_session()

    def test_whole_session_when_the_key_cannot_read_logs(self):
        """The logs endpoint answers 403: READY must come through the pod's own copy of the log."""
        STATE["logs_403"] = True
        self.whole_session()

    def test_request_timeout_is_a_message_not_a_traceback(self):
        """A server that accepts the request and never answers: an ApiError, code 0, flagged as a timeout."""
        with self.assertRaises(pod.ApiError) as cm:
            pod.request("GET", "http://127.0.0.1:{}/v2/slow".format(self.port), timeout=1)
        self.assertEqual(cm.exception.code, 0)
        self.assertTrue(cm.exception.timed_out, cm.exception.detail)

    def test_up_shows_the_image_download_and_warns_on_a_stalled_container(self):
        """The host fetches the image for 40 scripted seconds (16 real): the terminal must show the
        download, then the stall warning (STALL_MINUTES is 0.1 here), then READY once the container runs."""
        STATE["pull_seconds"] = 40
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=120)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("[RUNPOD] create container runpod/pytorch", out, out)
        self.assertIn("the host is fetching the pod's image", out, out)
        self.assertIn("[RUNPOD] image download: 3 layers", out, out)
        self.assertIn("WARNING: NO CONTAINER", out, out)
        self.assertIn("3 done · 0 extracting · 0 downloading", out, "the final summary once every layer is done")
        self.assertIn("[RUNPOD] start container for runpod/pytorch", out, out)
        self.assertIn("[AINVFX] step 1/5 health check", out, out)
        self.assertIn("READY", out, out)
        self.assertEqual(out.count("the host is fetching the pod's image"), 1, "the explanation is printed once")
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_up_survives_a_log_stream_that_does_not_answer(self):
        """The logs route sends nothing for the first 12 s of the pod (`up` reaches it after about 9 s,
        the stream timeout is 2 s here): once a traceback, now a [RUNPOD] line, then the normal follow."""
        STATE["logs_hang_for"] = 12
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=120)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("the log stream did not answer", out, out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_several_pods_at_once(self):
        """Two training pods by name, one pod attached from the account, down --all."""
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "train", "--name", "jar-lora", "-y", "--wait", "1", timeout=90)
        self.assertIn("READY", out, out)
        self.assertEqual(STATE["posts"][-1]["name"], "ainvfx-train-jar-lora")
        out, rc = self.run_pod("up", "train", "--name", "bottle-lora", "-y", "--wait", "1", timeout=90)
        self.assertIn("READY", out, out)
        out, rc = self.run_pod("up", "train", "--name", "jar-lora", "-y")
        self.assertIn("already exists as `jar-lora`", out, out)
        out, rc = self.run_pod("status")
        self.assertNotEqual(rc, 0, "two pods recorded: the command must ask which one")
        self.assertIn("jar-lora", out, out)
        out, rc = self.run_pod("status", "bottle-lora")
        self.assertIn("`bottle-lora`", out, out)
        out, rc = self.run_pod("status", "ainvfx-train-jar-lora")
        self.assertIn("`jar-lora`", out, "a pod name resolves to its tag")
        Handler.new_pod("ainvfx-image-console", "NVIDIA GeForce RTX 5090", "CA-MTL-1")   # created from the console
        out, rc = self.run_pod("list")
        self.assertIn("not recorded here", out, out)
        out, rc = self.run_pod("status", "fakepod3")
        self.assertIn("attached to ainvfx-image-console", out, out)
        out, rc = self.run_pod("list")
        self.assertIn("here: `ainvfx-image-console`", out, out)
        out, rc = self.run_pod("down", "--all", "-y")
        self.assertEqual([p["deleted"] for p in STATE["pods"].values()], [True, True, True], out)
        self.assertIn("Nothing bills", out, out)

    def test_no_stock_proposes_the_cheapest_secure_fit_and_never_any_host(self):
        """7 Oct 2026: no RTX 5090 on Secure Cloud. `up` must not create one there: it lists the GPUs in
        stock that fit the image profile, cheapest first, and creates the cheapest. Since a Community pod
        never started (9 Oct 2026), Community Cloud is left out unless --community is
        given, so the replacement is the Secure RTX PRO 4500 SE, not the cheaper Community RTX 5090. It
        leaves out what does not fit (24 GB, Hopper, a 1g MIG slice) and never picks on its own what
        costs more than 2.50 USD per hour."""
        STATE["no_5090"] = True
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=90)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("in stock in North America: none right now", out, out)
        self.assertIn("(not considered: add --community)", out, out)
        self.assertIn("never creates a pod where the catalog shows no stock", out, out)
        self.assertIn("Replacement: RTX PRO 4500 SE (32 GB of VRAM, 0.72 USD per hour, Secure Cloud, US-KS-2)", out, out)
        self.assertNotIn("Community Cloud, a host Runpod picks", out, "Community offers are not listed without --community")
        self.assertIn("add --community to include it", out, out)
        self.assertNotIn("B300 MIG", out, "a 1g slice is never listed: 111 s self-test against 16 s (7 Oct 2026)")
        self.assertIn("above 2.50 USD: only with --gpu", out, "the B300 is listed, never picked on its own")
        self.assertNotIn("RTX 4090", out, "24 GB: does not fit the image profile")
        self.assertNotIn("H100", out, "Hopper: older than Blackwell")
        self.assertNotIn("PRO 6000 MIG 24GB", out, out)
        self.assertIn('python pod.py up image --gpu "RTX PRO 6000"', out, out)
        self.assertEqual([(p["gpu"]["id"], p["cloud"], p.get("dataCenterIds")) for p in STATE["posts"]],
                         [("NVIDIA RTX PRO 4500 Blackwell Server Edition", "SECURE", ["US-KS-2"])], out)
        self.assertIn("host checked: no maintenance planned", out, out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_community_cloud_only_with_the_flag(self):
        """With --community: the Community RTX 5090 at 0.69 is the cheapest fit and
        is created with no data center named (Runpod picks the host)."""
        STATE["no_5090"] = True
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "--community", "-y", "--wait", "1", timeout=90)
        self.assertNotIn("Traceback", out, out)
        self.assertNotIn("not considered", out, out)
        self.assertIn("Replacement: RTX 5090 (32 GB of VRAM, 0.69 USD per hour, Community Cloud, a host Runpod picks)", out, out)
        self.assertIn("RTX PRO 4500 SE", out, out)
        self.assertEqual([(p["gpu"]["id"], p["cloud"], p.get("dataCenterIds")) for p in STATE["posts"]],
                         [("NVIDIA GeForce RTX 5090", "COMMUNITY", None)], out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_local_settings_and_command_line_overrides(self):
        """A models.json of one's own (a participant's question, 9 Oct 2026). settings.local.env replaces
        settings.env key by key and survives `git pull`; `--models-url` and `--set` replace a value for
        one pod; a `--set` without an AINVFX_ key creates nothing."""
        self.run_pod("setup", "-y")
        with open(os.path.join(os.path.dirname(POD_PY), "models.json"), "rb") as f:
            course = f.read()
        STATE["files"].update({"/files/mine/models.json": course, "/files/gist/models.json": course})
        mine = "http://127.0.0.1:{}/files/mine/models.json".format(self.port)
        with open(os.path.join(self.home, "settings.local.env"), "w", encoding="utf-8") as f:
            f.write("# mine\nAINVFX_MODELS_URL={}\nAINVFX_CUSTOM_NODES=https://github.com/kijai/ComfyUI-KJNodes\n".format(mine))
        out, rc = self.run_pod("doctor")
        self.assertIn("settings.local.env", out, out)
        self.assertIn("AINVFX_MODELS_URL", out, out)
        self.assertIn("(settings.local.env)", out, out)
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=90)
        self.assertIn("models list: {} (your own, not the course's)".format(mine), out, out)
        self.assertIn("AINVFX_CUSTOM_NODES: https://github.com/kijai/ComfyUI-KJNodes (yours)", out, out)
        sent = STATE["posts"][0].get("env") or {}
        self.assertEqual(sent.get("AINVFX_MODELS_URL"), mine)
        self.assertEqual(sent.get("AINVFX_CUSTOM_NODES"), "https://github.com/kijai/ComfyUI-KJNodes")
        self.assertEqual(sent.get("AINVFX_COMFY_TAG"), "v0.39.0", "settings.env still provides the rest")
        self.assertIn("READY", out, out)
        self.run_pod("down", "-y")
        gist = "http://127.0.0.1:{}/files/gist/models.json".format(self.port)
        out, rc = self.run_pod("up", "image", "--name", "second", "--models-url", gist, "--set", "AINVFX_COMFY_TAG=master",
                               "-y", "--wait", "1", timeout=90)
        sent = STATE["posts"][1].get("env") or {}
        self.assertEqual(sent.get("AINVFX_MODELS_URL"), gist, "--models-url wins over settings.local.env")
        self.assertEqual(sent.get("AINVFX_COMFY_TAG"), "master")
        self.assertIn("AINVFX_COMFY_TAG: master (yours)", out, out)
        self.assertIn("READY", out, out)
        self.run_pod("down", "second", "-y", "--no-pull")
        out, rc = self.run_pod("up", "image", "--set", "HF_TOKEN=abc", "-y", "--wait", "1", timeout=60)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("--set takes AINVFX_KEY=value", out, out)
        self.assertEqual(len(STATE["posts"]), 2, "nothing created")

    def test_own_models_list(self):
        """A pod of 13 Oct 2026 with a Google Drive link: a share link becomes the file's own
        link, the disk follows your list, and a web page stops `up` before anything is created."""
        self.assertEqual(pod.direct_url("https://drive.google.com/file/d/1DUMJh6QWX3P2ZF60qHHcb1DkY5sf4jsh/view?usp=sharing"),
                         "https://drive.google.com/uc?export=download&id=1DUMJh6QWX3P2ZF60qHHcb1DkY5sf4jsh")
        self.assertEqual(pod.direct_url("https://github.com/me/ainvfx-runpod/blob/main/models.json"),
                         "https://raw.githubusercontent.com/me/ainvfx-runpod/main/models.json")
        self.assertEqual(pod.direct_url("https://www.dropbox.com/scl/fi/abc/models.json?rlkey=x&dl=0"),
                         "https://www.dropbox.com/scl/fi/abc/models.json?rlkey=x&dl=1")
        self.assertEqual(pod.direct_url(pod.DEFAULT_MODELS_URL), pod.DEFAULT_MODELS_URL)
        with open(os.path.join(os.path.dirname(POD_PY), "models.json"), encoding="utf-8") as f:
            lst = json.load(f)
        lst["profiles"]["image"] = ["z_image_turbo", "z_image_turbo_text_encoder", "z_image_vae", "z_image_full"]
        STATE["files"].update({"/files/zimage.json": json.dumps(lst).encode(),
                               "/files/page.html": b"<!DOCTYPE html><html><body>Google Drive</body></html>"})
        base = "http://127.0.0.1:{}/files/".format(self.port)
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "--models-url", base + "page.html", "-y", "--wait", "1", timeout=60)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("not the JSON itself", out, out)
        self.assertEqual(len(STATE["posts"]), 0, "nothing created")
        typo = dict(lst, profiles=dict(lst["profiles"], image=["z_image_turbo", "z_image_vea"]))
        STATE["files"]["/files/typo.json"] = json.dumps(typo).encode()
        out, rc = self.run_pod("up", "image", "--models-url", base + "typo.json", "-y", "--wait", "1", timeout=60)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("names files that are not in its files list: z_image_vea", out, out)
        self.assertEqual(len(STATE["posts"]), 0, "nothing created")
        out, rc = self.run_pod("up", "image", "--models-url", base + "zimage.json", "-y", "--wait", "1", timeout=90)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("profile image: about 18 GB of models, a 100 GB container disk", out, out)
        self.assertEqual(STATE["posts"][0].get("disk"), 100, out)
        self.assertIn("READY", out, out)
        self.run_pod("down", "-y", "--no-pull")

    def test_a_host_under_maintenance_is_terminated_and_the_next_gpu_tried(self):
        """The host of 5 and 7 Oct 2026 (maintenance under way, server being removed), on the first two
        candidates: each pod is terminated at once and `up` goes on to the next GPU that fits."""
        STATE["no_5090"] = True
        STATE["maint_dcs"] = {"US-KS-2"}
        STATE["maint_community"] = True
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "--community", "-y", "--wait", "1", timeout=120)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("REFUSED: its host has maintenance since", out, out)
        self.assertIn("removed from the platform", out, out)
        self.assertEqual(out.count("terminated at once"), 2, out)
        self.assertEqual([(p["gpu"]["id"], p["cloud"], p.get("dataCenterIds")) for p in STATE["posts"]],
                         [("NVIDIA GeForce RTX 5090", "COMMUNITY", None),
                          ("NVIDIA RTX PRO 4500 Blackwell Server Edition", "SECURE", ["US-KS-2"]),
                          ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "SECURE", ["CA-MTL-1"])], out)
        self.assertEqual([STATE["pods"][p]["deleted"] for p in ("fakepod1", "fakepod2", "fakepod3")], [True, True, False], out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_up_with_a_combined_profile(self):
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image+3d", "-y", "--wait", "1", timeout=90)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("profile image+3d: about 148 GB of models, a 210 GB container disk", out, out)
        self.assertEqual(STATE["posts"][0].get("disk"), 210, out)
        self.assertEqual((STATE["posts"][0].get("env") or {}).get("AINVFX_PROFILE"), "image+3d", out)
        self.assertEqual(STATE["posts"][0]["gpu"]["id"], "NVIDIA GeForce RTX 5090", out)
        self.assertIn("READY", out, out)
        out, rc = self.run_pod("down", "image+3d", "-y", "--no-pull")
        self.assertEqual(rc, 0, out)
        out, rc = self.run_pod("up", "nope+3d", "-y", "--wait", "1", timeout=30)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("unknown profile(s): nope", out, out)

    def test_money_lines(self):
        """The price in the question, the credit and the hours it buys before it, then
        what the pod cost, the other pods still running with their price, and the credit left after down."""
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "--wait", "1", input="n\n", timeout=60)
        self.assertIn("Your Runpod credit: 212.40 USD, about 178 hours of RTX 5090 at 1.19 USD per hour.", out, out)
        self.assertIn("Create a RTX 5090 at 1.19 USD per hour, with a 170 GB disk, billed from creation? [Y/n]", out, out)
        self.assertEqual(len(STATE["posts"]), 0, "answered n: nothing created")
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=90)
        self.assertIn("READY", out, out)
        Handler.new_pod("ainvfx-video-other", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "US-NC-2")
        out, rc = self.run_pod("up", "image", "--name", "second", "--wait", "1", input="n\n", timeout=60)
        self.assertIn("your account already spends 3.68 USD per hour.", out, out)
        out, rc = self.run_pod("down", "image", "-y", "--no-pull")
        self.assertRegex(out, r"Terminated: ainvfx-image-\S+\. It ran [^.]+ at 1\.19 USD per hour: about \d+\.\d\d USD spent\.")
        self.assertIn("Attention: 1 other pod(s) still running, 2.49 USD per hour in total:", out, out)
        self.assertIn("- ainvfx-video-other (", out, out)
        self.assertIn("Credit left on Runpod: 212.40 USD (the last minutes may still be deducted).", out, out)
        STATE["balance"] = 2.0
        out, rc = self.run_pod("up", "image", "--wait", "1", input="n\n", timeout=60)
        self.assertIn("That is under 3 hours", out, out)

    def test_every_data_center_with_stock_is_tried(self):
        """Ten pods created in the same minutes: the first seven data centers listed for the RTX PRO 6000
        are full by the time this one asks, the eighth has a GPU. An earlier version stopped after six."""
        dcs = ["CA-MTL-1", "CA-MTL-3", "US-CA-2", "US-GA-1", "US-IL-1", "US-NC-2", "US-TX-3", "EUR-IS-1"]
        STATE["pro6000_dcs"] = tuple((d, "LOW") for d in dcs)
        STATE["full_dcs"] = set(dcs[:7])
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "video", "-y", "--wait", "1", timeout=90)
        self.assertNotIn("Traceback", out, out)
        self.assertEqual([p.get("dataCenterIds") for p in STATE["posts"]], [[d] for d in dcs], out)
        self.assertIn("created on SECURE EUR-IS-1", out, out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_gpu_named_on_the_command_line(self):
        """--gpu takes the short name from the list, in any case; with no stock for it, nothing is created and
        the GPUs in stock are listed with the command; a name that matches several GPU types is refused."""
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "--gpu", "rtx pro 4500 se", "-y", "--wait", "1", timeout=90)
        self.assertEqual((STATE["posts"][0]["gpu"]["id"], STATE["posts"][0]["dataCenterIds"]),
                         ("NVIDIA RTX PRO 4500 Blackwell Server Edition", ["US-KS-2"]), out)
        self.assertIn("READY", out, out)
        self.run_pod("down", "-y")
        STATE["no_5090"] = True
        out, rc = self.run_pod("up", "image", "--gpu", "RTX 5090", "-y", "--wait", "1", timeout=60)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("No RTX 5090 in stock right now on Secure Cloud", out, out)
        self.assertIn('python pod.py up image --gpu "RTX PRO 4500 SE"', out, out)
        self.assertNotIn("B300 MIG", out, out)
        self.assertEqual(len(STATE["posts"]), 1, "nothing created when the GPU named has no stock")
        out, rc = self.run_pod("up", "image", "--gpu", "RTX PRO", "-y")
        self.assertIn("matches several GPU types", out, out)
        self.assertEqual(len(STATE["posts"]), 1, out)
        out, rc = self.run_pod("up", "image", "--gpu", "B300 MIG 34GB", "-y", "--wait", "1", timeout=90)
        self.assertEqual((STATE["posts"][1]["gpu"]["id"], STATE["posts"][1]["dataCenterIds"]),
                         ("NVIDIA B300 SXM6 AC MIG 1g.34gb", ["US-WA-2"]), "a 1g slice named with --gpu is still created")
        self.assertIn("READY", out, out)
        self.run_pod("down", "-y")

    def test_a_pod_missing_here_is_found_by_its_short_name(self):
        """`down test10` when this machine lost the record of `up image --name test10` (7 Oct 2026), and
        `down image` for a pod of the default tag: both find the pod in the account and terminate it."""
        self.run_pod("setup", "-y")
        Handler.new_pod("ainvfx-image-test10", "NVIDIA GeForce RTX 5090", "EU-CZ-1")
        Handler.new_pod("ainvfx-image-1007-1250", "NVIDIA GeForce RTX 5090", "EU-CZ-1")
        out, rc = self.run_pod("down", "test10", "-y", "--no-pull")
        self.assertIn("attached to ainvfx-image-test10", out, out)
        self.assertTrue(STATE["pods"]["fakepod1"]["deleted"], out)
        self.assertEqual(rc, 0, out)
        out, rc = self.run_pod("down", "image", "-y", "--no-pull")
        self.assertIn("attached to ainvfx-image-1007-1250", out, out)
        self.assertTrue(STATE["pods"]["fakepod2"]["deleted"], out)
        self.assertEqual(rc, 0, out)

    def recorded(self, pods):
        """Write pod records into this test's configuration, as `up` or an attach would."""
        path = os.path.join(self.home, ".ainvfx-runpod", "config.json")
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.setdefault("pods", {}).update(pods)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return path

    def test_records_of_terminated_pods_are_forgotten(self):
        """Seven records, one pod alive (the others terminated from the console or
        elsewhere). A command without a name forgets the gone ones and takes the live one."""
        self.run_pod("setup", "-y")
        live = Handler.new_pod("ainvfx-image-3d-video-1021-2016", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "CA-MTL-3")
        path = self.recorded({
            "image-3d-video": {"id": live["id"], "name": live["name"], "profile": "image-3d-video", "created": ""},
            "image": {"id": "gone1", "name": "ainvfx-image-video-1019-2157", "profile": "image", "created": ""},
            "MySecondPod": {"id": "gone2", "name": "ainvfx-image-MySecondPod", "profile": "image", "created": ""}})
        out, rc = self.run_pod("status")
        self.assertIn("forgot 2 pod(s) no longer on Runpod", out, out)
        self.assertNotIn("several pods recorded", out, out)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)["pods"]), ["image-3d-video"])

    def test_list_forgets_records_of_terminated_pods(self):
        self.run_pod("setup", "-y")
        live = Handler.new_pod("ainvfx-image-1021-0900", "NVIDIA GeForce RTX 5090", "CA-MTL-1")
        path = self.recorded({"image": {"id": live["id"], "name": live["name"], "profile": "image", "created": ""},
                              "CST_B200_01": {"id": "gone3", "name": "ainvfx-image-video-CST_B200_01",
                                              "profile": "image-video", "created": ""}})
        out, rc = self.run_pod("list")
        self.assertEqual(rc, 0, out)
        self.assertIn("here: `image`", out, out)
        self.assertIn("Forgot 1 record(s) of pods no longer on Runpod: ainvfx-image-video-CST_B200_01.", out, out)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)["pods"]), ["image"])

    def test_two_live_pods_still_ask_for_a_name(self):
        self.run_pod("setup", "-y")
        a = Handler.new_pod("ainvfx-image-1021-0900", "NVIDIA GeForce RTX 5090", "CA-MTL-1")
        b = Handler.new_pod("ainvfx-video-1021-0910", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "CA-MTL-3")
        self.recorded({"image": {"id": a["id"], "name": a["name"], "profile": "image", "created": ""},
                       "video": {"id": b["id"], "name": b["name"], "profile": "video", "created": ""},
                       "old": {"id": "gone4", "name": "ainvfx-image-old", "profile": "image", "created": ""}})
        out, rc = self.run_pod("status")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("forgot 1 pod(s) no longer on Runpod: ainvfx-image-old", out, out)
        self.assertIn("several pods recorded, name one: image (ainvfx-image-1021-0900)  video (ainvfx-video-1021-0910)", out, out)

    def test_up_stops_on_a_failed_line(self):
        """bootstrap.sh 5.4 prints FAILED when the pod cannot work (here PyTorch out of reach on every route):
        `up` must stop on that line, say what to do, and exit non-zero, instead of waiting for READY."""
        STATE["boot"] = BOOT_FAILED
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "image", "-y", "--wait", "1", timeout=90)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("[AINVFX] FAILED: PYTORCH COULD NOT BE DOWNLOADED", out, out)
        self.assertIn("Terminate it and create another", out, out)
        self.assertNotIn("Still working", out, "`up` must stop on FAILED, not on its deadline")
        self.assertNotIn("[AINVFX] READY", out, out)
        self.assertNotEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_several_gpus_named(self):
        """`--gpu B200 --gpu B300` tries those two only, in the order given. With neither in stock, nothing
        is created; with the B300 alone in stock, the B300 is. One --gpu holding two names is refused with a hint."""
        STATE["schedule"] = {"NVIDIA B300 SXM6 AC": [(0, ())]}
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "video", "--gpu", "B200", "--gpu", "b300", "-y", "--wait", "1", timeout=60)
        self.assertNotIn("Traceback", out, out)
        self.assertNotEqual(rc, 0, out)
        self.assertIn("B200: availability NONE on Secure Cloud", out, out)
        self.assertIn("B300: availability NONE on Secure Cloud", out, out)
        self.assertIn("price per hour: 7.99 USD Secure, 5.98 USD Community", out, out)
        self.assertIn("No B200 or B300 in stock right now on Secure Cloud.", out, out)
        self.assertIn('To take one:  python pod.py up video --gpu "RTX PRO 6000"', out, out)
        self.assertIn("ERROR: nothing created.", out, out)
        self.assertEqual(STATE["posts"], [], out)
        STATE["schedule"] = {}
        out, rc = self.run_pod("up", "video", "--gpu", "B200", "--gpu", "B300", "-y", "--wait", "1", timeout=90)
        self.assertEqual([(p["gpu"]["id"], p.get("dataCenterIds")) for p in STATE["posts"]],
                         [("NVIDIA B300 SXM6 AC", ["EU-NL-1"])], out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")
        out, rc = self.run_pod("up", "video", "--gpu", "B200 B300", "-y")
        self.assertNotEqual(rc, 0, out)
        self.assertIn("For several GPUs, repeat --gpu: --gpu B200 --gpu B300.", out, out)
        self.assertEqual(len(STATE["posts"]), 1, out)

    def test_retry_waits_until_a_named_gpu_is_free(self):
        """19 Oct 2026: no B200 and no B300 on Secure Cloud. `up --gpu B200 --gpu B300 --retry` asks once, before
        the wait, then reads the catalog again until the B200 comes into stock (at the fourth read here), and
        creates it there: one pod, then the usual follow until READY."""
        STATE["schedule"] = {"NVIDIA B300 SXM6 AC": [(0, ())], "NVIDIA B200": [(4, ("US-TX-3",))]}
        self.env.update(AINVFX_RETRY_EVERY="1", AINVFX_RETRY_NOTE="0")
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "video", "--gpu", "B200", "--gpu", "B300", "--retry", "--wait", "1",
                               input="y\n", timeout=120)
        self.assertNotIn("Traceback", out, out)
        self.assertNotIn("ERROR", out, out)
        self.assertIn("No B200 or B300 in stock right now on Secure Cloud.", out, out)
        self.assertIn("Your Runpod credit: 212.40 USD, about 26 hours at 7.99 USD per hour, the highest price this "
                      "wait accepts.", out, out)
        self.assertRegex(out, r"Wait, then create the first B200 or B300 free \(up to 7\.99 USD per hour, a \d+ GB disk, "
                              r"billed from creation\)\? \[Y/n\]")
        self.assertEqual(out.count("[Y/n]"), 1, "one question, before the wait")
        self.assertIn("Waiting on Secure Cloud for B200 or B300. Every 1 s, `up` reads the catalog and creates the "
                      "first one free.", out, out)
        self.assertIn("still waiting: 1 read in under a minute, none free", out, out)
        self.assertIn("B200 free on Secure Cloud (US-TX-3): creating it", out, out)
        self.assertIn("The wait took under a minute (3 reads).", out, out)
        self.assertEqual([(p["gpu"]["id"], p.get("dataCenterIds")) for p in STATE["posts"]],
                         [("NVIDIA B200", ["US-TX-3"])], out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_retry_goes_on_when_the_machine_is_taken_in_the_meantime(self):
        """A B200 shows in the catalog, but its machine is gone by the time `up` asks for it (the 400 of
        19 Oct 2026: There are no longer any instances available): the wait goes on, and the next free
        machine is created. One pod, never more."""
        STATE["schedule"] = {"NVIDIA B200": [(3, ("US-GA-1",)), (5, ("US-TX-3",))]}
        STATE["full_dcs"] = {"US-GA-1"}
        self.env.update(AINVFX_RETRY_EVERY="1", AINVFX_RETRY_NOTE="0")
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "video", "--gpu", "B200", "--retry", "-y", "--wait", "1", timeout=120)
        self.assertNotIn("Traceback", out, out)
        self.assertNotIn("[Y/n]", out, "-y: no question at all")
        self.assertIn("SECURE US-GA-1: no machine (400", out, out)
        self.assertIn("taken in the meantime: still waiting", out, out)
        self.assertEqual([p.get("dataCenterIds") for p in STATE["posts"]], [["US-GA-1"], ["US-GA-1"], ["US-TX-3"]], out)
        self.assertEqual([p["deleted"] for p in STATE["pods"].values()], [False], "one pod, never more")
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_retry_without_gpu_keeps_the_price_ceiling(self):
        """`up video --retry` with no RTX PRO 6000 free and a B300 in stock at 7.89 USD per hour: the B300 is
        never created, since `up` takes nothing above 2.50 USD per hour on its own. The RTX PRO 6000 is, once it
        comes back (the third read here)."""
        STATE["schedule"] = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": [(3, ("US-NC-2",))]}
        self.env.update(AINVFX_RETRY_EVERY="1", AINVFX_RETRY_NOTE="0")
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "video", "--retry", "-y", "--wait", "1", timeout=120)
        self.assertNotIn("Traceback", out, out)
        self.assertIn('To take one above 2.50 USD per hour:  python pod.py up video --gpu "B300"', out, out)
        self.assertIn("Waiting on Secure Cloud for RTX PRO 6000, or any GPU that fits the video profile at up to "
                      "2.50 USD per hour.", out, out)
        self.assertIn("RTX PRO 6000 free on Secure Cloud (US-NC-2): creating it", out, out)
        self.assertEqual([p["gpu"]["id"] for p in STATE["posts"]], ["NVIDIA RTX PRO 6000 Blackwell Server Edition"], out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_retry_after_a_failed_first_attempt_asks_no_second_question(self):
        """The RTX PRO 6000 is in stock at the first read, but each machine is taken before `up` asks for it.
        With --retry, the one question is the usual one, with a line on what --retry will do; the wait that
        follows creates the pod without asking again."""
        full = ("US-NC-2", "CA-MTL-1", "EU-CZ-1")
        STATE["schedule"] = {"NVIDIA RTX PRO 6000 Blackwell Server Edition": [(0, full), (3, ("US-TX-3",))]}
        STATE["full_dcs"] = set(full)
        self.env.update(AINVFX_RETRY_EVERY="1", AINVFX_RETRY_NOTE="0")
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "video", "--retry", "--wait", "1", input="y\n", timeout=120)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("With --retry: if none can be created now, `up` reads the catalog every 1 s and creates the first "
                      "one free, at up to 2.50 USD per hour, without asking again.", out, out)
        self.assertIn("Create a RTX PRO 6000 at 2.49 USD per hour", out, out)
        self.assertEqual(out.count("[Y/n]"), 1, out)
        self.assertIn("None could be created now: waiting, as --retry asks.", out, out)
        self.assertEqual(len(STATE["posts"]), 7, "three full data centers twice, then US-TX-3")
        self.assertEqual(STATE["posts"][-1].get("dataCenterIds"), ["US-TX-3"], out)
        self.assertEqual([p["deleted"] for p in STATE["pods"].values()], [False], out)
        self.assertIn("READY", out, out)
        self.assertEqual(rc, 0, out)
        self.run_pod("down", "-y")

    def test_retry_asks_once_and_no_means_nothing(self):
        STATE["schedule"] = {"NVIDIA B300 SXM6 AC": [(0, ())]}
        self.run_pod("setup", "-y")
        out, rc = self.run_pod("up", "video", "--gpu", "B200", "--retry", input="n\n", timeout=60)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("Wait, then create the first B200 free (up to 7.99 USD per hour", out, out)
        self.assertIn("Nothing created.", out, out)
        self.assertNotIn("Waiting on", out, out)
        self.assertEqual(STATE["posts"], [], out)
        self.assertEqual(rc, 0, out)

    def test_retry_stops_when_another_up_takes_the_name(self):
        """A wait can last hours. When another `up` creates a pod under the same name in the meantime (here
        `up video --gpu "RTX PRO 6000"` in a second terminal), the wait stops when its GPU comes into stock,
        instead of creating a second pod that would bill unnoticed."""
        self.env.update(AINVFX_RETRY_EVERY="1", AINVFX_RETRY_NOTE="0")
        self.run_pod("setup", "-y")
        waiting = subprocess.Popen([sys.executable, POD_PY, "up", "video", "--gpu", "B200", "--retry", "-y", "--wait", "1"],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=self.env)
        deadline = time.time() + 60
        while STATE["catalog_reads"] < 2 and time.time() < deadline:
            time.sleep(0.1)
        out, rc = self.run_pod("up", "video", "--gpu", "RTX PRO 6000", "-y", "--wait", "1", timeout=90)
        self.assertIn("READY", out, out)
        STATE["schedule"] = {"NVIDIA B200": [(0, ("US-TX-3",))]}
        out, _ = waiting.communicate(timeout=60)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("a pod was recorded as `video` while this wait ran", out, out)
        self.assertNotEqual(waiting.returncode, 0, out)
        self.assertEqual([p["gpu"]["id"] for p in STATE["posts"]], ["NVIDIA RTX PRO 6000 Blackwell Server Edition"], out)
        self.run_pod("down", "-y")

    @unittest.skipIf(os.name == "nt", "Ctrl+C reaches the process as SIGINT on Linux and macOS only")
    def test_retry_stops_on_ctrl_c(self):
        """Ctrl+C during the wait: a message, exit code 130, no pod."""
        self.env.update(AINVFX_RETRY_EVERY="1", AINVFX_RETRY_NOTE="0")
        self.run_pod("setup", "-y")
        proc = subprocess.Popen([sys.executable, POD_PY, "up", "video", "--gpu", "B200", "--retry", "-y"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=self.env)
        deadline = time.time() + 60
        while STATE["catalog_reads"] < 3 and time.time() < deadline:
            time.sleep(0.1)
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=30)
        self.assertNotIn("Traceback", out, out)
        self.assertIn("Stopped waiting after", out, out)
        self.assertIn("this wait created no pod.", out, out)
        self.assertEqual(proc.returncode, 130, out)
        self.assertEqual(STATE["posts"], [], out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
