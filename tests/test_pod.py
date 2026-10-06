#!/usr/bin/env python3
"""Tests for pod.py. Standard library only, nothing reaches Runpod, nothing is billed, and the
real ~/.ainvfx-runpod is untouched (the end-to-end tests run with HOME in a temporary folder).

    python -m unittest discover tests      # from the repository root
    python tests/test_pod.py               # the same, directly
    pytest                                 # if you have it

Two groups. `Helpers` are unit tests of the pure functions (data center order, the place guessed
from the clock, the stream resume point, the error hints). `EndToEnd` starts a fake Runpod API v2
and a fake ComfyUI on 127.0.0.1 and runs the real commands (setup, doctor, up, status, pull, push,
list, down) through subprocess, as a user would. The fake reproduces what the real log stream was
seen doing on 4 and 5 October 2026: events carry a timestamp as id, a `since` cursor is exclusive
at one-second resolution, the stream stays open with keepalive lines, and the lines written after
SELFTEST OK (READY, remember) never arrive on a live connection: only a fresh request serves them.
Before 0.3.1, `up` therefore sat on SELFTEST OK until its deadline. Three scenarios: both sources
available; the pod's own copy of the log unavailable (READY must come from a reopened stream);
the log endpoint answering 403 (READY must come from the pod's own copy). Then several pods at
once: `up --name`, commands by name or id, attaching a pod created elsewhere, `down --all`. Then
what the 5 October pods taught: Runpod's system log (the image pull, layer by layer) shown as one
summary, a container that does not start (the stall warning), and a log route that does not answer
(a message, not a traceback).
"""
import json, os, shutil, subprocess, sys, tempfile, threading, time, unittest, urllib.parse
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
    (7, "[AINVFX] READY · ComfyUI https://fakepod1-8188.proxy.runpod.net · JupyterLab port 8888 · log /workspace/ComfyUI/input/ainvfx/bootstrap.log"),
    (7, "[AINVFX] remember: terminate the pod when you are done"),
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
         "pull_seconds": 0,       # scripted seconds the image pull lasts before the container starts
         "logs_hang_for": 0}      # real seconds after creation during which the logs route sends nothing, not even headers


def pod_seconds(pod):
    return (time.time() - pod["t0"]) / STEP


def container_started(pod):
    return pod_seconds(pod) >= STATE["pull_seconds"]


def all_events(pod):
    """Every scripted (second, source, line) of this pod, in order."""
    pull = STATE["pull_seconds"]
    ev = [(s, "system", l) for s, l in PULL_START if pull] + [(pull + s, "system", l) for s, l in PULL_END if pull]
    ev += [(pull + s, "container", l) for s, l in BOOT]
    return sorted(ev, key=lambda e: e[0])


def visible_events(pod):
    return [e for e in all_events(pod) if e[0] <= pod_seconds(pod)]


def visible_lines(pod):
    """The container lines written so far, as (second, line): what the pod's own log copy holds."""
    return [(s, l) for s, src, l in visible_events(pod) if src == "container"]


def pod_of(path):
    for pid, pod in STATE["pods"].items():
        if "/" + pid in path:
            return pid, pod
    return None, None


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
        if p.startswith("/v2/catalog/gpus/"):
            return self.send_json(200, {
                "id": "NVIDIA GeForce RTX 5090", "name": "RTX 5090", "availability": "LOW",
                "price": {"secure": 0.99, "community": 0.69},
                "dataCenters": [{"id": "EU-CZ-1", "availability": "LOW"}, {"id": "US-TX-3", "availability": "HIGH"},
                                {"id": "CA-MTL-1", "availability": "LOW"}, {"id": "EUR-NO-1", "availability": "NONE"}]})
        if p == "/v2/account/secrets":
            return self.send_json(200, {"secrets": []})
        if p == "/v2/account/ssh-keys":
            return self.send_json(200, {"keys": []})
        return self.problem(404, "no route " + p)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
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
            pod = self.new_pod(body["name"], body["gpu"]["id"], (body.get("dataCenterIds") or ["?"])[0])
            return self.send_json(201, self.view_of(pod))
        return self.problem(404, "no route")

    @staticmethod
    def new_pod(name, gpu, dc):
        STATE["count"] += 1
        pid = "fakepod{}".format(STATE["count"])
        pod = {"id": pid, "name": name, "cost": 0.99, "gpu": {"id": gpu, "count": 1}, "dataCenterId": dc,
               "cudaVersion": "13.3", "createdAt": ts_of(0), "template": None,
               "ssh": {"direct": {"host": "81.27.69.177", "port": 32554, "username": "root"}},
               "runtime": {"gpus": [{"util": 0, "memoryUtil": 0}]}, "t0": time.time(), "deleted": False}
        STATE["pods"][pid] = pod
        return pod

    @staticmethod
    def view_of(pod):
        v = {k: val for k, val in pod.items() if k not in ("t0", "deleted")}
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
                except BrokenPipeError:
                    return
                idx += 1
            if time.time() - last_keepalive > 1:   # keepalives defeat an idle timeout
                last_keepalive = time.time()
                try:
                    self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
                except BrokenPipeError:
                    return
            time.sleep(0.1)

    # ------------------------------------------------------------- fake ComfyUI behind the proxy
    def proxy_get(self, p, q):
        pid, pod = pod_of(p)
        up = pod and not pod["deleted"] and pod_seconds(pod) >= STATE["pull_seconds"] + 4
        if not up:
            return self.problem(502, "no comfy yet")
        if p.endswith("/system_stats"):
            return self.send_json(200, {"system": {"comfyui_version": "0.39.0"}})
        if p.endswith("/history"):
            done = pod_seconds(pod) >= STATE["pull_seconds"] + 7
            return self.send_json(200, {"p1": {"outputs": {"9": {"images": [{"filename": "ainvfx_selftest_00001_.png",
                                        "subfolder": "", "type": "output"}]}}, "status": {"completed": True}}} if done else {})
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
        old = pod.SETTINGS
        pod.SETTINGS = pod.Path(f)
        try:
            self.assertEqual(pod.load_settings(), {"AINVFX_COMFY_TAG": "master", "AINVFX_TORCH": "--pre torch torchvision",
                                                   "AINVFX_PYTHON": "3.14"})
            self.assertEqual(pod.comfy_tag(), "master")
            pod.SETTINGS = pod.Path(d) / "missing.env"
            self.assertEqual(pod.load_settings(), {})
            self.assertEqual(pod.comfy_tag(), "v0.39.0")
        finally:
            pod.SETTINGS = old
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

    def test_profiles_and_start_command(self):
        self.assertEqual(sorted(pod.PROFILES), ["image", "train", "video"])
        self.assertIn("bootstrap.sh", pod.START_CMD)
        self.assertTrue(pod.START_CMD.endswith('|| /start.sh"'))


# ----------------------------------------------------------------------------- end to end
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        STATE.update({"pods": {}, "posts": [], "logs_403": False, "no_proxy_log": False, "uploads": [], "count": 0,
                      "pull_seconds": 0, "logs_hang_for": 0})
        self.home = tempfile.mkdtemp(prefix="ainvfx-test-")
        self.env = dict(os.environ, HOME=self.home, USERPROFILE=self.home, TZ="America/Toronto", RUNPOD_API_KEY="fake-key",
                        AINVFX_OUTPUTS=os.path.join(self.home, "outputs"),
                        AINVFX_API_BASE="http://127.0.0.1:{}/v2".format(self.port),
                        AINVFX_PROXY_FMT="http://127.0.0.1:%d/proxy/{id}/{port}" % self.port,
                        AINVFX_STREAM_MAX_AGE="4", AINVFX_PROXY_POLL="3", AINVFX_STATUS_POLL="3",
                        AINVFX_PULL_SUMMARY="2", AINVFX_HEARTBEAT="4", AINVFX_STALL_MINUTES="0.1",
                        AINVFX_STREAM_IDLE="2",
                        AINVFX_SETTINGS=os.path.join(self.home, "settings.env"))
        with open(self.env["AINVFX_SETTINGS"], "w", encoding="utf-8") as f:
            f.write("AINVFX_COMFY_TAG=v0.39.0\nAINVFX_SELFTEST=0\n")

    def tearDown(self):
        shutil.rmtree(self.home, ignore_errors=True)

    def run_pod(self, *cmd, timeout=120):
        r = subprocess.run([sys.executable, POD_PY] + list(cmd), capture_output=True, text=True, env=self.env, timeout=timeout)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
