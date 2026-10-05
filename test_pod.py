#!/usr/bin/env python3
"""Tests for pod.py. Standard library only, nothing reaches Runpod, nothing is billed, and the
real ~/.ainvfx-runpod is untouched (the end-to-end tests run with HOME in a temporary folder).

    python -m unittest discover tests      # from the repository root
    python tests/test_pod.py               # the same, directly
    pytest                                 # if you have it

Two groups. `Helpers` are unit tests of the pure functions (data center order, the place guessed
from the clock, the stream resume point, the error hints). `EndToEnd` starts a fake Runpod API v2
and a fake ComfyUI on 127.0.0.1 and runs the real commands (setup, doctor, up, status, pull, push,
list, down) through subprocess, as a user would. The fake reproduces what the real log stream
does: events carry a timestamp as id, a `since` cursor is exclusive at one-second resolution, and
the server closes the stream after a few events. Before 0.3.0, READY (written in the same second
as SELFTEST OK) was skipped on reconnect and `up` waited for its deadline. A second scenario makes
the log endpoint answer 403 (a key without log access): READY must then come through the pod's
own copy of the log, served by ComfyUI.
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
    (0, "[AINVFX] bootstrap start · profile image · ComfyUI v0.38.2"),
    (0, "[AINVFX] no HF_TOKEN: the gated files (LTX, video sessions) will be skipped; the image models need none"),
    (1, "[AINVFX] step 0/5 starting Runpod's /start.sh in the background (SSH, JupyterLab on port 8888)"),
    (1, "[AINVFX] step 1/5 health check"),
    (1, "[AINVFX] GPU: NVIDIA GeForce RTX 5090, 32607 MiB, 610.43.02"),
    (2, "[AINVFX] disk: write 4284 MB/s · read 4306 MB/s"),
    (2, "[AINVFX] step 2/5 install (uv, Python 3.13, PyTorch cu130, ComfyUI v0.38.2)"),
    (3, "[AINVFX] ComfyUI v0.38.2 in /workspace/ComfyUI · environment /workspace/venv"),
    (3, "[AINVFX] step 3/5 start ComfyUI on port 8188"),
    (4, "[AINVFX] COMFYUI UP · https://fakepod1-8188.proxy.runpod.net"),
    (4, "[AINVFX] PROXY OK · https://fakepod1-8188.proxy.runpod.net answers from outside"),
    (4, "[AINVFX] step 4/5 models of profile image"),
    (5, "[AINVFX] models 1: downloading z_image_turbo_int8_convrot.safetensors (6.2 GB) from Comfy-Org/z_image_turbo"),
    (5, "[AINVFX] models 1: z_image_turbo_int8_convrot.safetensors in 10 s · 600 MB/s"),
    (6, "[AINVFX] MODELS DONE 14/14 present · 59G on disk · 59 GB downloaded in 91 s (650 MB/s)"),
    (6, "[AINVFX] step 5/5 self-test"),
    (7, "[AINVFX] SELFTEST OK · Z-Image Turbo 1024 x 1024, 8 steps, in 16.0 s (models loaded from disk) · output/ainvfx_selftest_00001_.png"),
    (7, "[AINVFX] READY · ComfyUI https://fakepod1-8188.proxy.runpod.net · JupyterLab port 8888 · log /workspace/ComfyUI/input/ainvfx/bootstrap.log"),
    (7, "[AINVFX] remember: terminate the pod when you are done"),
]
STEP = 0.4            # one scripted second of pod time = 0.4 real seconds
STATE = {"pod": None, "t0": None, "posts": [], "logs_403": False, "uploads": [], "deleted": False}


def pod_seconds():
    return (time.time() - STATE["t0"]) / STEP if STATE["t0"] else 0


def visible_lines():
    return [(s, l) for s, l in BOOT if s <= pod_seconds()]


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
        if self.headers.get("Authorization") != "Bearer fake-key":
            return self.problem(401, "bad key")
        if p == "/v2/pods":
            pods = [STATE["pod"]] if STATE["pod"] and not STATE["deleted"] else []
            return self.send_json(200, {"pods": pods, "pagination": {"nextCursor": None}})
        if p.startswith("/v2/pods/") and p.endswith("/logs"):
            return self.logs(q)
        if p.startswith("/v2/pods/"):
            if not STATE["pod"] or STATE["deleted"]:
                return self.problem(404, "pod not found")
            pod = dict(STATE["pod"])
            pod["status"] = "RUNNING" if pod_seconds() >= 1 else "STARTING"
            return self.send_json(200, pod)
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
            STATE["t0"] = time.time()
            STATE["pod"] = {"id": "fakepod1", "name": body["name"], "status": "STARTING", "cost": 0.99,
                            "gpu": {"id": body["gpu"]["id"], "count": 1}, "dataCenterId": (body.get("dataCenterIds") or ["?"])[0],
                            "cudaVersion": "13.3", "createdAt": ts_of(0), "template": None,
                            "ssh": {"direct": {"host": "81.27.69.177", "port": 32554, "username": "root"}},
                            "runtime": {"gpus": [{"util": 0, "memoryUtil": 0}]}}
            return self.send_json(201, STATE["pod"])
        return self.problem(404, "no route")

    def do_PUT(self):
        self.body()
        return self.send_json(200, {"keys": []})

    def do_DELETE(self):
        if self.headers.get("Authorization") != "Bearer fake-key":
            return self.problem(401, "bad key")
        STATE["deleted"] = True
        self.send_response(204)
        self.end_headers()

    def logs(self, q):
        if STATE["logs_403"]:
            return self.problem(403, "this key cannot read logs")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        since = q.get("since", [None])[0]
        tail = int(q.get("tail", ["100"])[0])
        sent = 0
        if since:
            # exclusive at second resolution, like the real cursor
            start = [i for i, (s, _) in enumerate(BOOT) if ts_of(s) > since]
            idx = start[0] if start else len(visible_lines())
        else:
            idx = max(0, len(visible_lines()) - tail)
        deadline = time.time() + 60
        while time.time() < deadline:
            vis = visible_lines()
            while idx < len(vis):
                s, line = vis[idx]
                ev = "id: {}\ndata: {}\n\n".format(ts_of(s), json.dumps({"ts": ts_of(s), "source": "container", "line": line}))
                try:
                    self.wfile.write(ev.encode()); self.wfile.flush()
                except BrokenPipeError:
                    return
                idx += 1; sent += 1
                if "SELFTEST OK" in line or sent >= 6:   # the server closes the stream often, and right after SELFTEST OK
                    return
            time.sleep(0.1)

    # ------------------------------------------------------------- fake ComfyUI behind the proxy
    def proxy_get(self, p, q):
        up = pod_seconds() >= 4 and not STATE["deleted"]
        if not up:
            return self.problem(502, "no comfy yet")
        if p.endswith("/system_stats"):
            return self.send_json(200, {"system": {"comfyui_version": "0.38.2"}})
        if p.endswith("/history"):
            done = pod_seconds() >= 7
            return self.send_json(200, {"p1": {"outputs": {"9": {"images": [{"filename": "ainvfx_selftest_00001_.png",
                                        "subfolder": "", "type": "output"}]}}, "status": {"completed": True}}} if done else {})
        if p.endswith("/view"):
            if q.get("type") == ["input"] and q.get("filename") == ["bootstrap.log"] and q.get("subfolder") == ["ainvfx"]:
                data = "\n".join(l for _, l in visible_lines()).encode()
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
        STATE.update({"pod": None, "t0": None, "posts": [], "logs_403": False, "uploads": [], "deleted": False})
        self.home = tempfile.mkdtemp(prefix="ainvfx-test-")
        self.env = dict(os.environ, HOME=self.home, USERPROFILE=self.home, TZ="America/Toronto", RUNPOD_API_KEY="fake-key",
                        AINVFX_OUTPUTS=os.path.join(self.home, "outputs"),
                        AINVFX_API_BASE="http://127.0.0.1:{}/v2".format(self.port),
                        AINVFX_PROXY_FMT="http://127.0.0.1:%d/proxy/{id}/{port}" % self.port)

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
        self.assertIn("SELFTEST OK", out, out)
        self.assertIn("READY", out, "READY must be printed even when the stream skips the second it was written in")
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
        out, rc = self.run_pod("down", "-y")
        self.assertTrue(STATE["deleted"], out)
        self.assertIn("Nothing bills", out, out)

    def test_whole_session_with_the_api_log_stream(self):
        """The stream has an exclusive timestamp cursor and closes often (right after SELFTEST OK)."""
        self.whole_session()

    def test_whole_session_when_the_key_cannot_read_logs(self):
        """The logs endpoint answers 403: READY must come through the pod's own copy of the log."""
        STATE["logs_403"] = True
        self.whole_session()


if __name__ == "__main__":
    unittest.main(verbosity=2)
