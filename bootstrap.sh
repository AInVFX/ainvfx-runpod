#!/usr/bin/env bash
# ainvfx-runpod · bootstrap.sh · runs ON THE POD as the container start command.
#
# What it does, in order (every line it prints starts with [AINVFX], so the log is easy to read):
#   0. start Runpod's own /start.sh in the background: SSH and JupyterLab are up within seconds
#   1. health check: GPU and driver, disk speed (the download speed is measured in step 4, on real files)
#   2. install: uv, Python 3.13, PyTorch stable for CUDA 13.0, ComfyUI at a pinned tag, the Manager
#   3. start ComfyUI, listening for Runpod's proxy on port 8188, and check the proxy from here
#   4. download the models of the profile (AINVFX_PROFILE: image, video or train), resumable
#   5. one test image with Z-Image Turbo (proves the GPU, the kernels and the models), then READY
#
# v4 (5 Oct 2026, after two real pods): the one-stream curl probe is gone (it read 21 to 39 MB/s on pods
# that then downloaded at 100 to 1300 MB/s with hf, and warned for nothing); the download speed is now
# judged on the first model file over 1 GB. The log lives in ComfyUI's input folder
# (input/ainvfx/bootstrap.log, with /workspace/ainvfx-bootstrap.log pointing to it), so pod.py can
# also read it through the proxy once ComfyUI answers. MODELS DONE reports the total time and the
# average speed. The unresolved-secret message is plain information, not a warning.
# v3 (4 Oct 2026): ComfyUI is fetched with git init + fetch, which tolerates a folder that already holds
# files; the install marker is written only when main.py exists; each model download logs its speed;
# the disk threshold is 300 MB/s (a healthy pod read 660 MB/s with dd and rendered in 2 s).
# Measured: EUR-IS-2 pod, 63 GB in 10 minutes, READY at 15 minutes; EU-CZ-1 pod, 59 GB in 90 seconds,
# READY at about 5 minutes. The self-test takes 16 to 106 s on the first load, 2 s once cached.
#
# Environment variables, all optional:
#   AINVFX_PROFILE    image (default) · video · train
#   AINVFX_COMFY_TAG  the ComfyUI git tag (default v0.38.2)
#   AINVFX_REPO_RAW   where to fetch models.json from (default: this repository on GitHub)
#   AINVFX_SELFTEST   1 (default) generates the test image at the end; 0 skips it
#   HF_TOKEN          a Hugging Face read token, for the gated files (LTX). Without it they are skipped.
#                     Runpod fills it from the secret `huggingface_token` when the value is
#                     {{ RUNPOD_SECRET_huggingface_token }}; an unresolved placeholder counts as absent.
#
# Idempotent: a second run (pod restart) skips what is already installed and downloaded.
# Nothing here depends on SSH: everything is visible in the pod's log in the Runpod console.

set -uo pipefail
PROFILE="${AINVFX_PROFILE:-image}"
TAG="${AINVFX_COMFY_TAG:-v0.38.2}"
RAW="${AINVFX_REPO_RAW:-https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main}"
SELFTEST="${AINVFX_SELFTEST:-1}"
ROOT=/workspace
COMFY=$ROOT/ComfyUI
VENV=$ROOT/venv
LOG=$COMFY/input/ainvfx/bootstrap.log   # inside ComfyUI's input folder: readable through the proxy
STAGE=$ROOT/.hfdl
PORT=8188
PROXY="https://${RUNPOD_POD_ID:-<pod id>}-$PORT.proxy.runpod.net"
export HF_XET_HIGH_PERFORMANCE=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$(dirname "$LOG")"
ln -sfn "$LOG" $ROOT/ainvfx-bootstrap.log
exec > >(tee -a "$LOG") 2>&1

say()  { echo "[AINVFX] $*"; }
warn() { echo "[AINVFX] WARNING: $*"; }
now()  { date +%s.%N; }
mbps() { python3 -c "import sys; b=float(sys.argv[1]); t=float(sys.argv[2]); print(int(b/1e6/max(t,0.001)))" "$1" "$2"; }

say "bootstrap start · profile $PROFILE · ComfyUI $TAG · $(date -u +'%F %T') UTC"

# A placeholder Runpod did not substitute (no secret in the account) must not reach Hugging Face:
# an invalid token makes it refuse even public files.
if [ -n "${HF_TOKEN:-}" ] && case "$HF_TOKEN" in *"{{"*) true;; *) false;; esac; then
  say "no Runpod secret named huggingface_token in this account: HF_TOKEN ignored"
  unset HF_TOKEN
fi
if [ -n "${HF_TOKEN:-}" ]; then say "HF_TOKEN present: gated files (LTX) allowed"; else say "no HF_TOKEN: the gated files (LTX, video sessions) will be skipped; the image models need none"; fi

# ---------------------------------------------------------------- 0. SSH and JupyterLab now
if [ -x /start.sh ] && ! pgrep -f "jupyter lab" >/dev/null 2>&1; then
  say "step 0/5 starting Runpod's /start.sh in the background (SSH, JupyterLab on port 8888)"
  nohup /start.sh >> $ROOT/runpod-start.log 2>&1 &
  START_PID=$!
else
  START_PID=""
fi

# ---------------------------------------------------------------- 1. health check
say "step 1/5 health check"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_LINE=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
  say "GPU: $GPU_LINE"
  DRV=$(echo "$GPU_LINE" | awk -F', ' '{print $3}' | cut -d. -f1)
  if [ -n "$DRV" ] && [ "$DRV" -lt 580 ] 2>/dev/null; then
    warn "DRIVER $DRV IS OLDER THAN 580: CUDA 13 kernels will not run, int8 models will be slow. Terminate this pod and create another."
  fi
else
  warn "nvidia-smi NOT FOUND: no GPU visible. Terminate this pod and create another."
fi
say "CPU: $(nproc) cores · RAM: $(free -g | awk '/Mem:/ {print $2}') GB · disk free on $ROOT: $(df -BG $ROOT | awk 'NR==2 {print $4}')"

T0=$(now); dd if=/dev/zero of=$ROOT/.probe bs=1M count=2048 oflag=direct status=none 2>/dev/null; T1=$(now)
WR=$(mbps 2147483648 "$(python3 -c "print($T1-$T0)")")
T0=$(now); dd if=$ROOT/.probe of=/dev/null bs=1M iflag=direct status=none 2>/dev/null; T1=$(now)
RD=$(mbps 2147483648 "$(python3 -c "print($T1-$T0)")")
rm -f $ROOT/.probe
say "disk: write $WR MB/s · read $RD MB/s"
[ "$RD" -lt 300 ] 2>/dev/null && warn "DISK READ UNDER 300 MB/s: models will load slowly. Consider another pod."


# ---------------------------------------------------------------- 2. install
say "step 2/5 install (uv, Python 3.13, PyTorch cu130, ComfyUI $TAG)"
command -v git  >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || pip install -q uv
fi
say "uv $(uv --version 2>/dev/null | awk '{print $2}')"
if [ ! -x $VENV/bin/python ]; then
  uv venv $VENV --python 3.13 --quiet || uv venv $VENV --python 3.12 --quiet
fi
# shellcheck disable=SC1091
source $VENV/bin/activate
MARK=$VENV/.ainvfx_install_$TAG
if [ ! -f "$MARK" ] || [ ! -f $COMFY/main.py ]; then
  uv pip install --quiet torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130 \
    || warn "PyTorch install failed: read the lines above"
  # git init + fetch instead of git clone: works in a folder that already holds models (a restart)
  if [ ! -d $COMFY/.git ]; then
    mkdir -p $COMFY
    (cd $COMFY && git init --quiet && git remote add origin https://github.com/Comfy-Org/ComfyUI) \
      || warn "git init of ComfyUI failed"
  fi
  (cd $COMFY && git fetch --quiet --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" && git checkout --quiet "$TAG") \
    || (cd $COMFY && git fetch --quiet --tags origin && git checkout --quiet "$TAG") \
    || warn "could not check out ComfyUI $TAG"
  uv pip install --quiet -r $COMFY/requirements.txt || warn "ComfyUI requirements failed"
  [ -f $COMFY/manager_requirements.txt ] && (uv pip install --quiet -r $COMFY/manager_requirements.txt || true)
  uv pip install --quiet -U huggingface_hub || warn "huggingface_hub install failed: model downloads will fail"
  [ -f $COMFY/main.py ] && python - <<'PY' && date -u +'%F %T' > "$MARK"
import torch, sys
ok = torch.cuda.is_available()
cu = torch.version.cuda or "0"
print("[AINVFX] torch", torch.__version__, "cuda", cu, "GPU", torch.cuda.get_device_name(0) if ok else "NOT VISIBLE")
if not ok:
    print("[AINVFX] WARNING: THE GPU IS NOT VISIBLE FROM PYTORCH. Terminate this pod and create another.")
    sys.exit(1)
if tuple(int(x) for x in cu.split(".")[:2]) < (13, 0):
    print("[AINVFX] WARNING: CUDA", cu, "under 13.0: int8 kernels off, everything slow.")
PY
else
  say "install already done ($(cat "$MARK") UTC)"
fi
[ -f $COMFY/main.py ] || warn "COMFYUI IS NOT INSTALLED ($COMFY/main.py missing): read the lines above"
say "ComfyUI $(cd $COMFY 2>/dev/null && git describe --tags 2>/dev/null || echo '?') in $COMFY · environment $VENV"

# ---------------------------------------------------------------- 3. start ComfyUI
say "step 3/5 start ComfyUI on port $PORT"
mkdir -p $COMFY/models/diffusion_models $COMFY/models/text_encoders $COMFY/models/loras \
         $COMFY/models/model_patches $COMFY/models/upscale_models $COMFY/models/latent_upscale_models \
         $COMFY/models/geometry_estimation $COMFY/models/checkpoints $COMFY/models/frame_interpolation \
         $COMFY/input $COMFY/output
if ! curl -fs "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
  (cd $COMFY && nohup python main.py --listen 0.0.0.0 --port $PORT --enable-manager --preview-method auto \
     >> $ROOT/comfy.log 2>&1 &)
  for i in $(seq 1 60); do
    sleep 3
    curl -fs "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break
  done
fi
if curl -fs "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
  say "COMFYUI UP · $PROXY"
  # the proxy, as a browser sees it: Runpod routes <pod id>-8188 to this container
  if curl -fs --max-time 15 "$PROXY/system_stats" >/dev/null 2>&1; then
    say "PROXY OK · $PROXY answers from outside"
  else
    warn "the proxy address does not answer yet ($PROXY): it can take a minute after COMFYUI UP"
  fi
else
  warn "COMFYUI DID NOT START within 3 minutes: read $ROOT/comfy.log (JupyterLab or ssh)"
fi

# ---------------------------------------------------------------- 4. models
say "step 4/5 models of profile $PROFILE"
MODELS_JSON=$ROOT/models.json
curl -fsSL --retry 3 "$RAW/models.json" -o "$MODELS_JSON.new" && mv -f "$MODELS_JSON.new" "$MODELS_JSON"
if [ ! -s "$MODELS_JSON" ]; then
  warn "models.json could not be fetched: no model downloaded. Use the Manager's model library."
else
  mkdir -p "$STAGE"
  TOTAL=0; DONE=0; SKIPPED=0; BYTES_ALL=0; T_ALL0=$(now); SPEED_JUDGED=0
  while IFS='|' read -r repo path dir gated gb; do
    [ -z "$repo" ] && continue
    TOTAL=$((TOTAL+1))
    name=$(basename "$path")
    dest=$COMFY/models/$dir
    mkdir -p "$dest"
    if [ -s "$dest/$name" ]; then
      DONE=$((DONE+1)); continue
    fi
    if [ "$gated" = "1" ] && [ -z "${HF_TOKEN:-}" ]; then
      say "models $TOTAL: SKIPPED $name (gated repo $repo, no HF_TOKEN)"
      SKIPPED=$((SKIPPED+1)); continue
    fi
    say "models $TOTAL: downloading $name ($gb GB) from $repo"
    T0=$(now)
    if hf download "$repo" "$path" --local-dir "$STAGE" >/dev/null 2>&1 && [ -s "$STAGE/$path" ]; then
      T1=$(now)
      BYTES=$(stat -c %s "$STAGE/$path" 2>/dev/null || echo 0)
      SECS=$(python3 -c "print(int($T1-$T0))")
      mv -f "$STAGE/$path" "$dest/$name"
      DONE=$((DONE+1)); BYTES_ALL=$((BYTES_ALL+BYTES))
      SPEED=$(mbps "$BYTES" "$(python3 -c "print($T1-$T0)")")
      say "models $TOTAL: $name in $SECS s · $SPEED MB/s"
      # the speed judgement, once, on the first file over 1 GB (small files measure latency, not speed)
      if [ "$SPEED_JUDGED" = "0" ] && [ "$BYTES" -gt 1000000000 ]; then
        SPEED_JUDGED=1
        [ "$SPEED" -lt 50 ] 2>/dev/null && warn "DOWNLOAD UNDER 50 MB/s: the models of this profile could take over 20 minutes. Consider another pod."
      fi
    else
      warn "download failed: $path from $repo (path changed, or access refused)"
    fi
  done < <(python - "$MODELS_JSON" "$PROFILE" <<'PY'
import json, sys
spec = json.load(open(sys.argv[1]))
files, profiles = spec["files"], spec["profiles"]
def resolve(name, seen=()):
    out = []
    for item in profiles[name]:
        if item.startswith("@"):
            if item[1:] not in seen:
                out += resolve(item[1:], seen + (name,))
        else:
            out.append(item)
    return out
keys = []
for k in resolve(sys.argv[2]):
    if k not in keys:
        keys.append(k)
for k in keys:
    f = files[k]
    print("|".join([f["repo"], f["path"], f["dir"], "1" if f.get("gated") else "0", str(f.get("gb", "?"))]))
PY
)
  rm -rf "$STAGE"
  T_ALL1=$(now)
  TOTALS=""
  [ "$BYTES_ALL" -gt 0 ] && TOTALS=" · $((BYTES_ALL/1000000000)) GB downloaded in $(python3 -c "print(int($T_ALL1-$T_ALL0))") s ($(mbps "$BYTES_ALL" "$(python3 -c "print($T_ALL1-$T_ALL0)")") MB/s)"
  say "MODELS DONE $DONE/$TOTAL present${SKIPPED:+ · $SKIPPED skipped (gated, no token)} · $(du -sh $COMFY/models 2>/dev/null | cut -f1) on disk$TOTALS"
  say "press r in ComfyUI to refresh the model lists"
fi

# ---------------------------------------------------------------- 5. self-test, then READY
say "step 5/5 self-test"
if [ "$SELFTEST" = "1" ] && [ -s $COMFY/models/diffusion_models/z_image_turbo_int8_convrot.safetensors ] \
   && [ -s $COMFY/models/text_encoders/qwen_3_4b_fp8_mixed.safetensors ] && [ -s $COMFY/models/vae/ae.safetensors ] \
   && curl -fs "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1; then
  python - "$PORT" <<'PY'
import json, sys, time, urllib.request
port = sys.argv[1]
base = "http://127.0.0.1:%s" % port
prompt = {
 "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "z_image_turbo_int8_convrot.safetensors", "weight_dtype": "default"}},
 "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b_fp8_mixed.safetensors", "type": "lumina2", "device": "default"}},
 "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
 "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text":
       "Studio packshot of a matte black protein powder jar on a seamless light grey background, soft studio lighting, product photography"}},
 "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
 "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
 "7": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 8, "cfg": 1.0, "sampler_name": "res_multistep",
       "scheduler": "simple", "denoise": 1.0, "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
 "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
 "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ainvfx_selftest", "images": ["8", 0]}},
}
def call(path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())
t0 = time.time()
try:
    pid = call("/prompt", {"prompt": prompt, "client_id": "ainvfx-bootstrap"})["prompt_id"]
except Exception as e:
    print("[AINVFX] WARNING: SELFTEST could not queue the test image:", str(e)[:300]); sys.exit(0)
while time.time() - t0 < 300:
    time.sleep(2)
    try:
        h = call("/history/" + pid).get(pid)
    except Exception:
        h = None
    if h:
        st = h.get("status", {})
        if st.get("status_str") == "error" or (st.get("completed") is False and st.get("status_str") == "error"):
            msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
            detail = msgs[-1][1].get("exception_message", "?")[:300] if msgs else "?"
            print("[AINVFX] WARNING: SELFTEST FAILED:", detail); sys.exit(0)
        outs = [i for o in h.get("outputs", {}).values() for i in o.get("images", [])]
        if outs:
            print("[AINVFX] SELFTEST OK · Z-Image Turbo 1024 x 1024, 8 steps, in %.1f s (models loaded from disk) · output/%s"
                  % (time.time() - t0, outs[0].get("filename")))
            sys.exit(0)
print("[AINVFX] WARNING: SELFTEST did not finish within 5 minutes: read %s/comfy.log" % "/workspace")
PY
else
  say "self-test skipped (AINVFX_SELFTEST=$SELFTEST, or the Z-Image Turbo files are missing)"
fi

say "READY · ComfyUI $PROXY · JupyterLab port 8888 · log $LOG"
say "remember: terminate the pod when you are done"
if [ -n "$START_PID" ]; then
  wait "$START_PID"
fi
sleep infinity
