#!/usr/bin/env bash
# ainvfx-runpod · bootstrap.sh · runs ON THE POD as the container start command.
#
# What it does, in order (every line it prints starts with [AINVFX], so the log is easy to read):
#   1. health check: GPU and driver, disk speed, download speed from Hugging Face
#   2. install: uv, Python 3.13, PyTorch stable for CUDA 13.0, ComfyUI at a pinned tag, the Manager
#   3. start ComfyUI, listening for Runpod's proxy on port 8188
#   4. download the models of the profile (AINVFX_PROFILE: image, video or train), resumable
#   5. hand over to Runpod's /start.sh (SSH and JupyterLab), so the pod behaves as usual
#
# Environment variables, all optional:
#   AINVFX_PROFILE    image (default) · video · train
#   AINVFX_COMFY_TAG  the ComfyUI git tag (default v0.38.2)
#   AINVFX_REPO_RAW   where to fetch models.json from (default: this repository on GitHub)
#   HF_TOKEN          a Hugging Face read token, for the gated files (LTX 2.5). Without it they are skipped.
#
# Idempotent: a second run (pod restart) skips what is already installed and downloaded.
# Nothing here depends on SSH: everything is visible in the pod's log in the Runpod console.

set -uo pipefail
PROFILE="${AINVFX_PROFILE:-image}"
TAG="${AINVFX_COMFY_TAG:-v0.38.2}"
RAW="${AINVFX_REPO_RAW:-https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main}"
ROOT=/workspace
COMFY=$ROOT/ComfyUI
VENV=$ROOT/venv
LOG=$ROOT/ainvfx-bootstrap.log
STAGE=$ROOT/.hfdl
PORT=8188
export HF_XET_HIGH_PERFORMANCE=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1

say()  { echo "[AINVFX] $*"; }
warn() { echo "[AINVFX] WARNING: $*"; }
now()  { date +%s.%N; }
mbps() { python3 -c "import sys; b=float(sys.argv[1]); t=float(sys.argv[2]); print(int(b/1e6/max(t,0.001)))" "$1" "$2"; }

say "bootstrap start · profile $PROFILE · ComfyUI $TAG · $(date -u +'%F %T') UTC"

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
[ "$RD" -lt 1000 ] 2>/dev/null && warn "DISK READ UNDER 1000 MB/s: models will load slowly. Consider another pod."

mkdir -p $COMFY/models/vae
PROBE_URL="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors"
PROBE_OUT=$COMFY/models/vae/ae.safetensors
if [ ! -s "$PROBE_OUT" ]; then
  T0=$(now); curl -fsSL "$PROBE_URL" -o "$PROBE_OUT"; T1=$(now)
  BYTES=$(stat -c %s "$PROBE_OUT" 2>/dev/null || echo 0)
  DL=$(mbps "$BYTES" "$(python3 -c "print($T1-$T0)")")
  say "download from Hugging Face: $DL MB/s (ae.safetensors, $((BYTES/1000000)) MB)"
  [ "$DL" -lt 100 ] 2>/dev/null && warn "DOWNLOAD UNDER 100 MB/s: 100 GB of models would take over 20 minutes. Consider another pod."
else
  say "download probe skipped: ae.safetensors already present"
fi

# ---------------------------------------------------------------- 2. install
say "step 2/5 install (uv, Python 3.13, PyTorch cu130, ComfyUI $TAG)"
command -v git  >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git)
command -v tmux >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq tmux) || true
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
if [ ! -f "$MARK" ]; then
  uv pip install --quiet torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu130 \
    || warn "PyTorch install failed: read the lines above"
  if [ ! -d $COMFY/.git ]; then
    git clone --quiet https://github.com/Comfy-Org/ComfyUI $COMFY || warn "git clone of ComfyUI failed"
  fi
  (cd $COMFY && git fetch --tags --quiet && git checkout --quiet "$TAG") || warn "could not check out $TAG"
  uv pip install --quiet -r $COMFY/requirements.txt || warn "ComfyUI requirements failed"
  [ -f $COMFY/manager_requirements.txt ] && (uv pip install --quiet -r $COMFY/manager_requirements.txt || true)
  uv pip install --quiet -U huggingface_hub || warn "huggingface_hub install failed: model downloads will fail"
  python - <<'EOF' && date -u +'%F %T' > "$MARK"
import torch, sys
ok = torch.cuda.is_available()
cu = torch.version.cuda or "0"
print("[AINVFX] torch", torch.__version__, "cuda", cu, "GPU", torch.cuda.get_device_name(0) if ok else "NOT VISIBLE")
if not ok:
    print("[AINVFX] WARNING: THE GPU IS NOT VISIBLE FROM PYTORCH. Terminate this pod and create another.")
    sys.exit(1)
if tuple(int(x) for x in cu.split(".")[:2]) < (13, 0):
    print("[AINVFX] WARNING: CUDA", cu, "under 13.0: int8 kernels off, everything slow.")
EOF
else
  say "install already done ($(cat "$MARK") UTC)"
fi
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
  say "COMFYUI UP · https://${RUNPOD_POD_ID:-<pod id>}-$PORT.proxy.runpod.net"
else
  warn "COMFYUI DID NOT START within 3 minutes: read $ROOT/comfy.log (JupyterLab or ssh)"
fi

# ---------------------------------------------------------------- 4. models
say "step 4/5 models of profile $PROFILE"
MODELS_JSON=$ROOT/models.json
curl -fsSL "$RAW/models.json" -o "$MODELS_JSON.new" && mv -f "$MODELS_JSON.new" "$MODELS_JSON"
if [ ! -s "$MODELS_JSON" ]; then
  warn "models.json could not be fetched: no model downloaded. Use the Manager's model library."
else
  mkdir -p "$STAGE"
  TOTAL=0; DONE=0; SKIPPED=0
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
    if hf download "$repo" "$path" --local-dir "$STAGE" >/dev/null 2>&1 && [ -s "$STAGE/$path" ]; then
      mv -f "$STAGE/$path" "$dest/$name"
      DONE=$((DONE+1))
    else
      warn "download failed: $path from $repo (path changed, or access refused)"
    fi
  done < <(python - "$MODELS_JSON" "$PROFILE" <<'EOF'
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
EOF
)
  rm -rf "$STAGE"
  say "MODELS DONE $DONE/$TOTAL present${SKIPPED:+ · $SKIPPED skipped (gated, no token)} · $(du -sh $COMFY/models 2>/dev/null | cut -f1) on disk"
  say "press r in ComfyUI to refresh the model lists"
fi

# ---------------------------------------------------------------- 5. hand over
say "step 5/5 READY · ComfyUI https://${RUNPOD_POD_ID:-<pod id>}-$PORT.proxy.runpod.net · JupyterLab port 8888 · log $LOG"
say "remember: terminate the pod when you are done"
if [ -x /start.sh ]; then
  exec /start.sh
fi
sleep infinity
