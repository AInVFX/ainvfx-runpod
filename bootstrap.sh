#!/usr/bin/env bash
# ainvfx-runpod · bootstrap.sh · runs ON THE POD as the container start command.
#
# What it does, in order (every line it prints starts with [AINVFX], so the log is easy to read):
#   0. start Runpod's own /start.sh in the background: SSH and JupyterLab are up within seconds
#   1. health check: GPU and driver, disk speed (the download speed is measured in step 4, on real files)
#   2. install: uv, Python 3.13, PyTorch stable for CUDA 13.0, ComfyUI at a pinned tag, the Manager
#   3. start ComfyUI, listening for Runpod's proxy on port 8188, and check the proxy from here
#   4. download the models of the profile (AINVFX_PROFILE: image, video or train), resumable; every
#      line says file k of n, GB done of GB planned, percent and the time left
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
# v5 (5 Oct 2026): every choice is an environment variable with a default, listed in settings.env at
# the root of the repository; pod.py sends that file's values with each pod, and a browser user sets
# the same variables on the deploy page. New: AINVFX_PYTHON, AINVFX_TORCH, AINVFX_TORCH_INDEX,
# AINVFX_MODELS_URL, AINVFX_CUSTOM_NODES, AINVFX_HEALTHCHECK, AINVFX_BOOTSTRAP_URL (a fork's own
# bootstrap, fetched and run instead of this one).
#
# v5.5 (7 Oct 2026, after the first pod on a B300 MIG slice): step 1 printed "[Insufficient Permissions]"
# as the GPU's memory. A MIG slice (Multi-Instance GPU: one part of a bigger GPU, with its own memory,
# here 1g.34gb of a B300) cannot read the whole card's memory from inside its container. Step 1 now reads
# the slice's size from its profile name (`nvidia-smi -L`), and the PyTorch line of step 2 adds the VRAM
# that PyTorch sees, on every GPU.
# v5.4 (6 Oct 2026, night, after a console pod that failed): three changes.
# (1) PyTorch. One host could not reach pypi.nvidia.com, where the PyTorch index sends its NVIDIA
# libraries, and the install failed. uv now waits longer and retries more (UV_HTTP_TIMEOUT 120,
# UV_HTTP_RETRIES 5). After a failure, PyPI is tried: it carries the same CUDA 13.0 build of the stable
# release, with the NVIDIA libraries on its own servers. The other installs retry once. A pod that still
# cannot work (no Python, no PyTorch, no GPU from PyTorch, no ComfyUI) prints a line starting with
# FAILED, says what to do, and stays up so the log can be read; pod.py 0.5.1 stops on that line.
# (2) Gated files. The token is checked once, and the log names its Hugging Face account. Each gated
# file is checked before its download. A file whose licence that account has not accepted is listed,
# just before the last line, with the link of its page and the command that fetches it once accepted:
# `bash <this script> models` (step 4 only, then exit; works from JupyterLab's terminal or ssh).
# (3) The last line gives JupyterLab's link with its token, so one click opens it (Adrien's choice for
# temporary teaching pods), or says that JupyterLab is off.
# v5.3 (6 Oct 2026): every download line says where it stands: file k of n, GB done of GB planned,
# percent, and the time left at the average speed so far (the plan is computed before the loop).
# v5.2 (6 Oct 2026): default ComfyUI tag v0.39.0 (tagged 5 Oct 22:49 UTC: Save EXR in 16-bit float by default,
# Save Video quality defaults, --offline). First pod on it: Adrien's gate before session 2.
# v5.1 (5 Oct 2026, comment only): the template image is runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404,
# the image of Runpod's own "Runpod Pytorch 2.8.0" template, which Runpod keeps on its hosts: the
# container starts within seconds. The 1.0.7-cu1300-torch291 tag tried on 5 Oct had to be fetched from
# Docker Hub on every pod (19 layers, about 9 GB, over 5 minutes). This script does not use the image's
# PyTorch or CUDA toolkit: the venv it builds takes the cu130 wheels, which carry their own CUDA 13
# libraries. A custom node that compiles a kernel against CUDA 13 would need
# `uv pip install nvidia-cuda-nvcc==13.0.88` in the venv first; no course step does.
#
# Environment variables, all optional (the defaults are the bootcamp's; settings.env documents them):
#   AINVFX_PROFILE        image (default) · video · train: which set of models.json to download
#   AINVFX_COMFY_TAG      the ComfyUI git tag or branch (default v0.39.0; master for the latest)
#   AINVFX_PYTHON         the Python version of the environment (default 3.13)
#   AINVFX_TORCH          the PyTorch packages, with pip flags if wanted (default: torch torchvision torchaudio;
#                         "--pre torch torchvision torchaudio" for nightlies)
#   AINVFX_TORCH_INDEX    the PyTorch wheel index (default https://download.pytorch.org/whl/cu130)
#   AINVFX_MODELS_URL     where models.json comes from (default: this repository on GitHub)
#   AINVFX_CUSTOM_NODES   git URLs of custom nodes to install, separated by spaces (default: none)
#   AINVFX_HEALTHCHECK    1 (default) measures the disk; 0 skips the measurement
#   AINVFX_SELFTEST       1 (default) generates the test image at the end; 0 skips it
#   AINVFX_BOOTSTRAP_URL  the raw URL of another bootstrap.sh (a fork): fetched and run in place of this one
#   HF_TOKEN              a Hugging Face read token, for the gated files (LTX). Without it they are skipped.
#                         Runpod fills it from the secret `huggingface_token` when the value is
#                         {{ RUNPOD_SECRET_huggingface_token }}; an unresolved placeholder counts as absent.
#
# One argument is understood: `models` runs step 4 alone (the files still missing), then exits.
#
# Idempotent: a second run (pod restart) skips what is already installed and downloaded.
# Nothing here depends on SSH: everything is visible in the pod's log in the Runpod console.

set -uo pipefail
# A fork's bootstrap takes over here, once (AINVFX_BOOTSTRAP_RAN guards against a loop).
if [ -n "${AINVFX_BOOTSTRAP_URL:-}" ] && [ -z "${AINVFX_BOOTSTRAP_RAN:-}" ]; then
  echo "[AINVFX] fetching the bootstrap named in AINVFX_BOOTSTRAP_URL: $AINVFX_BOOTSTRAP_URL"
  if curl -fsSL --retry 3 "$AINVFX_BOOTSTRAP_URL" -o /tmp/ainvfx-bootstrap-fork.sh; then
    AINVFX_BOOTSTRAP_RAN=1 exec bash /tmp/ainvfx-bootstrap-fork.sh "$@"
  fi
  echo "[AINVFX] WARNING: that bootstrap could not be fetched; continuing with this one"
fi

MODE="${1:-all}"
case "$MODE" in
  all|models) ;;
  *) echo "usage: bash $0 [models]   (models: fetch the files still missing, then exit)"; exit 2 ;;
esac
if [ "$MODE" = "models" ]; then
  # A terminal opened on the pod may lack the pod's variables: read them from the container's first process.
  for v in AINVFX_PROFILE AINVFX_MODELS_URL AINVFX_REPO_RAW HF_TOKEN; do
    if [ -z "${!v:-}" ] && [ -r /proc/1/environ ]; then
      val=$({ tr '\0' '\n' < /proc/1/environ; } 2>/dev/null | sed -n "s/^$v=//p" | head -1)
      [ -n "$val" ] && export "$v=$val"
    fi
  done
fi

PROFILE="${AINVFX_PROFILE:-image}"
TAG="${AINVFX_COMFY_TAG:-v0.39.0}"
PY="${AINVFX_PYTHON:-3.13}"
TORCH="${AINVFX_TORCH:-torch torchvision torchaudio}"
TORCH_INDEX="${AINVFX_TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
MODELS_URL="${AINVFX_MODELS_URL:-${AINVFX_REPO_RAW:-https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main}/models.json}"
CUSTOM_NODES="${AINVFX_CUSTOM_NODES:-}"
HEALTHCHECK="${AINVFX_HEALTHCHECK:-1}"
SELFTEST="${AINVFX_SELFTEST:-1}"
ROOT=/workspace
COMFY=$ROOT/ComfyUI
VENV=$ROOT/venv
LOG=$COMFY/input/ainvfx/bootstrap.log   # inside ComfyUI's input folder: readable through the proxy
STAGE=$ROOT/.hfdl
PORT=8188
PROXY="https://${RUNPOD_POD_ID:-<pod id>}-$PORT.proxy.runpod.net"
JUPYTER="https://${RUNPOD_POD_ID:-<pod id>}-8888.proxy.runpod.net"
SELF=$(readlink -f "$0" 2>/dev/null || echo "$0")   # this script on the pod, for the `models` command it prints
export HF_XET_HIGH_PERFORMANCE=1
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$(dirname "$LOG")"
ln -sfn "$LOG" $ROOT/ainvfx-bootstrap.log
exec > >(tee -a "$LOG") 2>&1

say()  { echo "[AINVFX] $*"; }
warn() { echo "[AINVFX] WARNING: $*"; }
now()  { date +%s.%N; }
mbps() { python3 -c "import sys; b=float(sys.argv[1]); t=float(sys.argv[2]); print(int(b/1e6/max(t,0.001)))" "$1" "$2"; }
fail() { echo "[AINVFX] FAILED: $*"; }
hold() {   # after FAILED: the pod stays up (SSH, JupyterLab, this log) until it is terminated
  say "the pod stays up so this log can be read, and it bills until you terminate it"
  if [ -n "${START_PID:-}" ]; then wait "$START_PID"; fi
  sleep infinity
}
# uv waits longer and retries more than its defaults (30 s, 3 retries): slow mirrors happen.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"
uvi() {    # uv pip install, once more after a pause if the first attempt fails
  uv pip install --quiet "$@" && return 0
  warn "install failed, one more try in 10 s: uv pip install $*"
  sleep 10
  uv pip install --quiet "$@"
}
# PyTorch: its own index first. That index sends its NVIDIA libraries to pypi.nvidia.com, which a host
# could not reach on 6 Oct 2026. PyPI carries the same CUDA 13.0 build of the stable release, with the
# NVIDIA libraries on its own servers, so it is the second route (stable cu130 only: PyPI has no other
# CUDA build and no nightlies). uv keeps finished downloads in its cache: a new attempt fetches only what failed.
install_torch() {
  # shellcheck disable=SC2086
  uv pip install --quiet $TORCH --index-url "$TORCH_INDEX" && return 0
  warn "PyTorch download from $TORCH_INDEX failed (the lines above name the file)"
  local pypi=0 attempt
  case "$TORCH_INDEX" in */whl/cu130|*/whl/cu130/) pypi=1 ;; esac
  case " $TORCH " in *" --pre "*) pypi=0 ;; esac
  for attempt in 1 2; do
    sleep 10
    if [ "$pypi" = "1" ]; then
      say "PyTorch: trying PyPI (attempt $attempt of 2): the same CUDA 13.0 build, with the NVIDIA libraries from PyPI"
      # shellcheck disable=SC2086
      uv pip install --quiet $TORCH && return 0
    else
      say "PyTorch: trying $TORCH_INDEX again (attempt $attempt of 2)"
      # shellcheck disable=SC2086
      uv pip install --quiet $TORCH --index-url "$TORCH_INDEX" && return 0
    fi
  done
  return 1
}
hf_state() {   # ok, gated, missing or error for one file of Hugging Face: one HEAD request, nothing downloaded
  local auth=() hdr code err
  [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  hdr=$(curl -sI --max-time 20 "${auth[@]}" "https://huggingface.co/$1/resolve/main/$2" 2>/dev/null | tr -d '\r')
  code=$(printf '%s\n' "$hdr" | awk '/^HTTP/ {c=$2} END {print c}')
  err=$(printf '%s\n' "$hdr" | awk -F': ' 'tolower($1) == "x-error-code" {e=$2} END {print e}')
  case "$code" in
    2*|3*) echo ok ;;
    401|403) if [ "$err" = "GatedRepo" ]; then echo gated; else echo error; fi ;;
    404) echo missing ;;
    *) echo error ;;
  esac
}
NEED_REPOS=""; NOTOKEN_REPOS=""; WAITING=0; SKIPPED=0; HF_USER=""
print_links() {   # one page link per repo, each repo once, in the order met
  local r
  # shellcheck disable=SC2086
  for r in $(printf '%s\n' $1 | awk 'NF && !seen[$0]++'); do say "   https://huggingface.co/$r"; done
}
access_summary() {   # the gated files that did not come, with the page of each and what to do
  if [ -n "$NEED_REPOS" ]; then
    say "ACTION NEEDED · $WAITING gated file(s) wait for a licence that the Hugging Face account${HF_USER:+ $HF_USER} has not accepted. Logged in to huggingface.co with that account, open each page and click « Agree and access repository » (the LTX pages approve at once):"
    print_links "$NEED_REPOS"
    say "   then fetch them on this pod, without a new one: bash $SELF models   (in JupyterLab's terminal, or after python pod.py ssh)"
    say "   a Read token works; a fine-grained token also needs its permission to read public gated repos"
  fi
  if [ -n "$NOTOKEN_REPOS" ]; then
    say "ACTION NEEDED · $SKIPPED gated file(s) skipped: this pod has no Hugging Face token. Store a Read token as the Runpod secret huggingface_token (python pod.py setup --hf-token), accept the licence on each page below, then create a new pod:"
    print_links "$NOTOKEN_REPOS"
  fi
}

if [ "$MODE" = "models" ]; then
  say "models only (bash $SELF models) · profile $PROFILE · $(date -u +'%F %T') UTC"
else
  say "bootstrap start · profile $PROFILE · ComfyUI $TAG · Python $PY · $(date -u +'%F %T') UTC"
  say "settings: torch '$TORCH' from $TORCH_INDEX · models $MODELS_URL${CUSTOM_NODES:+ · custom nodes: $CUSTOM_NODES}"
fi

# A placeholder Runpod did not substitute (no secret in the account) must not reach Hugging Face:
# an invalid token makes it refuse even public files.
if [ -n "${HF_TOKEN:-}" ] && case "$HF_TOKEN" in *"{{"*) true;; *) false;; esac; then
  say "no Runpod secret named huggingface_token in this account: HF_TOKEN ignored"
  unset HF_TOKEN
fi
# The token, checked once: a refused token is dropped (the public files then come without it), a valid one
# names its Hugging Face account, the one that must accept the gated licences.
if [ -n "${HF_TOKEN:-}" ]; then
  WHO=$(curl -s --max-time 15 -w '\n%{http_code}' -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami-v2 2>/dev/null)
  case "$(printf '%s\n' "$WHO" | tail -n 1)" in
    200) HF_USER=$(printf '%s\n' "$WHO" | sed '$d' | python3 -c "import json, sys; print(json.load(sys.stdin).get('name', ''))" 2>/dev/null) ;;
    401) warn "HUGGING FACE REFUSES THIS TOKEN (401): it is ignored, so the gated files will be skipped. Create a new Read token (huggingface.co, Settings, Access Tokens) and store it again as the Runpod secret huggingface_token"
         unset HF_TOKEN ;;
  esac
fi
if [ -n "${HF_TOKEN:-}" ]; then
  say "HF_TOKEN present${HF_USER:+, Hugging Face account $HF_USER}: gated files (LTX) allowed once their licence is accepted"
else
  say "no HF_TOKEN: the gated files (LTX, video sessions) will be skipped; the image models need none"
fi

if [ "$MODE" != "models" ]; then   # steps 0 to 3; the models mode goes straight to step 4
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
  GPU_NAME=$(echo "$GPU_LINE" | awk -F', ' '{print $1}')
  GPU_MEM=$(echo "$GPU_LINE" | awk -F', ' '{print $2}')
  DRV_FULL=$(echo "$GPU_LINE" | awk -F', ' '{print $3}')
  case "$GPU_MEM" in
    [0-9]*) ;;
    *)  # v5.5: a MIG slice cannot read the whole card's memory from its container ("[Insufficient
        # Permissions]"). Its own size is in its profile name, as `nvidia-smi -L` lists it:
        # "  MIG 1g.34gb     Device  0: (UUID: MIG-...)". PyTorch reports the exact VRAM in step 2.
        MIG=$(nvidia-smi -L 2>/dev/null | sed -n 's/^[[:space:]]*MIG \([^[:space:]]*\).*/\1/p' | head -1)
        MIG_GB=$(echo "$MIG" | grep -o '[0-9][0-9]*gb' | head -1 | sed 's/gb$//')
        if [ -n "$MIG_GB" ]; then
          GPU_MEM="MIG slice $MIG, $MIG_GB GB"
        elif [ -n "$MIG" ]; then
          GPU_MEM="MIG slice $MIG (its VRAM: see step 2)"
        else
          GPU_MEM="VRAM not readable here (see step 2)"
        fi ;;
  esac
  say "GPU: $GPU_NAME, $GPU_MEM, $DRV_FULL"
  DRV=$(echo "$DRV_FULL" | cut -d. -f1)
  if [ -n "$DRV" ] && [ "$DRV" -lt 580 ] 2>/dev/null; then
    warn "DRIVER $DRV IS OLDER THAN 580: CUDA 13 kernels will not run, int8 models will be slow. Terminate this pod and create another."
  fi
else
  warn "nvidia-smi NOT FOUND: no GPU visible. Terminate this pod and create another."
fi
say "CPU: $(nproc) cores · RAM: $(free -g | awk '/Mem:/ {print $2}') GB · disk free on $ROOT: $(df -BG $ROOT | awk 'NR==2 {print $4}')"

if [ "$HEALTHCHECK" = "1" ]; then
  T0=$(now); dd if=/dev/zero of=$ROOT/.probe bs=1M count=2048 oflag=direct status=none 2>/dev/null; T1=$(now)
  WR=$(mbps 2147483648 "$(python3 -c "print($T1-$T0)")")
  T0=$(now); dd if=$ROOT/.probe of=/dev/null bs=1M iflag=direct status=none 2>/dev/null; T1=$(now)
  RD=$(mbps 2147483648 "$(python3 -c "print($T1-$T0)")")
  rm -f $ROOT/.probe
  say "disk: write $WR MB/s · read $RD MB/s"
  [ "$RD" -lt 300 ] 2>/dev/null && warn "DISK READ UNDER 300 MB/s: models will load slowly. Consider another pod."
else
  say "disk measurement skipped (AINVFX_HEALTHCHECK=0)"
fi


# ---------------------------------------------------------------- 2. install
say "step 2/5 install (uv, Python $PY, PyTorch from $TORCH_INDEX, ComfyUI $TAG)"
command -v git  >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq git)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || pip install -q uv
fi
say "uv $(uv --version 2>/dev/null | awk '{print $2}')"
if [ ! -x $VENV/bin/python ]; then
  uv venv $VENV --python "$PY" --quiet || { sleep 10; uv venv $VENV --python "$PY" --quiet; } || {
    fail "PYTHON $PY COULD NOT BE INSTALLED on this machine (the lines above). Terminate this pod and create a new one: it lands on another machine."
    hold
  }
fi
# shellcheck disable=SC1091
source $VENV/bin/activate
MARK=$VENV/.ainvfx_install_${TAG}_py${PY}
if [ ! -f "$MARK" ] || [ ! -f $COMFY/main.py ]; then
  install_torch || {
    fail "PYTORCH COULD NOT BE DOWNLOADED on this machine: the network of this host timed out (the lines above). Your settings are not the cause. Terminate this pod and create a new one, which lands on another machine: python pod.py down, then python pod.py up $PROFILE; or Terminate in the console, then deploy again from the template."
    hold
  }
  # git init + fetch instead of git clone: works in a folder that already holds models (a restart)
  if [ ! -d $COMFY/.git ]; then
    mkdir -p $COMFY
    (cd $COMFY && git init --quiet && git remote add origin https://github.com/Comfy-Org/ComfyUI) \
      || warn "git init of ComfyUI failed"
  fi
  (cd $COMFY && git fetch --quiet --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG" && git checkout --quiet "$TAG") \
    || (cd $COMFY && git fetch --quiet --depth 1 origin "$TAG" && git checkout --quiet FETCH_HEAD) \
    || (cd $COMFY && git fetch --quiet --tags origin && git checkout --quiet "$TAG") \
    || warn "could not check out ComfyUI $TAG"
  uvi -r $COMFY/requirements.txt || warn "ComfyUI requirements failed"
  [ -f $COMFY/manager_requirements.txt ] && (uvi -r $COMFY/manager_requirements.txt || true)
  uvi -U huggingface_hub || warn "huggingface_hub install failed: model downloads will fail"
  # custom nodes named in AINVFX_CUSTOM_NODES: cloned into custom_nodes, their requirements installed
  for url in $CUSTOM_NODES; do
    name=$(basename "${url%.git}")
    if [ ! -d "$COMFY/custom_nodes/$name" ]; then
      git clone --quiet --depth 1 "$url" "$COMFY/custom_nodes/$name" && say "custom node $name installed" \
        || warn "custom node $url could not be cloned"
    fi
    [ -f "$COMFY/custom_nodes/$name/requirements.txt" ] && (uvi -r "$COMFY/custom_nodes/$name/requirements.txt" \
        || warn "the requirements of $name failed")
  done
  if [ -f $COMFY/main.py ]; then
    if python - <<'PY'
import torch, sys
ok = torch.cuda.is_available()
cu = torch.version.cuda or "0"
try:      # v5.5: the VRAM as PyTorch sees it, exact on every GPU, a MIG slice included; never fatal
    vram = " · {:.0f} MiB of VRAM".format(torch.cuda.get_device_properties(0).total_memory / 2**20) if ok else ""
except Exception:
    vram = ""
print("[AINVFX] torch", torch.__version__, "cuda", cu, "GPU", (torch.cuda.get_device_name(0) if ok else "NOT VISIBLE") + vram)
if not ok:
    print("[AINVFX] WARNING: THE GPU IS NOT VISIBLE FROM PYTORCH. Terminate this pod and create another.")
    sys.exit(1)
if tuple(int(x) for x in cu.split(".")[:2]) < (13, 0):
    print("[AINVFX] WARNING: CUDA", cu, "under 13.0: int8 kernels off, everything slow.")
PY
    then
      date -u +'%F %T' > "$MARK"
    else
      fail "PYTORCH DOES NOT START, OR DOES NOT SEE THE GPU, on this machine (the lines above). Terminate this pod and create a new one: it lands on another machine."
      hold
    fi
  fi
else
  say "install already done ($(cat "$MARK") UTC)"
fi
if [ ! -f $COMFY/main.py ]; then
  fail "COMFYUI IS NOT INSTALLED ($COMFY/main.py missing: the lines above say why). Terminate this pod and create a new one."
  hold
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
  for _ in $(seq 1 60); do
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
else   # models mode: the environment the first run built
  if [ -f $VENV/bin/activate ]; then
    # shellcheck disable=SC1091
    source $VENV/bin/activate
  else
    warn "no environment in $VENV: this pod never finished its install, so the models cannot be fetched from here"
    exit 1
  fi
fi   # end of steps 0 to 3

# ---------------------------------------------------------------- 4. models
say "step 4/5 models of profile $PROFILE"
MODELS_JSON=$ROOT/models.json
curl -fsSL --retry 3 "$MODELS_URL" -o "$MODELS_JSON.new" && mv -f "$MODELS_JSON.new" "$MODELS_JSON"
if [ ! -s "$MODELS_JSON" ]; then
  warn "models.json could not be fetched: no model downloaded. Use the Manager's model library."
else
  mkdir -p "$STAGE"
  # The plan first (v5.3): which files are missing, their number and their size, so every
  # download line says where it stands (k of n, GB done of GB total, percent, time left).
  # v5.4: each missing gated file is checked first (one HEAD request with the token), so a licence not
  # yet accepted is told with its page instead of failing a download. Every row ends with its state:
  # todo, present, notoken (gated, no token), gated (licence not accepted) or missing (path gone).
  LIST=$ROOT/models.list
  python - "$MODELS_JSON" "$PROFILE" "$COMFY/models" "${HF_TOKEN:-}" > "$LIST" <<'PY'
import json, os, sys, urllib.error, urllib.parse, urllib.request
spec = json.load(open(sys.argv[1]))
files, profiles, root, token = spec["files"], spec["profiles"], sys.argv[3], sys.argv[4]
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None                 # a redirect to the file's storage already means "allowed"
opener = urllib.request.build_opener(NoRedirect)
def access(repo, path):
    """todo, gated, missing, or todo again when the check itself fails (the download then decides)."""
    req = urllib.request.Request("https://huggingface.co/%s/resolve/main/%s" % (repo, urllib.parse.quote(path)),
                                 method="HEAD", headers={"Authorization": "Bearer " + token} if token else {})
    try:
        opener.open(req, timeout=20).close()
        return "todo"
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:
            return "todo"
        if e.code in (401, 403) and e.headers.get("X-Error-Code") == "GatedRepo":
            return "gated"
        if e.code == 404:
            return "missing"
        return "todo"
    except Exception:
        return "todo"
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
rows = []
for k in keys:
    f = files[k]
    name = os.path.basename(f["path"])
    present = os.path.isfile(os.path.join(root, f["dir"], name)) and os.path.getsize(os.path.join(root, f["dir"], name)) > 0
    gated = bool(f.get("gated"))
    if present:
        state = "present"
    elif gated and not token:
        state = "notoken"
    elif gated:
        state = access(f["repo"], f["path"])
    else:
        state = "todo"
    rows.append([f["repo"], f["path"], f["dir"], "1" if gated else "0", str(f.get("gb", "?")), state, f.get("gb") or 0])
todo = [r for r in rows if r[5] == "todo"]
n_todo = len(todo)
gb_todo = round(sum(r[6] for r in todo), 1)
k, before = 0, 0.0
for r in rows:
    if r[5] == "todo":
        k += 1
        print("|".join(r[:5] + [str(k), str(n_todo), "%.1f" % before, "%.1f" % gb_todo, r[5]]))
        before += r[6]
    else:
        print("|".join(r[:5] + ["", str(n_todo), "", "%.1f" % gb_todo, r[5]]))
PY
  N_ALL=$(grep -c . "$LIST" 2>/dev/null || echo 0)
  N_TODO=$(head -1 "$LIST" 2>/dev/null | cut -d'|' -f7); N_TODO=${N_TODO:-0}
  GB_TODO=$(head -1 "$LIST" 2>/dev/null | cut -d'|' -f9); GB_TODO=${GB_TODO:-0}
  # the rest of the plan, by state: present, gated without a token, waiting for a licence, moved
  PLAN_REST=$(awk -F'|' '{n[$10]++} END {s = ""
    if (n["present"]) s = s " · " n["present"] " already present"
    if (n["notoken"]) s = s " · " n["notoken"] " gated, no token"
    if (n["gated"])   s = s " · " n["gated"] " waiting for a licence"
    if (n["missing"]) s = s " · " n["missing"] " moved"
    print s}' "$LIST")
  say "models: $N_ALL files in profile $PROFILE · $N_TODO to download ($GB_TODO GB)$PLAN_REST"
  TOTAL=0; DONE=0; SKIPPED=0; BYTES_ALL=0; T_ALL0=$(now); SPEED_JUDGED=0
  while IFS='|' read -r repo path dir _ gb k n_todo gb_before gb_todo state; do
    [ -z "$repo" ] && continue
    TOTAL=$((TOTAL+1))
    name=$(basename "$path")
    dest=$COMFY/models/$dir
    mkdir -p "$dest"
    if [ -s "$dest/$name" ]; then
      DONE=$((DONE+1)); continue
    fi
    case "$state" in
      notoken)
        say "models: SKIPPED $name (gated repo $repo, no HF_TOKEN): https://huggingface.co/$repo"
        SKIPPED=$((SKIPPED+1)); NOTOKEN_REPOS="$NOTOKEN_REPOS $repo"; continue ;;
      gated)
        say "models: WAITING $name · its licence is not accepted yet: https://huggingface.co/$repo"
        WAITING=$((WAITING+1)); NEED_REPOS="$NEED_REPOS $repo"; continue ;;
      missing)
        warn "models: $path is no longer in $repo (moved or renamed): models.json needs an update"
        continue ;;
    esac
    say "models $k/$n_todo · $gb_before of $gb_todo GB done · downloading $name ($gb GB) from $repo"
    T0=$(now)
    if hf download "$repo" "$path" --local-dir "$STAGE" >/dev/null 2>&1 && [ -s "$STAGE/$path" ]; then
      T1=$(now)
      BYTES=$(stat -c %s "$STAGE/$path" 2>/dev/null || echo 0)
      SECS=$(python3 -c "print(int($T1-$T0))")
      mv -f "$STAGE/$path" "$dest/$name"
      DONE=$((DONE+1)); BYTES_ALL=$((BYTES_ALL+BYTES))
      SPEED=$(mbps "$BYTES" "$(python3 -c "print($T1-$T0)")")
      # where we stand: GB done of the plan, percent, and the time left at the average speed so far
      PROGRESS=$(python3 - "$gb_before" "$gb" "$gb_todo" "$BYTES_ALL" "$T_ALL0" "$T1" <<'PY'
import sys
before, gb, total, bytes_all, t0, t1 = [float(x) if x not in ("?", "") else 0.0 for x in sys.argv[1:]]
done = before + gb
pct = int(100 * done / total) if total else 100
avg = bytes_all / 1e6 / max(t1 - t0, 0.001)            # MB/s over every download so far
left = max(total - done, 0.0)
eta = int(left * 1000 / avg) if avg > 0 else 0
when = "done" if left <= 0.05 else ("about %d min %02d s left" % (eta // 60, eta % 60) if eta >= 60 else "about %d s left" % eta)
print("%.1f of %.1f GB (%d%%) · %s" % (done, total, pct, when))
PY
)
      say "models $k/$n_todo · $name in $SECS s · $SPEED MB/s · $PROGRESS"
      # the speed judgement, once, on the first file over 1 GB (small files measure latency, not speed)
      if [ "$SPEED_JUDGED" = "0" ] && [ "$BYTES" -gt 1000000000 ]; then
        SPEED_JUDGED=1
        [ "$SPEED" -lt 50 ] 2>/dev/null && warn "DOWNLOAD UNDER 50 MB/s: the models of this profile could take over 20 minutes. Consider another pod."
      fi
    else
      case "$(hf_state "$repo" "$path")" in
        gated)
          say "models: WAITING $name · its licence is not accepted yet: https://huggingface.co/$repo"
          WAITING=$((WAITING+1)); NEED_REPOS="$NEED_REPOS $repo" ;;
        missing)
          warn "download failed: $path is no longer in $repo (moved or renamed): models.json needs an update" ;;
        *)
          warn "download failed: $name from $repo (network): bash $SELF models tries again" ;;
      esac
    fi
  done < "$LIST"
  rm -rf "$STAGE"
  T_ALL1=$(now)
  TOTALS=""
  [ "$BYTES_ALL" -gt 0 ] && TOTALS=" · $((BYTES_ALL/1000000000)) GB downloaded in $(python3 -c "print(int($T_ALL1-$T_ALL0))") s ($(mbps "$BYTES_ALL" "$(python3 -c "print($T_ALL1-$T_ALL0)")") MB/s)"
  SK=""; [ "$SKIPPED" -gt 0 ] 2>/dev/null && SK=" · $SKIPPED skipped (gated, no token)"
  [ "$WAITING" -gt 0 ] 2>/dev/null && SK="$SK · $WAITING waiting for a licence (the pages are listed at the end)"
  say "MODELS DONE $DONE/$TOTAL present$SK · $(du -sh $COMFY/models 2>/dev/null | cut -f1) on disk$TOTALS"
  say "press r in ComfyUI to refresh the model lists"
fi
if [ "$MODE" = "models" ]; then
  access_summary
  sleep 1   # lets the log copy catch up before the prompt returns
  exit 0
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

access_summary
# JupyterLab runs only when the pod was created with "Start Jupyter notebook" on (pod.py always asks for it):
# Runpod then sets JUPYTER_PASSWORD, the token the page asks for, and the link carries it, so one click
# opens JupyterLab (Adrien's choice, 6 Oct 2026: temporary teaching pods). Like the ComfyUI address, the
# line opens the pod to whoever reads it, and this log is served through the proxy: keep the address private.
if pgrep -f "jupyter-lab|jupyter lab" >/dev/null 2>&1; then
  if [ -n "${JUPYTER_PASSWORD:-}" ]; then JL="JupyterLab $JUPYTER/lab?token=$JUPYTER_PASSWORD"; else JL="JupyterLab $JUPYTER"; fi
else
  JL="JupyterLab off (the pod was created without « Start Jupyter notebook »)"
fi
say "READY · ComfyUI $PROXY · $JL · log $LOG"
say "remember: terminate the pod when you are done"
if [ -n "$START_PID" ]; then
  wait "$START_PID"
fi
sleep infinity
