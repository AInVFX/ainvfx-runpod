<p align="center">
  <a href="https://www.ainvfx.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.ainvfx.com/assets/images/logo.webp"><img src="https://www.ainvfx.com/assets/images/logo-light.png" alt="AInVFX" height="40"></picture></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.runpod.io/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047dba6113c1d43463ddd_logo-white.svg"><img src="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047c6e55021705127c8da_runpod-logo-black.svg" alt="Runpod" height="40"></picture></a>
</p>

# ainvfx-runpod

Rent a GPU on [Runpod](https://www.runpod.io/), get [ComfyUI](https://github.com/Comfy-Org/ComfyUI) running on it with the right models, work, download your results, terminate. From the browser or from one script, on Windows, macOS or Linux.

Built by [AInVFX](https://www.ainvfx.com/) for its [Generative AI Bootcamp for Film and TV](https://www.ainvfx.com/bootcamp/), in partnership with Runpod, the bootcamp's GPU infrastructure partner. Anyone may use it: the ComfyUI version and the model list are plain files you can change.

## How it works

A **pod** is a Linux machine with a GPU, rented by the second. You create one from the AInVFX template, the pod installs ComfyUI and the course models on its own, and you reach ComfyUI from your browser at `https://<pod id>-8188.proxy.runpod.net`. When you are done, you download your results and terminate the pod. Nothing stays on Runpod between sessions, so you pay only for the hours you work.

Two ways to create a pod. Both give the same machine.

| | What you need | What you do |
|---|---|---|
| **Browser** | a Runpod account with credit | open the template's deploy page, pick a GPU, deploy, read the log until `READY` |
| **Script** | the same, plus Python 3.8 or newer and Git | `python pod.py up image`, then `open`, `pull`, `down` |

## Quick start, browser only

1. Open [the AInVFX template](https://console.runpod.io/deploy?template=4i789znkrd&ref=iad0yzht) (template id `4i789znkrd`).
2. Pick the GPU: **RTX 5090** for image work. If it is out of stock, open the **Filter** button of the Compute section and set **Cloud type** to **Community** (machines owned by third parties, cheaper and often better stocked; the console shows a warning), or pick a slice of the RTX PRO 6000.
3. Click **Deploy On-Demand**. The pod appears under Pods; its **Logs** button shows the install running.
4. Wait for the line `READY`: 5 to 15 minutes depending on the data center (the log's first lines are Runpod's own: the image found on the host, the container started; then every line of ours starts with `[AINVFX]`). Open `https://<pod id>-8188.proxy.runpod.net` (also behind the pod's **Connect** button): that is your ComfyUI. The `READY` line also gives JupyterLab's link, its token included. If the button opens a blank "403" page, press Enter in the address bar: ComfyUI refuses a page opened by a click from another site (its protection against sites that would drive it), never the address typed or opened by `pod.py open`.
5. When you are done: download your results (the Assets panel in ComfyUI), then **Terminate** the pod. Stop is not enough: a stopped pod keeps a dead entry and may keep billing storage.

## Quick start, script

```bash
git clone https://github.com/AInVFX/ainvfx-runpod
cd ainvfx-runpod
python pod.py setup        # once: your Runpod API key, your nearest data centers, your SSH key (optional)
python pod.py setup --hf-token   # optional: your Hugging Face token, stored on Runpod as a secret (gated files)
python pod.py up image     # creates the pod, follows its log until READY
python pod.py open         # opens ComfyUI in your browser
python pod.py pull         # downloads the pod's outputs into outputs/<pod name>/
python pod.py down         # pulls, then terminates (billing stops)
```

The other commands: `logs` (follow the log again), `status` (GPU, cost so far, address, last log lines), `push file.png` (copy a file into the pod's input folder), `ssh` (a terminal on the pod), `list` (every pod of your account), `doctor` (check your setup). `python pod.py --help` lists them all. Ctrl+C while a log is followed stops the following only; the pod keeps running until `down`.

While `up` waits, two kinds of lines appear. `[RUNPOD]` lines are Runpod's own system log: what the host does before our script runs (the image found on the host or fetched from Docker Hub, the container started). `[AINVFX]` lines are our script on the pod. If the host has to fetch the image, `up` prints one summary line every 15 seconds (`image download: 19 layers · 7 done · 9 extracting · 3 downloading (2 min 30 s)`), so a long pull never looks like a hang; after 8 minutes without a container, it prints a warning in capitals. The pod bills from its creation, whatever the host is doing: the answer to a stalled pod is `python pod.py down`, then `up` again, usually in another data center. An error after the pod exists is always a message that names the pod, never a traceback.

**Several pods at once** (LoRA training, for example): `up train --name jar-lora` creates a second pod of the same profile, known to the script as `jar-lora`. Every command then takes that name, the pod's full name or its id: `status jar-lora`, `pull ainvfx-train-jar-lora`, `down jar-lora`. `list` shows every pod of your account and which ones this machine knows; a pod created from the console or from another machine is attached by its first command (`python pod.py status <name or id>`). `down --all` terminates every pod recorded here.

Profiles: `up image` rents an RTX 5090 with a 100 GB disk; `up video` an RTX PRO 6000 (96 GB) with a 200 GB disk, for LTX 2.5; `up train` the same card with 250 GB, for LoRA training.

**What you need on your machine**

- **Python 3.8 or newer.** Windows: [python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.12`. macOS and Linux usually have it: `python3 --version`. Or install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run `uv run pod.py ...`, which fetches a Python for you.
- **Git**, to clone and update this repository. Without Git, GitHub's "Download ZIP" button works too.
- **A Runpod API key**: in the console, Account, Credentials, API Keys, "Create API Key". Name it `ainvfx-runpod` (any name works; this one says what it is for). Choose *Restricted*, then set **api.runpod.io/graphql** to **Read / Write** (the API the script uses for pods, the catalog, secrets and SSH keys) and leave **api.runpod.ai** on **None** (Serverless endpoints, not used here). The old Settings page now redirects to Credentials. The script keeps the key in `~/.ainvfx-runpod/config.json`, readable by your account only. Never paste it in a chat or a slide.
- **`ssh` is optional.** Only `pod.py ssh` uses it. Everything else goes through the browser and Runpod's proxy.

The script uses Python's standard library only: nothing to install.

## What the pod does when it starts

The template's start command runs [`bootstrap.sh`](bootstrap.sh) from this repository. Every line it writes starts with `[AINVFX]`; read them in the pod's log (Runpod console, `python pod.py logs` or `python pod.py status`).

| Step | What happens | The log says |
|---|---|---|
| before | Runpod puts the image on the host. The template uses the image of Runpod's own "Runpod Pytorch 2.8.0" template, which the hosts already hold, so this takes seconds; a host that has to fetch an image from Docker Hub takes several minutes (19 layers, about 9 GB) | `[RUNPOD]` lines in `pod.py`; system log in the console |
| 0 | SSH and JupyterLab start in the background, before anything else | JupyterLab answers within two minutes |
| 1 | **Health check**: the GPU and its driver (580 or newer, so the CUDA 13 kernels of the int8 models run), disk write and read speed | a `WARNING` in capitals when a value is bad: terminate and create again, usually in another data center |
| 2 | **Install**: uv, Python 3.13, PyTorch for CUDA 13.0, ComfyUI at tag `v0.39.0`, the Manager, then the custom nodes named in `settings.env`. The same lines as a manual install on your own machine. A download that fails is retried, and PyTorch then comes from PyPI, the same CUDA 13.0 build | `ComfyUI v0.39.0`; `FAILED` when the pod cannot work on that machine (below) |
| 3 | **Start**: ComfyUI listens for the proxy | `COMFYUI UP` with the address, then `PROXY OK` |
| 4 | **Models**: the files of the profile, from [`models.json`](models.json), one by one. A first line gives the plan (files to download, GB); then every line says file k of n, the GB done of the GB planned, the percent and the time left at the average speed so far. The first file over 1 GB judges the download speed (a `WARNING` under 50 MB/s); files already present are skipped. Each gated file is checked first: one whose licence your Hugging Face account has not accepted gets a `WAITING` line with its page, and the end of the log lists them all | `MODELS DONE 14/14` with the total time and the average speed |
| 5 | **Self-test**: one Z-Image Turbo image, 1024 x 1024 in 8 steps, saved in `output/` | `SELFTEST OK` with the time, then `READY` with two links: ComfyUI, and JupyterLab with its token |

`up` and `logs` read the log from two places, because Runpod's live log stream has been seen staying open and silent while the pod wrote its last lines: the stream is reopened every minute a few seconds back, and every 15 seconds the script reads the copy of the log that the pod serves itself. Whichever shows `READY` first ends the wait.

If GitHub cannot be reached, the start command falls back to Runpod's own `/start.sh`: the pod still boots with SSH and JupyterLab, and the error can be read.

**When the log says `FAILED`.** The pod cannot work on that machine: Python, PyTorch or ComfyUI could not be installed, or PyTorch cannot see the GPU. The line says which, and what to do. The pod stays up so the log can be read, and it bills: terminate it and create another (`python pod.py down`, then `up`), which lands on another machine. `up` stops on that line.

**When the log says `ACTION NEEDED`.** Gated files did not come, and the lines below it give the Hugging Face page of each. Logged in with the account the log names, click « Agree and access repository » on each page, then fetch the files on the same pod: `bash /tmp/ainvfx-bootstrap.sh models`, in JupyterLab's terminal (File, New, Terminal) or after `python pod.py ssh`. A pod without a token needs one first (below), then a new pod.

Profile sizes (Hugging Face, 4 October 2026): `image` 14 files, about 63 GB; `video` 36 files, about 150 GB; `train` 38 files, about 215 GB.

Measured on two RTX 5090 pods on Secure Cloud (4 and 5 October 2026), on the image above: the container up within a minute or two of creation, PyTorch installed in 90 seconds, ComfyUI answering after about 4 minutes; the 63 GB of the `image` profile took 10 minutes in EUR-IS-2 (about 100 MB/s) and 90 seconds in EU-CZ-1 (300 to 1300 MB/s per file); the self-test 16 to 106 seconds on the first load, 2 seconds once cached; `READY` 5 to 15 minutes after creation. A third pod on 5 October, created with "any data center" after the first choice had no stock, landed on a host that Runpod had scheduled for removal: it never started its container, the API still said RUNNING, and the log stream did not answer. That pod is why `up` now warns after 8 minutes without a container and why every API error is a message. On 7 October a pod created the same way landed on such a host again, although the catalog showed no stock for that GPU: since 0.6.0, `up` never creates a pod without stock and checks the host after creation (see How `up` picks the machine, below).

The log is written to `/workspace/ComfyUI/input/ainvfx/bootstrap.log` (also reachable as `/workspace/ainvfx-bootstrap.log`). Because it lives in ComfyUI's input folder, the pod serves it through its own proxy once ComfyUI answers (so anyone with the pod's address can read it, the JupyterLab token of its last line included: keep the address to yourself), and `pod.py` reads it there as well as through the API stream: `up` and `logs` catch `READY` either way.

## Your own settings: settings.env

Every choice the pod makes is one line of [`settings.env`](settings.env), next to the script. Edit the file, then `python pod.py up`: the values travel with the pod as environment variables, and `bootstrap.sh` reads them. An empty value means the default; `python pod.py doctor` shows what will be sent. From the browser, the same variables go in the Environment variables section of the template's deploy page. Deploy without touching anything and you get the course's pod.

**`AINVFX_PROFILE`** · which set of `models.json` to download. Default `image`.
`image` (63 GB) · `video` (adds LTX 2.5, 150 GB) · `train` (adds the training models, 215 GB). `up video` and `up train` set it for you.

**`AINVFX_COMFY_TAG`** · the ComfyUI release. Default `v0.39.0`.
A git tag (`v0.39.0`) · a branch (`master` for the latest commit).

**`AINVFX_PYTHON`** · the Python of the pod's environment, installed by uv. Default `3.13`.
`3.12` · `3.13` · `3.14`.

**`AINVFX_TORCH`** · the PyTorch packages, with pip flags if wanted. Default `torch torchvision torchaudio`.
`torch torchvision torchaudio` (stable) · `--pre torch torchvision torchaudio` (nightlies) · a pinned `torch==2.9.1 torchvision==0.24.1`.

**`AINVFX_TORCH_INDEX`** · the wheel index PyTorch comes from. Default: the cu130 stable index.
`https://download.pytorch.org/whl/cu130` (stable, CUDA 13.0) · `https://download.pytorch.org/whl/nightly/cu130` (nightlies) · the `cu128` index with a 12.8 driver.

**`AINVFX_MODELS_URL`** · where the list of models comes from. Default: this repository's `models.json` on `main`.
The raw URL of a `models.json` shaped like this one: fork, edit, point here.

**`AINVFX_CUSTOM_NODES`** · custom nodes installed at start. Default: none.
Git URLs separated by spaces, for example `https://github.com/kijai/ComfyUI-KJNodes https://github.com/rgthree/rgthree-comfy`. Each is cloned into `custom_nodes` and its `requirements.txt` installed.

**`AINVFX_HEALTHCHECK`** · the disk measurement at start. Default `1`.
`1` measures and warns when slow · `0` skips it.

**`AINVFX_SELFTEST`** · the test image at the end. Default `1`.
`1` renders one Z-Image Turbo image, which proves the GPU, the kernels and the models · `0` skips it (`up --no-selftest` does the same).

**`AINVFX_BOOTSTRAP_URL`** · a fork's own `bootstrap.sh`, run in place of this repository's. Default: this repository's on `main`.
The raw URL of the script.

**`HF_TOKEN`** · the Hugging Face token for the gated files (LTX). Never written in this file.
`setup --hf-token` stores it as the Runpod secret `huggingface_token`; `up` sends the reference `{{ RUNPOD_SECRET_huggingface_token }}`. Without it the gated files are skipped and the log says so; the image models need none.

Values with spaces need no quotes (`AINVFX_TORCH=--pre torch torchvision torchaudio` is read whole); lines starting with `#` are comments.

## The Hugging Face token

The gated repositories (LTX 2.5) need a Hugging Face read token: a free Hugging Face account, the licence accepted once on the model's page ("Agree and access repository"), then a token of type Read from Settings, Access Tokens. The token never travels in a slide, a chat or a file:

- `python pod.py setup --hf-token` stores it on Runpod as the secret `huggingface_token` (or do it in the console: Account, Credentials, Secrets, same name).
- Pods receive it as the variable `HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}`, which Runpod replaces with the value when the pod boots.
- Without a secret of that name, the pod receives the placeholder unchanged, notices it, ignores it and skips the gated files. The log says so, with the page of each file. Nothing else breaks.
- The pod checks the token once and names its Hugging Face account in the log. A token that Hugging Face refuses is dropped, with a `WARNING`.
- Each gated repository has its own licence: the LTX 2.5 models, then each LTX IC-LoRA. A licence not accepted yet stops nothing else: the end of the log lists the page of each file waiting (`ACTION NEEDED`, above).

## Files in and out

- **Workflow files** (`.json`): drag and drop them onto the ComfyUI canvas, as at home.
- **Images and videos in**: the upload button of a Load Image or Load Video node, or `python pod.py push file1 file2`.
- **Results out**: the Assets panel in ComfyUI's sidebar, or `python pod.py pull`, which downloads every output listed in the pod's history that you do not have yet, into `outputs/<pod name>/` next to the script (or the folder named in `AINVFX_OUTPUTS`).
- **Many files at once**: JupyterLab, from the link on the `READY` line (its token is in it) or from the pod's Connect button. Its terminal is under File, New, Terminal.

## How `up` picks the machine

The script reads Runpod's catalog once per cloud: every GPU type with its stock on hosts with CUDA 13.0 or newer. It orders the data centers by distance to you: your country first (Runpod names its data centers by country: `CA-MTL-1`, `US-TX-3`, `EU-FR-1`, `EUR-IS-2`), then the rest of your region (Europe or North America), then the other region. Country and region are guessed from this computer's clock (`America/Toronto` gives Canada; `Europe/Paris` gives France) and changed with `setup --region EU --country FR`. `up` prints where the profile's GPU is in stock, in your region and elsewhere, and its hourly price.

`up` creates a pod only where the catalog shows stock, and always names the Secure Cloud data center. With no data center named, Runpod picks any host, and on 5 and 7 October 2026 that host was one being removed from the platform. When the profile's GPU has no stock on Secure Cloud, `up` lists every GPU in stock that fits the profile (32 GB of VRAM or more for `image`, 96 GB for `video` and `train`, a Blackwell chip or newer), cheapest first. It marks the one it will create and prints the command to take another one, for example `python pod.py up image --gpu "RTX PRO 4500 SE"` (the name from the GPU column, or the full id). It never takes a GPU above 2.50 USD per hour on its own (`--max-price` changes that). If a GPU is taken between the catalog read and the creation, `up` tries the next one that fits, cheapest first. Community Cloud stock comes with no data center in the catalog, so a Community pod is the one case where Runpod picks the host.

Right after each creation, `up` reads the pod's host through Runpod's GraphQL API (the REST API does not report it). A host with maintenance under way, or starting within 24 hours, is terminated at once (a few seconds billed), and the next GPU is tried. `--secure-only` leaves out Community Cloud; `--gpu` tries only the GPU you name.

By default `up` creates the pod from the AInVFX template (`4i789znkrd`); `setup --template <id>` points it at another. Without a template the script describes the pod itself: the same image, ports 8188 and 8888 over HTTP and 22 over TCP, the same start command.

**The image** is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, the image of Runpod's own "Runpod Pytorch 2.8.0" template. Runpod keeps the images of its own templates on its hosts, so the container starts within seconds; any other tag has to be fetched from Docker Hub by the host first (the `1.0.7-cu1300-torch291` tag tried on 5 October 2026: 19 layers, about 9 GB, over 5 minutes before we gave up). The image's own PyTorch and CUDA toolkit are not used: `bootstrap.sh` builds the pod's environment with the cu130 PyTorch wheels, which carry their own CUDA 13 libraries, and the GPU driver is the host's. The toolkit of the image only matters to a custom node that compiles a CUDA kernel from source; none of the course's does, and if one did, `uv pip install nvidia-cuda-nvcc==13.0.88` in the venv provides the CUDA 13 compiler.

## Costs, as read on 4 October 2026

RTX 5090: 0.99 USD per hour on Secure Cloud, 0.69 on Community. RTX PRO 6000 (96 GB): 2.09 and 1.69. The container disk costs 0.10 USD per GB per month while the pod runs, nothing after termination. Billing is by the second, from creation to termination. Current prices: [Runpod pricing](https://docs.runpod.io/pods/pricing).

## Security notes

- The proxy address is public: the pod id is its only protection, and ComfyUI has no login. Do not share the address; terminate the pod when you are done.
- Community Cloud pods run on machines owned by third parties. Fine for public material; keep client material on Secure Cloud, or at home.
- The API key is written to one file with restricted permissions (`chmod 600`, or an `icacls` grant to your account on Windows). `RUNPOD_API_KEY` in the environment takes precedence, which lets you avoid the file entirely.
- Nothing survives termination. That is the design: your work travels with you.

## Updating

```bash
git pull
```

The ComfyUI tag and the other choices live in `settings.env` (and, for browser users, in the template's variables); the model list in `models.json`. A pod created after a change uses the new values; a running pod keeps its own.

## For contributors

`python -m unittest discover tests` (or `pytest`) runs the tests in `tests/test_pod.py`: unit tests of the helpers, then every command end to end against a fake Runpod API and a fake ComfyUI started on your own machine. Nothing is billed, your configuration is untouched, and the log stream's reconnect behaviour that once hid `READY` is reproduced. Standard library only, in the script and in the tests; keep it that way.

## Credits and licence

Maintained by [AInVFX](https://www.ainvfx.com/) for its [Generative AI Bootcamp for Film and TV](https://www.ainvfx.com/bootcamp/) and open to everyone, in partnership with [Runpod](https://www.runpod.io/), the bootcamp's GPU infrastructure partner. Not affiliated with [Comfy Org](https://www.comfy.org/): [ComfyUI](https://github.com/Comfy-Org/ComfyUI) is theirs, under the [GPL-3.0 licence](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE). The models come from their authors on [Hugging Face](https://huggingface.co/), each under its own licence (see the model page named in [`models.json`](models.json)); the LTX models are gated and need a Hugging Face token. The Runpod logo belongs to [Runpod](https://www.runpod.io/) and the AInVFX logo to [AInVFX](https://www.ainvfx.com/). This code is released under the [Apache License 2.0](LICENSE). Questions and bugs: [the issues page](https://github.com/AInVFX/ainvfx-runpod/issues), or [the AInVFX Discord](https://discord.gg/6CKSa3b5ks).
