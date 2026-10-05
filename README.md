<p align="center">
  <a href="https://www.ainvfx.com/"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://www.ainvfx.com/assets/images/logo.webp>
    <img src="https://www.ainvfx.com/assets/images/logo.webp" alt="AInVFX" height="40">
  </picture></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.runpod.io/"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047dba6113c1d43463ddd_logo-white.svg">
    <img src="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047dba6113c1d43463ddd_logo-white.svg" alt="Runpod" height="40">
  </picture></a>
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
2. Pick the GPU: **RTX 5090** for image work. If it is out of stock, switch to Community Cloud or pick a slice of the RTX PRO 6000.
3. Click **Deploy On-Demand**. The pod appears under Pods; its **Logs** button shows the install running.
4. Wait for the line `READY`, about 15 minutes. Open `https://<pod id>-8188.proxy.runpod.net` (also behind the pod's **Connect** button): that is your ComfyUI.
5. When you are done: download your results (the Assets panel in ComfyUI), then **Terminate** the pod. Stop is not enough: a stopped pod keeps a dead entry and may keep billing storage.

## Quick start, script

```bash
git clone https://github.com/AInVFX/ainvfx-runpod
cd ainvfx-runpod
python pod.py setup        # once: your Runpod API key, your region, your SSH key (optional)
python pod.py up image     # creates the pod, follows its log until READY
python pod.py open         # opens ComfyUI in your browser
python pod.py pull         # downloads the pod's outputs into outputs/<pod name>/
python pod.py down         # pulls, then terminates (billing stops)
```

The other commands: `logs` (follow the log again), `status` (GPU, cost so far, address, last log lines), `push file.png` (copy a file into the pod's input folder), `ssh` (a terminal on the pod), `list` (every pod of your account), `doctor` (check your setup). `python pod.py --help` lists them all.

Profiles: `up image` rents an RTX 5090 with a 100 GB disk; `up video` an RTX PRO 6000 (96 GB) with a 200 GB disk, for LTX 2.5; `up train` the same card with 250 GB, for LoRA training.

**What you need on your machine**

- **Python 3.8 or newer.** Windows: [python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.12`. macOS and Linux usually have it: `python3 --version`. Or install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run `uv run pod.py ...`, which fetches a Python for you.
- **Git**, to clone and update this repository. Without Git, GitHub's "Download ZIP" button works too.
- **A Runpod API key**: in the console, Settings, API Keys, "Create API Key", permission *Restricted* with Read and Write on Pods (add Secrets if you use `setup --hf-token`, and SSH Keys if you use `pod.py ssh`). The script keeps the key in `~/.ainvfx-runpod/config.json`, readable by your account only. Never paste it in a chat or a slide.
- **`ssh` is optional.** Only `pod.py ssh` uses it. Everything else goes through the browser and Runpod's proxy.

The script uses Python's standard library only: nothing to install.

## What the pod does when it starts

The template's start command runs [`bootstrap.sh`](bootstrap.sh) from this repository. Every line it writes starts with `[AINVFX]`; read them in the pod's log (Runpod console, `python pod.py logs` or `python pod.py status`).

| Step | What happens | The log says |
|---|---|---|
| 0 | SSH and JupyterLab start in the background, before anything else | JupyterLab answers within two minutes |
| 1 | **Health check**: the GPU and its driver (580 or newer, so the CUDA 13 kernels of the int8 models run), disk speed, download speed from Hugging Face on one real file | a `WARNING` in capitals when a value is bad: terminate and create again, usually in another data center |
| 2 | **Install**: uv, Python 3.13, PyTorch for CUDA 13.0, ComfyUI at tag `v0.38.2`, the Manager. The same lines as a manual install on your own machine | `ComfyUI v0.38.2` |
| 3 | **Start**: ComfyUI listens for the proxy | `COMFYUI UP` with the address, then `PROXY OK` |
| 4 | **Models**: the files of the profile, from [`models.json`](models.json), one by one, each with its time and speed; files already present are skipped | `MODELS DONE 14/14` |
| 5 | **Self-test**: one Z-Image Turbo image, 1024 x 1024 in 8 steps, saved in `output/` | `SELFTEST OK` with the time, then `READY` |

If GitHub cannot be reached, the start command falls back to Runpod's own `/start.sh`: the pod still boots with SSH and JupyterLab, and the error can be read.

Profile sizes (Hugging Face, 4 October 2026): `image` 14 files, about 63 GB; `video` 36 files, about 150 GB; `train` 38 files, about 215 GB.

Measured on the first real pod (RTX 5090, Secure Cloud, EUR-IS-2, 4 October 2026): PyTorch installed in 90 seconds, ComfyUI answering after 4 minutes, the 63 GB of the `image` profile in about 10 minutes (about 100 MB/s), the self-test in 106 seconds on the first load and 2 seconds on the second, `READY` about 15 minutes after creation.

## The Hugging Face token

The gated repositories (LTX 2.5) need a Hugging Face read token. The token never travels in a slide, a chat or a file:

- `python pod.py setup --hf-token` stores it on Runpod as the secret `huggingface_token` (or do it in the console: Settings, Secrets, same name).
- Pods receive it as the variable `HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}`, which Runpod replaces with the value when the pod boots.
- Without a secret of that name, the pod receives the placeholder unchanged, notices it, ignores it and skips the gated files. The log says so. Nothing else breaks.

## Files in and out

- **Workflow files** (`.json`): drag and drop them onto the ComfyUI canvas, as at home.
- **Images and videos in**: the upload button of a Load Image or Load Video node, or `python pod.py push file1 file2`.
- **Results out**: the Assets panel in ComfyUI's sidebar, or `python pod.py pull`, which downloads every output listed in the pod's history that you do not have yet, into `outputs/<pod name>/`.
- **Many files at once**: JupyterLab at `https://<pod id>-8888.proxy.runpod.net` (the token is under Connect in the console).

## How `up` picks the machine

The script asks Runpod's catalog where the profile's GPU is in stock on hosts with CUDA 13.0 or newer, puts your region's data centers first (EU or NA, guessed from your clock, changed with `setup --region`), prints the hourly price, asks you once, then tries Secure Cloud data center by data center, then Secure Cloud anywhere, then Community Cloud. If the profile's first GPU is out everywhere, it moves to the next one in the list (for `image`: RTX 5090, then a 48 GB slice of the RTX PRO 6000, then the full card). `--secure-only` keeps it on Runpod's own machines; `--gpu` names a card yourself.

By default `up` creates the pod from the AInVFX template (`4i789znkrd`); `setup --template <id>` points it at another. Without a template the script describes the pod itself: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, ports 8188 and 8888 over HTTP and 22 over TCP, the same start command.

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

The ComfyUI tag lives in `pod.py` (`COMFY_TAG`) and in the template; the model list in `models.json`. A pod created after a change uses the new values; a running pod keeps its own.

## Credits and licence

Maintained by AInVFX for its bootcamp and open to everyone, in partnership with Runpod. Not affiliated with Comfy Org; ComfyUI, the models and their licences belong to their authors (see each repository on Hugging Face). This code is released under the Apache License 2.0.
