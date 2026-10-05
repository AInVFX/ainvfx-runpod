# ainvfx-runpod

Create a GPU pod on [Runpod](https://www.runpod.io/), get ComfyUI running on it with the right models, work, download your results, terminate. One script, one command per step, from Windows, macOS or Linux.

Built by [AInVFX](https://www.ainvfx.com/) for the [Generative AI Bootcamp for Film and TV](https://www.ainvfx.com/bootcamp/), where Runpod is the course's GPU partner. Anyone may use it: the models and the ComfyUI version are the ones the bootcamp teaches, and both are listed in plain files you can change.

## What it does

A **pod** is a Linux machine with a GPU, rented by the second. The script creates one from a known recipe, the pod installs itself, and you reach ComfyUI from your browser through Runpod's HTTPS proxy. Nothing is kept on Runpod between sessions: you pull your results and terminate, so a session costs only the hours it ran.

```
python pod.py setup          once: your Runpod API key, your region, your SSH key (optional)
python pod.py up image       a pod for image work (RTX 5090, 100 GB disk)
python pod.py open           ComfyUI in your browser
python pod.py push a.png     a file into the pod's input folder
python pod.py pull           the pod's outputs into outputs/<pod name>/
python pod.py down           pull, then terminate (billing stops)
```

`up video` picks an RTX PRO 6000 (96 GB) and a 200 GB disk for LTX 2.5; `up train` the same card and 250 GB for LoRA training. `status`, `list`, `ssh` and `doctor` are described by `python pod.py --help`.

## What you need on your machine

- **Python 3.8 or newer.** Windows: [python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.12`; macOS and Linux usually have it (`python3 --version`). Or [uv](https://docs.astral.sh/uv/getting-started/installation/) and run `uv run pod.py ...`, which fetches a Python for you.
- **Git**, to clone and update this repository, or the "Download ZIP" button on GitHub.
- **A Runpod account** with credit, and an **API key**: console, Settings, API Keys, "Create API Key", permission *Restricted* with Read/Write on Pods. The key is stored in `~/.ainvfx-runpod/config.json`, readable by your account only. Never paste it in a chat or a slide.
- **`ssh` is optional.** It only serves `pod.py ssh`. Everything else works through the browser and the proxy, which is what makes a 14-year-old laptop a valid machine for this course.

No Python package is installed: the script uses the standard library only.

## What happens when a pod starts

The pod runs [`bootstrap.sh`](bootstrap.sh) as its start command. Every line it prints starts with `[AINVFX]`, in the pod's log (Runpod console, or `python pod.py status`).

1. **Health check.** GPU and driver (580 or newer, so the CUDA 13 kernels of the int8 models run), disk write and read speed, download speed from Hugging Face measured on a real file. Below a threshold the log shouts a `WARNING` in capitals: terminate and create again, usually in another data center.
2. **Install.** uv, Python 3.13, PyTorch stable for CUDA 13.0, ComfyUI at tag `v0.38.2`, the Manager. The same lines as a manual install on your own machine.
3. **Start.** ComfyUI listens on port 8188; the log says `COMFYUI UP` with the address `https://<pod id>-8188.proxy.runpod.net`.
4. **Models.** The files of the profile, from [`models.json`](models.json), one by one, resumable; already-present files are skipped. Gated repositories (LTX 2.5) need a Hugging Face read token in `HF_TOKEN`; without it they are skipped and the log says so.
5. **Hand over** to Runpod's own start script: SSH and JupyterLab (port 8888) as on any Runpod pod. The log ends with `READY`.

Profiles and sizes (Hugging Face, 4 October 2026): `image` 14 files, about 63 GB; `video` 36 files, about 150 GB; `train` 38 files, about 215 GB.

## How `up` chooses the machine

The script asks Runpod's catalog where the profile's GPU is in stock on hosts with CUDA 13.0 or newer, sorts the data centers by your region (EU or NA, guessed from your clock, changed with `setup --region`), prints the hourly price, asks once, then tries Secure Cloud data center by data center, then Secure Cloud anywhere, then Community Cloud. If the first GPU of the profile is out everywhere, it moves to the next one in the list (for `image`: RTX 5090, then a 48 GB slice of the RTX PRO 6000, then the full PRO 6000). `--secure-only` keeps it on Runpod's own machines; `--gpu` names a card yourself.

A pod can be created from a saved Runpod **template** (`setup --template <id>`) or from the script alone: image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, ports 8188 and 8888 over HTTP and 22 over TCP, the start command above. Both give the same pod.

## Files in and out

- **Workflow files** (`.json`) drag and drop onto the ComfyUI canvas from your machine, as always.
- **Images and videos in**: the upload button of a Load Image or Load Video node, or `python pod.py push file1 file2`.
- **Results out**: the Assets panel of ComfyUI's sidebar, or `python pod.py pull`, which downloads every output listed in the pod's history that you do not have yet, into `outputs/<pod name>/`.
- **Many files**: JupyterLab at `https://<pod id>-8888.proxy.runpod.net` (password under Connect in the console).

## Security notes

- The proxy address is public: the pod id is its only protection, and ComfyUI has no login. Do not share the address; terminate the pod when you are done.
- Community Cloud pods run on machines owned by third parties. Fine for public material; keep client material on Secure Cloud, or at home.
- The API key is written to one file with restricted permissions (`chmod 600`, or an `icacls` grant to your account on Windows). `RUNPOD_API_KEY` in the environment takes precedence, which lets you avoid the file entirely.
- Nothing survives termination. That is the design: your work travels with you.

## Costs, as read on 4 October 2026

RTX 5090: 0.99 USD per hour on Secure Cloud, 0.69 on Community. RTX PRO 6000 (96 GB): 2.09 and 1.69. The container disk costs 0.10 USD per GB per month while the pod runs, nothing after termination. Current prices: [Runpod pricing](https://docs.runpod.io/pods/pricing).

## Updating

```
git pull
```

The ComfyUI tag and the model list live in `pod.py` (`COMFY_TAG`) and `models.json`. A pod created after a change uses the new values; a running pod keeps its own.

## Credits and licence

Maintained by AInVFX for its bootcamp and open to everyone. Runpod is the bootcamp's GPU infrastructure partner. Not affiliated with Runpod or Comfy Org; ComfyUI, the models and their licences belong to their authors (see each repository on Hugging Face). This code is released under the Apache License 2.0.
