<p align="center">
  <a href="https://www.ainvfx.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.ainvfx.com/assets/images/logo.webp"><img src="https://www.ainvfx.com/assets/images/logo-light.png" alt="AInVFX" height="40"></picture></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.runpod.io/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047dba6113c1d43463ddd_logo-white.svg"><img src="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047c6e55021705127c8da_runpod-logo-black.svg" alt="Runpod" height="40"></picture></a>
</p>

# ainvfx-runpod

Rent a GPU on [Runpod](https://www.runpod.io/), get [ComfyUI](https://github.com/Comfy-Org/ComfyUI) running on it with the right models, work, download your results, terminate. From the browser or from one script, on Windows, macOS or Linux.

Built by [AInVFX](https://www.ainvfx.com/) for its [Generative AI Bootcamp for Film and TV](https://www.ainvfx.com/bootcamp/), with Runpod as the bootcamp's GPU infrastructure partner. Anyone may use it: the ComfyUI version and the list of models are plain files you can change.

## How it works

A **pod** is a Linux machine with a GPU, rented by the second. You create it from the AInVFX template. On its own, the pod installs ComfyUI and the models of the **profile** you chose (a set of models for one block of the course). You open ComfyUI in your browser at `https://<pod id>-8188.proxy.runpod.net`. At the end, you download your results and terminate the pod. Nothing stays on Runpod, so you pay only for the hours you work.

There are two ways to create a pod. Both give the same machine.

| | You need | You do |
|---|---|---|
| **Browser** | a Runpod account with credit | open the template's deploy page, pick a GPU, deploy, read the log until `READY` |
| **Script** | the same, plus Python 3.8 or newer and Git | `python pod.py up image`, then `open`, `pull`, `down` |

## Quick start, browser

1. Open [the AInVFX template](https://console.runpod.io/deploy?template=4i789znkrd&ref=iad0yzht) (template id `4i789znkrd`).
2. Pick the GPU: **RTX 5090** for the `image` and `3d` profiles, **RTX PRO 6000** (96 GB) for `video` and `train`. If it is out of stock, pick another GPU with as much memory. Community Cloud (machines owned by third parties) is cheaper and often in stock, but Runpod picks the host and it can fail to start.
3. For another profile, set the `AINVFX_PROFILE` environment variable (for example `image+3d`). The template's 200 GB **Container Disk** holds any single profile; for a combination, raise it to the size given in [Profiles](#profiles).
4. Click **Deploy On-Demand**. The pod appears under Pods; its **Logs** button shows the install.
5. Wait for the line `READY`, 5 to 15 minutes after creation depending on the data center. Then open `https://<pod id>-8188.proxy.runpod.net`, also behind the pod's **Connect** button: that is your ComfyUI. The `READY` line also gives the JupyterLab link, with its token.
6. When you are done, download your results (the Assets panel in ComfyUI), then **Terminate** the pod. Stop is not enough: a stopped pod keeps an entry and can keep billing storage.

## Quick start, script

```bash
git clone https://github.com/AInVFX/ainvfx-runpod
cd ainvfx-runpod
python pod.py setup              # once: your Runpod API key, nearest data centers, SSH key (optional)
python pod.py setup --hf-token   # optional: your Hugging Face token, for the gated models
python pod.py up image           # creates the pod, follows its log until READY
python pod.py open               # opens ComfyUI in your browser
python pod.py pull               # downloads the pod's output folder into outputs/<pod name>/
python pod.py down               # pulls, then terminates: billing stops
```

The script uses Python's standard library only: nothing to install. On your machine you need:

- **Python 3.8 or newer.** Windows: [python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.12`. macOS and Linux usually have it (`python3 --version`). With [uv](https://docs.astral.sh/uv/getting-started/installation/), `uv run pod.py ...` fetches a Python for you.
- **Git**, to clone and update this repository. GitHub's "Download ZIP" button works too.
- **A Runpod API key.** In the console: Account, Credentials, API Keys, Create API Key. Choose *Restricted*, set **api.runpod.io/graphql** to **Read / Write** (pods, catalog, secrets, SSH keys) and leave **api.runpod.ai** on **None** (Serverless, not used). `setup` stores the key in `~/.ainvfx-runpod/config.json`, readable by your account only. Never paste it in a chat or a slide.
- **`ssh`**, optional: only `pod.py ssh` uses it.

## Commands

| Command | What it does |
|---|---|
| `setup` | Stores the API key, guesses your nearest data centers from your clock, registers your SSH key. `--hf-token` stores your Hugging Face token; `--region EU --country FR` sets the data centers to try first |
| `up <profiles>` | Creates a pod and follows its log until `READY` (below: [How `up` picks a GPU](#how-up-picks-a-gpu)) |
| `status [pod]` | GPU, data center, cost so far, credit left, addresses, last log lines |
| `logs [pod]` | Follows the log again until `READY` |
| `open [pod]` | Opens ComfyUI in your browser |
| `push [pod] FILES` | Copies files into the pod's input folder |
| `pull [pod]` | Downloads every new file of the pod's output folder into `outputs/<pod name>/` |
| `down [pod]` | Pulls, terminates, then says what the pod cost and the credit left. `--all` for every pod recorded here, `--no-pull` to skip the download |
| `ssh [pod]` | A terminal on the pod |
| `list` | Every pod of your account, and which ones this machine knows |
| `doctor` | Checks Python, ssh, the API key and the settings |

`[pod]` is the profile, the name given to `up --name`, the pod's name or its id. You can leave it out when one pod is recorded. Ctrl+C while a log is followed stops the following only: the pod keeps running until `down`. `python pod.py --help` lists every option.

**Several pods at once.** `up video+train --name jar-lora` creates a second pod, known here as `jar-lora`; every command then takes that name. A pod created from the console or another machine is attached by its first command (`python pod.py status <name or id>`). A pod terminated elsewhere is forgotten by the next `list`, or by the next command given without a name.

## Profiles

Each profile is a set of models in [`models.json`](models.json). Join them with `+` (`up image+3d`), or ask for `all`. A profile downloads only its own files: `video` does not include `image`.

| Profile | Contents | GPU `up` rents | Models | Container disk |
|---|---|---|---|---|
| `image` | Z-Image, Qwen-Image-Edit, ControlNet, SAM 3.1, SeedVR2, the Qwen-Image-2512 panorama set, FLUX 3 Image Klein, MOSS-Image | RTX 5090 (32 GB) | 117 GB | 170 GB |
| `3d` | TRELLIS.2, Pixal3D, TripoSplat, SAM 3D Body, MoGe, BiRefNet | RTX 5090 | 32 GB | 100 GB |
| `video` | LTX 2.5, the LTX IC-LoRAs (ainvfx-fluid and LTX Ripple included), SeedVR2 7B, RIFE | RTX PRO 6000 (96 GB) | 85 GB | 130 GB |
| `train` | the bf16 LTX 2.5 files of the training kit | RTX PRO 6000 | 68 GB | 110 GB |
| `image+3d` | | RTX 5090 | 148 GB | 210 GB |
| `image+video` | | RTX PRO 6000 | 202 GB | 270 GB |
| `all` | | RTX PRO 6000 | 301 GB | 380 GB |

Sizes read from `models.json` on 21 October 2026. A combination rents the GPU of its largest profile. The disk is 115% of the models plus 30 GB for the environment, rounded up to 10 GB, never under 100 GB. `up` computes it from the list the pod will download, yours included, and prints it before it creates anything; `--disk` overrides it.

## Your own settings

Every choice the pod makes is one line of [`settings.env`](settings.env). The values travel with the pod as environment variables, and `bootstrap.sh` reads them on the pod. From the browser, set the same variables in the Environment variables section of the deploy page. Change nothing and you get the course's pod.

Three ways to change a value, from the most lasting to the most temporary:

1. **`settings.local.env`**, a file you create next to `settings.env`, with the same `KEY=value` lines. Each key there replaces the same key of `settings.env`. Git ignores it, so `git pull` never touches it. Do not edit `settings.env` itself: `git pull` would stop on your change.
2. **`up --set AINVFX_KEY=VALUE`**, repeatable, for one pod: `python pod.py up image --set AINVFX_COMFY_TAG=master`.
3. **`up --models-url URL`**, a shortcut for `--set AINVFX_MODELS_URL=URL`.

`up` prints every value that differs from `settings.env` before it creates the pod, and `doctor` shows them all.

| Variable | Default | What it sets |
|---|---|---|
| `AINVFX_PROFILE` | `image` | The profiles to download. `up <profiles>` sets it for you |
| `AINVFX_COMFY_TAG` | `v0.42.1` | The ComfyUI release: a git tag, or a branch (`master` for the latest commit) |
| `AINVFX_PYTHON` | `3.13` | The Python of the pod's environment, installed by uv |
| `AINVFX_TORCH` | `torch torchvision torchaudio` | The PyTorch packages, with pip flags if wanted (`--pre ...` for nightlies) |
| `AINVFX_TORCH_INDEX` | the CUDA 13.0 stable index | The wheel index PyTorch comes from (`.../whl/nightly/cu130` for nightlies) |
| `AINVFX_MODELS_URL` | this repository's `models.json` | Your own list of models (below) |
| `AINVFX_CUSTOM_NODES` | none | Git URLs separated by spaces; each is cloned into `custom_nodes` and its `requirements.txt` installed |
| `AINVFX_HEALTHCHECK` | `1` | `0` skips the disk speed test at start |
| `AINVFX_SELFTEST` | `1` | `0` skips the test image at the end (`up --no-selftest` does the same). It needs the `image` profile |
| `AINVFX_BOOTSTRAP_URL` | this repository's | The raw URL of a fork's own `bootstrap.sh`, run in place of this one |

Values with spaces need no quotes; lines starting with `#` are comments.

**Your own list of models.** Copy [`models.json`](models.json) and keep its shape. In `profiles`, each profile names its entries (`@image` inside a profile means "everything of image, plus"). In `files`, each entry gives the Hugging Face `repo`, the `path` of the file in it, the `dir` under `ComfyUI/models` where it goes, its size in `gb`, and `gated: true` when the repository needs a token. Put the file where the pod can download it: a fork of this repository, a GitHub Gist (its Raw button gives the URL), or a Google Drive, Dropbox or GitHub share link open to anyone with the link. Then set `AINVFX_MODELS_URL` in `settings.local.env`, or give `--models-url` for one pod. `up` reads your list before it creates the pod, and stops if the link gives a web page instead of the file. For a file needed once, the pod's terminal is enough: `cd /workspace/ComfyUI/models/<folder> && hf download <repo> <path> --local-dir .`

## The Hugging Face token

Some repositories are **gated**: Hugging Face serves their files only to an account that has accepted their licence. Here: FLUX 3 Image Klein in `image`, LTX 2.5 in `video` and `train` (`gated: true` in `models.json`). They need a free Hugging Face account, the licence accepted on each model's page ("Agree and access repository"), and a token of type Read (Settings, Access Tokens).

- `python pod.py setup --hf-token` stores the token on Runpod as the secret `huggingface_token`, never on your machine. From the console: Account, Credentials, Secrets, same name.
- Each pod receives `HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}`, which Runpod replaces with the value when the pod boots. Without that secret, the pod ignores the placeholder and skips the gated files. Nothing else breaks.
- The log names the Hugging Face account of the token, lists each gated repository with how its licence is approved (at once, or by its authors), and ends with the page of every file still waiting.

## Files in and out

- **Workflows** (`.json`): drag them onto the ComfyUI canvas, as at home.
- **Images and videos in**: the upload button of a Load Image or Load Video node, or `python pod.py push file1 file2`.
- **Results out**: the Assets panel in ComfyUI, or `python pod.py pull`. It downloads every file of the pod's output folder that you do not have yet, subfolders kept: images, videos, 3D files, and files a node saves without showing them, such as a trained LoRA. It walks the folder through the pod's JupyterLab, with the token of the `READY` line. When JupyterLab is off, it falls back to the files ComfyUI's history lists. Results go to `outputs/<pod name>/` next to the script, or to the folder named in `AINVFX_OUTPUTS`. `down` runs the same pull first.
- **Many files at once**: JupyterLab, from the link on the `READY` line. Its terminal is under File, New, Terminal.

## How `up` picks a GPU

1. It reads Runpod's catalog: every GPU type in stock on hosts with CUDA 13.0 or newer (driver 580 or newer, which the course's int8 models need).
2. It tries the profile's GPU on **Secure Cloud** (Runpod's own data centers), in the nearest data center first: your country, then your region, then the other region. Your country comes from your computer's clock (`America/Toronto` gives Canada); `setup --region EU --country FR` changes it.
3. When that GPU has no stock, it lists every GPU in stock that fits the profile, cheapest first, and proposes the cheapest. To fit, a GPU needs the profile's memory (32 GB for `image` and `3d`, 96 GB for `video` and `train`), a Blackwell chip or newer, and must not be a 1g MIG slice (the smallest share of a bigger GPU: one was seven times slower than an RTX 5090 in our tests). `up` never takes more than 2.50 USD per hour on its own (`--max-price` changes it).
4. It creates a pod only where the catalog shows stock, and always names the data center. Left to choose, Runpod has twice placed a pod on a host being removed from the platform.
5. Right after creation, it checks the host. A host with maintenance under way, or due within 24 hours, is terminated at once (a few seconds billed), and the next machine is tried.

Before its question, `up` shows your credit and how many hours it buys at the proposed price. `--gpu NAME` takes one GPU from the list, by its short name (`"RTX PRO 4500 SE"`) or full id; repeat it to accept several, tried in the order given (`--gpu B200 --gpu B300`). `--community` also considers Community Cloud.

**Waiting for a GPU.** With `--retry`, `up` waits instead of stopping when nothing that fits is free. It reads the catalog every 15 seconds (`AINVFX_RETRY_EVERY`) and creates the first GPU free, by the same rules: one pod, never more. It asks its question once, before the wait (`-y` skips it). Ctrl+C ends the wait, and nothing bills before a pod exists. One read every 15 seconds is 240 requests an hour, well under Runpod's limit of 3,600.

```bash
python pod.py up image+video --gpu B200 --gpu B300 --retry
```

## What the pod does when it starts

The template's start command runs [`bootstrap.sh`](bootstrap.sh) from this repository. Every line it writes starts with `[AINVFX]`. Before it, `up` shows Runpod's own system log as `[RUNPOD]` lines: the host fetching the image, then starting the container.

| Step | What happens | The log says |
|---|---|---|
| 0 | SSH and JupyterLab start in the background | JupyterLab answers within two minutes |
| 1 | Health check: the GPU, its driver (580 or newer), the disk speed | a `WARNING` in capitals when a value is bad |
| 2 | Install: uv, Python, PyTorch for CUDA 13.0, ComfyUI at the tag of `AINVFX_COMFY_TAG`, the Manager, your custom nodes. A failed download is retried; PyTorch then comes from PyPI, the same CUDA 13.0 build | `FAILED` when the pod cannot work on that machine |
| 3 | ComfyUI starts for Runpod's proxy | `COMFYUI UP`, then `PROXY OK` |
| 4 | The models of the profile, one by one. Each line gives file k of n, GB done, percent and time left. Files already present are skipped | `MODELS DONE` with the total time and speed |
| 5 | A test image by MOSS-Image, 1024 x 1024 in 25 steps: a thank-you poster with this repository's address over eight VFX stills. It needs the `image` profile; without it, the step is skipped with a note | `SELFTEST OK`, then `READY` with the ComfyUI and JupyterLab links |

The image is `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, the one of Runpod's own "Runpod Pytorch 2.8.0" template. Runpod keeps it on its hosts, so the container starts within seconds. Its own PyTorch and CUDA toolkit are not used: `bootstrap.sh` installs PyTorch built for CUDA 13.0, which carries its own CUDA libraries.

The log is written to `/workspace/ComfyUI/input/ainvfx/bootstrap.log`. Because it sits in ComfyUI's input folder, the pod serves it through its proxy, and `pod.py` reads it there as well as through Runpod's log stream: that stream has been seen staying silent while the pod wrote its last lines.

## When something goes wrong

- **`FAILED` in the log.** The pod cannot work on that machine: Python, PyTorch or ComfyUI could not be installed, or PyTorch cannot see the GPU. The line says which. The pod stays up so the log can be read, and it bills. Terminate it and create another (`python pod.py down`, then `up`): it lands on another machine. `up` stops on that line.
- **`ACTION NEEDED` in the log.** Some gated files did not come. Logged in to Hugging Face with the account the log names, click "Agree and access repository" on each page listed. Then fetch them on the same pod: `bash /tmp/ainvfx-bootstrap.sh models`, in JupyterLab's terminal or after `python pod.py ssh`. A pod without a token needs one first (`setup --hf-token`), then a new pod.
- **No container after 8 minutes.** `up` prints a warning in capitals: the host is still fetching the image, or does not respond. The pod bills meanwhile. Run `python pod.py down`, then `up` again, usually in another data center.
- **A blank "403" page from the Connect button.** Press Enter in the address bar. ComfyUI refuses a page opened by a click from another site (its protection against sites that would drive it), never an address typed or opened by `pod.py open`.
- **GitHub unreachable at start.** The start command falls back to Runpod's own `/start.sh`: the pod still boots with SSH and JupyterLab, so the error can be read.
- **Any other error after the pod exists.** It is a message that names the pod, never a traceback. `python pod.py list` shows your pods, `down` terminates one. `AINVFX_DEBUG=1` prints the full trace for a bug report.

## Security

- ComfyUI has no login, and its proxy address is public: the pod id is its only protection. The log, served at that address, holds JupyterLab's token. Keep the address to yourself, and terminate the pod when you are done.
- Community Cloud runs on machines owned by third parties. Keep client material on Secure Cloud, or at home.
- The API key sits in one file readable by your account only (`chmod 600`, or an `icacls` grant on Windows). `RUNPOD_API_KEY` in the environment takes precedence, so you can avoid the file entirely.
- The Hugging Face token stays on Runpod as a secret; pods receive it at boot.
- Nothing survives termination. That is the design: your work travels with you.

## Updating

```bash
git pull
```

A pod created after the update uses the new values; a running pod keeps its own. Prices change: `up` always shows the live price, and [Runpod's pricing page](https://docs.runpod.io/pods/pricing) lists them all.

## For contributors

`python -m unittest discover tests` runs [`tests/test_pod.py`](tests/test_pod.py): unit tests of the helpers, then every command end to end against a fake Runpod API, a fake ComfyUI and a fake JupyterLab on your own machine. Nothing is billed and your configuration is untouched. Standard library only, in the script and in the tests: keep it that way.

## Credits and licence

Maintained by [AInVFX](https://www.ainvfx.com/) for its [Generative AI Bootcamp for Film and TV](https://www.ainvfx.com/bootcamp/) and open to everyone, in partnership with [Runpod](https://www.runpod.io/). Not affiliated with [Comfy Org](https://www.comfy.org/): [ComfyUI](https://github.com/Comfy-Org/ComfyUI) is theirs, under the [GPL-3.0 licence](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE). The models come from their authors on [Hugging Face](https://huggingface.co/), each under its own licence (see the repository named in [`models.json`](models.json)). The Runpod logo belongs to Runpod and the AInVFX logo to AInVFX. This code is released under the [Apache License 2.0](LICENSE). Questions and bugs: [the issues page](https://github.com/AInVFX/ainvfx-runpod/issues), or [the AInVFX Discord](https://discord.gg/6CKSa3b5ks).
