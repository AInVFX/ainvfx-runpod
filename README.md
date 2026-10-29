<p align="center">
  <a href="https://www.ainvfx.com/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://www.ainvfx.com/assets/images/logo.webp"><img src="https://www.ainvfx.com/assets/images/logo-light.png" alt="AInVFX" height="40"></picture></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.runpod.io/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047dba6113c1d43463ddd_logo-white.svg"><img src="https://cdn.prod.website-files.com/69ce570adca53340abab8376/69f047c6e55021705127c8da_runpod-logo-black.svg" alt="Runpod" height="40"></picture></a>
</p>

# ainvfx-runpod

Rent a GPU on [Runpod](https://www.runpod.io/), run [ComfyUI](https://github.com/Comfy-Org/ComfyUI) on it with the right models, train your own LoRAs, download your results, terminate. From the browser, or from one script on Windows, macOS or Linux. A **LoRA** is a small add-on file that teaches a model a style, a character, an object or an effect.

Built by [AInVFX](https://www.ainvfx.com/) for its [Generative AI Bootcamp for Film and TV](https://www.ainvfx.com/bootcamp/), with Runpod as the bootcamp's GPU infrastructure partner. Anyone may use it: the ComfyUI version, the models and the training recipes are plain files you can change.

## How it works

A **pod** is a Linux machine with a GPU, rented by the second. You create it from the AInVFX template. The pod then installs ComfyUI and the models of the **profile** you chose: the set of models for one block of the course. You open ComfyUI in your browser at `https://<pod id>-8188.proxy.runpod.net`, work, download your results and terminate the pod. Nothing stays on Runpod, so you pay only for the hours you work.

There are two ways to create a pod. Both give the same machine.

| | You need | You do |
|---|---|---|
| **Browser** | a Runpod account with credit | open the template's deploy page, pick a GPU, deploy, read the log until `READY` |
| **Script** | the same, plus Python 3.8 or newer and Git | `python pod.py up image`, then `open`, `pull`, `down` |

## Quick start, browser

1. Open [the AInVFX template](https://console.runpod.io/deploy?template=4i789znkrd&ref=iad0yzht).
2. Pick the GPU: **RTX 5090** for the `image` and `3d` profiles, **RTX PRO 6000** (96 GB) for `video` and `train`. If it is out of stock, pick another GPU with as much memory. Community Cloud (machines owned by third parties) is cheaper and often in stock, but its pods can fail to start.
3. For another profile, set `AINVFX_PROFILE` under Environment variables (for example `image+3d`). The template's 200 GB **Container Disk** holds any single profile; for a combination, raise it to the size given in [Profiles](#profiles).
4. Click **Deploy On-Demand**. The pod appears under Pods; its **Logs** button shows the install.
5. Wait for the line `READY`, 5 to 15 minutes after creation depending on the data center. Then open ComfyUI from the pod's **Connect** button, or at the address above. The `READY` line also gives the JupyterLab link.
6. When you are done, download your results (the Assets panel in ComfyUI), then **Terminate** the pod. Stop is not enough: a stopped pod can keep billing storage.

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

`pod.py` uses Python's standard library only, so there is nothing to install. On your computer you need:

- **Python 3.8 or newer.** Windows: [python.org](https://www.python.org/downloads/) or `winget install Python.Python.3.12`. macOS and Linux usually have it (`python3 --version`).
- **Git**, to clone and update this repository. GitHub's "Download ZIP" button works too.
- **A Runpod API key.** In the console: Account, Credentials, API Keys, Create API Key. Choose *Restricted*, set **api.runpod.io/graphql** to **Read / Write** and leave **api.runpod.ai** on **None**. `setup` stores the key in `~/.ainvfx-runpod/config.json`, readable by your account only. Never paste it in a chat or a slide.
- **`ssh`**, optional: only `pod.py ssh` uses it.

## Commands

| Command | What it does |
|---|---|
| `setup` | Stores the API key, guesses your nearest data centers from your clock, registers your SSH key. `--hf-token` stores your Hugging Face token; `--region EU --country FR` sets the data centers to try first |
| `up <profiles>` | Creates a pod and follows its log until `READY` ([How `up` picks a GPU](#how-up-picks-a-gpu)) |
| `status [pod]` | GPU, data center, cost so far, credit left, addresses, last log lines |
| `logs [pod]` | Follows the log again until `READY` |
| `open [pod]` | Opens ComfyUI in your browser |
| `push [pod] FILES` | Copies files into ComfyUI's input folder. `--dataset NAME` sends files and folders into `/workspace/dataset/NAME`, for training |
| `pull [pod]` | Downloads every new file of the pod's output folder into `outputs/<pod name>/` |
| `down [pod]` | Pulls, terminates, then gives the pod's cost and the credit left. `--all` for every pod recorded here, `--no-pull` to skip the download |
| `ssh [pod]` | A terminal on the pod |
| `list` | Every pod of your account, and which ones this computer knows |
| `doctor` | Checks Python, ssh, the API key and the settings |

`[pod]` is the profile, the name given to `up --name`, the pod's name or its id, matched exactly: `image` is the pod of `up image`, never the pod of `up image+3d`. You can leave it out when one pod is recorded. Ctrl+C while a log is followed stops the following only: the pod keeps running until `down`. `python pod.py --help` lists every option.

**Several pods at once.** `up video+train --name jar-lora` creates a second pod, known here as `jar-lora`. A pod created from the console or from another computer is attached by its first command (`python pod.py status <name or id>`).

## Profiles

A profile is a set of models in [`models.json`](models.json). Join several with `+` (`up image+3d`), or ask for `all`. A profile downloads only its own files: `video` does not include `image`.

| Profile | Contents | GPU `up` rents | Models | Container disk |
|---|---|---|---|---|
| `image` | Z-Image, Qwen-Image-Edit, ControlNet, SAM 3.1, BiRefNet, SeedVR2, the Qwen-Image-2512 panorama set, FLUX 3 Image Klein, MOSS-Image and its colour and depth control | RTX 5090 (32 GB) | 118 GB | 170 GB |
| `3d` | TRELLIS.2, Pixal3D, TripoSplat, SAM 3D Body, MoGe, BiRefNet | RTX 5090 | 32 GB | 100 GB |
| `video` | LTX 2.5 and its IC-LoRAs (LoRAs that follow a reference video: ainvfx-fluid and LTX Ripple among them), SeedVR2 7B, RIFE | RTX PRO 6000 (96 GB) | 88 GB | 140 GB |
| `train` | the two trainers ([Training a LoRA](#training-a-lora)) | RTX PRO 6000 | none at start, up to 70 GB per run | 190 GB |
| `image+3d` | | RTX 5090 | 148 GB | 210 GB |
| `image+video` | | RTX PRO 6000 | 206 GB | 270 GB |
| `image+train` | | RTX PRO 6000 | 118 GB, then up to 70 GB per run | 320 GB |
| `video+train` | | RTX PRO 6000 | 88 GB, then up to 68 GB per run | 280 GB |
| `all` | | RTX PRO 6000 | 236 GB, then up to 68 GB per run | 450 GB |

A combination rents the GPU of its largest profile. The disk is 115% of the models plus 30 GB for the environment, rounded up to 10 GB and never under 100 GB. With `train`, it counts the largest recipe's base files, plus 70 GB for the trainers, the caption model, caches and checkpoints. `up` computes the disk from the list the pod will download, yours included, and prints it before it creates anything; `--disk` overrides it.

## Your own settings

Every choice the pod makes is one line of [`settings.env`](settings.env). The values are sent to the pod as environment variables, and `bootstrap.sh` reads them there. From the browser, set the same variables in the Environment variables section of the deploy page. Change nothing and you get the course's pod.

Three ways to change a value, from the most lasting to the most temporary:

1. **`settings.local.env`**, a file you create next to `settings.env`, with the same `KEY=value` lines. Each key there replaces the same key of `settings.env`. Git ignores it, so `git pull` never touches it. Do not edit `settings.env` itself: `git pull` would stop on your change.
2. **`up --set AINVFX_KEY=VALUE`**, repeatable, for one pod: `python pod.py up image --set AINVFX_COMFY_TAG=master`.
3. **`up --models-url URL`**, a shortcut for `--set AINVFX_MODELS_URL=URL`.

`up` prints every value that differs from `settings.env` before it creates the pod, and `doctor` shows them all. Values with spaces need no quotes; lines starting with `#` are comments.

| Variable | Default | What it sets |
|---|---|---|
| `AINVFX_PROFILE` | `image` | The profiles to download. `up <profiles>` sets it for you |
| `AINVFX_COMFY_TAG` | `v0.43.0` | The ComfyUI release: a git tag, or a branch (`master` for the latest commit) |
| `AINVFX_PYTHON` | `3.13` | The Python of the pod's environment, installed by uv |
| `AINVFX_TORCH` | `torch torchvision torchaudio` | The PyTorch packages, with pip flags if wanted (`--pre ...` for nightlies) |
| `AINVFX_TORCH_INDEX` | `https://download.pytorch.org/whl/cu130` | The wheel index PyTorch comes from (`.../whl/nightly/cu130` for nightlies) |
| `AINVFX_MODELS_URL` | the `models.json` of `AINVFX_REPO_RAW` | Your own list of models (below) |
| `AINVFX_CUSTOM_NODES` | none | Git URLs separated by spaces; each is cloned into `custom_nodes` and its `requirements.txt` installed |
| `AINVFX_HEALTHCHECK` | `1` | `0` skips the disk speed test at start |
| `AINVFX_SELFTEST` | `1` | `0` skips the test image at the end (`up --no-selftest` does the same). It needs the `image` profile |
| `AINVFX_REPO_RAW` | the release path the template names | Repository release path: the raw GitHub URL of the release the pod runs, such as `.../ainvfx-runpod/v1.2.0`; a branch or a fork works too (below) |

**Your own list of models.** Copy [`models.json`](models.json) and keep its shape. In `profiles`, each profile names its entries (`@image` inside a profile means "everything of image, plus"). In `files`, each entry gives the Hugging Face `repo`, the `path` of the file in it, the `dir` under `ComfyUI/models` where it goes, its size in `gb`, and `gated: true` when the repository needs a token. Put the file where the pod can download it: a fork of this repository, a GitHub Gist (its Raw button gives the address), or a Google Drive, Dropbox or GitHub share link open to anyone with the link. Then set `AINVFX_MODELS_URL` in `settings.local.env`, or give `--models-url` for one pod. `up` reads your list before it creates the pod, and stops if the link gives a web page instead of the file. For a file needed once, the pod's terminal is enough: `cd /workspace/ComfyUI/models/<folder> && hf download <repo> <path> --local-dir .`

**The repository release path.** `AINVFX_REPO_RAW` is the raw GitHub URL of this repository at the release the pod runs, for example `https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/v1.2.0`. The template sets it, so you never need to. The pod takes `bootstrap.sh`, `models.json` and `train.py` from that path, so a change on `main` reaches no pod before the next release. A models list of your own needs only `AINVFX_MODELS_URL`. To run a branch or a fork, give its path: `--set AINVFX_REPO_RAW=https://raw.githubusercontent.com/<owner>/ainvfx-runpod/<branch>`.

## The Hugging Face token

Some repositories are **gated**: Hugging Face serves their files only to an account that has accepted their licence. They are marked `gated: true` in `models.json`: FLUX 3 Image Klein in `image`, and LTX 2.5 and Lightricks' IC-LoRAs in `video` and in the LTX training recipes. They need a free Hugging Face account, the licence accepted on each model's page ("Agree and access repository"), and a token of type Read (Settings, Access Tokens).

- `python pod.py setup --hf-token` stores the token on Runpod as the secret `huggingface_token`, never on your computer. From the console: Account, Credentials, Secrets, same name.
- Each pod receives `HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}`, which Runpod replaces with the value when the pod starts. Without that secret, the pod skips the gated files, and nothing else breaks.
- The log names the Hugging Face account of the token, lists each gated repository with how its licence is approved (at once, or by its authors), and ends with the page of every file still waiting.

## Files in and out

- **Workflows** (`.json`): drag them onto the ComfyUI canvas.
- **Images and videos in**: the upload button of a Load Image or Load Video node, or `python pod.py push file1 file2`.
- **A training dataset in**: `python pod.py push --dataset NAME FOLDER` ([Training a LoRA](#training-a-lora)).
- **Results out**: the Assets panel in ComfyUI, or `python pod.py pull`. It downloads every file of the pod's output folder that you do not have yet, subfolders kept: images, videos, 3D files, and files a node saves without showing them, such as a trained LoRA. Results go to `outputs/<pod name>/` next to the script, or to the folder named in the `AINVFX_OUTPUTS` environment variable. Without JupyterLab, `pull` gets only the files that ComfyUI's history lists. `down` runs the same pull first.
- **Many files at once**: JupyterLab, from the link of the `READY` line. Its terminal is under File, New, Terminal.

## Training a LoRA

The `train` profile makes the pod a training machine too. After ComfyUI, it installs two trainers, each at one tested commit and in its own environment: [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) in [AkaneTendo25's fork](https://github.com/AkaneTendo25/musubi-tuner), the one that trains LTX 2.5, and [sd-scripts](https://github.com/kohya-ss/sd-scripts) for MOSS-Image. Then it adds the `train` command to the pod. No base model comes at start: the first run of a recipe downloads that recipe's files, once.

| Recipe | It learns | Dataset | Base files | Steps · learning rate · rank |
|---|---|---|---|---|
| `zimage` | a style, a character or an object, on Z-Image Base | images | 21 GB | 1,500 · 1e-4 · 32 |
| `qwen_edit` | one edit, on Qwen-Image-Edit-2511 | image pairs | 58 GB | 1,250 · 5e-5 · 16 |
| `moss` | a style, a character or an object, on MOSS-Image Base | images | 20 GB | 1,500 · 1e-4 · 16 |
| `ltx` | a style or a motion, on LTX 2.5 | clips | 70 GB | 1,500 · 1e-4 · 32 |
| `ltx_images` | a look, on LTX 2.5, from still images | images | 70 GB | 1,500 · 1e-4 · 32 |
| `ltx_iclora` | a control video turned into a result (an IC-LoRA) | clip pairs | 70 GB | 2,000 · 1e-4 · 32 |
| `ltx_av` | the picture and the sound together, on LTX 2.5 | clips with sound | 70 GB | 1,500 · 1e-4 · 32 |

A dataset is one folder. **Images**: each image with its caption, a `.txt` file of the same name. **Clips**: each clip of 121 frames or more with its caption; the trainer reads its first 121 frames at 25 fps. **Pairs**: a `targets` folder (the results, with their captions) and a `control` folder (what the model is given: the image to edit, or the control clip), with the same file names.

### A first run

On your computer:

```bash
python pod.py up train                          # or image+train, to test the LoRA with the course's image models
python pod.py push --dataset mychar mychar/     # the folder's files go to /workspace/dataset/mychar on the pod
```

Then in a terminal on the pod (JupyterLab, from the link of the `READY` line: File, New, Terminal):

```bash
train caption mychar --trigger mychr            # writes a caption for every image that has none
train zimage mychar_v1 --dataset mychar         # trains in the background; its log follows on screen
```

- **Read every caption before training.** JupyterLab opens the `.txt` files under `workspace/dataset/mychar`. The caption model makes mistakes, and a wrong word is learned like the others. `train caption` ends with the words found in most captions: a word that describes the style goes in `--avoid`, then `train caption mychar --trigger mychr --avoid "watercolor,brush" --overwrite`.
- **During the run**, Ctrl+C stops the display, never the run. `train log` shows it again, `train status` says where it stands, `train stop` stops it. Run no workflow in ComfyUI meanwhile: the run needs the GPU's memory.
- **Every 250 steps**, a checkpoint and test images (samples) land in `ComfyUI/output/train/<run>/`, and the checkpoint appears in ComfyUI's LoRA list under `train/<run>/` (press r in ComfyUI to refresh it). Judge the samples, not the loss curve: stop when they stop improving.
- **At the end**, `python pod.py pull` or `down` brings the run home with the rest of the output folder. An LTX checkpoint weighs about 0.65 GB in each of its two formats.

### Options of a run

| Option | What it does |
|---|---|
| `--dataset NAME` | a folder of `/workspace/dataset`; or `--dataset FILE.toml`, a dataset config for several folders or other settings |
| `--steps N` | the number of training steps |
| `--lr X` | the learning rate: how fast the LoRA learns |
| `--rank N` | the LoRA's rank (its size); alpha changes by the same factor, so the learning speed stays the same |
| `--from RUN` | start from the final LoRA of an earlier run, or from one of its checkpoints by file name; with `--steps` |
| `--prompts FILE` | your own sample prompts (by default: two of your captions, then the recipe's own subject with your trigger) |
| `--trigger WORD` | the trigger word of the third sample prompt, when the captions do not start with it |
| `--then RUN [options]` | another stage after this one, in the same background run |

Each run writes the dataset config it used (`dataset.toml`) into its folder: copy it to change a setting. A chain of three stages, here the three resolutions of an IC-LoRA, each stage starting from the one before:

```bash
train ltx_iclora fx_s1 --dataset fx/low.toml \
  --then fx_s2 --from fx_s1 --steps 1000 --lr 5e-5 --dataset fx/mid.toml \
  --then fx --from fx_s2 --steps 500 --lr 2e-5 --dataset fx/high.toml
```

The musubi-tuner recipes repeat settings proven on RTX PRO 6000 pods. The `moss` recipe follows the MOSS-Image guide of sd-scripts and is not yet proven on a pod. FLUX 3 Image Klein has no recipe: neither trainer supports it yet.

## How `up` picks a GPU

1. It reads Runpod's catalog: every GPU type in stock on hosts with CUDA 13.0 or newer (driver 580 or newer, which the course's int8 models need).
2. It tries the profile's GPU on **Secure Cloud** (Runpod's own data centers), nearest first: your country, then your region, then the other region. Your country comes from your computer's clock (`America/Toronto` gives Canada); `setup --region EU --country FR` changes it.
3. When that GPU has no stock, it proposes the cheapest GPU in stock that fits the profile: the profile's memory (32 GB for `image` and `3d`, 96 GB for `video` and `train`), a Blackwell chip or newer, and never a 1g MIG slice (the smallest share of a larger GPU: one was seven times slower than an RTX 5090 in our tests). It never takes more than 2.50 USD per hour on its own (`--max-price` changes it).
4. It creates a pod only where the catalog shows stock, and always names the data center: left to choose, Runpod twice placed a pod on a host that was being removed.
5. Right after creation, it checks the host. A host with maintenance under way, or due within 24 hours, is terminated at once (a few seconds billed), and the next machine is tried.

Before it asks you to confirm, `up` shows your credit and how many hours it buys at the proposed price. `--gpu NAME` takes one GPU from the list, by its short name (`"RTX PRO 4500 SE"`) or full id; repeat it to accept several, tried in the order given. `--community` also considers Community Cloud.

**Waiting for a GPU.** With `--retry`, `up` waits instead of stopping when nothing that fits is free. It reads the catalog every 15 seconds (`AINVFX_RETRY_EVERY`) and creates the first fitting GPU that frees up: one pod, never more. Ctrl+C ends the wait, and nothing bills before a pod exists.

```bash
python pod.py up image+video --gpu B200 --gpu B300 --retry
```

## What the log says

The template's start command runs [`bootstrap.sh`](bootstrap.sh). Every line it writes starts with `[AINVFX]`. Before it, `up` shows Runpod's own system log as `[RUNPOD]` lines: the host fetching the image, then starting the container. These words mark the steps:

| The log says | It means |
|---|---|
| `COMFYUI UP`, then `PROXY OK` | ComfyUI runs and answers at its public address. The models still download |
| `MODELS DONE` | Every file of the profile is on the pod, with the total time and speed |
| `TRAINERS OK` | With `train`: the trainers and the `train` command are ready |
| `SELFTEST OK` | MOSS-Image made a test image: a thank-you poster with this repository's address. It needs the `image` profile |
| `READY` | The pod is ready. The line gives the ComfyUI and JupyterLab links |
| `WARNING` | Something is wrong or slow, such as an old driver or a slow disk. The line says what to do |
| `ACTION NEEDED` | Gated files did not come (below) |
| `FAILED` | The pod cannot work on that machine. It stays up so the log can be read, and it bills |

The log is also written to `/workspace/ComfyUI/input/ainvfx/bootstrap.log`, so the pod serves it at its ComfyUI address too. [ARCHITECTURE.md](ARCHITECTURE.md) describes every step.

## When something goes wrong

- **`FAILED` in the log.** The pod cannot work on that machine: Python, PyTorch or ComfyUI could not be installed, or PyTorch cannot see the GPU. The line says which. Terminate the pod and create another (`python pod.py down`, then `up`): it lands on another machine. `up` stops on that line.
- **`ACTION NEEDED` in the log.** Some gated files did not come. Logged in to Hugging Face with the account the log names, click "Agree and access repository" on each page listed. Then fetch them on the same pod: `bash /tmp/ainvfx-bootstrap.sh models`, in JupyterLab's terminal or after `python pod.py ssh`. A pod without a token needs one first (`setup --hf-token`), then a new pod.
- **The trainers did not install.** The log says why. `bash /tmp/ainvfx-bootstrap.sh trainer` tries again on the same pod.
- **No container after 8 minutes.** `up` prints a warning in capitals: the host is still fetching the image, or does not respond. The pod bills meanwhile. Run `python pod.py down`, then `up` again, usually in another data center.
- **A blank "403" page from the Connect button.** Press Enter in the address bar. ComfyUI refuses a page opened by a click from another site (a protection against sites that would control it), never an address typed or opened by `pod.py open`.
- **GitHub unreachable at start.** The start command falls back to Runpod's own `/start.sh`: the pod still starts SSH and JupyterLab, so you can read the error.
- **Any other error after the pod exists.** `pod.py` prints a message that names the pod, never a traceback. `python pod.py list` shows your pods, `down` terminates one. `AINVFX_DEBUG=1` prints the full trace for a bug report.

## Security

- ComfyUI has no login, and its proxy address is public: the pod id is its only protection. The log, served at that address, holds JupyterLab's token. Keep the address to yourself, and terminate the pod when you are done.
- Community Cloud runs on machines owned by third parties. Keep client material on Secure Cloud, or on your own computer.
- The API key sits in one file readable by your account only (`chmod 600`, or an `icacls` grant on Windows). `RUNPOD_API_KEY` in the environment takes precedence, so you can avoid the file entirely.
- The Hugging Face token stays on Runpod as a secret; pods receive it when they start.
- Nothing survives termination: `down` downloads your results first.

## Updating

```bash
git pull
```

A pod created after the update uses the new values; a running pod keeps its own. Prices change: `up` always shows the live price, and [Runpod's pricing page](https://docs.runpod.io/pods/pricing) lists them all.

## Contributing

[ARCHITECTURE.md](ARCHITECTURE.md) explains how the code is built, how to test it without a Runpod account (`python -m unittest discover -s tests`) and how a release is made. Bugs and questions: [the issues page](https://github.com/AInVFX/ainvfx-runpod/issues), or [the AInVFX Discord](https://discord.gg/6CKSa3b5ks).

## Credits and licence

Maintained by [AInVFX](https://www.ainvfx.com/) for its [Generative AI Bootcamp for Film and TV](https://www.ainvfx.com/bootcamp/) and open to everyone, in partnership with [Runpod](https://www.runpod.io/). Not affiliated with [Comfy Org](https://www.comfy.org/): [ComfyUI](https://github.com/Comfy-Org/ComfyUI) is theirs, under the [GPL-3.0 licence](https://github.com/Comfy-Org/ComfyUI/blob/master/LICENSE). The models come from their authors on [Hugging Face](https://huggingface.co/), each under its own licence (see the repository named in [`models.json`](models.json)). The trainers keep their own licences: musubi-tuner and sd-scripts are Apache 2.0. The Runpod logo belongs to Runpod and the AInVFX logo to AInVFX. This code is released under the [Apache License 2.0](LICENSE).
