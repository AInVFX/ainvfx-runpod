# ainvfx-runpod: architecture

This document explains how ainvfx-runpod is built and where each part lives, for anyone who maintains the code. [README.md](README.md) is the user guide. A change to the code updates this file in the same commit.

## 1. What ainvfx-runpod does

ainvfx-runpod sets up ComfyUI and its models on a rented Runpod GPU, then terminates the pod when the work is done, so nothing stays on Runpod and nothing more is billed. Three programs share the work:

- **`pod.py`** runs on the user's computer. It creates the pod through Runpod's API, follows its log until the pod is ready, moves files in and out, and terminates the pod.
- **`bootstrap.sh`** runs on the pod, as the container's start command. It installs ComfyUI, downloads the models, installs the trainers and renders a test image.
- **`train.py`** runs on the pod, as the `train` command. It trains LoRAs in the background, with musubi-tuner or sd-scripts.

Two data files drive them: **`settings.env`** holds every choice the pod makes, and **`models.json`** lists every model file, grouped in profiles and training families.

## 2. Words used here and in the code

| Word | Meaning |
|---|---|
| Pod | A Runpod container with a GPU, rented by the second. Its container disk holds everything and is erased when the pod is terminated. |
| Template | The public Runpod template `4i789znkrd` ("AInVFX bootcamp ComfyUI"): the container image, the ports, the environment variables and the start command. Every pod is created from it, in the browser or by `pod.py`. |
| Release | A git tag `vX.Y.Z` of this repository. |
| Release path | `AINVFX_REPO_RAW`: the raw GitHub URL of this repository at a release, such as `https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/v1.2.0`. The template sets it, and a pod takes `bootstrap.sh`, `models.json` and `train.py` from it. |
| Profile | A named set of model files in `models.json`: `image`, `3d`, `video`, `train`. `AINVFX_PROFILE` joins several with `+`. |
| Training family | The base files of one kind of training (`zimage`, `qwen_edit`, `moss`, `ltx`), named by role in the `training` part of `models.json`. The first run that needs them downloads them. |
| Recipe | One kind of LoRA in `train.py`: its training family, its data, its trainer and its settings. |
| Run | One training job, written into `ComfyUI/output/train/<run>/`. A chain (`--then`) is one job with several stages, one run each. |
| Record | What `pod.py` keeps about a pod in `~/.ainvfx-runpod/config.json`, under the pod's tag. |
| Tag | The short name commands take for a pod: its profile (`image+3d`) or the name given to `up --name`. |
| Log words | The fixed words `bootstrap.sh` prints at each milestone (`COMFYUI UP`, `MODELS DONE`, `TRAINERS OK`, `READY`, `FAILED`). `pod.py` reacts to them. |

## 3. The big picture

```
YOUR COMPUTER                        RUNPOD                             THE POD
pod.py                               api.runpod.io                      start command (from the template):
  settings.env + settings.local.env    REST v2: catalog, create,          curl $AINVFX_REPO_RAW/bootstrap.sh
  models.json (disk size)              status, terminate, log stream      bash bootstrap.sh
  ~/.ainvfx-runpod/config.json         GraphQL: host maintenance,           models.json, train.py from the same folder
                                       account credit                       ComfyUI on port 8188
                                     proxy https://<id>-8188 ---------->  ComfyUI (log copy, /upload/image, /history)
                                     proxy https://<id>-8888 ---------->  JupyterLab (pull, push --dataset)
                                                                          musubi-tuner, sd-scripts, `train` (profile train)
```

`pod.py` sends the settings as the pod's environment variables, and Runpod merges them with the template's, key by key. A browser deploy gets the template's values. Both paths then run the same `bootstrap.sh`, so they give the same pod.

## 4. Design rules

- **`pod.py` is one file, standard library only, Python 3.8 or newer**, on Windows, macOS and Linux. A participant clones the repository and runs it, with nothing to install. CI checks the Python 3.8 syntax (vermin) and runs the pod tests on Python 3.8.
- **`bootstrap.sh` is one file.** Its Python helpers are inline heredocs, so a pod needs one download to start.
- **An optional step never stops a pod.** A slow disk, a gated file without a licence, a failed custom node, failed trainers or a failed test image give a `WARNING`, and the pod still reaches `READY`. Only a pod that cannot work (no Python, no PyTorch, no GPU from PyTorch, no ComfyUI) prints `FAILED` and holds: it stays up, so its log can be read.
- **Every step can run again.** An install marker per set of install settings (ComfyUI tag, Python, PyTorch, custom nodes), files already present, one marker per trainer commit: a restarted pod skips what it has. `bash bootstrap.sh models` and `bash bootstrap.sh trainer` repeat one step on a live pod.
- **`pod.py` reads the log.** Every bootstrap line starts with `[AINVFX]`. The log words are fixed, because `pod.py` and the tests react to them: a changed word changes `bootstrap.sh`, `pod.py` and `tests/test_pod.py` in the same commit. The trainers end on `TRAINERS OK`, never on a line containing `READY`, because `pod.py` stops following on that word.
- **One place per value.** A setting lives in `settings.env`, a model in `models.json`, a recipe in `train.py`, a trainer pin in `bootstrap.sh`. Where a copy must exist (the defaults in `bootstrap.sh`, the tables of README.md), `tests/test_repo.py` checks that it agrees.
- **Secrets stay out of files and logs.** The API key lives in `~/.ainvfx-runpod/config.json` (readable by its owner only) or in `RUNPOD_API_KEY`. The Hugging Face token is a Runpod secret: the pod receives a reference, which Runpod resolves when the pod starts. One deliberate exception: the `READY` line carries JupyterLab's token, so that `pull` and `push --dataset` can use JupyterLab. The pod's address therefore stays private.
- **Messages, never tracebacks.** A user error ends in one sentence that says what to do. `AINVFX_DEBUG=1` shows the trace.
- **Plain words.** Messages and documents use short sentences, one idea each: many users read English as a second language.

## 5. The life of a pod

1. **`up`** reads `settings.env`, then `settings.local.env`, then `--set`. It computes the disk from `models.json`, reads the GPU catalog, picks a machine (README, "How `up` picks a GPU") and creates the pod through REST v2 with the template and those values.
2. It reads the host's maintenance window through GraphQL, and replaces a host under maintenance or due within 24 hours (`AINVFX_MAINT_HOURS`).
3. The container starts. The template's start command fetches `bootstrap.sh` from `AINVFX_REPO_RAW` (default: `main`) and runs it. If GitHub cannot be reached, it runs Runpod's own `/start.sh`, so SSH and JupyterLab still start.
4. **`bootstrap.sh`** runs its steps (section 6). Meanwhile `pod.py` follows Runpod's log stream (`GET /pods/{id}/logs`, Server-Sent Events). The stream can stay open without delivering the last lines, so `pod.py` reopens it every 60 seconds a few seconds back, and reads the pod's own copy of the log through the ComfyUI proxy every 15 seconds. It stops on `READY` or `FAILED`, and warns when no container has started 8 minutes after creation.
5. The user works. **`push`** sends files to ComfyUI's `/upload/image`; with `--dataset`, through JupyterLab's contents API, in 8 MiB chunks written to a `.part` file and renamed at the end. **`pull`** walks the output folder through JupyterLab, or falls back to the files ComfyUI's history lists.
6. **`down`** pulls, terminates, then reports the pod's cost and the credit left. Every command first forgets the records of pods gone from the account.

## 6. What `bootstrap.sh` does

| Step | Work | Log words |
|---|---|---|
| 0 | Runpod's `/start.sh` in the background: SSH and JupyterLab. Skipped when JupyterLab already runs (a second run on a live pod) | |
| 1 | Health check: the GPU, its memory and driver (580 or newer), CPU, RAM, free disk, disk write and read speed (`AINVFX_HEALTHCHECK`) | `WARNING` on a bad value |
| 2 | uv, Python `AINVFX_PYTHON` in `/workspace/venv`, PyTorch from `AINVFX_TORCH_INDEX` (PyPI as a second route for stable CUDA 13.0), ComfyUI at `AINVFX_COMFY_TAG` (git init and a shallow fetch, so a folder that already holds models works), its requirements and the Manager's, `huggingface_hub`, the custom nodes of `AINVFX_CUSTOM_NODES`, then a PyTorch check on the GPU | `FAILED` when Python, PyTorch or ComfyUI cannot work |
| 3 | ComfyUI on port 8188, then one request through the public proxy | `COMFYUI UP`, `PROXY OK` |
| 4 | The models plan: every file of the profile with its state (to download, present, gated without a token, waiting for a licence, moved). Then each file through `hf download` into a staging folder, moved into place when complete, with the time left | `MODELS DONE`, `WAITING`, `SKIPPED` |
| 4b | With `train`: both trainers, the fork's sampling fix, `train.py` and the `train` command (section 7) | `TRAINERS OK` |
| 5 | The self-test: one MOSS-Image poster through ComfyUI's API, core nodes only, skipped without the image profile's MOSS-Image files | `SELFTEST OK` |
| end | The gated files still missing, each with its page; then the ComfyUI and JupyterLab links | `ACTION NEEDED`, `READY` |

**Why this container image.** The template uses `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, the image of Runpod's own PyTorch template. Runpod keeps it on its hosts, so the container starts within seconds; another image is fetched from Docker Hub at the host's speed. Its PyTorch and CUDA toolkit are not used: `bootstrap.sh` installs PyTorch built for CUDA 13.0, which carries its own CUDA libraries.

**Why the log sits in ComfyUI's input folder.** `/workspace/ComfyUI/input/ainvfx/bootstrap.log` is served by ComfyUI at `/view?filename=bootstrap.log&subfolder=ainvfx&type=input`, so `pod.py` can read it when Runpod's log stream stays silent.

**A placeholder token.** A pod deployed without the `huggingface_token` secret receives the literal text `{{ RUNPOD_SECRET_huggingface_token }}`. `bootstrap.sh` and `train.py` drop it: Hugging Face refuses even public files to an invalid token.

## 7. Training

**The trainers** (`install_trainers` in `bootstrap.sh`). musubi-tuner in AkaneTendo25's fork trains LTX 2.5, Z-Image and Qwen-Image-Edit; sd-scripts trains MOSS-Image. Each sits at a pinned commit (`MUSUBI_COMMIT`, `SDSCRIPTS_COMMIT`) in its own Python 3.12 environment, with PyTorch 2.8.0 for CUDA 12.8, the build the recipes were proven on. Two environments, because sd-scripts pins a transformers release that the fork does not accept. ComfyUI's environment (Python 3.13, CUDA 13.0) is never touched: every uv call names its environment. `patch_fork` fixes one known bug of the fork at that commit: a failed training sample left the model in evaluation mode, so the next step ran without gradient checkpointing and out of memory.

**A run** (`train.py`).

1. `train <recipe> <run> --dataset ...` checks everything before any GPU time: the dataset's folders, files, captions and pairs (`check_blocks`), the sample prompts (`make_prompts`), the base files (`ensure_models`: downloaded once into `ComfyUI/models`, with a disk check first).
2. It writes the job record `ComfyUI/output/train/.job.json`, asks ComfyUI to free the GPU (`/free`), starts itself in the background in a new session (`train _run`), then follows the log.
3. The background process runs each stage's commands in order: the trainer's caching scripts (musubi-tuner), then `accelerate launch` of the training script. Every 30 seconds it publishes the new checkpoints (`publish`): Z-Image's converted for ComfyUI once untouched for 15 seconds, LTX's `.comfy.safetensors` copies, the others as they are, each linked under `ComfyUI/models/loras/train/<run>/`.
4. `log` and `status` read the job record and the run's `train.log`. `stop` signals the whole process group: SIGTERM, then SIGKILL after 30 seconds.

A recipe is data (`RECIPES`): argument lists with `{name}` placeholders, filled per run (`fill`). `--steps`, `--lr`, `--rank` and `--from` edit those lists (`set_flag`, `with_rank`). `train caption` writes captions with Qwen3-VL-8B-Instruct (downloaded at first use, 17.6 GB).

## 8. Where files go

On the user's computer:

| Path | What |
|---|---|
| `~/.ainvfx-runpod/config.json` | The API key, the data center preferences and the pod records |
| `settings.local.env` | The user's own settings, next to `settings.env`, ignored by git |
| `outputs/<pod name>/` | What `pull` downloads (`AINVFX_OUTPUTS` moves it) |

On the pod:

| Path | What |
|---|---|
| `/tmp/ainvfx-bootstrap.sh` | The bootstrap that runs; `models` and `trainer` call it again |
| `/workspace/venv` | ComfyUI's Python environment, with its install marker |
| `/workspace/ComfyUI` | ComfyUI, its models, input and output folders |
| `/workspace/ComfyUI/input/ainvfx/bootstrap.log` | The log (linked as `/workspace/ainvfx-bootstrap.log`) |
| `/workspace/comfy.log`, `/workspace/runpod-start.log` | The logs of ComfyUI and of Runpod's `/start.sh` |
| `/workspace/models.json` | This pod's models list, which `train.py` also reads |
| `/workspace/musubi-tuner`, `/workspace/sd-scripts` | The trainers, each with its `.venv` |
| `/workspace/ainvfx/train.py`, `/usr/local/bin/train` | The training command |
| `/workspace/dataset/<name>` | The datasets; the trainers' caches go to `/workspace/cache` |
| `/workspace/ComfyUI/output/train/<run>` | Each run; `.job.json` and `.job.log` there describe the last job |

## 9. Code map

### `pod.py`, in file order

| Part | Main functions | Notes |
|---|---|---|
| Constants | `VERSION`, `TEMPLATE_ID`, `PROFILES`, the `AINVFX_*` tuning numbers | Every delay and limit can be changed by an environment variable, which the tests use |
| Settings and models | `read_env_file`, `load_settings`, `parse_profiles`, `read_models`, `profile_spec` | `profile_spec` sizes the disk |
| Config and records | `load_config`, `save_config`, `restrict`, `remember_pod` | `restrict` makes the config file private, on Windows too |
| HTTP | `request`, `rp` (REST v2), `graphql`, `ApiError` | A 429 is retried five times (Retry-After, or 2 to 32 seconds) |
| Pods and names | `get_pod`, `list_pods`, `which_pod`, `forget_gone`, `named_for` | A name matches exactly |
| GPU choice | `read_catalogs`, `make_offers`, `fits_profile`, `pick_candidates`, `wait_for_stock` | `--retry` reads the catalog every 15 seconds |
| Host and money | `pod_machine`, `maintenance_of`, `host_check`, `account_money` | The two GraphQL reads |
| Logs | `log_stream`, `follow_logs`, `ImagePull` | `ImagePull` turns Docker's layer lines into one progress line |
| Creation | `create_pod`, `try_create` | |
| Commands | `cmd_setup`, `cmd_up`, `cmd_status`, `cmd_logs`, `cmd_open`, `cmd_push`, `cmd_pull`, `cmd_down`, `cmd_list`, `cmd_ssh`, `cmd_doctor` | |
| Files | `jupyter_tree`, `jupyter_upload`, `push_dataset`, `history_files`, `fetch_to`, `pull` | Downloads go through a `.part` file |
| Entry | `main` | argparse |

### `bootstrap.sh`, in file order

The header (steps, variables, modes), then the settings and paths, the helpers (`say`, `warn`, `fail`, `hold`, `uvi`, `install_torch`, `hf_state`, `access_summary`), the trainers (`install_repo`, `patch_fork`, `install_trainers`, with their pins), the modes (`models`, `trainer`), the token check, steps 0 to 5, and `READY`.

### `train.py`, in file order

The paths, the recipes (`COMMON`, the `LTX_*` parts, `RECIPES`, `DATA`, `CAPTION_PROMPTS`), the helpers, the job record, the base files, the datasets, the sample prompts, a run (`parse_stages`, `stage_commands`, `publish`, `cmd_train`, `run_job`, `run_stage`, `follow`), the other commands, the captions, and `main`.

### Data files

- **`models.json`**: `profiles` (lists of file keys; `@name` includes another profile), `training` (per family, a file key per role), `files` (per key: `repo`, `path`, `dir` under `ComfyUI/models`, `gb`, and `gated` when the repository needs a token).
- **`settings.env`**: one commented `AINVFX_*` line per setting. An empty value means the template's value, or `bootstrap.sh`'s default.

## 10. Tests

```bash
python -m unittest discover -s tests                    # every test: about 5 minutes, nothing billed
python -m unittest discover -s tests -p "test_repo.py"  # under a second: the files agree with each other
```

| Module | What it proves | Needs |
|---|---|---|
| `test_pod.py` | `Helpers`: the pure functions (settings files, profiles and disks, offers and prices, maintenance windows, names and records, log parsing, dataset plans). `EndToEnd`: every command against a fake Runpod API (REST v2, GraphQL, the catalog, a log stream that behaves as the real one), a fake ComfyUI and a fake JupyterLab, run through subprocess with `HOME` in a temporary folder. Each test's docstring names the incident it reproduces | Python 3.8 or newer |
| `test_train.py` | `train.py` against fake trainers in a temporary `/workspace`: recipes and their base files, dataset configs and checks, sample prompts, stages, a run end to end, a chain, a MOSS-Image run, a failure, `stop`, captions in a dry run, and README's recipe table | Python 3.11 or newer (skipped below) |
| `test_repo.py` | The files agree: `models.json` is whole, `pod.py` knows every profile, README's tables match `models.json` and `settings.env`, `bootstrap.sh`'s defaults match `settings.env`, the self-test's files are in the image profile, `bootstrap.sh` turns a Google Drive, Dropbox or GitHub share link into the same file link as `pod.py` | Python 3.8 or newer (`sed` for the last test) |

**CI** (`.github/workflows/tests.yml`) runs on every push and pull request: pyflakes, shellcheck and vermin; the whole suite on Python 3.12 and 3.14; the pod and repository tests on Python 3.8.

**What only a pod can test.** `bootstrap.sh` runs only on a real pod (shellcheck checks it statically), and the real trainers only on a GPU. When a change touches them, run this on an RTX PRO 6000 pod before the release, with the branch's address in `AINVFX_REPO_RAW`:

1. `python pod.py up image+train --set AINVFX_REPO_RAW=https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/<branch>`: the log reaches `TRAINERS OK`, `SELFTEST OK` and `READY`.
2. `train recipes`, then a 10-image dataset through `push --dataset`, then `train caption`.
3. `train zimage t1 --steps 250`, `train moss m1 --steps 250`, and a two-pair `ltx_iclora` run of 250 steps; each LoRA loads in ComfyUI.
4. `pull` brings the runs home, `down` terminates the pod.

## 11. How to make common changes

- **Add a model to a profile.** Add the file to `files` in `models.json` (repo, path, folder, the size read on Hugging Face, `gated` when needed), then name it in the profile. Update README's Profiles table with the sizes that `test_repo.py` reports.
- **Add a profile.** `profiles` in `models.json`, `PROFILES` and `PROFILE_ORDER` in `pod.py` (its memory and GPU), and a row in README's Profiles table.
- **Move the default ComfyUI version.** `settings.env`, the default in `bootstrap.sh` and README's settings table (the tests check all three), then the template's `AINVFX_COMFY_TAG`. Run the course files on a pod first.
- **Add a setting.** A commented line in `settings.env`, its default in `bootstrap.sh`, a row in README's settings table.
- **Add a recipe.** An entry in `RECIPES`, a training family in `models.json` when it needs new base files, a fake script in `test_train.py` when it calls a new training script, and a row in README's recipe table.
- **Move a trainer.** `MUSUBI_COMMIT` or `SDSCRIPTS_COMMIT` in `bootstrap.sh`. Check that `patch_fork` still finds its anchor. A running pod reinstalls with `bash /tmp/ainvfx-bootstrap.sh trainer`.
- **Change a log word.** `bootstrap.sh`, `pod.py` and the fake logs at the top of `tests/test_pod.py`, in one commit.

## 12. Releases and the template

A release is a tag, and pods run the release the template names. A change on `main` therefore reaches no new pod until the template moves.

1. Set `VERSION` in `pod.py`. Run the tests, and the pod check of section 10 when `bootstrap.sh` or `train.py` changed.
2. Commit, then tag and push `main` and the tag together, so that both arrive or neither does:

   ```bash
   git tag -a vX.Y.Z -m 'ainvfx-runpod X.Y.Z'
   git push --atomic origin main vX.Y.Z
   gh release create vX.Y.Z --verify-tag --latest --title 'ainvfx-runpod X.Y.Z' --notes-file <notes>
   ```

3. Move the template: set its release path, `AINVFX_REPO_RAW`, to `https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/vX.Y.Z`. Update its README field when the profiles, sizes or defaults changed (the console holds that field; the REST API does not).

A published tag never moves: a fix is the next patch release.

**The template's settings.**

| Field | Value |
|---|---|
| Name | AInVFX bootcamp ComfyUI |
| Container image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container disk | 200 GB: any single profile fits; `up` sizes the disk for a combination |
| Ports | `8188/http` (ComfyUI), `8888/http` (JupyterLab), `22/tcp` (SSH) |
| CUDA versions | 13.0 and newer (extend the list when Runpod adds one) |
| Environment | `AINVFX_PROFILE=image`, `AINVFX_COMFY_TAG=v0.43.0`, `AINVFX_REPO_RAW=<the release path>`, `HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}` |

The start command fetches `bootstrap.sh` from `AINVFX_REPO_RAW`, and falls back to Runpod's `/start.sh` when GitHub cannot be reached:

```bash
bash -c "curl -fsSL --retry 5 --retry-delay 3 ${AINVFX_REPO_RAW:-https://raw.githubusercontent.com/AInVFX/ainvfx-runpod/main}/bootstrap.sh -o /tmp/ainvfx-bootstrap.sh && bash /tmp/ainvfx-bootstrap.sh || /start.sh"
```

The variable appears once, so the command works whether Runpod runs it directly or through a shell first.

## 13. Known limits and planned work

- **GraphQL, for two reads.** `pod.py` reads the pod's host with its maintenance window (`MACHINE_QUERY`) and the account's credit (`MONEY_QUERY`) through Runpod's GraphQL API, because REST v2 offers neither: its specification (`https://api.runpod.io/v2/openapi.json`, version 2.0.0, read on 29 October 2026) has no maintenance window, no balance and no spend rate. Runpod's own CLI shows the same GraphQL fields (`clientBalance`, `currentSpendPerHr` in `runpodctl user`). Runpod deprecates GraphQL for integrators in early 2027. When GraphQL stops answering, `pod.py` keeps working: `up` skips the host check and the credit line, and `test_up_and_down_work_without_graphql` proves it. Move both reads to REST v2 once Runpod adds them there.
- **The `moss` recipe** follows sd-scripts' MOSS-Image guide and has not yet run on a pod.
- **FLUX 3 Image Klein** has no recipe: neither pinned trainer supports it.
- **Custom nodes are not pinned.** `AINVFX_CUSTOM_NODES` clones each node's default branch at its latest commit.
- **The pod's address is its only protection.** ComfyUI has no login, and the log it serves holds JupyterLab's token. ComfyUI's own origin check refuses a click from another site, which is why the console's Connect button can show an empty 403 page: kept on purpose.
