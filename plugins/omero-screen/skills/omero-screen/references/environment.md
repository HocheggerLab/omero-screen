# Install, configure, check

For an installed user (the commands are on the `PATH`, no source checkout).
For a developer checkout, see `references/developer.md`.

Full guide, with the expected output at each step:
https://hocheggerlab.github.io/omero-screen/ (Install & configure).

---

## Install

```bash
curl -LsSf https://raw.githubusercontent.com/HocheggerLab/omero-screen/main/install.sh | sh
```

The installer gets uv, downloads the latest release into
`~/.local/share/omero-screen/<release>` (`current` points at it), links the
commands into `~/.local/bin`, compiles (a quiet step that can take a while),
fetches the stock Cellpose models, then runs `omero-screen setup` and
`omero-screen doctor`. Running the same line again continues after a network
error or a timeout.

Installer options (environment variables before `sh`, e.g.
`curl … | OMERO_SCREEN_HOME=/work/me/os sh`):

| Variable | Effect |
|---|---|
| `OMERO_SCREEN_HOME` | install location (on a cluster: the work space, beside uv's cache) |
| `OMERO_SCREEN_HEADLESS` | `1` no napari GUI, `0` with it; default headless on a SLURM cluster |
| `OMERO_SCREEN_VERSION` | a specific release |
| `OMERO_SCREEN_NO_SETUP=1` | skip setup and doctor |

Update: `omero-screen self update [--check]`. Uninstall:
`omero-screen self uninstall [--purge]` (`--purge` also removes the settings
and password file; the CellView database, image cache and models are never
deleted). Ask the user before either.

## Configure and check

```bash
omero-screen setup         # site, OMERO user, password (owner-only file); on a cluster also the image cache
omero-screen doctor        # checks everything and prints a fix for each failure
omero-screen config show   # every setting in effect and where it comes from (password masked)
```

The user runs `setup` themselves, in their terminal: it asks for the password.
Never ask for the password in the chat, and never read or print
`~/.config/omero-screen/` (it holds the password file).

Settings live in `~/.config/omero-screen/config.toml`; `setup` writes it and
keeps any other settings already there. What a user may add by hand:

```toml
site = "sussex"

[omero]
username = "ab123"
# group = "lab-group"            # default: the user's OMERO default group

[segmentation]
# model_set = "hocheggerlab"     # after: omero-screen models pull hocheggerlab

[segmentation.models]            # per cell line, over the model set
# RPE = "RPE-1_Tub_Hoechst"

[paths]
# cache = "~/omero-cache"        # downloaded images; on a cluster, the work space
# cellview_db = "~/.cellview/cellview.duckdb"

[env]                            # any OMERO_SCREEN_* variable
# OMERO_SCREEN_USE_GPU = "0"
```

Precedence, highest first: command-line options, environment variables, a
developer `.env` file, the user config, the site profile (`sussex` holds the
server, stitch calibrations and model sets), built-in defaults.

A password still in the system keychain (setups before the password file):
`doctor` shows a warning; the fix is to run `omero-screen setup` again.

## Models

```bash
omero-screen models pull stock            # stock Cellpose 3 models (the installer does this)
omero-screen models pull hocheggerlab     # the lab's Cellpose models
omero-screen models publish NAME.pt       # upload a classifier (+ its .json) for --inference; ask first
```

With no models configured: Cellpose 4 (`cpsam`) on an NVIDIA GPU, stock
Cellpose 3 elsewhere. Every run records the models used on the plate
(namespace `omero-screen/provenance`).

## HPC cluster

Install on a login node (headless, automatically). Put the install and uv's
cache in the work space, not home:

```bash
export UV_CACHE_DIR=/mnt/lustre/users/<group>/$USER/.cache/uv
export OMERO_SCREEN_HOME=/mnt/lustre/users/<group>/$USER/omero-screen-install
```

(in `~/.bashrc`). Never run the pipeline on the login node: submit jobs with
`omero-screen-sbatch` (`references/pipeline.md`). Test a GPU node first:

```bash
sbatch -p gpu --gres=gpu:1 --mem=8G --time=00:10:00 -o doctor-%j.out \
  --wrap "$(command -v omero-screen) doctor"
```

`doctor` in that job should report `Compute device: CUDA (…)`.

## Useful environment variables

```bash
OMERO_SCREEN_USE_GPU=0            # force CPU
OMERO_SCREEN_LOG_LEVEL=DEBUG      # more detail in the log
OMERO_SCREEN_LOG_FILE=path|none   # where the file log goes
OMERO_SCREEN_CACHE_PATH=~/omero-cache
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Anything | `omero-screen doctor`, and fix what it reports |
| `command not found: omero-screen` | Open a new terminal (the installer updated `PATH`), or run `uv tool update-shell` |
| Can't connect to OMERO | VPN; then `omero-screen setup` |
| A setting isn't what you expect | `omero-screen config show` names its source |
| `Failed to hardlink files` during install | Install and uv's cache are on different filesystems: set `OMERO_SCREEN_HOME` beside the cache |
| `Compute device: CPU only` on the cluster | That's the login node; check in a GPU job (above) |
