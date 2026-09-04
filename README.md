# OMERO-Screen

**End-to-end high-content bio-image analysis powered by AI.**

OMERO-Screen takes a 96-well immunofluorescence plate from an
[OMERO](https://www.openmicroscopy.org/omero/) server to a publication figure:
segmentation, feature extraction, cell-cycle assignment, storage,
classification and plotting as one reproducible workflow. It is built for cell
biologists running high-content screens.

### 📖 [Read the documentation →](https://hocheggerlab.github.io/omero-screen/)

[![version](https://img.shields.io/badge/version-0.11.0-blue)](https://github.com/HocheggerLab/omero-screen/releases)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> [!WARNING]
> **Active development.** OMERO-Screen is developed alongside ongoing research,
> so expect frequent changes and occasional breaking ones. A stable release is
> planned for later in 2026. Pin a commit if you need an analysis to stay
> reproducible.

```bash
omero-screen 1234 --stitch          # segment a plate, write results back to OMERO
cellview import plate 1234          # pull the measurements into a local database
```

## What makes it different

- **Rapid model adaptation** — builds on [Cellpose](https://www.cellpose.org/)
  and [Trackastra](https://github.com/weigertlab/trackastra), with a fast
  generation-and-deployment loop for classifiers. Models are selected from the
  command line, so an improved model swaps in without touching analysis code.
- **4i / cyclic IF and automated tracking, without the coding** — registration
  and tracking pipelines produce highly multiplexed single-cell data from a few
  commands.
- **Agentic AI integration** — the whole platform is driven by terminal
  commands, so AI agents can run it. Bundled agent skills execute the pipeline
  from plate analysis to figure production from natural-language prompts.
- **Data integration and management** — raw data, masks and analysis files stay
  together on an OMERO server; single-cell measurements are queried from a local
  DuckDB database. Terabytes stay organised, queryable and reproducible.

## Install

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/). Not tested on
Windows.

```bash
git clone https://github.com/HocheggerLab/omero-screen.git
cd omero-screen
uv sync --dev
```

Then create a `.env.development` file with your OMERO credentials and database
path. See
[Install & configure](https://hocheggerlab.github.io/omero-screen/user-guide/installation.html)
for the full list and the hardware requirements — in short, Cellpose 3 runs on
Apple silicon or NVIDIA, while Cellpose 4 and tracking want an NVIDIA GPU.

Verify the connection:

```bash
uv run omero-integration-test e2e_connection
```

## The packages

Six packages in one `uv` workspace. Each has its own documentation site with
generated API and CLI reference.

| Package | What it is |
|---|---|
| **omero-screen** | The analysis pipeline: segmentation, feature extraction, cell-cycle phases, flatfield correction, stitching, tracking |
| **[cellview](https://hocheggerlab.github.io/omero-screen/cellview/)** | DuckDB-backed single-cell database with a CLI and a Python API |
| **[omero-screen-plots](https://hocheggerlab.github.io/omero-screen/plots/)** | Statistical plots built for publication figures |
| **[omero-screen-napari](https://hocheggerlab.github.io/omero-screen/napari/)** | napari widgets for browsing plates, inspecting galleries and labelling training data |
| **[cellclass](https://hocheggerlab.github.io/omero-screen/cellclass/)** | CNN classifier training, sweeps and inference |
| **[omero-utils](https://hocheggerlab.github.io/omero-screen/utils/)** | Connection handling, attachments, annotations and image I/O |

## Documentation

| | |
|---|---|
| [Analyse your first plate](https://hocheggerlab.github.io/omero-screen/user-guide/first-plate.html) | One continuous journey from a plate ID to a figure |
| [How it works](https://hocheggerlab.github.io/omero-screen/user-guide/architecture.html) | Pipeline stages, data model, benchmarks |
| [CLI reference](https://hocheggerlab.github.io/omero-screen/reference/cli/) | Generated from the command definitions |
| [API reference](https://hocheggerlab.github.io/omero-screen/reference/) | Public Python API |

## Development

```bash
uv sync --dev
pre-commit install          # ruff, mypy and formatting run on every commit
pytest tests/unit_tests     # the full unit suite
```

A parallel OMERO server for integration tests runs at `127.0.0.2:4064`:

```bash
./scripts/manage_test_server.sh start|stop|status
```

To populate it, `scripts/load_plates.sh` imports plates into a "Screens"
project. It expects the Operetta layout, with `*/Images/Index.idx.xml` under
each plate directory. `-x` executes the import (without it the script is a dry
run), `-d` sets the source directory, and `-s` prompts for host, port and
username instead of using the defaults.

```bash
./scripts/load_plates.sh                        # show options
./scripts/load_plates.sh -d /path/to/plates -x  # import
```

### Versioning

[Semantic Versioning](https://semver.org/) with
[Conventional Commits](https://www.conventionalcommits.org/); versions are
bumped by CI. `feat:` → minor, `fix:` → patch, `!` or `BREAKING CHANGE:` →
major. Only the root package is bumped by default — name the scope to bump
another, e.g. `feat(omero-utils): ...`.

Use `cz commit` rather than `git commit` to get the format right.

### Contributing

Fork, branch, commit with `cz commit`, and open a pull request. Please run the
unit suite and let the pre-commit hooks pass before pushing.

## Citing

If OMERO-Screen contributes to work you publish, please cite it. A paper is in
preparation; until then, cite this repository.

## Authors

Built by **Helfrid Hochegger** and **Alex Herbert** in the
[Hochegger lab](https://www.sussex.ac.uk/lifesci/hocheggerlab/), University of
Sussex. Contact: hh65@sussex.ac.uk

## License

MIT — see [LICENSE](LICENSE).

OMERO-Screen is an independent project built on OMERO. OMERO is a trademark of
the University of Dundee.
