# omero-screen


## Status

Version: ![version](https://img.shields.io/badge/version-0.1.1-blue)

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- omero-screen: an end to end high content image analysis pipeline
- omero-utils: helper functions to work with the omero-py API
- omero-screen-napari: Napari plugins to interact with the data

## Installation

```bash
# Install uv
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/Helfrid/omero-screen.git
# Change into the project directory
cd omero-screen
# Create and activate virtual environment
uv sync --dev
source .venv/bin/activate

```

## Project Structure

```
.
├── .env.development
├── .env.production
├── .github
│   └── workflows
│       ├── ci.yml
│       └── release.yml
├── .gitignore
├── .python-version
├── CHANGELOG.md
├── LICENSE
├── README.md
├── logs
│   └── app.log
├── packages
│   ├── omero-screen-napari
│   │   ├── README.md
│   │   ├── pyproject.toml
│   │   └── src
│   └── omero-utils
│       ├── CHANGELOG.md
│       ├── README.md
│       ├── pyproject.toml
│       └── src
├── pyproject.toml
├── src
│   └── omero_screen
│       ├── __init__.py
│       └── __pycache__
├── test.log
├── tests
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   └── conftest.cpython-312-pytest-8.3.4.pyc
│   ├── conftest.py
│   ├── e2e
│   │   └── omero_utils
│   └── unit
│       ├── omero_screen
│       └── omero_utils
└── uv.lock
```

## Versioning

This project uses [Semantic Versioning](https://semver.org/) and [Conventional Commits](https://www.conventionalcommits.org/).

Version bumping occurs via CI when pushing the commits.

Major: Incremented for breaking changes, indicated by BREAKING CHANGE: or ! in commit messages.

- Example: feat!: update API endpoint to v2 or fix: remove legacy support BREAKING CHANGE: old endpoint removed

Minor: Incremented for new features, indicated by feat in the commit message.

- Example: feat: add support for OAuth2 authentication

Patch: Incremented for bug fixes, indicated by fix in the commit message.

- Example: fix: correct typo in login validation

By default only the root package omero-screen is updated. To bump the version of the other packages indicate the scope in the commit message

- Example: feat(omero-utils): new feature

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`cz commit`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Authors

Created by Helfrid Hochegger
Email: hh65@sussex.ac.uk

## Dependencies

Requires Python 3.12 or greater

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- References
