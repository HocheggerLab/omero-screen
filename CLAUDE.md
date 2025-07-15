# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

Use `uv` for package management:
```bash
# Install dependencies and activate virtual environment
uv sync --dev
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install
```

## Code Quality Commands

Run these commands to maintain code quality:
```bash
# Run all unit tests
pytest -v

# Run specific test modules
pytest tests/unit_tests/omero_screen_tests
pytest tests/unit_tests/omero_utils_tests

# Run linting and formatting
ruff check src/
ruff format src/

# Run type checking
mypy src/

# Run end-to-end tests manually
omero-integration-test e2e_excel
omero-integration-test e2e_connection
```

## Project Architecture

This is a monorepo with a workspace structure containing multiple packages:

### Core Packages
- **omero-screen**: Main package for high-content image analysis pipeline
- **omero-utils**: Helper functions for OMERO-py API interactions
- **omero-screen-napari**: Napari plugins for data interaction
- **omero-screen-plots**: Visualization and plotting utilities
- **cellview**: Database management and data export tools

### Environment Configuration
The project uses environment-specific configuration files (`.env.development`, `.env.production`, `.env.e2etest`). The `config.py` module handles loading these configurations and provides structured logging.

Required environment variables:
- `USERNAME`, `PASSWORD`, `HOST`: OMERO server credentials
- `LOG_LEVEL`, `LOG_FILE_PATH`: Logging configuration
- `ENABLE_CONSOLE_LOGGING`, `ENABLE_FILE_LOGGING`: Logging toggles

### OMERO Test Server
For development, use the test server management script:
```bash
# Start/stop/restart test server
./scripts/manage_test_server.sh start
./scripts/manage_test_server.sh stop

# Load test plates
./scripts/load_plates.sh -d /path/to/plates -x
```

## Testing Strategy

### Unit Tests
Located in `tests/unit_tests/` with comprehensive fixtures in `conftest.py`:
- Mock OMERO connections and objects
- Environment variable management
- Automatic cleanup procedures
- Session-scoped fixtures for expensive operations

### End-to-End Tests
Located in `tests/e2e_tests/` - run against real OMERO test server:
- Use dedicated `.env.e2etest` configuration
- Test complete workflows from data import to analysis
- Manual execution via `omero-integration-test` command

## Entry Points

The project provides these CLI commands:
- `omero-screen`: Main application entry point
- `omero-integration-test`: E2E test runner
- `cellview`: Database management tool

## Code Style

- Python 3.12+ required
- Line length: 79 characters (enforced by ruff)
- Google docstring convention
- Strict mypy type checking enabled
- Pre-commit hooks for code quality

## Commit Conventions

Uses Conventional Commits with semantic versioning:
- `feat`: New features (minor version bump)
- `fix`: Bug fixes (patch version bump)
- `feat!` or `BREAKING CHANGE:`: Breaking changes (major version bump)
- Package-specific bumps: `feat(omero-utils): description`

Use `cz commit` for guided commit message creation.
