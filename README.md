# omero-screen



## Status

Version: ![version](https://img.shields.io/badge/version-0.2.2-blue)

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

## Test Server Setup

For local development and testing, you can run a separate OMERO test server that runs in parallel with your main OMERO server. The test server uses a different IP address (127.0.0.2) to avoid conflicts with your main OMERO server.

To manage the test server, use the provided script:

```bash
# Start the test server
./scripts/manage_test_server.sh start

# Check server status
./scripts/manage_test_server.sh status

# Stop the test server
./scripts/manage_test_server.sh stop

# Restart the test server
./scripts/manage_test_server.sh restart
```

The test server will be accessible at:
- OMERO.server: 127.0.0.2:4064

Default credentials:
- Username: root
- Password: omero

## Loading Test Data

To load test data into your OMERO server, you can use the `load_plates.sh` script. This script helps import plate data from a specified directory into your OMERO instance.

Basic usage:
```bash
# Show help and available options
./scripts/load_plates.sh

# Load plates from a specific directory
./scripts/load_plates.sh -d /path/to/plates -x

# Load plates with custom server settings
./scripts/load_plates.sh -d /path/to/plates -s -x
```

Options:
- `-x`: Execute the import (required to actually perform the import)
- `-d`: Specify the directory containing plate data (defaults to current directory)
- `-s`: Use custom server settings (prompts for host, port, and username)

The script will:
1. Check for an active OMERO session and log out if found
2. Connect to the OMERO server (using default or custom settings)
3. Create a new Project named "Screens"
4. Import all plates found in the specified directory

Note: The script expects plate data to be organized in a specific structure with `*/Images/Index.idx.xml` files.

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

## Development

This project uses [pre-commit](https://pre-commit.com/) to create actions to validate changes to the source code for each `git commit`.
Install the hooks for your development repository clone using:

    pre-commit install

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

## ENV variables and project setup

The project uses a flexible environment variable system that supports both development and production environments. Here's how it works:

### Environment Selection
- The system first checks for an `ENV` environment variable, defaulting to "development" if not set
- Environment-specific configurations are loaded from `.env.{ENV}` files (e.g., `.env.development`, `.env.production`)
- If no environment-specific file exists, it falls back to a default `.env` file

### Required Environment Variables
The following variables are required for proper operation:

1. **OMERO Server Configuration**:
   - `USERNAME`: OMERO server username
   - `PASSWORD`: OMERO server password
   - `HOST`: OMERO server host address

2. **Logging Configuration**:
   - `LOG_LEVEL`: The logging level (e.g., DEBUG, INFO, WARNING)
   - `LOG_FORMAT`: Format string for log messages
   - `ENABLE_CONSOLE_LOGGING`: Boolean to enable/disable console logging
   - `ENABLE_FILE_LOGGING`: Boolean to enable/disable file logging
   - `LOG_FILE_PATH`: Path to the log file (defaults to "logs/app.log")
   - `LOG_MAX_BYTES`: Maximum size of log files before rotation (defaults to 1MB)
   - `LOG_BACKUP_COUNT`: Number of backup log files to keep (defaults to 5)

### Logging Setup
The logging system is configured through the `get_logger()` function in `config.py`:
- Creates a hierarchical logger structure based on module names
- Supports both console and file logging
- Automatically creates log directories if they don't exist
- Implements log rotation to manage file sizes
- Suppresses verbose OMERO logs by default (set to WARNING level)

### OMERO Server Selection
The OMERO server connection is configured through environment variables:
- For development/testing, a local test server can be run on 127.0.0.2:4064
- Production server settings are configured through the environment variables
- The system validates all required server credentials before establishing connections

### Error Handling
- The system raises an `OSError` if:
  - No configuration files are found
  - Required environment variables are missing
  - Invalid logging configurations are provided

This setup allows for easy switching between different environments and provides robust logging capabilities while maintaining security through environment-based configuration.

## Tests

The project includes both unit tests and end-to-end (e2e) tests to ensure code quality and functionality.

### Unit Tests
Unit tests are organized in the `tests/unit_tests` directory and cover:
- `omero_screen_tests`: Tests for the main omero-screen package
- `omero_utils_tests`: Tests for utility functions
- `config_tests`: Tests for configuration management

Key features of the unit testing setup:
- Uses pytest as the testing framework
- Includes fixtures for common test scenarios
- Environment variable management for testing
- Mock OMERO server connections
- Cleanup procedures for test artifacts

### End-to-End Tests
End-to-end tests are designed to simulate production-like conditions while using a test server. They are located in `tests/e2e_tests` and include:
- `e2e_connection.py`: Tests for OMERO server connectivity
- `e2e_excel.py`: Tests for Excel file handling
- `e2e_pixelsize.py`: Tests for pixel size calculations
- `e2e_run.py`: Main test runner
- `e2e_setup.py`: Test environment setup

#### E2E Test Environment
The e2e tests use a dedicated environment configuration (`.env.e2etest`) that:
- Simulates production mode logging (saved but not shown on console)
- Connects to a test OMERO server
- Uses production-like settings for realistic testing

Example `.env.e2etest` configuration:
```bash
USERNAME=root
PASSWORD=omero
HOST=localhost
PROJECT_ID=1
DATA_PATH = 'omero-napari-data'

# Logging configuration
LOG_LEVEL=DEBUG
LOG_FILE_PATH=logs/app.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s
ENABLE_CONSOLE_LOGGING=False
ENABLE_FILE_LOGGING=True
LOG_MAX_BYTES=1048576        # 1MB
LOG_BACKUP_COUNT=5
```

#### Running E2E Tests
E2E tests are run manually using the `omero-integration-test` command:
```bash
# Run a specific e2e test
omero-integration-test testname

# Example: Run Excel handling tests
omero-integration-test e2e_excel
```

The e2e tests:
- Run against a dedicated test OMERO server instance
- Test complete workflows from data import to analysis
- Include cleanup procedures to maintain test environment
- Can be run in parallel with the main OMERO server
- Use production-like logging configuration for realistic testing

### Running Unit Tests
To run the tests:

```bash
# Run all unit tests
pytest -v

# Run specific unit test modules
pytest tests/unit_tests/omero_screen_tests
pytest tests/unit_tests/omero_utils_tests
```

### Test Configuration
The test suite uses:
- `conftest.py` for shared fixtures and configuration
- Environment-specific test settings
- Automatic cleanup of test artifacts
- Mock objects for external dependencies

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
